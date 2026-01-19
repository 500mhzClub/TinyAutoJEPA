import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import glob
import os
import cv2
import random
from tqdm import tqdm

from networks import TinyEncoder, Predictor

BATCH_SIZE = 256       
EPOCHS = 50            
LR = 5e-4              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_HORIZON = 15      
SEQUENCE_LEN = PRED_HORIZON + 1 
ENCODER_PATH = "./models/encoder_mixed_final.pth" 
NUM_WORKERS = 8        

class ShardedSeqDataset(IterableDataset):
    def __init__(self):
        self.files = sorted(glob.glob("./data_*/*.npz"))
        self.epoch = 0 
        print(f"Found {len(self.files)} files.")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        rng = random.Random(self.epoch + worker_id + 999)
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)
        
        buffer = []
        for f in my_files:
            try:
                with np.load(f) as data:
                    # Load Obs, Actions, and optionally Speed
                    obs = data['states'] if 'states' in data else data.get('obs')
                    act = data['actions'] if 'actions' in data else data.get('action')
                    
                    # Try to load speed, else zeros
                    if 'speed' in data:
                        spd = data['speed']
                    else:
                        spd = np.zeros((len(obs),), dtype=np.float32)

                    if obs is None or act is None: continue
                    if len(obs) < SEQUENCE_LEN + 4: continue # +4 for stacking history
                    
                    if obs.shape[1] != 64:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])

                    # Stride 
                    stride = 4 
                    # We start at index 3 so we have history [0,1,2,3]
                    for t in range(3, len(obs) - SEQUENCE_LEN, stride):
                        
                        # Prepare sequence of STACKS
                        stack_seq = []
                        speed_seq = []
                        action_seq = []
                        
                        # We need a sequence of T steps.
                        # At each step k (relative to t), input is stack [t+k-3 ... t+k]
                        for k in range(SEQUENCE_LEN):
                            current_idx = t + k
                            # Stack frames: Oldest -> Newest
                            # stack indices: current_idx-3, -2, -1, 0
                            s_idx = [current_idx-3, current_idx-2, current_idx-1, current_idx]
                            
                            # (4, 64, 64, 3)
                            stack_imgs = obs[s_idx]
                            # Flatten to (12, 64, 64) happens in _format later
                            stack_seq.append(stack_imgs)
                            
                            # Speed at current frame
                            speed_seq.append(spd[current_idx])
                            
                            # Action at current frame (to get to next)
                            if k < SEQUENCE_LEN - 1:
                                action_seq.append(act[current_idx])

                        buffer.append((stack_seq, action_seq, speed_seq))

                        if len(buffer) >= 500: 
                             rng.shuffle(buffer)
                             for b in buffer: yield self._format(*b)
                             buffer = []
            except Exception: pass
            
        rng.shuffle(buffer)
        for b in buffer: yield self._format(*b)

    def _format(self, stack_list, act_list, speed_list):
        # stack_list is list of T arrays, each (4, 64, 64, 3)
        # Convert to Tensor (T, 4, 3, 64, 64) -> (T, 12, 64, 64)
        
        stack_np = np.array(stack_list) # (T, 4, 64, 64, 3)
        T, F_stack, H, W, C = stack_np.shape
        # Permute to (T, F, C, H, W) -> flatten F*C
        t_stack = torch.from_numpy(stack_np).float().div_(255.0)
        t_stack = t_stack.permute(0, 1, 4, 2, 3).reshape(T, 12, H, W)
        
        t_act = torch.from_numpy(np.array(act_list)).float()
        t_spd = torch.from_numpy(np.array(speed_list)).float()
        
        return t_stack, t_act, t_spd

@torch.no_grad()
def validate(encoder, predictor, val_loader):
    encoder.eval()
    predictor.eval()
    total_loss = 0
    count = 0
    
    for i, (obs, act, spd) in enumerate(val_loader):
        if i >= 20: break 
        obs, act, spd = obs.to(DEVICE), act.to(DEVICE), spd.to(DEVICE)
        
        B, T, C, H, W = obs.shape # C=12
        
        # Encoder takes 12-channel stacks
        z_gt = encoder(obs.view(-1, C, H, W)).view(B, T, 512, 8, 8)
            
        z_curr = z_gt[:, 0]
        
        for t in range(PRED_HORIZON):
            # Pass speed at current time step
            z_pred = predictor(z_curr, act[:, t], spd[:, t])
            z_true = z_gt[:, t+1]
            total_loss += F.mse_loss(z_pred, z_true).item()
            z_curr = z_pred 
            
        count += 1
        
    print(f"  >>> Val MSE: {total_loss/count if count else 0:.5f}")
    predictor.train()

def train():
    os.makedirs("models", exist_ok=True)
    
    print(f"Loading Encoder: {ENCODER_PATH}")
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    
    predictor = Predictor().to(DEVICE)
    optimizer = optim.AdamW(predictor.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    dataset = ShardedSeqDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    print(f"Starting Training...")
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        predictor.train()
        
        pbar = tqdm(loader, desc=f"Ep {epoch+1}")
        for obs, act, spd in pbar:
            obs = obs.to(DEVICE, non_blocking=True)
            act = act.to(DEVICE, non_blocking=True)
            spd = spd.to(DEVICE, non_blocking=True)
            
            B, T, C, H, W = obs.shape
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                     # Encode all stacks in sequence
                    z_all = encoder(obs.view(-1, C, H, W)).view(B, T, 512, 8, 8)
                
                loss = 0
                z_curr = z_all[:, 0] 
                
                for t in range(PRED_HORIZON):
                    z_target = z_all[:, t+1]
                    # Predict using latent + action + SPEED
                    z_pred = predictor(z_curr, act[:, t], spd[:, t])
                    
                    mse = F.mse_loss(z_pred, z_target)
                    cos = 1.0 - F.cosine_similarity(z_pred, z_target, dim=1).mean()
                    loss += mse + (0.05 * cos)
                    
                    # Teacher forcing
                    if random.random() < 0.5:
                        z_curr = z_target
                    else:
                        z_curr = z_pred   
                
                loss = loss / PRED_HORIZON

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        if (epoch+1) % 5 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_ep{epoch+1}.pth")
            validate(encoder, predictor, loader)

    torch.save(predictor.state_dict(), "models/predictor_final.pth")

if __name__ == "__main__":
    train()