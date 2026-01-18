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

# --- IMPORT YOUR NETWORKS ---
from networks import TinyEncoder, Predictor

# --- R9700 OPTIMIZED CONFIG ---
BATCH_SIZE = 256       # Increased to saturate 32GB VRAM
EPOCHS = 50            # Increased for fine-tuning
LR = 5e-4              # Increased to match larger batch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_HORIZON = 15      # How many steps into the future to predict
SEQUENCE_LEN = PRED_HORIZON + 1 
ENCODER_PATH = "./models/encoder_mixed_final.pth" 
NUM_WORKERS = 8        # Safe limit for shared memory
MAX_SEQS_PER_WORKER = 5000 

# --- DATASET ---
class ShardedSeqDataset(IterableDataset):
    def __init__(self):
        # Load BOTH Race and Recovery to learn "Stay on track" AND "Get back on track"
        self.files = sorted(glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz"))
        self.epoch = 0 
        
        print(f"--- Predictor Training Setup ---")
        print(f"Found {len(self.files)} trajectory files.")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Shuffle files uniquely per epoch to prevent static batches
        rng = random.Random(self.epoch + worker_id + 999)
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)
        
        buffer = []
        
        for f in my_files:
            try:
                with np.load(f) as data:
                    # Handle different naming conventions in .npz
                    if 'states' in data: 
                        obs = data['states']
                        act = data['actions']
                    elif 'obs' in data: 
                        obs = data['obs']
                        act = data['action']
                    else: continue
                
                # Filter short sequences
                if len(obs) < SEQUENCE_LEN: continue
                
                # Resize if necessary (CPU side, done once per file)
                if obs.shape[1] != 64:
                    obs = np.array([cv2.resize(img, (64, 64)) for img in obs])

                # Sliding Window Generation
                # Stride of 4 reduces redundancy and speeds up training 4x
                stride = 4 
                for t in range(0, len(obs) - SEQUENCE_LEN, stride):
                    o_seq = obs[t : t + SEQUENCE_LEN]
                    a_seq = act[t : t + SEQUENCE_LEN - 1] # Actions are N-1
                    buffer.append((o_seq, a_seq))

                    # Yield chunks to prevent RAM spikes
                    if len(buffer) >= 1000: 
                         rng.shuffle(buffer)
                         for b_o, b_a in buffer:
                             yield self._format(b_o, b_a)
                         buffer = []
            except Exception: pass
            
        # Yield any remaining items
        rng.shuffle(buffer)
        for b_o, b_a in buffer:
            yield self._format(b_o, b_a)

    def _format(self, o, a):
        # 0-1 Scaling (Matches Encoder Training)
        o_t = torch.from_numpy(o).float() / 255.0
        o_t = o_t.permute(0, 3, 1, 2) # T, C, H, W
        a_t = torch.from_numpy(a).float()
        return o_t, a_t

# --- VALIDATION ---
@torch.no_grad()
def validate(encoder, predictor, val_loader):
    encoder.eval()
    predictor.eval()
    total_loss = 0
    count = 0
    
    print("  >>> Validating (Autoregressive Rollout)...")
    for i, (obs, act) in enumerate(val_loader):
        if i >= 20: break # Validate on 20 batches
        
        obs, act = obs.to(DEVICE), act.to(DEVICE)
        B, T, C, H, W = obs.shape
        
        # Get Ground Truth Latents
        with torch.no_grad():
            z_gt = encoder(obs.view(-1, C, H, W)).view(B, T, -1)
            
        z_curr = z_gt[:, 0, :]
        batch_loss = 0
        
        # Pure Autoregressive Loop (No Teacher Forcing)
        for t in range(PRED_HORIZON):
            z_pred = predictor(z_curr, act[:, t, :])
            z_true = z_gt[:, t+1, :]
            
            batch_loss += F.mse_loss(z_pred, z_true)
            z_curr = z_pred # Use own prediction
            
        total_loss += batch_loss.item()
        count += 1
        
    avg = total_loss / count
    print(f"  >>> Val MSE: {avg:.5f}")
    predictor.train()

# --- TRAINING ---
def train():
    # Enable Kernel Autotuning for Speed
    torch.backends.cudnn.benchmark = True
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}")

    # 1. Load Frozen Encoder
    print(f"Loading Encoder from {ENCODER_PATH}...")
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    
    # 2. Setup Predictor
    predictor = Predictor().to(DEVICE)
    optimizer = optim.AdamW(predictor.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # 3. Data
    dataset = ShardedSeqDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 4. Loop
    print(f"Starting Training on {DEVICE} (Batch: {BATCH_SIZE}, LR: {LR})")
    
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        predictor.train()
        
        # Teacher Forcing Schedule: 1.0 -> 0.0 over 20 epochs
        tf_prob = max(0.0, 1.0 - (epoch / 20.0))
        
        pbar = tqdm(loader, desc=f"Ep {epoch+1} [TF={tf_prob:.2f}]")
        epoch_loss = 0
        steps = 0
        
        for obs, act in pbar:
            obs = obs.to(DEVICE, non_blocking=True)
            act = act.to(DEVICE, non_blocking=True)
            
            B, T, C, H, W = obs.shape
            
            with torch.amp.autocast('cuda'):
                # Encode ALL observations at once (Fastest)
                with torch.no_grad():
                    z_all = encoder(obs.view(-1, C, H, W)).view(B, T, -1)
                
                loss = 0
                z_curr = z_all[:, 0, :] # Start state
                
                # Rollout
                for t in range(PRED_HORIZON):
                    # Ground Truth next state
                    z_target = z_all[:, t+1, :]
                    
                    # Predict
                    z_pred = predictor(z_curr, act[:, t, :])
                    
                    # Loss: MSE (Magnitude) + Cosine (Direction)
                    mse = F.mse_loss(z_pred, z_target)
                    cos = 1.0 - F.cosine_similarity(z_pred, z_target, dim=1).mean()
                    
                    # Small weight on Cosine helps steering accuracy
                    loss += mse + (0.05 * cos)
                    
                    # Teacher Forcing Logic
                    if random.random() < tf_prob:
                        z_curr = z_target 
                    else:
                        z_curr = z_pred   
                
                loss = loss / PRED_HORIZON

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # End of Epoch
        if (epoch+1) % 5 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_ep{epoch+1}.pth")
            validate(encoder, predictor, loader)

    torch.save(predictor.state_dict(), "models/predictor_final.pth")
    print("Predictor Training Complete.")

if __name__ == "__main__":
    train()