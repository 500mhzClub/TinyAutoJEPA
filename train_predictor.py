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
BATCH_SIZE = 256       
EPOCHS = 50            
LR = 5e-4              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_HORIZON = 15      
SEQUENCE_LEN = PRED_HORIZON + 1 
ENCODER_PATH = "./models/encoder_mixed_final.pth" 
NUM_WORKERS = 8        
MAX_SEQS_PER_WORKER = 5000 

# --- DATASET ---
class ShardedSeqDataset(IterableDataset):
    def __init__(self):
        # CHANGED: Look for .npz (The originals with actions) instead of .npy
        self.files = sorted(
            glob.glob("./data_expert/*.npz") + 
            glob.glob("./data_recover/*.npz") + 
            glob.glob("./data_random/*.npz")
        )
        
        print(f"--- Predictor Training Setup ---")
        if len(self.files) == 0:
            print("❌ ERROR: No .npz files found!")
            print("   Did you delete them? If so, you need to re-generate data.")
        else:
            print(f"✅ Found {len(self.files)} .npz files (Source Data).")

        self.epoch = 0 

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Shuffle files uniquely per epoch
        rng = random.Random(self.epoch + worker_id + 999)
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)
        
        buffer = []
        
        for f in my_files:
            try:
                # Load .npz (Context Manager for automatic closing)
                with np.load(f) as data:
                    obs, act = None, None

                    # Check for standard keys
                    if 'states' in data and 'actions' in data:
                        obs = data['states']
                        act = data['actions']
                    elif 'obs' in data and 'action' in data:
                        obs = data['obs']
                        act = data['action']

                    # Skip if missing required data
                    if obs is None or act is None:
                        continue

                    # Filter short sequences
                    if len(obs) < SEQUENCE_LEN: continue
                    
                    # Resize logic (Vectorized resize is hard, loop is okay for file-level chunks)
                    if obs.shape[1] != 64:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])

                    # Sliding Window Generation (Stride 4)
                    stride = 4 
                    for t in range(0, len(obs) - SEQUENCE_LEN, stride):
                        o_seq = obs[t : t + SEQUENCE_LEN]
                        a_seq = act[t : t + SEQUENCE_LEN - 1] # Actions are N-1
                        buffer.append((o_seq, a_seq))

                        if len(buffer) >= 1000: 
                             rng.shuffle(buffer)
                             for b_o, b_a in buffer:
                                 yield self._format(b_o, b_a)
                             buffer = []
            except Exception: pass
            
        rng.shuffle(buffer)
        for b_o, b_a in buffer:
            yield self._format(b_o, b_a)

    def _format(self, o, a):
        # 0-1 Scaling
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
        if i >= 20: break 
        
        obs, act = obs.to(DEVICE), act.to(DEVICE)
        B, T, C, H, W = obs.shape
        
        with torch.no_grad():
            z_gt = encoder(obs.view(-1, C, H, W)).view(B, T, -1)
            
        z_curr = z_gt[:, 0, :]
        batch_loss = 0
        
        for t in range(PRED_HORIZON):
            z_pred = predictor(z_curr, act[:, t, :])
            z_true = z_gt[:, t+1, :]
            
            batch_loss += F.mse_loss(z_pred, z_true)
            z_curr = z_pred 
            
        total_loss += batch_loss.item()
        count += 1
        
    if count > 0:
        avg = total_loss / count
        print(f"  >>> Val MSE: {avg:.5f}")
    else:
        print("  >>> Val MSE: N/A (No Data)")
    
    predictor.train()

# --- TRAINING ---
def train():
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
    
    # 3. Data Check
    dataset = ShardedSeqDataset()
    if len(dataset.files) == 0:
        print("❌ CRITICAL ERROR: No .npz files found. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 4. Loop
    print(f"Starting Training on {DEVICE} (Batch: {BATCH_SIZE}, LR: {LR})")
    
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        predictor.train()
        
        tf_prob = max(0.0, 1.0 - (epoch / 20.0))
        
        pbar = tqdm(loader, desc=f"Ep {epoch+1} [TF={tf_prob:.2f}]")
        epoch_loss = 0
        steps = 0
        
        for obs, act in pbar:
            obs = obs.to(DEVICE, non_blocking=True)
            act = act.to(DEVICE, non_blocking=True)
            
            B, T, C, H, W = obs.shape
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z_all = encoder(obs.view(-1, C, H, W)).view(B, T, -1)
                
                loss = 0
                z_curr = z_all[:, 0, :] 
                
                for t in range(PRED_HORIZON):
                    z_target = z_all[:, t+1, :]
                    z_pred = predictor(z_curr, act[:, t, :])
                    
                    mse = F.mse_loss(z_pred, z_target)
                    cos = 1.0 - F.cosine_similarity(z_pred, z_target, dim=1).mean()
                    
                    loss += mse + (0.05 * cos)
                    
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
            
        if (epoch+1) % 5 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_ep{epoch+1}.pth")
            validate(encoder, predictor, loader)

    torch.save(predictor.state_dict(), "models/predictor_final.pth")
    print("Predictor Training Complete.")

if __name__ == "__main__":
    train()