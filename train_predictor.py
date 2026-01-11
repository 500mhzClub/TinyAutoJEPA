import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import glob
import os
import cv2
import random
import math
from tqdm import tqdm
from networks import TinyEncoder, Predictor

BATCH_SIZE = 64     
EPOCHS = 40         
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sequence Settings
PRED_HORIZON = 15    
SEQUENCE_LEN = PRED_HORIZON + 1 
ENCODER_PATH = "./models/encoder_mixed_final.pth" 

NUM_WORKERS = 8

MAX_SEQS_PER_WORKER = 20000 

class ShardedSeqDataset(IterableDataset):
    def __init__(self):
        self.files = sorted(glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz"))
        self.total_seqs = 0
        self.epoch = 0 # Track epoch for shuffling
        
        print(f"Predictor Training Setup")
        print(f"Scanning {len(self.files)} files for sequence counts...")
        
        # quick scan
        for f in tqdm(self.files):
            try:
                with np.load(f, mmap_mode='r') as d:
                    if 'states' in d: n = d['states'].shape[0]
                    elif 'obs' in d: n = d['obs'].shape[0]
                    else: continue
                    if n > SEQUENCE_LEN:
                        self.total_seqs += (n - SEQUENCE_LEN)
            except: pass
            
        print(f"Total Sequences Available: {self.total_seqs:,}")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # 1. Deterministic Shuffling based on Epoch
        # This ensures every epoch sees data in a different order
        rng = random.Random(self.epoch + worker_id + 1337)
        
        # Partition files
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files) # Shuffle files for this worker
        
        sequences_buffer = []
        
        for f in my_files:
            # Stop loading if buffer is full to preserve RAM
            if len(sequences_buffer) >= MAX_SEQS_PER_WORKER: 
                break
                
            try:
                with np.load(f) as data:
                    if 'states' in data: 
                        obs_raw = data['states']
                        act_raw = data['actions']
                    elif 'obs' in data: 
                        obs_raw = data['obs']
                        act_raw = data['action']
                    else: continue
                
                length = len(obs_raw)
                if length < SEQUENCE_LEN: continue

                # Resize if necessary (vectorized)
                if obs_raw.shape[1] != 64:
                    obs_raw = np.array([cv2.resize(img, (64, 64)) for img in obs_raw])

                valid_starts = range(0, length - SEQUENCE_LEN)
                
                for start_t in valid_starts:
                    if len(sequences_buffer) >= MAX_SEQS_PER_WORKER: break
                    
                    end_t = start_t + SEQUENCE_LEN
                    o_chunk = obs_raw[start_t:end_t]     # Shape: (Seq, 64, 64, 3)
                    a_chunk = act_raw[start_t:end_t-1]   # Shape: (Seq-1, ActionDim)
                    
                    sequences_buffer.append((o_chunk, a_chunk))
            except Exception as e: 
                # print(f"Error loading {f}: {e}")
                pass
            
        # Shuffle the buffer so we don't get highly correlated sequences (consecutive frames) in one batch
        rng.shuffle(sequences_buffer)
        
        for o_chunk, a_chunk in sequences_buffer:
            # Normalize and Permute here (Yield time)
            obs_seq = torch.from_numpy(o_chunk).float() / 255.0
            obs_seq = obs_seq.permute(0, 3, 1, 2) # T, C, H, W
            act_seq = torch.from_numpy(a_chunk).float()
            yield obs_seq, act_seq
            
    def __len__(self): 
        # Approximate length (since workers might drop data due to buffer limit)
        return self.total_seqs

@torch.no_grad()
def validate(encoder, predictor, val_loader):
    """Checks how well the model predicts WITHOUT teacher forcing (Autoregressive)"""
    encoder.eval()
    predictor.eval()
    total_loss = 0
    steps = 0
    
    # Check just 50 batches to save time
    for i, (obs_seq, act_seq) in enumerate(val_loader):
        if i > 50: break
        
        obs_seq, act_seq = obs_seq.to(DEVICE), act_seq.to(DEVICE)
        B, T_plus_1, C, H, W = obs_seq.shape
        
        # Encode inputs
        obs_flat = obs_seq.view(-1, C, H, W)
        z_flat = encoder(obs_flat)
        z_seq = z_flat.view(B, T_plus_1, -1)
        
        loss = 0
        z_current = z_seq[:, 0, :]
        
        # PURE AUTOREGRESSIVE (Hard mode)
        for t in range(PRED_HORIZON):
            action = act_seq[:, t, :]
            z_target = z_seq[:, t+1, :]
            
            z_next_pred = predictor(z_current, action)
            loss += nn.MSELoss()(z_next_pred, z_target)
            
            # In validation, always use own prediction
            z_current = z_next_pred
            
        total_loss += (loss.item() / PRED_HORIZON)
        steps += 1
        
    avg_loss = total_loss / steps if steps > 0 else 0
    print(f"  >>> Validation Loss (Autoregressive): {avg_loss:.6f}")
    
    encoder.eval() # Keep encoder frozen
    predictor.train() # Reset predictor to train mode
    return avg_loss

def train():
    print(f"Training Predictor on {DEVICE}")
    
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Encoder not found! Run train_encoder.py first.")

    #Setup Encoder -Frozen
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    #Setup Predictor
    predictor = Predictor().to(DEVICE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    #Data
    dataset = ShardedSeqDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Training Loop
    os.makedirs("models", exist_ok=True)
    # Estimate steps per epoch
    steps_per_epoch = len(dataset) // BATCH_SIZE 

    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch) # Update shuffling seed
        predictor.train()
        
        # Curriculum Learning: Teacher Forcing Ratio
        # Epoch 0: 1.0 (Always use Ground Truth) -> Epoch 40: 0.0 (Always use Prediction)
        tf_ratio = max(0.0, 1.0 - (epoch / (EPOCHS * 0.8))) 
        
        pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Ep {epoch+1} (TF={tf_ratio:.2f})")
        epoch_loss = 0
        batch_count = 0

        for obs_seq, act_seq in pbar:
            obs_seq = obs_seq.to(DEVICE, non_blocking=True)
            act_seq = act_seq.to(DEVICE, non_blocking=True)
            
            B, T_plus_1, C, H, W = obs_seq.shape
            
            # Flatten to encode all frames at once
            obs_flat = obs_seq.view(-1, C, H, W)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z_flat = encoder(obs_flat)
                    z_seq = z_flat.view(B, T_plus_1, -1)
                
                loss = 0
                z_current = z_seq[:, 0, :] 
                
                for t in range(PRED_HORIZON):
                    action = act_seq[:, t, :]    # Action taken at step t
                    z_target = z_seq[:, t+1, :]  # Resulting state at t+1
                    
                    z_next_pred = predictor(z_current, action)
                    loss += criterion(z_next_pred, z_target)
                    
                    # Teacher Forcing Logic:
                    # If random < ratio, use Ground Truth (Target) for next step
                    # Else, use the model's own Prediction
                    if random.random() < tf_ratio:
                        z_current = z_target
                    else:
                        z_current = z_next_pred
                
                # Normalize loss by horizon so gradients are stable
                loss = loss / PRED_HORIZON

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            curr_loss = loss.item()
            epoch_loss += curr_loss
            batch_count += 1
            pbar.set_postfix(loss=f"{curr_loss:.6f}")

        # End of Epoch
        if batch_count > 0:
            print(f"Epoch {epoch+1} Avg Train Loss: {epoch_loss/batch_count:.6f}")

        # Checkpoint & Validation
        if (epoch+1) % 5 == 0:
            validate(encoder, predictor, dataloader)
            torch.save(predictor.state_dict(), f"models/predictor_ep{epoch+1}.pth")

    torch.save(predictor.state_dict(), "models/predictor_final.pth")
    print("Predictor Training Complete.")

if __name__ == "__main__":
    train()