import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import glob
import os
import cv2
import random
from tqdm import tqdm
from networks import TinyEncoder, Predictor

# --- CONFIGURATION ---
BATCH_SIZE = 64     
EPOCHS = 40         
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_HORIZON = 15    
SEQUENCE_LEN = PRED_HORIZON + 1 
ENCODER_PATH = "./models/encoder_mixed_final.pth" 
NUM_WORKERS = 8

class ShardedSeqDataset(IterableDataset):
    def __init__(self):
        self.files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
        random.shuffle(self.files)
        
        print(f"--- High-RAM Predictor Training ---")
        print(f"Scanning headers for exact sequence count...")
        
        self.total_seqs = 0
        for f in tqdm(self.files):
            try:
                with np.load(f, mmap_mode='r') as d:
                    if 'states' in d: n = d['states'].shape[0]
                    elif 'obs' in d: n = d['obs'].shape[0]
                    if n > SEQUENCE_LEN:
                        self.total_seqs += (n - SEQUENCE_LEN)
            except: pass
            
        print(f"✅ Total Sequences: {self.total_seqs:,}")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            my_files = self.files[worker_info.id::worker_info.num_workers]
        else:
            my_files = self.files
        
        # Buffer to hold sequences in RAM
        sequences_buffer = []
        MAX_SEQS_PER_WORKER = 30_000 
        
        # 1. LOAD PHASE
        for f in my_files:
            if len(sequences_buffer) >= MAX_SEQS_PER_WORKER: break
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

                valid_starts = range(0, length - SEQUENCE_LEN)
                
                for start_t in valid_starts:
                    if len(sequences_buffer) >= MAX_SEQS_PER_WORKER: break
                    end_t = start_t + SEQUENCE_LEN
                    
                    o_chunk = obs_raw[start_t:end_t]
                    a_chunk = act_raw[start_t:end_t-1]
                    
                    if o_chunk.shape[1] != 64:
                         o_chunk = np.array([cv2.resize(img, (64, 64)) for img in o_chunk])
                    
                    sequences_buffer.append((o_chunk, a_chunk))
            except: pass
            
        # 2. STREAM PHASE
        random.shuffle(sequences_buffer)
        for o_chunk, a_chunk in sequences_buffer:
            obs_seq = torch.from_numpy(o_chunk).float() / 255.0
            obs_seq = obs_seq.permute(0, 3, 1, 2) 
            act_seq = torch.from_numpy(a_chunk).float()
            yield obs_seq, act_seq
            
    def __len__(self): return self.total_seqs

def train():
    print(f"Training Predictor on {DEVICE}")
    
    if not os.path.exists(ENCODER_PATH):
        print("Wait for Encoder to finish first!")
        return

    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    predictor = Predictor().to(DEVICE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    dataset = ShardedSeqDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # We use a smaller number for tqdm total if we rely on the RAM buffer limit
    # because workers might drop data to save RAM.
    # The iterator handles the actual end.
    steps_per_epoch = len(dataset) // BATCH_SIZE

    for epoch in range(EPOCHS):
        predictor.train()
        pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for obs_seq, act_seq in pbar:
            obs_seq = obs_seq.to(DEVICE, non_blocking=True)
            act_seq = act_seq.to(DEVICE, non_blocking=True)
            
            B, T_plus_1, C, H, W = obs_seq.shape
            obs_flat = obs_seq.view(-1, C, H, W)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z_flat = encoder(obs_flat)
                    z_seq = z_flat.view(B, T_plus_1, -1)
                
                loss = 0
                z_current = z_seq[:, 0, :] 
                
                for t in range(PRED_HORIZON):
                    action = act_seq[:, t, :]
                    z_target = z_seq[:, t+1, :]
                    
                    z_next_pred = predictor(z_current, action)
                    loss += criterion(z_next_pred, z_target)
                    
                    if np.random.rand() < 0.1: z_current = z_target
                    else: z_current = z_next_pred

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item() / PRED_HORIZON:.6f}")

        if (epoch+1) % 5 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_multistep_ep{epoch+1}.pth")

    torch.save(predictor.state_dict(), "models/predictor_multistep_final.pth")
    print("Predictor Training Complete.")

if __name__ == "__main__":
    train()