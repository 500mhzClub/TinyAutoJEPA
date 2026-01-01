import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import glob
import os
import cv2
import random
from tqdm import tqdm
from networks import TinyEncoder, Predictor

# --- CONFIG ---
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_HORIZON = 15    
SEQUENCE_LEN = PRED_HORIZON + 1 
STEPS_PER_EPOCH = 1000 # 1000 batches * 64 size = 64,000 samples per epoch

ENCODER_PATH = "./models/encoder_mixed_final.pth" 

class StreamingSeqDataset(IterableDataset):
    def __init__(self):
        self.files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
        random.shuffle(self.files)
        print(f"Streaming {len(self.files)} files...")

    def __iter__(self):
        while True:
            # Pick random file
            f = random.choice(self.files)
            try:
                with np.load(f) as arr:
                    if 'states' in arr: o, a = arr['states'], arr['actions']
                    elif 'obs' in arr: o, a = arr['obs'], arr['action']
                    else: continue
                    
                    if len(o) < SEQUENCE_LEN: continue

                    if o.shape[1] != 64:
                        o = np.array([cv2.resize(img, (64, 64)) for img in o])

                    o = np.transpose(o, (0, 3, 1, 2))
                    
                    # Yield sequences from this file
                    # We can pick random start points to keep it fresh
                    num_valid_starts = len(o) - SEQUENCE_LEN
                    
                    # Yield 50 random sequences from this file, then move to next file
                    # This prevents IO bottleneck (opening file just for 1 sequence)
                    for _ in range(50):
                        start_t = random.randint(0, num_valid_starts)
                        end_t = start_t + SEQUENCE_LEN
                        
                        obs_seq = torch.from_numpy(o[start_t:end_t]).float() / 255.0
                        act_seq = torch.from_numpy(a[start_t:end_t-1]).float()
                        yield obs_seq, act_seq
            except: pass

def train():
    print(f"Training Streaming Predictor on {DEVICE}")
    
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    predictor = Predictor().to(DEVICE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    dataset = StreamingSeqDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    data_iter = iter(dataloader)

    for epoch in range(EPOCHS):
        predictor.train()
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for _ in pbar:
            try:
                obs_seq, act_seq = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                obs_seq, act_seq = next(data_iter)

            B, T_plus_1, C, H, W = obs_seq.shape
            obs_flat = obs_seq.view(-1, C, H, W).to(DEVICE, non_blocking=True)
            act_seq = act_seq.to(DEVICE, non_blocking=True)

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
                    
                    if np.random.rand() < 0.1:
                        z_current = z_target
                    else:
                        z_current = z_next_pred

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item() / PRED_HORIZON:.6f}")

        if (epoch+1) % 5 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_multistep_ep{epoch+1}.pth")

    torch.save(predictor.state_dict(), "models/predictor_multistep_final.pth")
    print("Streaming Predictor Training Complete.")

if __name__ == "__main__":
    train()