import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm
from networks import TinyEncoder, Predictor

# --- CONFIGURATION ---
BATCH_SIZE = 64     
EPOCHS = 40         
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UPGRADE: LONG HORIZON TRAINING ---
PRED_HORIZON = 15    
SEQUENCE_LEN = PRED_HORIZON + 1 

ENCODER_PATH = "./models/encoder_mixed_final.pth" 

class MultiStepDataset(Dataset):
    def __init__(self):
        # Only load high quality data
        self.files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
        print(f"Loading Episodes from {len(self.files)} files...")
        
        self.episodes = [] 
        self.valid_indices = []
        
        for f in tqdm(self.files, desc="Indexing"):
            try:
                with np.load(f) as arr:
                    if 'states' in arr: o, a = arr['states'], arr['actions']
                    elif 'obs' in arr: o, a = arr['obs'], arr['action']
                    else: continue
                    
                    if len(o) < SEQUENCE_LEN: continue

                    if o.shape[1] != 64:
                        o = np.array([cv2.resize(img, (64, 64)) for img in o])

                    o = np.transpose(o, (0, 3, 1, 2))
                    
                    self.episodes.append({'obs': o, 'act': a})
                    
                    ep_idx = len(self.episodes) - 1
                    num_valid_starts = len(o) - SEQUENCE_LEN + 1
                    for t in range(0, num_valid_starts, 2): 
                        self.valid_indices.append((ep_idx, t))
            except: pass
            
        print(f"Loaded {len(self.episodes)} episodes.")
        print(f"Training Sequences: {len(self.valid_indices):,}")

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_idx, start_t = self.valid_indices[idx]
        data = self.episodes[ep_idx]
        
        end_t = start_t + SEQUENCE_LEN
        
        obs_seq = data['obs'][start_t : end_t]      
        act_seq = data['act'][start_t : end_t - 1]  
        
        obs_seq = torch.from_numpy(obs_seq).float() / 255.0
        act_seq = torch.from_numpy(act_seq).float()
        
        return obs_seq, act_seq

def train():
    print(f"Training Long-Horizon Predictor (Steps={PRED_HORIZON}) on {DEVICE}")
    
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Encoder not found! Run train_encoder.py first.")

    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    predictor = Predictor().to(DEVICE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    dataset = MultiStepDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=8, pin_memory=True, drop_last=True)

    for epoch in range(EPOCHS):
        predictor.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for obs_seq, act_seq in pbar:
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
                    
                    # Drift Correction (10% Teacher Forcing)
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
    print("Predictor Training Complete.")

if __name__ == "__main__":
    train()