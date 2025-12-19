import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import cv2 # <--- NEW
from tqdm import tqdm
from networks import TinyEncoder, Predictor

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Points to the encoder we just fine-tuned
ENCODER_PATH = "./models/encoder_final_mixed.pth" 
if not os.path.exists(ENCODER_PATH): 
    ENCODER_PATH = "./models/encoder_ep20.pth"

class CombinedDynamicsDataset(Dataset):
    def __init__(self):
        self.files = glob.glob("./data/*.npz") + glob.glob("./data_race/*.npz")
        print(f"Loading Dynamics from {len(self.files)} files...")
        
        self.obs_list, self.act_list, self.next_list = [], [], []
        
        for f in self.files:
            try:
                with np.load(f) as arr:
                    if 'states' in arr: o, a = arr['states'], arr['actions']
                    elif 'obs' in arr: o, a = arr['obs'], arr['action']
                    else: continue
                    
                    if len(o) != len(a): min_l = min(len(o), len(a)); o=o[:min_l]; a=a[:min_l]
                    if len(o) < 2: continue

                    # --- AUTO-RESIZE FIX ---
                    if o.shape[1] == 96:
                        o = np.array([cv2.resize(img, (64, 64)) for img in o])
                    # -----------------------

                    self.obs_list.append(o[:-1])
                    self.act_list.append(a[:-1])
                    self.next_list.append(o[1:])
            except: pass
            
        self.obs = np.concatenate(self.obs_list, axis=0)
        self.actions = np.concatenate(self.act_list, axis=0)
        self.next_obs = np.concatenate(self.next_list, axis=0)
        
        self.obs = np.transpose(self.obs, (0, 3, 1, 2))
        self.next_obs = np.transpose(self.next_obs, (0, 3, 1, 2))
        print(f"Total Triplets: {len(self.obs)}")

    def __len__(self): return len(self.obs)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.obs[idx]).float()/255.0,
                torch.from_numpy(self.actions[idx]).float(),
                torch.from_numpy(self.next_obs[idx]).float()/255.0)

def train():
    print(f"Training Predictor (New Physics) on {DEVICE}")
    
    encoder = TinyEncoder().to(DEVICE)
    print(f"Loading Encoder from: {ENCODER_PATH}")
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    predictor = Predictor().to(DEVICE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    dataset = CombinedDynamicsDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(EPOCHS):
        predictor.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for obs, act, next_obs in pbar:
            obs, act, next_obs = obs.to(DEVICE, non_blocking=True), act.to(DEVICE, non_blocking=True), next_obs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z_t = encoder(obs)
                    z_t1_true = encoder(next_obs)
                
                z_t1_pred = predictor(z_t, act)
                loss = criterion(z_t1_pred, z_t1_true)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        if (epoch+1) % 10 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_race_ep{epoch+1}.pth")

    torch.save(predictor.state_dict(), "models/predictor_race_final.pth")
    print("Predictor Training Complete.")

if __name__ == "__main__":
    train()