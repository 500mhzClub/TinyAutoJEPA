import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import cv2 # <--- NEW
from tqdm import tqdm
from networks import TinyEncoder, TinyDecoder

BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_PATH = "./models/encoder_final_mixed.pth"
if not os.path.exists(ENCODER_PATH): 
    ENCODER_PATH = "./models/encoder_ep20.pth"

class CombinedImageDataset(Dataset):
    def __init__(self):
        self.files = glob.glob("./data/*.npz") + glob.glob("./data_race/*.npz")
        self.data_list = []
        for f in self.files:
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue

                    # --- AUTO-RESIZE FIX ---
                    if obs.shape[1] == 96:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                    # -----------------------

                    self.data_list.append(obs)
            except: pass
        self.data = np.concatenate(self.data_list, axis=0)
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        print(f"Decoder Training on {len(self.data)} frames.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.from_numpy(self.data[idx]).float()/255.0

def train():
    print(f"Training Decoder on {DEVICE}")
    
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    dataset = CombinedImageDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    for epoch in range(EPOCHS):
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                with torch.no_grad(): z = encoder(imgs)
                recon = decoder(z)
                loss = criterion(recon, imgs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch+1) % 5 == 0:
            torch.save(decoder.state_dict(), f"models/decoder_race_ep{epoch+1}.pth")

    torch.save(decoder.state_dict(), "models/decoder_race_final.pth")

if __name__ == "__main__":
    train()