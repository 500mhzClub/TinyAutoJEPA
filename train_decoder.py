import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import numpy as np
import glob
import os
import cv2 
from tqdm import tqdm
from networks import TinyEncoder, TinyDecoder

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "./models/encoder_mixed_final.pth"

class SimpleDataset(Dataset):
    def __init__(self):
        # Load high quality data for decoding
        self.files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
        self.data_list = []
        
        print(f"Loading {len(self.files)} files for Decoder...")
        for f in tqdm(self.files):
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue

                    if obs.shape[1] != 64:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])

                    # Take a random subset from each file to save RAM
                    # (We don't need EVERY frame to learn decoding)
                    indices = np.random.choice(len(obs), size=min(len(obs), 50), replace=False)
                    self.data_list.append(obs[indices])
            except: pass
            
        self.data = np.concatenate(self.data_list, axis=0)
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        print(f"Decoder Dataset: {len(self.data)} frames.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.from_numpy(self.data[idx]).float()/255.0

def train():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Encoder not found! Run train_encoder.py first.")
        
    print(f"Training Decoder on {DEVICE}")
    
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    os.makedirs("visuals", exist_ok=True)

    for epoch in range(EPOCHS):
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        last_imgs = None
        last_recon = None
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad(): 
                    z = encoder(imgs)
                recon = decoder(z)
                loss = criterion(recon, imgs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            last_imgs = imgs
            last_recon = recon

        if (epoch+1) % 5 == 0:
            comparison = torch.cat([last_imgs[:8], last_recon[:8]], dim=0)
            save_image(comparison, f"visuals/reconstruct_ep{epoch+1}.png", nrow=8)
            torch.save(decoder.state_dict(), f"models/decoder_parallel_ep{epoch+1}.pth")

    torch.save(decoder.state_dict(), "models/decoder_parallel_ep40.pth")
    print("Decoder Training Complete.")

if __name__ == "__main__":
    train()