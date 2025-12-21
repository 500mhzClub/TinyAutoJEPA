import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import numpy as np
import glob
import os
import cv2 
import re
from tqdm import tqdm
from networks import TinyEncoder, TinyDecoder

# --- CONFIG FOR SECONDARY GPU ---
# Force use of the second GPU (Index 1)
if torch.cuda.device_count() > 1:
    DEVICE = torch.device("cuda:1")
else:
    print("⚠️  Only 1 GPU found! Falling back to cuda:0 (Might be slow)")
    DEVICE = torch.device("cuda:0")

print(f"🚀 Launching Decoder Training on: {torch.cuda.get_device_name(DEVICE)}")

BATCH_SIZE = 256 # Kept high for your 16GB card
EPOCHS = 50      # Run longer since it's a background task
LR = 1e-3

def find_latest_encoder():
    """Finds the most recent checkpoint saved by the main training loop"""
    # Look for intermediate checkpoints first
    files = glob.glob("models/encoder_mixed_ep*.pth")
    
    # If none, look for final
    if not files:
        if os.path.exists("models/encoder_mixed_final.pth"):
            return "models/encoder_mixed_final.pth"
        return None
    
    # Sort by epoch number (e.g. ep5, ep10)
    def extract_epoch(f):
        match = re.search(r'ep(\d+)', f)
        return int(match.group(1)) if match else 0
        
    latest = sorted(files, key=extract_epoch)[-1]
    return latest

class RaceDataset(Dataset):
    def __init__(self, limit_files=100):
        # Only load a subset to save System RAM (RAM is shared with the main training!)
        self.files = glob.glob("./data_race/*.npz")[:limit_files]
        self.data_list = []
        
        print(f"📂 Loading {len(self.files)} files for Background Decoder Training...")
        for f in self.files:
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue

                    if obs.shape[1] != 64:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])

                    self.data_list.append(obs)
            except: pass
            
        self.data = np.concatenate(self.data_list, axis=0)
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        print(f"✅ Decoder Buffer: {len(self.data)} frames loaded.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.from_numpy(self.data[idx]).float()/255.0

def train():
    # 1. Wait/Search for Encoder
    encoder_path = find_latest_encoder()
    if not encoder_path:
        print("❌ No encoder checkpoints found yet. Let the main training run for 5 epochs first!")
        return
    
    print(f"🔗 Locking onto Encoder: {encoder_path}")
    
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Limit files to prevent OOM on System RAM
    dataset = RaceDataset(limit_files=50) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    os.makedirs("visuals", exist_ok=True)

    for epoch in range(EPOCHS):
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # Keep track of last batch for visualization
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

        # --- VISUAL VERIFICATION ---
        if (epoch+1) % 1 == 0: # Save image every epoch
            # Stack top 8 images: Original on Top, Reconstruction on Bottom
            comparison = torch.cat([last_imgs[:8], last_recon[:8]], dim=0)
            save_image(comparison, f"visuals/reconstruct_ep{epoch+1}.png", nrow=8)
            
        # Save Model
        if (epoch+1) % 5 == 0:
            torch.save(decoder.state_dict(), f"models/decoder_parallel_ep{epoch+1}.pth")
            
            # Check if a newer encoder is available and reload if so
            latest_enc = find_latest_encoder()
            if latest_enc != encoder_path:
                print(f"\n🔄 NEW ENCODER DETECTED! Switching to {latest_enc}")
                encoder.load_state_dict(torch.load(latest_enc, map_location=DEVICE))
                encoder_path = latest_enc

    torch.save(decoder.state_dict(), "models/decoder_parallel_final.pth")

if __name__ == "__main__":
    train()