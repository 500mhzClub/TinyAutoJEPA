import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import sys
from tqdm import tqdm
from networks import TinyEncoder, TinyDecoder

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 20 # Decoder learns fast
LR = 1e-3
DATA_PATH = "./data/*.npz"
# Ensure this matches the one you are using for the Predictor
ENCODER_PATH = "./models/encoder_ep20.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Robust Image Dataset ---
class ImageDataset(Dataset):
    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)
        if not self.files:
            print(f"CRITICAL: No files found at {file_pattern}")
            sys.exit(1)
            
        self.data_list = []
        print(f"Found {len(self.files)} files. Loading images...")
        
        success_count = 0
        for f in self.files:
            try:
                with np.load(f) as arr:
                    # Robust Key Search (Simpler than Predictor, we only need images)
                    obs = None
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    elif 'arr_0' in arr: obs = arr['arr_0']
                    
                    if obs is None:
                        continue
                        
                    self.data_list.append(obs)
                    success_count += 1
            except Exception:
                pass
        
        if success_count == 0:
            print("CRITICAL: Failed to load any images.")
            sys.exit(1)

        print(f"Concatenating {success_count} files...")
        self.data = np.concatenate(self.data_list, axis=0)
        del self.data_list # Free RAM
        
        # NHWC -> NCHW
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        print(f"Total training images: {len(self.data)}")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float() / 255.0

def train():
    print(f"Training Decoder on {DEVICE}")
    
    # 1. Load Frozen Encoder
    encoder = TinyEncoder().to(DEVICE)
    if os.path.exists(ENCODER_PATH):
        print(f"Loading Frozen Eye from {ENCODER_PATH}")
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    else:
        print("WARNING: Encoder weights not found!")
    
    encoder.eval() 
    for param in encoder.parameters(): param.requires_grad = False

    # 2. Setup Decoder
    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # 3. Load Data
    try:
        dataset = ImageDataset(DATA_PATH)
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 4. Train Loop
    for epoch in range(EPOCHS):
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Get Latent (Ground Truth)
                with torch.no_grad():
                    z = encoder(imgs)
                
                # Reconstruct
                recons = decoder(z)
                
                # Loss = Input Image vs Reconstructed Image
                loss = criterion(recons, imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Save occasionally
        if (epoch+1) % 5 == 0:
            torch.save(decoder.state_dict(), f"models/decoder_ep{epoch+1}.pth")

    torch.save(decoder.state_dict(), "models/decoder_final.pth")
    print("Decoder Training Complete.")

if __name__ == "__main__":
    train()