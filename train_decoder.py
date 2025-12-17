import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from tqdm import tqdm
from networks import TinyEncoder, TinyDecoder

# This is a classic Auto-Encoder training setup
# We use the Frozen Encoder to get Z, then train Decoder to reconstruct X

BATCH_SIZE = 256
EPOCHS = 20 # Decoder learns fast
LR = 1e-3
DATA_PATH = "./data/*.npz"
ENCODER_PATH = "./models/encoder_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    # Reusing a simplified version of the encoder dataset
    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)
        self.data = []
        for f in self.files:
            try:
                with np.load(f) as arr:
                    obs = arr['obs'] if 'obs' in arr else arr['arr_0']
                    self.data.append(obs)
            except: pass
        self.data = np.concatenate(self.data, axis=0)
        self.data = np.transpose(self.data, (0, 3, 1, 2)) # NCHW

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float() / 255.0

def train():
    print(f"Training Decoder on {DEVICE}")
    encoder = TinyEncoder().to(DEVICE)
    decoder = TinyDecoder().to(DEVICE)
    
    if os.path.exists(ENCODER_PATH):
        encoder.load_state_dict(torch.load(ENCODER_PATH))
    encoder.eval() # Freeze Encoder

    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    dataset = ImageDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Decoder Epoch {epoch+1}/{EPOCHS}")
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z = encoder(imgs) # Get latent
                
                recons = decoder(z)   # Reconstruct image
                loss = criterion(recons, imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    torch.save(decoder.state_dict(), "models/decoder_final.pth")
    print("Decoder Training Complete.")

if __name__ == "__main__":
    train()