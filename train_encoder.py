import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
import cv2 
from tqdm import tqdm
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4    

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

class BalancedDataset(Dataset):
    def __init__(self):
        print("--- Constructing High-Quality Dataset ---")
        
        # 1. Load Expert and Recovery Data
        # WE INTENTIONALLY IGNORE THE RANDOM 'data' FOLDER NOW
        self.race_data     = self.load_from_folder("./data_race/*.npz", "Expert/Race")
        self.recovery_data = self.load_from_folder("./data_recovery/*.npz", "Recovery/Drift")
        
        # 2. Combine
        if len(self.race_data) == 0 or len(self.recovery_data) == 0:
            print("Warning: Missing data! Ensure you ran collect_race_data and collect_recovery_data.")
        
        self.data = np.concatenate([self.race_data, self.recovery_data], axis=0)
        
        # Free memory
        del self.race_data
        del self.recovery_data
        
        # NHWC -> NCHW
        self.data = np.transpose(self.data, (0, 3, 1, 2)) 
        print(f"Final Dataset Size: {len(self.data):,} frames")

        # 3. Augmentations (FIXED CROP)
        self.transform = transforms.Compose([
            # Scale changed from 0.7 -> 0.9 to prevent losing context (zoomed in grass)
            transforms.RandomResizedCrop(64, scale=(0.9, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
        ])

    def load_from_folder(self, glob_pattern, label):
        files = glob.glob(glob_pattern)
        print(f"Loading {label}: Found {len(files)} files...")
        
        data_list = []
        for f in tqdm(files, desc=f"Reading {label}"):
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue
                    
                    if obs.shape[1] != 64:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                        
                    data_list.append(obs)
            except: pass
            
        if not data_list: return np.array([])
        return np.concatenate(data_list, axis=0)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float() / 255.0
        return self.transform(img), self.transform(img)

def train():
    try:
        dataset = BalancedDataset()
    except MemoryError:
        print("OOM Error: Dataset too large for RAM.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        encoder.train()
        projector.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x1, x2 in pbar:
            x1, x2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                z1 = projector(encoder(x1))
                z2 = projector(encoder(x2))
                loss = vicreg_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch+1) % 5 == 0:
            torch.save(encoder.state_dict(), f"models/encoder_mixed_ep{epoch+1}.pth")

    torch.save(encoder.state_dict(), "models/encoder_mixed_final.pth")
    print("Encoder Training Complete.")

if __name__ == "__main__":
    train()