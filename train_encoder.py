import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
import cv2 
import re
from tqdm import tqdm
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- Configuration ---
BATCH_SIZE = 128 # Reduced for stability
EPOCHS = 30
LR = 1e-4    
MAX_RAM_FRAMES = 500_000 # Limit dataset to ~6GB RAM to prevent OOM

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

class BalancedDataset(Dataset):
    def __init__(self):
        print("--- Constructing Balanced Dataset (RAM Safe) ---")
        
        # 1. Load File Lists
        race_files = glob.glob("./data_race/*.npz")
        recovery_files = glob.glob("./data_recovery/*.npz")
        
        # 2. Estimate frames per file to limit loading
        # Assume ~2000 frames per file on average
        # We want MAX_RAM_FRAMES total, split 50/50
        target_per_class = MAX_RAM_FRAMES // 2
        files_needed_race = max(1, target_per_class // 2000)
        files_needed_rec  = max(1, target_per_class // 1000)
        
        # Shuffle and Slice File Lists (Sampling files instead of frames saves loading time)
        np.random.shuffle(race_files)
        np.random.shuffle(recovery_files)
        
        race_files = race_files[:files_needed_race]
        recovery_files = recovery_files[:files_needed_rec]
        
        print(f"Loading subset: {len(race_files)} Race files, {len(recovery_files)} Recovery files")

        self.race_data     = self.load_from_folder(race_files, "Expert/Race")
        self.recovery_data = self.load_from_folder(recovery_files, "Recovery/Drift")
        
        if len(self.race_data) == 0 or len(self.recovery_data) == 0:
            raise ValueError("Missing data! Ensure you ran data collection.")
            
        # 3. Balance perfectly
        min_len = min(len(self.race_data), len(self.recovery_data))
        print(f"Balancing to {min_len} frames per class...")
        
        idx_race = np.random.choice(len(self.race_data), min_len, replace=False)
        idx_rec  = np.random.choice(len(self.recovery_data), min_len, replace=False)
        
        self.data = np.concatenate([self.race_data[idx_race], self.recovery_data[idx_rec]], axis=0)
        
        del self.race_data
        del self.recovery_data
        
        self.data = np.transpose(self.data, (0, 3, 1, 2)) 
        print(f"✅ Final RAM Dataset: {len(self.data):,} frames")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.9, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
        ])

    def load_from_folder(self, files, label):
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
        print("OOM Error: Still too large. Reduce MAX_RAM_FRAMES.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # --- Robust Resume Logic ---
    start_epoch = 0
    
    # Find all checkpoints
    checkpoints = glob.glob("./models/encoder_mixed_ep*.pth")
    
    if checkpoints:
        # Sort by epoch number
        def get_epoch(f):
            match = re.search(r'ep(\d+)', f)
            return int(match.group(1)) if match else 0
            
        latest = max(checkpoints, key=get_epoch)
        print(f"--- RESUMING from {latest} ---")
        
        checkpoint = torch.load(latest, map_location=DEVICE)
        encoder.load_state_dict(checkpoint)
        start_epoch = get_epoch(latest)
    else:
        print("--- STARTING FRESH ---")

    os.makedirs("models", exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
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