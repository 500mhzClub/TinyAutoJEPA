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
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4    

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

class MmapDataset(Dataset):
    def __init__(self):
        print("--- Constructing Full Dataset (Memory Mapped) ---")
        
        # 1. Gather files
        self.files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
        
        # 2. Index files without loading them
        # We need to know how many frames are in each file to map indices
        self.file_indices = []
        self.total_frames = 0
        
        print(f"Indexing {len(self.files)} files...")
        for f in tqdm(self.files):
            try:
                # 'r' mode just reads the header, doesn't load data
                with np.load(f, mmap_mode='r') as data:
                    if 'states' in data: n = data['states'].shape[0]
                    elif 'obs' in data: n = data['obs'].shape[0]
                    else: continue
                    
                    self.file_indices.append((f, self.total_frames, n))
                    self.total_frames += n
            except: pass
            
        print(f"✅ Indexed {self.total_frames:,} frames across all files.")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.9, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
        ])

    def __len__(self): return self.total_frames

    def __getitem__(self, idx):
        # Binary search or simple iteration to find which file owns this global idx
        # (Iteration is fast enough since we only have ~60 files)
        for fname, start_idx, count in self.file_indices:
            if start_idx <= idx < start_idx + count:
                local_idx = idx - start_idx
                
                # LAZY LOAD: Open file, grab one frame, close.
                # mmap makes this instant.
                try:
                    with np.load(fname, mmap_mode='r') as data:
                        if 'states' in data: raw = data['states'][local_idx]
                        else: raw = data['obs'][local_idx]
                        
                        # Resize on the fly if needed
                        if raw.shape[0] != 64: 
                             raw = cv2.resize(raw, (64, 64))
                        
                        # Copy to ensure we own the memory (detached from mmap)
                        img = torch.from_numpy(np.array(raw)).float() / 255.0
                        return self.transform(img), self.transform(img)
                except:
                    # Fallback for corrupted frames (return black)
                    return torch.zeros((3,64,64)), torch.zeros((3,64,64))
        
        return torch.zeros((3,64,64)), torch.zeros((3,64,64))

def train():
    # num_workers must be > 0 to hide disk latency
    dataset = MmapDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # Resume Logic
    start_epoch = 0
    checkpoints = glob.glob("./models/encoder_mixed_ep*.pth")
    if checkpoints:
        def get_epoch(f):
            match = re.search(r'ep(\d+)', f)
            return int(match.group(1)) if match else 0
        latest = max(checkpoints, key=get_epoch)
        print(f"--- RESUMING from {latest} ---")
        encoder.load_state_dict(torch.load(latest, map_location=DEVICE))
        start_epoch = get_epoch(latest)

    os.makedirs("models", exist_ok=True)
    encoder.train()
    projector.train()

    for epoch in range(start_epoch, EPOCHS):
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