import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
import numpy as np
import glob
import os
import cv2 
import re
import random
from tqdm import tqdm
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4    

# 64GB RAM OPTIMIZATION
# We will use 8 workers. 
# Total Data = ~32GB. 
# Each worker will hold ~4GB in RAM.
NUM_WORKERS = 8 

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

class ShardedMemoryDataset(IterableDataset):
    def __init__(self):
        self.files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
        random.shuffle(self.files)
        
        # Calculate total frames for the progress bar
        # (We estimate based on average file size to avoid opening them all in the main process)
        self.est_total_frames = len(self.files) * 2000 
        print(f"--- 🚀 High-RAM Mode Enabled (64GB Detected) ---")
        print(f"    Targeting {len(self.files)} files.")
        print(f"    Estimated Data Size: ~32 GB")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.9, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
        ])

    def __iter__(self):
        # --- WORKER ISOLATION LOGIC ---
        # This code runs inside each worker process independently.
        worker_info = get_worker_info()
        
        if worker_info is not None:
            # Divide files among workers
            # Worker 0 gets files [0, 8, 16...]
            # Worker 1 gets files [1, 9, 17...]
            my_files = self.files[worker_info.id::worker_info.num_workers]
            worker_id = worker_info.id
        else:
            my_files = self.files
            worker_id = 0

        # --- MASSIVE CHUNK LOADING ---
        # Load ALL assigned files into RAM as a single huge array.
        # This takes ~30 seconds at startup but makes training blazing fast.
        
        data_buffer = []
        
        # Determine strict allocation limit per worker to prevent OOM
        # 32GB Total / 8 Workers = 4GB per worker limit
        # 4GB / (64*64*3 bytes) = ~325,000 frames per worker
        MAX_FRAMES_PER_WORKER = 350_000 
        
        frames_loaded = 0
        
        random.shuffle(my_files)
        
        for f in my_files:
            if frames_loaded >= MAX_FRAMES_PER_WORKER: break
            
            try:
                with np.load(f) as data:
                    if 'states' in data: raw = data['states']
                    elif 'obs' in data: raw = data['obs']
                    else: continue
                
                # Resize immediately to save RAM (if they aren't already 64x64)
                if raw.shape[1] != 64:
                    raw = np.array([cv2.resize(img, (64, 64)) for img in raw])
                
                data_buffer.append(raw)
                frames_loaded += len(raw)
            except: pass
            
        if len(data_buffer) > 0:
            # Concatenate into one massive contiguous block in RAM
            self.memory_chunk = np.concatenate(data_buffer, axis=0)
            # Shuffle indices for randomness
            self.indices = np.random.permutation(len(self.memory_chunk))
            
            # Streaming loop
            for idx in self.indices:
                img_raw = self.memory_chunk[idx]
                
                # To Tensor
                img = torch.from_numpy(img_raw).float() / 255.0
                img = img.permute(2, 0, 1) # HWC -> CHW
                if img.shape[0] != 3: img = img.permute(2, 0, 1) # Fix double permute edge case
                
                yield self.transform(img), self.transform(img)
        else:
            # Fallback if file load failed
            return

    def __len__(self):
        return self.est_total_frames

def train():
    dataset = ShardedMemoryDataset()
    
    # PIN_MEMORY=True speeds up transfer to GPU
    # PREFETCH_FACTOR=2 ensures the GPU never waits for data
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                            pin_memory=True, prefetch_factor=2)

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

    # Steps calculation (Approximate)
    steps_per_epoch = dataset.est_total_frames // BATCH_SIZE

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        # We wrap the dataloader. The first time this runs, there will be a 
        # ~30s delay while workers load their RAM chunks.
        pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
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