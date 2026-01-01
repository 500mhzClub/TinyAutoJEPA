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
NUM_WORKERS = 8 # Optimized for your CPU

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

class ChunkedBalancedDataset(IterableDataset):
    def __init__(self):
        print("--- Constructing Chunked Dataset ---")
        self.race_files = glob.glob("./data_race/*.npz")
        self.rec_files  = glob.glob("./data_recovery/*.npz")
        
        # 1. Balance the File Lists
        # We want equal probability of seeing Race vs Recovery
        # If we have 100 race and 20 recovery, we upsample recovery
        max_files = max(len(self.race_files), len(self.rec_files))
        
        # Simple balancing: create a master list that is 50/50
        # We will cycle the smaller list to match the larger one
        self.balanced_files = []
        
        # Interleave them: [Race, Rec, Race, Rec...]
        # This ensures the model sees both even if the epoch cuts short
        r_idx = 0
        rec_idx = 0
        
        # Create enough pairs to cover all data at least once
        total_pairs = max_files
        
        for _ in range(total_pairs):
            self.balanced_files.append(self.race_files[r_idx % len(self.race_files)])
            self.balanced_files.append(self.rec_files[rec_idx % len(self.rec_files)])
            r_idx += 1
            rec_idx += 1
            
        print(f"Balanced File List: {len(self.balanced_files)} files (50/50 Mix)")
        
        # Estimate total frames (Assume ~2000 per file avg) for Progress Bar
        self.est_total_frames = len(self.balanced_files) * 2000 

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.9, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
        ])

    def __iter__(self):
        worker_info = get_worker_info()
        
        # Distribute files among workers
        if worker_info is not None:
            my_files = self.balanced_files[worker_info.id::worker_info.num_workers]
        else:
            my_files = self.balanced_files

        # Shuffle MY files so workers don't march in lockstep
        random.shuffle(my_files)

        for f in my_files:
            try:
                # FAST LOAD: Load whole file into RAM
                with np.load(f) as data:
                    if 'states' in data: raw = data['states']
                    elif 'obs' in data: raw = data['obs']
                    else: continue
                
                # Resize if needed
                if raw.shape[1] != 64:
                    raw = np.array([cv2.resize(img, (64, 64)) for img in raw])

                # Shuffle frames INSIDE the file (Local Randomness)
                indices = np.random.permutation(len(raw))
                
                for idx in indices:
                    img_raw = raw[idx]
                    
                    # To Tensor
                    img = torch.from_numpy(img_raw).float() / 255.0
                    img = img.permute(2, 0, 1) # HWC -> CHW
                    if img.shape[0] != 3: img = img.permute(2, 0, 1)

                    yield self.transform(img), self.transform(img)

            except Exception as e:
                continue

    def __len__(self):
        return self.est_total_frames

def train():
    dataset = ChunkedBalancedDataset()
    
    # Persistent workers keep the processes alive, avoiding startup overhead
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

    # Steps for tqdm
    steps_per_epoch = dataset.est_total_frames // BATCH_SIZE

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
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