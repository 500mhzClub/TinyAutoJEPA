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
LR = 3e-4
NUM_WORKERS = 20
WEIGHT_DECAY = 0.05
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BalancedMixedDataset(IterableDataset):
    def __init__(self):
        self.random_files = glob.glob("./data/*.npz")
        self.race_files = glob.glob("./data_race/*.npz")
        self.recovery_files = glob.glob("./data_recovery/*.npz")
        self.edge_files = glob.glob("./data_edge/*.npz")
        
        # Create balanced 4-way mix
        max_files = max(len(self.random_files), len(self.race_files), len(self.recovery_files), len(self.edge_files))
        self.balanced_files = []
        for i in range(max_files):
            if self.random_files: self.balanced_files.append(self.random_files[i % len(self.random_files)])
            if self.race_files: self.balanced_files.append(self.race_files[i % len(self.race_files)])
            if self.recovery_files: self.balanced_files.append(self.recovery_files[i % len(self.recovery_files)])
            if self.edge_files: self.balanced_files.append(self.edge_files[i % len(self.edge_files)])
            
        print(f"Balanced Dataset: {len(self.balanced_files)} files.")
        
        # Calculate total frames
        self.total_frames = 0
        for f in tqdm(self.balanced_files, desc="Scanning Dataset"):
            try:
                with np.load(f, mmap_mode='r') as d:
                    if 'states' in d: self.total_frames += d['states'].shape[0]
            except: pass
        print(f"Total Frames: {self.total_frames:,}")

        # TRANSFORM: Removed HorizontalFlip to preserve directionality
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomRotation(degrees=3), 
        ])

    def __iter__(self):
        worker_info = get_worker_info()
        my_files = self.balanced_files[worker_info.id::worker_info.num_workers] if worker_info else self.balanced_files
        random.shuffle(my_files)

        for f in my_files:
            try:
                with np.load(f) as data:
                    raw = data['states'] if 'states' in data else data['obs']
                
                if raw.shape[1] != 64: raw = np.array([cv2.resize(i, (64,64)) for i in raw])
                indices = np.random.permutation(len(raw))
                
                for idx in indices:
                    img = torch.from_numpy(raw[idx]).float() / 255.0
                    img = img.permute(2, 0, 1) # HWC -> CHW
                    yield self.transform(img), self.transform(img)
            except: continue

    def __len__(self): return self.total_frames

@torch.no_grad()
def validate_encoder(encoder, projector, val_loader, epoch):
    encoder.eval()
    projector.eval()
    all_embeddings = []
    
    for i, (x1, _) in enumerate(val_loader):
        if i >= 10: break
        x1 = x1.to(DEVICE)
        z = encoder(x1)
        all_embeddings.append(z.cpu())
        
    z_all = torch.cat(all_embeddings, dim=0)
    std_per_dim = z_all.std(dim=0)
    dead_dims = (std_per_dim < 0.01).sum().item()
    
    print(f"\n[Validation Epoch {epoch}]")
    print(f"  Dead dimensions: {dead_dims}/512")
    print(f"  Avg Std: {std_per_dim.mean().item():.4f}")
    
    encoder.train()
    projector.train()

def train():
    dataset = BalancedMixedDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    
    # Validation loader
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    checkpoints = glob.glob("./models/encoder_mixed_ep*.pth")
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(re.search(r'ep(\d+)', f).group(1)))
        print(f"Resuming from {latest}")
        encoder.load_state_dict(torch.load(latest, map_location=DEVICE))
        start_epoch = int(re.search(r'ep(\d+)', latest).group(1))
        for _ in range(start_epoch): scheduler.step()

    os.makedirs("models", exist_ok=True)
    encoder.train()
    projector.train()

    steps = dataset.total_frames // BATCH_SIZE

    for epoch in range(start_epoch, EPOCHS):
        pbar = tqdm(dataloader, total=steps, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x1, x2 in pbar:
            x1, x2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                loss = vicreg_loss(projector(encoder(x1)), projector(encoder(x2)))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()
        if (epoch+1) % 5 == 0:
            validate_encoder(encoder, projector, val_loader, epoch+1)
            torch.save(encoder.state_dict(), f"models/encoder_mixed_ep{epoch+1}.pth")

    torch.save(encoder.state_dict(), "models/encoder_mixed_final.pth")

if __name__ == "__main__":
    train()