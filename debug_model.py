import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.transforms import v2 
import numpy as np
import time
import os
import glob
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 512
DEVICE = "cuda"

print(f"DEBUG: Torch Version: {torch.__version__}")
print(f"DEBUG: Device: {torch.cuda.get_device_name(0)}")

# --- COMPONENTS ---
class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048)
        )
    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        z = self.projector(h)
        return h, z

class GPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=False), # Changed antialias=False for speed check
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    def forward(self, x):
        if x.shape[-1] == 3: # Handle HWC -> CHW
            x = x.permute(0, 3, 1, 2)
        return self.transforms(x)

def vicreg_loss(x, y):
    sim_loss = F.mse_loss(x, y)
    std_loss = torch.mean(F.relu(1.0 - torch.sqrt(x.var(dim=0) + 1e-04))) + \
               torch.mean(F.relu(1.0 - torch.sqrt(y.var(dim=0) + 1e-04)))
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (BATCH_SIZE - 1)
    cov_y = (y.T @ y) / (BATCH_SIZE - 1)
    cov_loss = (cov_x.pow(2).sum() - cov_x.diag().pow(2).sum()) / 2048 + \
               (cov_y.pow(2).sum() - cov_y.diag().pow(2).sum()) / 2048
    return 25*sim_loss + 25*std_loss + 1*cov_loss

# --- TEST PHASES ---

def run_phase(name, loader_fn, augment=False):
    print(f"\n--- STARTING {name} ---")
    model = VICRegModel().to(DEVICE)
    augmentor = GPUAugment().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Warmup
    print("Warming up GPU...")
    for _ in range(5):
        dummy = torch.randn(BATCH_SIZE, 3, 64, 64, device=DEVICE)
        model(dummy)
    torch.cuda.synchronize()
    
    start_time = time.time()
    steps = 50
    
    for i in tqdm(range(steps), desc=name):
        # 1. Get Data
        batch = loader_fn()
        
        # 2. Augment (Conditional)
        if augment:
            batch = batch.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                v1 = augmentor(batch)
                v2 = augmentor(batch)
        else:
            # Fake views if no augment
            batch = batch.to(DEVICE, non_blocking=True)
            v1 = batch
            v2 = batch

        # 3. Model Step
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = vicreg_loss(z1, z2)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Force sync to measure TRUE time per step
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    print(f"RESULT: {name} finished in {total_time:.2f}s")
    print(f"SPEED:  {steps/total_time:.2f} it/s  ({total_time/steps:.4f} s/it)")

# --- LOADERS ---

# Phase 1: Zero Copy. Data exists on GPU already.
def synthetic_gpu_loader():
    return torch.randn(BATCH_SIZE, 3, 64, 64, device=DEVICE)

# Phase 3: Real Data (Lazy Load simulation)
all_data = None
def setup_real_data():
    global all_data
    print("Loading Real Data for Phase 3...")
    files = glob.glob("data/*.npz")[:20] # Load just enough for testing
    data = [np.load(f)['states'] for f in files]
    all_data = torch.from_numpy(np.concatenate(data, axis=0))
    # CRITICAL: Pin memory!
    all_data = all_data.pin_memory()
    print(f"Loaded {len(all_data)} images into Pinned RAM.")

def real_data_loader():
    idx = torch.randint(0, len(all_data), (BATCH_SIZE,))
    return all_data[idx] # Slice from CPU RAM

# --- MAIN ---

if __name__ == "__main__":
    # PHASE 1: Raw Compute (Is the GPU kernel crashing/stalling?)
    run_phase("PHASE 1 (Synthetic Data on GPU, No Augs)", synthetic_gpu_loader, augment=False)
    
    # PHASE 2: Augmentation (Are transforms broken on RDNA4?)
    # Note: We pass raw 64x64 floats pretending to be images
    run_phase("PHASE 2 (Synthetic Data + GPU Augs)", synthetic_gpu_loader, augment=True)
    
    # PHASE 3: Transfer (Is the PCIe bus choking?)
    try:
        setup_real_data()
        run_phase("PHASE 3 (Real Data Transfer + Augs)", real_data_loader, augment=True)
    except Exception as e:
        print(f"Skipping Phase 3: {e}")