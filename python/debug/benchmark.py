import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# --- CONFIG ---
BATCH_SIZE = 512
DEVICE = "cuda:0"

# --- DUMMY COMPONENTS ---
class DummyDataset(Dataset):
    def __init__(self, size=5000):
        self.data = torch.randint(0, 255, (size, 64, 64, 3), dtype=torch.uint8)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches your TinyEncoder complexity
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 512) 
        )
    def forward(self, x): return self.net(x)

def simple_loss(p1, p2):
    # Simulates VICReg matrix math
    cov = (p1.T @ p1) / (p1.shape[0] - 1)
    return cov.sum() + (p1 - p2).pow(2).sum()

# --- BENCHMARK ---
def run_training_bench():
    print(f"\n--- DIAGNOSTIC: FULL TRAINING STEP ---")
    
    # Setup
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    
    # 1. Setup Model (Standard)
    model = SimpleEncoder().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    
    # Augmentation
    gpu_aug = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        v2.RandomResizedCrop(64, scale=(0.85, 1.0), antialias=True),
    ]).to(DEVICE)
    
    # Warmup
    print("Warming up...")
    iter_dl = iter(loader)
    for _ in range(10):
        try:
            raw = next(iter_dl).to(DEVICE)
        except StopIteration:
            iter_dl = iter(loader)
            raw = next(iter_dl).to(DEVICE)
            
        with torch.amp.autocast("cuda"):
            x = gpu_aug(raw)
            z1, z2 = model(x), model(x)
            loss = simple_loss(z1, z2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() # <--- FIXED: Added update
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # MEASURE
    print("Running 50 Full Training Steps...")
    start = time.time()
    for i in range(50):
        try:
            raw = next(iter_dl).to(DEVICE)
        except StopIteration:
            iter_dl = iter(loader)
            raw = next(iter_dl).to(DEVICE)
        
        with torch.amp.autocast("cuda"):
            x = gpu_aug(raw)
            z1, z2 = model(x), model(x)
            loss = simple_loss(z1, z2)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() # <--- FIXED: Added update
        optimizer.zero_grad()
        
    torch.cuda.synchronize()
    total_time = time.time() - start
    
    print(f"RESULT: {50 / total_time:.2f} it/s  ({total_time / 50:.4f} s/it)")

if __name__ == "__main__":
    run_training_bench()