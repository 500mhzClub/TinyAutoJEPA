import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import numpy as np
import glob
import os
import sys
from vicreg import vicreg_loss

# --- CONFIG ---
DEVICE = "cuda"

# --- MOCK DATASET (To test Model without loading huge files) ---
# If this fails, the GPU is broken. If this works, your DATA is corrupt.
class MockDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(256, 3, 64, 64) # Random noise
    def __len__(self): return 256
    def __getitem__(self, idx): return self.data[idx]

# --- REAL DATASET (To test your actual data) ---
class RealDataset(Dataset):
    def __init__(self, data_dir):
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.images = []
        for f in files[:2]: # Only load 2 chunks
            data = np.load(f)
            self.images.append(data['states'])
        self.images = np.concatenate(self.images, axis=0)
        self.images = np.transpose(self.images, (0, 3, 1, 2))
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return torch.from_numpy(self.images[idx])

# --- MODEL ---
class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.projector(h)
        return h, z

def inspect_tensor(name, x):
    if torch.isnan(x).any():
        print(f"❌ {name}: CONTAINS NaN!")
        return False
    if torch.isinf(x).any():
        print(f"❌ {name}: CONTAINS Infinity!")
        return False
    
    print(f"✅ {name}: Min={x.min().item():.4f}, Max={x.max().item():.4f}, Mean={x.mean().item():.4f}")
    return True

def run_diagnostic():
    print(f"--- DIAGNOSTIC MODE: {DEVICE} ---")
    
    # 1. Test Model Logic with Fake Data
    print("\n[Test 1] Running with Random Noise (Checking GPU Math)...")
    dataset = MockDataset()
    loader = DataLoader(dataset, batch_size=32)
    model = VICRegModel().to(DEVICE)
    
    batch = next(iter(loader)).to(DEVICE)
    h, z = model(batch)
    
    if not inspect_tensor("Random Input -> Encoder Output", h): sys.exit(1)
    if not inspect_tensor("Random Input -> Projector Output", z): sys.exit(1)
    print(">> GPU Math seems OK.")

    # 2. Test Real Data
    print("\n[Test 2] Running with REAL Data (Checking Dataset)...")
    try:
        real_dataset = RealDataset("data")
        real_loader = DataLoader(real_dataset, batch_size=32, shuffle=True)
        real_batch = next(iter(real_loader))
        
        # Check raw data
        real_batch_float = real_batch.float() / 255.0
        inspect_tensor("Raw Data Batch", real_batch_float)
        
        real_batch_gpu = real_batch_float.to(DEVICE)
        
        # Check outputs
        h, z = model(real_batch_gpu)
        inspect_tensor("Real Data -> Encoder Output", h)
        inspect_tensor("Real Data -> Projector Output", z)
        
        # Check Loss Calculation
        loss = vicreg_loss(z, z) # Self-loss
        print(f"Self-Loss Value: {loss.item()}")
        
    except Exception as e:
        print(f"\n❌ DATASET ERROR: {e}")
        print("Your .npz files might be corrupt or empty.")

if __name__ == "__main__":
    run_diagnostic()