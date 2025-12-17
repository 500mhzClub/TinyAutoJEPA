import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.transforms import v2 
import numpy as np
import glob
import os
import psutil
from tqdm import tqdm

# --- CONFIG ---
# CPU OPTIMIZATION:
# Smaller batch size fits entirely in the Ryzen 3600X's L3 Cache (32MB).
# This prevents RAM stalling and makes CPU training extremely fast.
BATCH_SIZE = 128  
EPOCHS = 20
LEARNING_RATE = 3e-4
DEVICE = "cpu"

# --- LOSS (Standard) ---
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    x = x.flatten()[:-1]
    x = x.view(n - 1, n + 1)
    x = x[:, 1:].flatten()
    return x

def vicreg_loss(x, y):
    # Standard FP32 math
    sim_loss = F.mse_loss(x, y)
    
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(y.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))
    
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (BATCH_SIZE - 1)
    cov_y = (y.T @ y) / (BATCH_SIZE - 1)
    
    cov_loss = off_diagonal(cov_x).pow(2).sum() / x.shape[1] + \
               off_diagonal(cov_y).pow(2).sum() / y.shape[1]
    
    return (25.0 * sim_loss) + (25.0 * std_loss) + (1.0 * cov_loss)

# --- AUGMENTATION ---
class CPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            v2.ToDtype(torch.float32, scale=True),
            # Antialias=False is significantly faster on CPU (no floating point filtering)
            v2.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=False), 
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    def forward(self, x):
        # Permute (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return self.transforms(x)

# --- DATA LOADING ---
def load_all_data(data_dir):
    print(f"Loading data from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    np_images = []
    for f in tqdm(files, desc="Reading Disk"):
        try:
            np_images.append(np.load(f)['states']) 
        except: pass
    
    np_images = np.concatenate(np_images, axis=0)
    print(f"Numpy Loaded. Shape: {np_images.shape}")
    
    print("Moving to RAM Tensor...")
    # No pinning needed for CPU
    return torch.from_numpy(np_images)

# --- MODEL ---
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

def train():
    os.makedirs("models", exist_ok=True)
    
    # --- CPU THREAD TUNING ---
    # Ryzen 3600X has 6 physical cores. 
    # Using more threads than physical cores often slows down AVX math.
    torch.set_num_threads(6)
    
    all_data = load_all_data("data")
    num_samples = len(all_data)
    
    # Memory Format: Channels Last allows AVX2 vectorization to process 
    # pixels in contiguous blocks.
    model = VICRegModel().to(DEVICE, memory_format=torch.channels_last)
    augmentor = CPUAugment().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE} (AVX2 Optimized)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        indices = torch.randperm(num_samples)
        num_batches = num_samples // BATCH_SIZE
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i in pbar:
            batch_idx = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch = all_data[batch_idx]
            
            # Augment
            with torch.no_grad():
                v1 = augmentor(batch)
                v2 = augmentor(batch)
            
            # Force channels_last for speed
            v1 = v1.to(memory_format=torch.channels_last)
            v2 = v2.to(memory_format=torch.channels_last)
            
            optimizer.zero_grad()
            
            # Forward
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = vicreg_loss(z1, z2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / num_batches:.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/encoder_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "models/vicreg_encoder_final.pth")

if __name__ == "__main__":
    train()