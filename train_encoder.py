import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2 
import numpy as np
import glob
import os
import psutil
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = "cuda"

# --- FIXED VICREG LOSS ---
def off_diagonal(x):
    # Dynamic shape handling (Works for 512, 2048, or any dim)
    n, m = x.shape
    assert n == m
    # Flatten and remove the last element
    x = x.flatten()[:-1]
    # Reshape to (n-1, n+1) to shift diagonals
    x = x.view(n - 1, n + 1)
    # Remove the first column (which contains the diagonals)
    x = x[:, 1:].flatten()
    return x

def vicreg_loss(x, y):
    # x, y are (Batch_Size, 2048)
    
    # 1. Invariance Loss (MSE)
    sim_loss = F.mse_loss(x, y)
    
    # 2. Variance Loss (Hinge)
    # Calculate std across batch
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(y.var(dim=0) + 1e-04)
    # Target std is 1.0. Penalize if < 1.0
    std_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))
    
    # 3. Covariance Loss
    batch_size, num_features = x.shape
    
    # Center the vectors
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    
    # Calculate Covariance Matrix (2048x2048)
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    
    # Penalize off-diagonal elements
    cov_loss = off_diagonal(cov_x).pow(2).sum() / num_features + \
               off_diagonal(cov_y).pow(2).sum() / num_features
    
    # Weights: 25.0 for Sim/Var, 1.0 for Cov
    return (25.0 * sim_loss) + (25.0 * std_loss) + (1.0 * cov_loss)

# --- GPU AUGMENTATION ---
class GPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    def forward(self, x):
        return self.transforms(x)

# --- DATASET ---
class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # Load all data into RAM
        print(f"Loading {len(files)} files into RAM...")
        for f in tqdm(files, desc="Loading Data"):
            try:
                self.images.append(np.load(f)['states']) 
            except: pass
                
        self.images = np.concatenate(self.images, axis=0)
        print(f"RAM Data: {self.images.nbytes / 1e9:.2f} GB | Shape: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Direct memory slice
        return torch.from_numpy(self.images[idx]).permute(2, 0, 1)

# --- MODEL ---
class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # ResNet18 output before FC is 512
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projector: 512 -> 2048 -> 2048 -> 2048
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
    dataset = CarRacingDataset("data")
    
    # Workers=0 prevents memory thrashing
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )
    
    model = VICRegModel().to(DEVICE)
    augmentor = GPUAugment().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Starting training on {DEVICE} (Workers=0, GPU Augs)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            # 1. Fetch & Augment
            batch = batch.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                v1 = augmentor(batch)
                v2 = augmentor(batch)
            
            # 2. Train
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = vicreg_loss(z1, z2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/encoder_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "models/vicreg_encoder_final.pth")

if __name__ == "__main__":
    train()