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

# --- LOSS ---
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    x = x.flatten()[:-1]
    x = x.view(n - 1, n + 1)
    x = x[:, 1:].flatten()
    return x

def vicreg_loss(x, y):
    # Invariance
    sim_loss = F.mse_loss(x, y)
    
    # Variance
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(y.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))
    
    # Covariance
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (BATCH_SIZE - 1)
    cov_y = (y.T @ y) / (BATCH_SIZE - 1)
    
    cov_loss = off_diagonal(cov_x).pow(2).sum() / x.shape[1] + \
               off_diagonal(cov_y).pow(2).sum() / y.shape[1]
    
    return (25.0 * sim_loss) + (25.0 * std_loss) + (1.0 * cov_loss)

# --- GPU AUGMENTATION ---
class GPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            # 1. Permute on GPU (Fastest)
            v2.PermuteDimensions(dims=(0, 3, 1, 2)), # (B, H, W, C) -> (B, C, H, W)
            # 2. Convert to Float & Augment
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    def forward(self, x):
        return self.transforms(x)

# --- DATASET (SHARED MEMORY) ---
class SharedMemoryDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Zero processing. Just return the bytes.
        # Returns (64, 64, 3) ByteTensor
        return self.data[idx]

def load_data_to_shared_memory(data_dir):
    print(f"Loading data from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    # 1. Load Numpy
    np_images = []
    for f in tqdm(files, desc="Reading Disk"):
        try:
            np_images.append(np.load(f)['states']) 
        except: pass
    
    np_images = np.concatenate(np_images, axis=0)
    print(f"Numpy Loaded. Shape: {np_images.shape}")
    
    # 2. Convert to Shared Tensor
    # We use uint8 (ByteTensor) to save 4x memory
    print("Moving to Shared Memory Tensor...")
    shared_tensor = torch.from_numpy(np_images)
    shared_tensor.share_memory_() # <--- THE MAGIC SAUCE
    
    print(f"Shared Tensor Ready. RAM: {shared_tensor.nelement() / 1e9:.2f} GB")
    return shared_tensor

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
    
    # Load data ONCE in main process
    shared_data = load_data_to_shared_memory("data")
    dataset = SharedMemoryDataset(shared_data)
    
    # 8 Workers reading from Shared Memory = Fast
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,       # Safe now because of share_memory_()
        pin_memory=False,    # Disabled to rule out ROCm pinning issues
        drop_last=True,
        persistent_workers=True
    )
    
    model = VICRegModel().to(DEVICE)
    augmentor = GPUAugment().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            # batch is (B, 64, 64, 3) ByteTensor
            batch = batch.to(DEVICE, non_blocking=True)
            
            # Augment (Includes Permute B,H,W,C -> B,C,H,W)
            with torch.no_grad():
                v1 = augmentor(batch)
                v2 = augmentor(batch)
            
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
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/encoder_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "models/vicreg_encoder_final.pth")

if __name__ == "__main__":
    train()