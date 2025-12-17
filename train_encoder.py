import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2  # GPU-accelerated transforms
import numpy as np
import glob
import os
import psutil
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 512  # Increased: GPU augs allow larger batches
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = "cuda"

# --- LOSS ---
def variance_loss(x, gamma=1.0):
    std = torch.sqrt(x.var(dim=0) + 1e-04)
    loss = torch.mean(F.relu(gamma - std))
    return loss

def covariance_loss(x):
    batch_size, dim = x.shape
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (batch_size - 1)
    off_diag = cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
    loss = off_diag.pow(2).sum() / dim
    return loss

def vicreg_loss(x, y):
    sim_loss = F.mse_loss(x, y)
    std_loss = variance_loss(x) + variance_loss(y)
    cov_loss = covariance_loss(x) + covariance_loss(y)
    return (25.0 * sim_loss) + (25.0 * std_loss) + (1.0 * cov_loss)

# --- GPU AUGMENTATION MODULE ---
class GPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            # Input is (B, 3, 64, 64) uint8 on GPU
            v2.ToDtype(torch.float32, scale=True), # 0-255 -> 0.0-1.0
            v2.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    def forward(self, x):
        return self.transforms(x)

# --- DATASET (Now Dumb & Fast) ---
class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # Check RAM
        mem = psutil.virtual_memory()
        print(f"System RAM Available: {mem.available / 1e9:.2f} GB")
        
        for f in tqdm(files, desc="Loading Chunks"):
            try:
                data = np.load(f)
                self.images.append(data['states']) 
            except Exception as e:
                print(f"Skipping {f}: {e}")
                
        if not self.images:
            raise RuntimeError("No data found!")

        self.images = np.concatenate(self.images, axis=0)
        print(f"Dataset Loaded. {len(self.images)} images. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # FAST PATH: No processing here. Just strict copy.
        img = self.images[idx] # (64, 64, 3)
        # Permute needed because PyTorch wants (C, H, W)
        # We return ByteTensor (uint8) to save bandwidth moving to GPU
        return torch.from_numpy(img).permute(2, 0, 1)

class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
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

def train():
    os.makedirs("models", exist_ok=True)
    
    dataset = CarRacingDataset("data")
    
    # 4 Workers is plenty when they do zero processing
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True
    )
    
    model = VICRegModel().to(DEVICE)
    augmentor = GPUAugment().to(DEVICE) # Augmentation lives on GPU now
    
    # Compile both for max speed on RDNA 4
    # (Note: sometimes v2 transforms don't like compile, if it crashes, remove compile on augmentor)
    print("Compiling model...")
    model = torch.compile(model)
    # augmentor = torch.compile(augmentor) # Optional: Uncomment if stable
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Starting training on {DEVICE} (GPU Augmentation Mode)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            # 1. Move Raw Uint8 Batch to GPU (Very fast)
            batch = batch.to(DEVICE, non_blocking=True)
            
            # 2. Augment on GPU (Parallel & Fast)
            with torch.no_grad():
                view_1 = augmentor(batch)
                view_2 = augmentor(batch)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 3. Forward & Loss (BFloat16)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, z1 = model(view_1)
                _, z2 = model(view_2)
                loss = vicreg_loss(z1, z2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/encoder_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "models/vicreg_encoder_final.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train()