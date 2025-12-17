import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import glob
import os
import sys
from tqdm import tqdm
from vicreg import vicreg_loss
import kornia.augmentation as K

# --- CONFIG ---
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = "cuda"

# --- KORNIA GPU PIPELINE ---
# Kornia implements transforms using differentiable matrix math (grid_sample),
# which bypasses the specific "upsample" driver bug on RDNA 4.
class KorniaAugment(nn.Module):
    def __init__(self):
        super().__init__()
        # We wrap it in a Sequential container
        # same_on_batch=False ensures every image gets a DIFFERENT random crop
        self.aug = K.AugmentationSequential(
            K.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), antialias=True),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                        std=torch.tensor([0.229, 0.224, 0.225])),
            data_keys=["input"]
        )

    def forward(self, x):
        return self.aug(x)

class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        for f in tqdm(files, desc="Loading Chunks"):
            try:
                data = np.load(f)
                self.images.append(data['states']) 
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
                
        if not self.images:
            raise RuntimeError("No data found!")

        self.images = np.concatenate(self.images, axis=0)
        # Kornia expects (B, C, H, W) in float 0-1
        self.images = np.transpose(self.images, (0, 3, 1, 2))
        
        print(f"Dataset Loaded. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return raw uint8 tensor
        return torch.from_numpy(self.images[idx])

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
    
    # We reduce workers because the CPU does almost nothing now.
    # It just hands memory to the GPU.
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    model = VICRegModel().to(DEVICE)
    augmentor = KorniaAugment().to(DEVICE) # The Magic Fix
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    print(f"Starting training on {DEVICE} (Kornia GPU Pipeline)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_imgs in pbar:
            # 1. Move Raw Uint8 to GPU (Fast Transfer)
            batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
            
            # 2. Convert to Float (0-1) on GPU
            batch_imgs = batch_imgs.float() / 255.0
            
            # 3. Kornia Augmentation (Runs via matrix math, bypassing broken kernels)
            view_1 = augmentor(batch_imgs)
            view_2 = augmentor(batch_imgs)
            
            # 4. Forward & Backward
            _, z1 = model(view_1)
            _, z2 = model(view_2)
            
            loss = vicreg_loss(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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