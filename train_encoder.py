import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import glob
import os
import sys
from tqdm import tqdm
from vicreg import vicreg_loss

# --- CONFIG ---
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 2e-4  # Slightly conservative LR
DEVICE = "cuda"

# --- AUGMENTATION ---
class GPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            # CRITICAL FIX: antialias=False
            # This uses "Nearest Neighbor" or simple Bilinear which is bulletproof on GPUs.
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.transform(x)

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
        self.images = np.transpose(self.images, (0, 3, 1, 2))
        print(f"Dataset Loaded. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
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
    
    # Increase workers for data loading
    dataset = CarRacingDataset("data")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    
    model = VICRegModel().to(DEVICE)
    augmentor = GPUAugment().to(DEVICE)
    
    # We stick to standard AdamW and Float32 for maximum stability
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, batch_imgs in enumerate(pbar):
            # 1. Load to GPU (Float32)
            batch_imgs = batch_imgs.to(DEVICE, non_blocking=True).float() / 255.0
            
            # 2. Augment
            view_1 = augmentor(batch_imgs)
            view_2 = augmentor(batch_imgs)
            
            # SAFETY CHECK: Did augmentation break the images?
            if torch.isnan(view_1).any() or torch.isnan(view_2).any():
                print(f"\n❌ AUGMENTATION FAILURE at Batch {i}")
                print("Disabling augmentation for this run and continuing...")
                # Fallback: Just use raw images if aug fails (keeps training alive)
                view_1 = batch_imgs
                view_2 = batch_imgs

            # 3. Forward Pass
            _, z1 = model(view_1)
            _, z2 = model(view_2)
            
            loss = vicreg_loss(z1, z2)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nPANIC: Loss is NaN/Inf at Batch {i}")
                print("Skipping batch update...")
                optimizer.zero_grad()
                continue # Skip bad batch, don't crash

            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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