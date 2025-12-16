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
LEARNING_RATE = 3e-4
DEVICE = "cuda"

# --- CPU AUGMENTATION PIPELINE ---
# We run this on the CPU to bypass the GPU driver bug.
# CPU operations in PyTorch are extremely stable.
cpu_transform = transforms.Compose([
    transforms.ToPILImage(), # Convert tensor/numpy to PIL for standard transforms
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True), # Safe on CPU!
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.ToTensor(), # Convert back to Tensor (0.0 to 1.0)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        # We keep data as Channel Last (H, W, C) for PIL compatibility
        print(f"Dataset Loaded. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get Raw Image (uint8)
        img = self.images[idx]
        
        # 2. Augment TWICE on CPU
        # This runs in a background process, so it doesn't block the GPU
        view_1 = cpu_transform(img)
        view_2 = cpu_transform(img)
        
        return view_1, view_2

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
    
    # CRITICAL: num_workers=12 uses your Ryzen 3600X fully.
    # pin_memory=True speeds up the CPU -> GPU transfer.
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    model = VICRegModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    print(f"Starting training on {DEVICE} (CPU Augmentation Pipeline)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for view_1, view_2 in pbar:
            # 1. Move Clean Tensors to GPU
            view_1 = view_1.to(DEVICE, non_blocking=True)
            view_2 = view_2.to(DEVICE, non_blocking=True)
            
            # 2. Forward Pass
            _, z1 = model(view_1)
            _, z2 = model(view_2)
            
            loss = vicreg_loss(z1, z2)
            
            # 3. Backward
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