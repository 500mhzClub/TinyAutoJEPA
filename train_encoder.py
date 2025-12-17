import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import glob
import os
import cv2  # The secret weapon for CPU speed
from tqdm import tqdm
from vicreg import vicreg_loss

# --- CONFIG ---
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = "cuda"

# --- OPENCV AUGMENTATION (True Parallel CPU) ---
def turbo_augment(img):
    """
    Performs augmentation using OpenCV (C++).
    This releases the Python GIL, allowing true multi-core processing.
    """
    # img is (64, 64, 3) uint8
    h, w, c = img.shape
    
    # 1. Random Resized Crop
    # We cheat slightly: Random crop, then fast resize back to 64x64
    scale = np.random.uniform(0.8, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    
    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    
    # Slice (Zero copy, very fast)
    crop = img[top:top+new_h, left:left+new_w]
    # Linear resize is much faster than Bicubic and fine for ML
    img = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    # 2. Random Horizontal Flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        
    # 3. Brightness/Contrast (Fast Matrix Ops)
    if np.random.rand() > 0.2:
        # alpha = contrast, beta = brightness
        contrast = np.random.uniform(0.8, 1.2)
        brightness = np.random.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        
    # 4. Gaussian Blur (Optional, helps VICReg)
    if np.random.rand() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    return img

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
        
        # CRITICAL: OpenCV expects (H, W, Channels). 
        # The .npz data is (N, 64, 64, 3). We keep it that way for OpenCV!
        # (We do NOT transpose to (3, 64, 64) yet)
        
        print(f"Dataset Loaded. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get raw image (64, 64, 3)
        img = self.images[idx]
        
        # 2. Augment using C++ OpenCV (Releases GIL)
        v1 = turbo_augment(img)
        v2 = turbo_augment(img)
        
        # 3. Manual ToTensor (HWC -> CHW, 0-255 -> 0.0-1.0)
        # Doing this manually is faster than torchvision.transforms.ToTensor
        v1 = torch.from_numpy(v1).permute(2, 0, 1).float() / 255.0
        v2 = torch.from_numpy(v2).permute(2, 0, 1).float() / 255.0
        
        return v1, v2

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
    
    # 12 Workers to fully utilize the Ryzen 3600X
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

    print(f"Starting training on {DEVICE} (OpenCV Turbo Mode)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for view_1, view_2 in pbar:
            # 1. Move Clean Tensors to GPU
            view_1 = view_1.to(DEVICE, non_blocking=True)
            view_2 = view_2.to(DEVICE, non_blocking=True)
            
            # 2. Forward Pass (Pure Matrix Math on GPU)
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