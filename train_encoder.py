import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm
import psutil

# --- CRITICAL FIX 1: Prevent Thread Explosion ---
# Forces OpenCV to run single-threaded. 
# We rely on PyTorch DataLoaders (multiprocessing) for parallelism instead.
cv2.setNumThreads(0)

# --- CONFIG ---
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = "cuda"

# --- VICReg LOSS FUNCTIONS ---
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

# --- AUGMENTATION ---
def turbo_augment(img):
    h, w, c = img.shape
    
    # 1. Random Resized Crop
    scale = np.random.uniform(0.8, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    
    crop = img[top:top+new_h, left:left+new_w]
    img = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    # 2. Random Horizontal Flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        
    # 3. Brightness/Contrast
    if np.random.rand() > 0.2:
        contrast = np.random.uniform(0.8, 1.2)
        brightness = np.random.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        
    # 4. Gaussian Blur
    if np.random.rand() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    return img

class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # Memory Safety Check
        mem = psutil.virtual_memory()
        print(f"RAM Available: {mem.available / 1e9:.2f} GB")
        
        for f in tqdm(files, desc="Loading Chunks"):
            try:
                data = np.load(f)
                self.images.append(data['states']) 
            except Exception as e:
                print(f"Skipping {f}: {e}")
                
        if not self.images:
            raise RuntimeError("No data found!")

        self.images = np.concatenate(self.images, axis=0)
        print(f"Dataset Loaded. Size: {len(self.images)} images. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Augment twice
        v1 = turbo_augment(img)
        v2 = turbo_augment(img)
        
        # HWC -> CHW, Normalize
        v1 = torch.from_numpy(v1).permute(2, 0, 1).float() / 255.0
        v2 = torch.from_numpy(v2).permute(2, 0, 1).float() / 255.0
        
        return v1, v2

class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use standard ResNet18
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # 3-layer Projector per VICReg paper
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
    
    # --- CRITICAL FIX 2: Tuned Workers ---
    # Ryzen 3600X (6 cores) gets clogged with 12 workers + heavy augmentation.
    # 6 is optimal.
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True
    )
    
    model = VICRegModel().to(DEVICE)
    
    # --- CRITICAL FIX 3: Torch Compile ---
    # Fuses kernels for higher GPU utilization
    print("Compiling model for RDNA 4...")
    model = torch.compile(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    # --- CRITICAL FIX 4: Mixed Precision Scaler ---
    scaler = torch.amp.GradScaler('cuda')

    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for view_1, view_2 in pbar:
            # Move to GPU
            view_1 = view_1.to(DEVICE, non_blocking=True)
            view_2 = view_2.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # --- CRITICAL FIX 5: BFloat16 Autocast ---
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