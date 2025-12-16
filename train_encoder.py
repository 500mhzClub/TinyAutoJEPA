import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import glob
import os
from tqdm import tqdm
from vicreg import vicreg_loss

# --- CONFIG ---
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 3e-4 # Standard VICReg rate
DEVICE = "cuda"

# --- AUGMENTATION (GPU - Eager Mode) ---
# We will NOT compile this class. It runs fast enough in eager mode.
class GPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # Normalization helps prevent NaNs by keeping inputs centered
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.transform(x)

class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # Load all chunks
        for f in tqdm(files, desc="Loading Chunks"):
            try:
                data = np.load(f)
                self.images.append(data['states']) 
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
                
        if not self.images:
            raise RuntimeError("No data found!")

        self.images = np.concatenate(self.images, axis=0)
        # Transpose to (N, 3, 64, 64)
        self.images = np.transpose(self.images, (0, 3, 1, 2))
        
        print(f"Dataset Loaded. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return raw uint8 to speed up transfer to GPU
        return torch.from_numpy(self.images[idx])

class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # Projector
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

# --- TRAINING LOOP ---
def train():
    os.makedirs("models", exist_ok=True)
    
    # 8 Workers for fast data loading
    dataset = CarRacingDataset("data")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    
    model = VICRegModel().to(DEVICE)
    augmentor = GPUAugment().to(DEVICE)
    
    # FIX: Compile ONLY the model, NOT the augmentor
    # The augmentor was causing the "Graph Breaks" and slowing things down.
    try:
        print("Compiling ResNet Model (Standard Mode)...")
        model = torch.compile(model)
    except Exception as e:
        print(f"Compilation skipped: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_imgs in pbar:
            # 1. Move to GPU
            batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
            
            # 2. Convert to Float
            batch_imgs = batch_imgs.float() / 255.0
            
            # 3. Augment (Eager Mode - No Compiler Issues)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                view_1 = augmentor(batch_imgs)
                view_2 = augmentor(batch_imgs)
                
                # 4. Forward Pass
                _, z1 = model(view_1)
                _, z2 = model(view_2)
                
                # FIX: Force Loss Calculation to Float32 to prevent NaN
                loss = vicreg_loss(z1.float(), z2.float())
            
            # 5. Backward Pass
            scaler.scale(loss).backward()
            
            # FIX: Gradient Clipping (Prevents Exploding Gradients)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Check for NaN again just in case
            if torch.isnan(loss):
                print("PANIC: Loss is NaN! Reducing LR or checking data...")
                break
                
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