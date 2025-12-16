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
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # 1. OPTIMIZATION: Only load 'states' to avoid duplicates
        # 'next_states' is just 'states' shifted by one, so we don't need it for the Encoder.
        for f in tqdm(files, desc="Loading Chunks"):
            try:
                data = np.load(f)
                self.images.append(data['states']) 
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
                
        if not self.images:
            raise RuntimeError("No data found! Did you copy the 'data' folder?")

        # 2. OPTIMIZATION: Keep as uint8 in RAM! (1 byte per pixel vs 8 bytes)
        self.images = np.concatenate(self.images, axis=0)
        
        # Current Shape: (N, 64, 64, 3) -> We want (N, 3, 64, 64)
        # Transpose is fast on uint8
        self.images = np.transpose(self.images, (0, 3, 1, 2))
        
        print(f"Dataset Loaded. RAM Usage: {self.images.nbytes / 1e9:.2f} GB")
        print(f"Total Images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 3. ON-THE-FLY CONVERSION:
        # Only convert a single image to float when requested.
        # This keeps the massive array in RAM small.
        img = self.images[idx]
        
        # Convert numpy uint8 -> torch float32 and Normalize (0-1)
        return torch.from_numpy(img).float() / 255.0

# --- AUGMENTATION PIPELINE ---
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: ResNet18 (No fc layer)
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # Projector: 3-layer MLP
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
        h = self.encoder(x)          # [Batch, 512, 1, 1]
        h = h.view(h.size(0), -1)    # [Batch, 512]
        z = self.projector(h)        # [Batch, 2048]
        return h, z

# --- TRAINING LOOP ---
def train():
    os.makedirs("models", exist_ok=True)
    
    # Setup Data
    # pin_memory=True speeds up transfer from RAM to GPU
    dataset = CarRacingDataset("data")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Setup Model
    model = VICRegModel().to(DEVICE)
    try:
        model = torch.compile(model)
        print("PyTorch 2.0 Compiler Active.")
    except Exception as e:
        print(f"Compiler skipped: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda") 

    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_imgs in pbar:
            batch_imgs = batch_imgs.to(DEVICE)
            
            # Create two augmented views
            view_1 = augment_transform(batch_imgs)
            view_2 = augment_transform(batch_imgs)
            
            optimizer.zero_grad()
            
            # AutoCast to BFloat16 (Native on RDNA 4)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    train()