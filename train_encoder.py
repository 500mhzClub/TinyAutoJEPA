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

# --- AUGMENTATION (MOVED TO CPU) ---
# We define this globally so the CPU workers can access it.
# The CPU handles the resizing (bypassing the GPU driver bug) 
# and hands a finished tensor to the GPU.
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # Load only 'states' to save memory
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
        # 1. Get image as uint8 numpy array
        img = self.images[idx]
        
        # 2. Convert to Tensor on CPU
        # This creates a Float tensor (0.0 to 1.0)
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        # 3. Augment on CPU (The Fix!)
        # The CPU workers handle this in parallel.
        v1 = augment_transform(img_tensor)
        v2 = augment_transform(img_tensor)
        
        return v1, v2

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
    
    # We increase num_workers to 8 so the Ryzen 3600X can crush the image resizing 
    # faster than the GPU can train.
    dataset = CarRacingDataset("data")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VICRegModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda") 

    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # The dataloader now yields TWO pre-augmented views
        for v1, v2 in pbar:
            # Move to GPU only AFTER augmentation is done
            v1 = v1.to(DEVICE, non_blocking=True)
            v2 = v2.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # The GPU now ONLY does Matrix Math (BF16), which it excels at.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, z1 = model(v1)
                _, z2 = model(v2)
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