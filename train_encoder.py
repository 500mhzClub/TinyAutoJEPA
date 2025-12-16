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
BATCH_SIZE = 256        # RDNA 4 loves large batches
EPOCHS = 50             # Should converge quickly
LEARNING_RATE = 3e-4
DEVICE = "cuda"         # We know it's available now!

class CarRacingDataset(Dataset):
    def __init__(self, data_dir):
        print(f"Loading data from {data_dir}...")
        self.images = []
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        # Load all chunks into RAM (It's only ~2-4GB, you have 32GB RAM)
        for f in tqdm(files, desc="Loading Chunks"):
            try:
                data = np.load(f)
                # We combine 'states' and 'next_states' to double our data
                # Shape is (N, 64, 64, 3)
                self.images.append(data['states'])
                self.images.append(data['next_states'])
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
                
        if not self.images:
            raise RuntimeError("No data found! Did you copy the 'data' folder?")

        self.images = np.concatenate(self.images, axis=0)
        # Convert to Channel-First (N, 3, 64, 64) and float 0-1
        self.images = np.transpose(self.images, (0, 3, 1, 2)) / 255.0
        self.images = torch.FloatTensor(self.images)
        print(f"Dataset Loaded. Total Images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

# --- AUGMENTATION PIPELINE ---
# The core of VICReg: Generate two different views of the same image
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Encoder (Backbone)
        # We use ResNet18 but remove the final classification layer
        resnet = models.resnet18(weights=None)
        # Remove the fully connected layer and the pooling layer to get raw features if needed,
        # but for simplicity, we just remove the fc layer and keep the average pool.
        # ResNet18 output before fc is 512.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # 2. Expander (Projector)
        # VICReg requires a large MLP projector for the loss calculation
        # We throw this away after training
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
        # x shape: [Batch, 3, 64, 64]
        h = self.encoder(x)          # [Batch, 512, 1, 1]
        h = h.view(h.size(0), -1)    # [Batch, 512]
        z = self.projector(h)        # [Batch, 2048]
        return h, z

# --- TRAINING LOOP ---
def train():
    os.makedirs("models", exist_ok=True)
    
    # 1. Setup Data
    dataset = CarRacingDataset("data")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Setup Model
    model = VICRegModel().to(DEVICE)
    
    # Optional: Compile for speed on RDNA 4
    try:
        model = torch.compile(model)
        print("PyTorch 2.0 Compiler Active.")
    except Exception as e:
        print(f"Compiler skipped: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    # Mixed Precision Scaler (Crucial for 9060 XT performance)
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
                # Forward Pass
                _, z1 = model(view_1)
                _, z2 = model(view_2)
                
                loss = vicreg_loss(z1, z2)
            
            # Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/encoder_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "models/vicreg_encoder_final.pth")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    train()