import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 50
LR = 3e-4
DATA_PATH = "./data/*.npz" # Ensure this matches your actual path

# ROCm maps 'cuda' calls to the AMD GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset (Unchanged) ---
class CarRacingDataset(Dataset):
    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)
        self.data = []
        if not self.files:
            raise FileNotFoundError(f"No files found at {file_pattern}. Check your path.")
            
        print(f"Loading {len(self.files)} files...")
        for f in self.files:
            try:
                with np.load(f) as arr:
                    # Robust loading: checks for 'obs', 'arr_0', or just takes the first key
                    if 'obs' in arr:
                        obs = arr['obs']
                    elif 'arr_0' in arr:
                        obs = arr['arr_0']
                    else:
                        obs = arr[list(arr.keys())[0]]
                    self.data.append(obs)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        self.data = np.concatenate(self.data, axis=0)
        # Convert NHWC -> NCHW (Channels First)
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        print(f"Total frames: {len(self.data)}")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Memory optimization: Convert to tensor on the fly rather than storing all as float32 in RAM
        img = torch.from_numpy(self.data[idx]).float() / 255.0
        
        view_1 = self.transform(img)
        view_2 = self.transform(img)
        return view_1, view_2

# --- Training ---
def train():
    # RDNA 2 Optimization: Ensure TF32 is off (usually irrelevant for AMD, but good practice)
    # and strictly use the modern AMP backend.
    
    print(f"Training on {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    dataset = CarRacingDataset(DATA_PATH)
    # pinned_memory=True speeds up CPU->GPU transfer
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector(input_dim=512, output_dim=1024).to(DEVICE)

    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    
    # Modern scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        encoder.train()
        projector.train()
        total_loss = 0
        
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()

            # UPDATED: Use torch.amp instead of torch.cuda.amp
            with torch.amp.autocast('cuda'):
                y1 = encoder(x1)
                y2 = encoder(x2)
                
                z1 = projector(y1)
                z2 = projector(y2)

                loss = vicreg_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

        if (epoch+1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(encoder.state_dict(), f"models/encoder_ep{epoch+1}.pth")
            print(f"Saved checkpoint.")

if __name__ == "__main__":
    # Optional: If you encounter 'HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION', uncomment the line below
    # os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    train()