import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
from tqdm import tqdm  # Progress bar

# Import our helper modules
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 50
LR = 3e-4
DATA_PATH = "./data/*.npz" 

# Setup Device (ROCm automatically maps 'cuda' to the GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class CarRacingDataset(Dataset):
    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)
        self.data = []
        
        if not self.files:
            raise FileNotFoundError(f"No files found at {file_pattern}. Please check your data folder.")
            
        print(f"Loading {len(self.files)} files...")
        
        # Load all npz files
        for f in self.files:
            try:
                with np.load(f) as arr:
                    # Robust key checking
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
        
        # Convert NHWC -> NCHW (Channels First) for PyTorch
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        print(f"Total frames: {len(self.data)}")

        # VICReg Augmentations (Crucial for learning)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to tensor on the fly (saves RAM)
        img = torch.from_numpy(self.data[idx]).float() / 255.0
        
        # Create two different views of the same image
        view_1 = self.transform(img)
        view_2 = self.transform(img)
        return view_1, view_2

# --- Training Loop ---
def train():
    print(f"Training on {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load Data
    dataset = CarRacingDataset(DATA_PATH)
    # num_workers=8 utilizes your 5950X efficiently
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # Initialize Models
    encoder = TinyEncoder().to(DEVICE)
    projector = Projector(input_dim=512, output_dim=1024).to(DEVICE)

    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    
    # Modern Mixed Precision Scaler for ROCm
    scaler = torch.amp.GradScaler('cuda')

    # Create model directory
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        encoder.train()
        projector.train()
        total_loss = 0
        
        # Progress Bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch_idx, (x1, x2) in enumerate(pbar):
            # Move data to GPU (non_blocking helps overlap transfer with compute)
            x1, x2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward Pass
            with torch.amp.autocast('cuda'):
                # Encode -> Representation
                y1 = encoder(x1)
                y2 = encoder(x2)
                
                # Project -> Embedding
                z1 = projector(y1)
                z2 = projector(y2)

                # Calculate Loss
                loss = vicreg_loss(z1, z2)

            # Backward Pass & Optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Summary | Average Loss: {avg_loss:.4f}")

        # Save Checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(encoder.state_dict(), f"models/encoder_ep{epoch+1}.pth")
            print(f"Saved models/encoder_ep{epoch+1}.pth")

    # Save final model
    torch.save(encoder.state_dict(), "models/encoder_final.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train()