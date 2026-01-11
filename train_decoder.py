import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm
from networks import TinyEncoder, TinyDecoder

BATCH_SIZE = 128   
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ENCODER_PATH = "./models/encoder_mixed_final.pth"
FINAL_MODEL_PATH = "models/decoder_vicreg_final.pth"

class DecoderDataset(Dataset):
    def __init__(self):
        """
        Loads race and recovery data for decoder training.
        Stores data as uint8 in RAM to maximize diversity without OOM.
        """
        # We focus on race/recovery as these are the "clean" images we want to reconstruct well
        self.files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
        self.data = []
        
        print(f"Loading {len(self.files)} files for Decoder training...")
        
        frame_count = 0
        
        # Load data with a progress bar
        for f in tqdm(self.files, desc="Loading Data"):
            try:
                with np.load(f) as arr:
                    # specific to your data structure
                    if 'states' in arr: 
                        obs = arr['states']
                    elif 'obs' in arr: 
                        obs = arr['obs']
                    else: 
                        continue
                
                # Resize if not 64x64
                if obs.shape[1] != 64:
                    # Quick list comprehension resize
                    obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                
                # Keep as uint8 (0-255) to save 4x RAM compared to float
                self.data.append(obs)
                frame_count += len(obs)
                    
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
        
        if not self.data:
            raise ValueError("No data loaded! Check your data paths.")

        # Concatenate into one massive array
        self.data = np.concatenate(self.data, axis=0)
        
        # Transpose from NHWC (OpenCV/Numpy) -> NCHW (PyTorch)
        # N, H, W, C -> N, C, H, W
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        
        print(f"Decoder Dataset Loaded: {len(self.data):,} frames.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to float and normalize on the fly
        # Matches encoder: float().div_(255.0)
        return torch.from_numpy(self.data[idx]).float() / 255.0

def train():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}! Run train_encoder.py first.")

    print(f"Initializing Decoder Training on {DEVICE}")
    
    print(f"Loading encoder from: {ENCODER_PATH}")
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    #Initialize Decoder
    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    # Data Loader
    dataset = DecoderDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,      
        num_workers=4, 
        drop_last=True,
        pin_memory=True
    )
    
    # Output directories
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        last_imgs = None
        last_recon = None
        epoch_loss = 0.0
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    # Encoder forward (No Grad)
                    with torch.no_grad():
                        z = encoder(imgs)
                    
                    # Decoder forward
                    recon = decoder(z)
                    loss = criterion(recon, imgs)
                
                # Backprop
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback for CPU
                with torch.no_grad():
                    z = encoder(imgs)
                recon = decoder(z)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            last_imgs = imgs
            last_recon = recon
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f}")

        # save Checkpoints & Visuals
        if (epoch+1) % 5 == 0 or (epoch+1) == EPOCHS:
            # Save Checkpoint
            torch.save(decoder.state_dict(), f"models/decoder_vicreg_ep{epoch+1}.pth")
            
            # Take first 8 images from the last batch
            comparison = torch.cat([last_imgs[:8], last_recon[:8]], dim=0)
            save_image(comparison, f"visuals/reconstruct_ep{epoch+1}.png", nrow=8)
            print(f"Saved visual comparison to visuals/reconstruct_ep{epoch+1}.png")

    # Final Save
    torch.save(decoder.state_dict(), FINAL_MODEL_PATH)
    print(f"Decoder Training Complete. Final model saved to: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train()