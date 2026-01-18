import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision.utils import save_image
import numpy as np
import glob
import os
import cv2
import random
from tqdm import tqdm

# --- IMPORT YOUR NETWORKS ---
# Ensure networks.py is in the same directory
from networks import TinyEncoder, TinyDecoder

# --- CONFIG ---
BATCH_SIZE = 1024      # Increased for R9700/3090 class GPUs
EPOCHS = 30            # 30 Epochs is sufficient for convergence
LR = 1e-3              # Standard Adam LR for this batch size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ENCODER_PATH = "./models/encoder_mixed_final.pth"
FINAL_MODEL_PATH = "models/decoder_final.pth"

class StreamingDecoderDataset(IterableDataset):
    def __init__(self):
        """
        Streams data from all available .npz folders.
        """
        # Search for data in all likely locations
        self.files = sorted(
            glob.glob("./data_race/*.npz") + 
            glob.glob("./data_recovery/*.npz") +
            glob.glob("./data_expert/*.npz") + 
            glob.glob("./data_random/*.npz")
        )
        self.epoch = 0
        
        print(f"--- Decoder Dataset Setup ---")
        if not self.files:
            print("❌ ERROR: No .npz files found! Checked: data_race, data_recovery, etc.")
            raise FileNotFoundError("No data found.")
        else:
            print(f"✅ Found {len(self.files)} files.")
            print(f"   (Batch Size: {BATCH_SIZE} | Device: {DEVICE})")
        
        # Rough estimate for progress bar (assuming ~30k frames per file)
        # This doesn't need to be exact, just gives tqdm a target
        self.total_frames = len(self.files) * 30000 

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Seeding for distinct data stream per worker/epoch
        seed = 42 + self.epoch + worker_id
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        # Shard files across workers
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)
        
        for f in my_files:
            try:
                with np.load(f) as data:
                    # Robust key handling
                    if 'states' in data: obs = data['states']
                    elif 'obs' in data: obs = data['obs']
                    else: continue
                
                # Resize if needed (vectorized for speed)
                if obs.shape[1] != 64:
                    # If data is not 64x64, resize it
                    obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                
                # Shuffle frames inside the file
                indices = np_rng.permutation(len(obs))
                
                for idx in indices:
                    # Convert to Tensor, Normalize 0-1
                    # Input: (64, 64, 3) uint8 -> Output: (3, 64, 64) float
                    img = torch.from_numpy(obs[idx]).float().div_(255.0)
                    yield img.permute(2, 0, 1) 
                    
            except Exception:
                continue

    def __len__(self):
        return self.total_frames

def train():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}! Check path.")

    print(f"Initializing Decoder Training...")
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1. Load Frozen Encoder
    print(f"Loading Frozen Encoder: {ENCODER_PATH}")
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    
    # 2. Initialize Decoder
    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # 3. Setup Data
    dataset = StreamingDecoderDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=8,      # High worker count to keep GPU fed
        pin_memory=True,
        persistent_workers=False
    )
    
    # 4. Training Loop
    print("Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        decoder.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0
        batches = 0
        
        # Holders for visualization
        last_imgs, last_recon = None, None
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                # Encode (Frozen)
                with torch.no_grad():
                    z = encoder(imgs)
                
                # Decode (Trainable)
                recon = decoder(z)
                loss = criterion(recon, imgs)
            
            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Grab a batch for visualization every ~500 steps
            if batches % 500 == 0:
                last_imgs, last_recon = imgs, recon
        
        # --- End of Epoch Tasks ---
        
        # 1. Save Visuals
        if last_imgs is not None:
            # Create a grid: Top 8 Real, Bottom 8 Fake
            comparison = torch.cat([last_imgs[:8], last_recon[:8]], dim=0)
            save_image(comparison, f"visuals/decoder_ep{epoch+1}.png", nrow=8)
            print(f"  --> Visuals saved: visuals/decoder_ep{epoch+1}.png")

        # 2. Save Checkpoint
        if (epoch+1) % 5 == 0:
            torch.save(decoder.state_dict(), f"models/decoder_ep{epoch+1}.pth")

    # Save Final
    torch.save(decoder.state_dict(), FINAL_MODEL_PATH)
    print(f"✅ Decoder Training Complete. Saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train()