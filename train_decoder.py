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

# --- CRITICAL FIX FOR STALLS ---
# Prevents OpenCV from spawning threads inside PyTorch workers.
cv2.setNumThreads(0) 

# --- IMPORT YOUR NETWORKS ---
from networks import TinyEncoder, TinyDecoder

# --- CONFIG ---
BATCH_SIZE = 1024      
EPOCHS = 30            
LR = 1e-3              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ENCODER_PATH = "./models/encoder_mixed_final.pth"
FINAL_MODEL_PATH = "models/decoder_final.pth"

class StreamingDecoderDataset(IterableDataset):
    def __init__(self):
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
            print("❌ ERROR: No .npz files found!")
            raise FileNotFoundError("No data found.")
        else:
            print(f"✅ Found {len(self.files)} files.")
            print(f"   (Batch Size: {BATCH_SIZE} | Device: {DEVICE})")
        
        # Rough estimate for progress bar
        self.total_frames = len(self.files) * 20000 

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        seed = 42 + self.epoch + worker_id
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)
        
        for f in my_files:
            try:
                with np.load(f) as data:
                    if 'states' in data: obs = data['states']
                    elif 'obs' in data: obs = data['obs']
                    else: continue
                
                # Resize if needed
                if obs.shape[1] != 64:
                    obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                
                indices = np_rng.permutation(len(obs))
                
                for idx in indices:
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
    if os.path.exists("models/decoder_latest.pth"):
        print("Resuming from decoder_latest.pth...")
        decoder.load_state_dict(torch.load("models/decoder_latest.pth", map_location=DEVICE))
        
    optimizer = optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # 3. Setup Data
    dataset = StreamingDecoderDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4,      # Safe value
        pin_memory=True,
        persistent_workers=False
    )
    
    print("Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        decoder.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        last_imgs, last_recon = None, None
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z = encoder(imgs)
                recon = decoder(z)
                loss = criterion(recon, imgs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Save visuals occasionally for monitoring
            if random.random() < 0.01:
                last_imgs, last_recon = imgs, recon
        
        # --- End of Epoch ---
        if last_imgs is not None:
            comparison = torch.cat([last_imgs[:8], last_recon[:8]], dim=0)
            save_image(comparison, f"visuals/decoder_ep{epoch+1}.png", nrow=8)
            print(f"  --> Visuals saved: visuals/decoder_ep{epoch+1}.png")

        # Save Checkpoint
        torch.save(decoder.state_dict(), f"models/decoder_ep{epoch+1}.pth")
        torch.save(decoder.state_dict(), "models/decoder_latest.pth")

    torch.save(decoder.state_dict(), FINAL_MODEL_PATH)
    print(f"✅ Decoder Training Complete. Saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train()