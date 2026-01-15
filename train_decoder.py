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
from networks import TinyEncoder, TinyDecoder

BATCH_SIZE = 128   
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ENCODER_PATH = "./models/encoder_mixed_final.pth"
FINAL_MODEL_PATH = "models/decoder_vicreg_final.pth"

class StreamingDecoderDataset(IterableDataset):
    def __init__(self):
        """
        Streams Race and Recovery data from disk.
        Uses ~200MB RAM regardless of dataset size.
        """
        self.files = sorted(glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz"))
        self.epoch = 0
        
        if not self.files:
            raise ValueError("No .npz files found! Check ./data_race and ./data_recovery")
            
        print(f"--- Decoder Dataset ---")
        print(f"Found {len(self.files)} files.")
        
        # Optional: Quick scan for total length (for progress bar)
        # If this is too slow, just set self.total_frames = 5700000 (hardcoded)
        self.total_frames = 0
        print("Scanning dataset size...")
        for f in tqdm(self.files):
            try:
                with np.load(f, mmap_mode='r') as d:
                    if 'states' in d: self.total_frames += d['states'].shape[0]
                    elif 'obs' in d: self.total_frames += d['obs'].shape[0]
            except: pass
        print(f"Total Frames: {self.total_frames:,}")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # Seeding for reproducible shuffling that changes every epoch
        seed = 42 + self.epoch + worker_id
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        # Shard files across workers
        worker_files = self.files[worker_id::num_workers]
        rng.shuffle(worker_files)
        
        for f in worker_files:
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue
                
                # Resize if needed (vectorized)
                if obs.shape[1] != 64:
                    obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                
                # Shuffle frames within the file
                indices = np_rng.permutation(len(obs))
                
                for idx in indices:
                    # Normalize to 0-1 float
                    img = torch.from_numpy(obs[idx]).float().div_(255.0)
                    # Yield simple (img) since input==target
                    yield img.permute(2, 0, 1) # HWC -> CHW
                    
            except Exception as e:
                continue

    def __len__(self):
        return self.total_frames

def train():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}! Run train_encoder.py first.")

    print(f"Initializing Decoder Training on {DEVICE}")
    
    # --- Load Frozen Encoder ---
    print(f"Loading encoder from: {ENCODER_PATH}")
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    dataset = StreamingDecoderDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=6,     # Safe to increase now since RAM usage is low
        pin_memory=True,
        persistent_workers=False # Safe for set_epoch
    )

    os.makedirs("visuals", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Starting Training Loop...")
    
    # Handle steps manually for IterableDataset
    steps_per_epoch = len(dataset) // BATCH_SIZE

    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch) # Shuffle data differently each time
        decoder.train()
        
        pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        last_imgs = None
        last_recon = None
        epoch_loss = 0.0
        batches = 0
        
        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    # Encoder forward (Frozen)
                    with torch.no_grad():
                        z = encoder(imgs)
                    
                    # Decoder forward
                    recon = decoder(z)
                    loss = criterion(recon, imgs)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    z = encoder(imgs)
                recon = decoder(z)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            
            # Save for visualization
            if batches % 100 == 0:
                last_imgs = imgs
                last_recon = recon
        
        if batches > 0:
            avg_loss = epoch_loss / batches
            print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f}")

        # Checkpoints & Visuals
        if (epoch+1) % 5 == 0 or (epoch+1) == EPOCHS:
            torch.save(decoder.state_dict(), f"models/decoder_vicreg_ep{epoch+1}.pth")
            
            if last_imgs is not None:
                # Top row: Original, Bottom row: Reconstructed
                comparison = torch.cat([last_imgs[:8], last_recon[:8]], dim=0)
                save_image(comparison, f"visuals/reconstruct_ep{epoch+1}.png", nrow=8)
                print(f"Saved visual comparison to visuals/reconstruct_ep{epoch+1}.png")

    torch.save(decoder.state_dict(), FINAL_MODEL_PATH)
    print(f"Decoder Training Complete.")

if __name__ == "__main__":
    train()