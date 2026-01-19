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

cv2.setNumThreads(0) 

from networks import TinyEncoder, TinyDecoder

BATCH_SIZE = 1024      
EPOCHS = 30            
LR = 1e-3              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "./models/encoder_mixed_final.pth"
FINAL_MODEL_PATH = "models/decoder_final.pth"

class StreamingDecoderDataset(IterableDataset):
    def __init__(self):
        self.files = sorted(glob.glob("./data_*/*.npz"))
        self.epoch = 0
        self.total_frames = len(self.files) * 5000 

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        seed = 42 + self.epoch + worker_id
        np_rng = np.random.default_rng(seed)
        
        my_files = self.files[worker_id::num_workers]
        random.shuffle(my_files)
        
        for f in my_files:
            try:
                with np.load(f) as data:
                    if 'states' in data: obs = data['states']
                    elif 'obs' in data: obs = data['obs']
                    else: continue
                
                if obs.shape[1] != 64:
                    obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                
                # We need index >= 3 for stacking
                if len(obs) < 4: continue
                
                indices = np_rng.permutation(np.arange(3, len(obs)))
                
                for idx in indices:
                    # Target Image: The current one (idx)
                    target_img = torch.from_numpy(obs[idx]).float().div_(255.0).permute(2, 0, 1)
                    
                    # Input Stack: idx-3 ... idx
                    stack_frames = obs[idx-3 : idx+1] # 4 frames
                    stack_t = torch.from_numpy(stack_frames).float().div_(255.0)
                    stack_t = stack_t.permute(0, 3, 1, 2).reshape(12, 64, 64)
                    
                    yield stack_t, target_img
                    
            except Exception: continue

    def __len__(self): return self.total_frames

def train():
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    
    decoder = TinyDecoder().to(DEVICE)
    optimizer = optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    dataset = StreamingDecoderDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        decoder.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        last_target, last_recon = None, None
        
        for stack, target in pbar:
            stack = stack.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z = encoder(stack) # (B, 512, 8, 8)
                recon = decoder(z)     # (B, 3, 64, 64)
                loss = criterion(recon, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if random.random() < 0.01:
                last_target, last_recon = target, recon
        
        if last_target is not None:
            comparison = torch.cat([last_target[:8], last_recon[:8]], dim=0)
            save_image(comparison, f"visuals/decoder_ep{epoch+1}.png", nrow=8)

        torch.save(decoder.state_dict(), f"models/decoder_ep{epoch+1}.pth")

    torch.save(decoder.state_dict(), FINAL_MODEL_PATH)

if __name__ == "__main__":
    train()