import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
import cv2 
import random
from tqdm import tqdm
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4    
STEPS_PER_EPOCH = 2000 # Since we stream infinite data, we define an "Epoch" as 2000 batches

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

class StreamingDataset(IterableDataset):
    def __init__(self):
        # We just list files. We DON'T load them yet.
        self.race_files = glob.glob("./data_race/*.npz")
        self.rec_files  = glob.glob("./data_recovery/*.npz")
        
        print(f"--- Streaming Dataset Initialized ---")
        print(f"   Expert Files:   {len(self.race_files)}")
        print(f"   Recovery Files: {len(self.rec_files)}")
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.9, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
        ])

    def load_random_file(self, file_list):
        # Pick a random file and load it
        if not file_list: return np.array([])
        
        f = random.choice(file_list)
        try:
            with np.load(f) as arr:
                if 'states' in arr: obs = arr['states']
                elif 'obs' in arr: obs = arr['obs']
                else: return np.array([])

                # Auto-resize if needed
                if obs.shape[1] != 64:
                    obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                
                # N H W C -> N C H W
                obs = np.transpose(obs, (0, 3, 1, 2))
                return obs
        except:
            return np.array([])

    def __iter__(self):
        # This runs inside every Worker Thread
        while True:
            # 50% chance to pick from Race, 50% from Recovery
            if random.random() < 0.5:
                data = self.load_random_file(self.race_files)
            else:
                data = self.load_random_file(self.rec_files)
                
            if len(data) == 0: continue
            
            # Shuffle the file's content
            indices = np.random.permutation(len(data))
            
            # Yield frames one by one
            for idx in indices:
                img_float = torch.from_numpy(data[idx]).float() / 255.0
                yield self.transform(img_float), self.transform(img_float)

def train():
    # num_workers=4 means 4 background processes act as "DJ's", 
    # constantly loading files and mixing them for the GPU.
    dataset = StreamingDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # Resume Logic
    start_epoch = 0
    import re
    checkpoints = glob.glob("./models/encoder_mixed_ep*.pth")
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(re.search(r'ep(\d+)', f).group(1)))
        print(f"--- RESUMING from {latest} ---")
        encoder.load_state_dict(torch.load(latest, map_location=DEVICE))
        start_epoch = int(re.search(r'ep(\d+)', latest).group(1))

    os.makedirs("models", exist_ok=True)
    encoder.train()
    projector.train()

    # Training Loop
    # Since dataset is infinite, we loop over the dataloader manually
    data_iter = iter(dataloader)

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for _ in pbar:
            try:
                x1, x2 = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader) # Reset if somehow exhausted
                x1, x2 = next(data_iter)

            x1, x2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                z1 = projector(encoder(x1))
                z2 = projector(encoder(x2))
                loss = vicreg_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch+1) % 5 == 0:
            torch.save(encoder.state_dict(), f"models/encoder_mixed_ep{epoch+1}.pth")

    torch.save(encoder.state_dict(), "models/encoder_mixed_final.pth")
    print("Encoder Training Complete.")

if __name__ == "__main__":
    train()