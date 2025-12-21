import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
import cv2 
from tqdm import tqdm
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 30  # Increased slightly for the new diverse data
LR = 1e-4    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Balanced Dataset ---
class BalancedDataset(Dataset):
    def __init__(self):
        print("--- ⚖️  Constructing Balanced Dataset ---")
        
        # 1. Load Datasets Separately
        self.random_data = self.load_from_folder("./data/*.npz", "Random/Exploration")
        self.race_data   = self.load_from_folder("./data_race/*.npz", "Expert/Race")
        
        # 2. Calculate the Bottleneck
        n_random = len(self.random_data)
        n_race = len(self.race_data)
        min_len = min(n_random, n_race)
        
        print(f"\n📊 Balancing Strategy:")
        print(f"   - Random Frames: {n_random:,}")
        print(f"   - Race Frames:   {n_race:,}")
        print(f"   - Target Count:  {min_len:,} per group (50/50 split)")
        
        # 3. Undersample the Majority Class to match Minority
        # This prevents the model from ignoring the "rare" events (physics on grass)
        idx_random = np.random.choice(n_random, min_len, replace=False)
        idx_race   = np.random.choice(n_race, min_len, replace=False)
        
        balanced_random = self.random_data[idx_random]
        balanced_race   = self.race_data[idx_race]
        
        # 4. Merge
        self.data = np.concatenate([balanced_random, balanced_race], axis=0)
        
        # Free up memory immediately
        del self.random_data
        del self.race_data
        
        # NHWC -> NCHW
        self.data = np.transpose(self.data, (0, 3, 1, 2)) 
        print(f"✅ Final Dataset Size: {len(self.data):,} frames")

        # 5. Augmentations (Crucial for Representation Learning)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5), # Flip 50% of time
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(3),
        ])

    def load_from_folder(self, glob_pattern, label):
        files = glob.glob(glob_pattern)
        print(f"Loading {label}: Found {len(files)} files...")
        
        data_list = []
        for f in tqdm(files, desc=f"Reading {label}"):
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue
                    
                    # Auto-Resize protection
                    if obs.shape[1] != 64:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                        
                    data_list.append(obs)
            except: pass
            
        if not data_list: return np.array([])
        return np.concatenate(data_list, axis=0)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        # Float conversion happens here to save RAM
        img = torch.from_numpy(self.data[idx]).float() / 255.0
        return self.transform(img), self.transform(img)

# --- Training Loop ---
def train():
    print(f"🚀 Training Encoder on {DEVICE}")
    
    # Initialize Dataset
    try:
        dataset = BalancedDataset()
    except MemoryError:
        print("!!! OOM Error: Dataset too large for RAM. Reduce 'min_len' manually or add swap.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # --- Resume Logic ---
    path = None
    if os.path.exists("./models/encoder_mixed_final.pth"): path = "./models/encoder_mixed_final.pth"
    elif os.path.exists("./models/encoder_final.pth"): path = "./models/encoder_final.pth"

    if path:
        print(f"--- RESUMING from {path} ---")
        checkpoint = torch.load(path, map_location=DEVICE)
        encoder.load_state_dict(checkpoint)
    else:
        print("--- STARTING FRESH (No previous encoder found) ---")

    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        encoder.train()
        projector.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x1, x2 in pbar:
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

        # Save checkpoint
        if (epoch+1) % 5 == 0:
            torch.save(encoder.state_dict(), f"models/encoder_mixed_ep{epoch+1}.pth")

    # Save Final
    torch.save(encoder.state_dict(), "models/encoder_mixed_final.pth")
    print("\n✅ Encoder Training Complete.")
    print("The model now understands both 'Exploration' (Physics) and 'Racing' (Task) features.")

if __name__ == "__main__":
    train()