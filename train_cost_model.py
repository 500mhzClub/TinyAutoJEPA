import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm
from networks import TinyEncoder, CostModel

# --- CONFIG ---
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "./models/encoder_mixed_final.pth"

class DiscriminatorDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        
        # 1. Load POSITIVE samples (Expert/Race Data) -> Label 0 (Low Energy)
        race_files = glob.glob("./data_race/*.npz")
        print(f"Loading {len(race_files)} Expert files (Label 0)...")
        self._load_files(race_files, label=0.0)
        
        # 2. Load NEGATIVE samples (Random Data) -> Label 1 (High Energy)
        # We limit random files to match race files to keep classes balanced
        rand_files = glob.glob("./data/*.npz")[:len(race_files)]
        print(f"Loading {len(rand_files)} Random files (Label 1)...")
        self._load_files(rand_files, label=1.0)
        
        self.data = np.concatenate(self.data, axis=0)
        self.data = np.transpose(self.data, (0, 3, 1, 2)) # NHWC -> NCHW
        self.labels = np.array(self.labels, dtype=np.float32)
        
        print(f"✅ Total Balanced Samples: {len(self.data)}")

    def _load_files(self, files, label):
        for f in tqdm(files):
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue
                    
                    # subsample to save RAM (take every 5th frame)
                    obs = obs[::5]
                    
                    if obs.shape[1] != 64:
                        obs = np.array([cv2.resize(img, (64, 64)) for img in obs])
                        
                    self.data.append(obs)
                    self.labels.extend([label] * len(obs))
            except: pass

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx]).float()/255.0, 
                torch.tensor([self.labels[idx]]))

def train():
    # 1. Load Frozen Encoder
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    
    # 2. Setup Cost Model
    cost_model = CostModel().to(DEVICE)
    optimizer = optim.Adam(cost_model.parameters(), lr=LR)
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    dataset = DiscriminatorDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    print("Training Energy-Based Cost Model...")
    for epoch in range(EPOCHS):
        cost_model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                z = encoder(imgs)
            
            # Predict Energy
            energy = cost_model(z)
            loss = criterion(energy, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    torch.save(cost_model.state_dict(), "./models/cost_model_final.pth")
    print("Cost Model Saved.")

if __name__ == "__main__":
    train()