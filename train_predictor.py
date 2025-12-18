import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import sys
from tqdm import tqdm
from networks import TinyEncoder, Predictor

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-4
DATA_PATH = "./data/*.npz"
# Pointing to your checkpoint from Epoch 20
ENCODER_PATH = "./models/encoder_ep20.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Debugging Dataset ---
class DynamicsDataset(Dataset):
    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)
        if not self.files:
            print(f"CRITICAL ERROR: No files found at {file_pattern}")
            print(f"Current working directory: {os.getcwd()}")
            sys.exit(1)

        self.obs_list = []
        self.action_list = []
        self.next_obs_list = []
        
        print(f"Found {len(self.files)} files. Inspecting first file...")
        
        # 1. Inspect the first file to debug keys
        try:
            with np.load(self.files[0]) as first_arr:
                print(f"Keys in first file: {list(first_arr.keys())}")
        except Exception as e:
            print(f"Could not read first file: {e}")

        print("Loading data...")
        success_count = 0
        
        for f in self.files:
            try:
                with np.load(f) as arr:
                    # Try to grab data with fallback keys
                    o = arr['obs'] if 'obs' in arr else arr.get('arr_0')
                    a = arr['action'] if 'action' in arr else arr.get('arr_1')
                    
                    # Robust check
                    if o is None:
                        # Try finding whatever key has the images
                        keys = list(arr.keys())
                        if len(keys) > 0: o = arr[keys[0]]
                    
                    if a is None:
                        # If action is missing, we can't train the predictor
                        # Check if maybe it's under a different name?
                        if 'actions' in arr: a = arr['actions']

                    if o is None or a is None:
                        print(f"Skipping {os.path.basename(f)}: Missing data (Found keys: {list(arr.keys())})")
                        continue

                    # Length check
                    if len(o) != len(a):
                        # Sometimes recording stops early, truncate to shortest
                        min_len = min(len(o), len(a))
                        o = o[:min_len]
                        a = a[:min_len]

                    # Create triplets: (State_t, Action_t) -> State_t+1
                    # We need at least 2 frames to make a prediction
                    if len(o) < 2:
                        continue

                    # Append (excluding last frame for input, excluding first for target)
                    self.obs_list.append(o[:-1])
                    self.action_list.append(a[:-1])
                    self.next_obs_list.append(o[1:])
                    success_count += 1
                    
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        if success_count == 0:
            raise ValueError("Failed to load ANY valid files. Check the 'Skipping' messages above.")

        print(f"Successfully loaded {success_count} files. Concatenating...")
        
        self.obs = np.concatenate(self.obs_list, axis=0)
        self.actions = np.concatenate(self.action_list, axis=0)
        self.next_obs = np.concatenate(self.next_obs_list, axis=0)
        
        # Free memory
        del self.obs_list, self.action_list, self.next_obs_list
        
        # NHWC -> NCHW
        self.obs = np.transpose(self.obs, (0, 3, 1, 2))
        self.next_obs = np.transpose(self.next_obs, (0, 3, 1, 2))
        
        print(f"Total triplets loaded: {len(self.obs)}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx]).float() / 255.0
        # Check action shape. If it's just (3,), fine. 
        action = torch.from_numpy(self.actions[idx]).float()
        next_obs = torch.from_numpy(self.next_obs[idx]).float() / 255.0
        return obs, action, next_obs

def train():
    print(f"Training Predictor on {DEVICE}")
    
    # Load Models
    encoder = TinyEncoder().to(DEVICE)
    if os.path.exists(ENCODER_PATH):
        print(f"Loading Frozen Eye from {ENCODER_PATH}")
        # map_location ensures we don't try to load onto GPU 1 if we are restricted to GPU 0
        state_dict = torch.load(ENCODER_PATH, map_location=DEVICE)
        encoder.load_state_dict(state_dict)
    else:
        print(f"ERROR: Checkpoint {ENCODER_PATH} not found.")
        return

    encoder.eval()
    for param in encoder.parameters(): param.requires_grad = False

    predictor = Predictor(input_dim=512, action_dim=3).to(DEVICE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # Load Data
    try:
        dataset = DynamicsDataset(DATA_PATH)
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(EPOCHS):
        predictor.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for obs, action, next_obs in pbar:
            obs, action, next_obs = obs.to(DEVICE, non_blocking=True), action.to(DEVICE, non_blocking=True), next_obs.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    z_t = encoder(obs)
                    z_t1_true = encoder(next_obs)

                z_t1_pred = predictor(z_t, action)
                loss = criterion(z_t1_pred, z_t1_true)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        if (epoch+1) % 10 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_ep{epoch+1}.pth")

    torch.save(predictor.state_dict(), "models/predictor_final.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train()