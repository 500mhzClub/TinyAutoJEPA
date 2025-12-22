import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import cv2
from tqdm import tqdm
from networks import TinyEncoder, Predictor

# --- CONFIGURATION ---
BATCH_SIZE = 128    # Lowered slightly because sequences take more VRAM
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MULTI-STEP CONFIG
PRED_HORIZON = 5    # Predict 5 steps into the future (0.5s of physics)
SEQUENCE_LEN = PRED_HORIZON + 1 # We need T to T+5

ENCODER_PATH = "./models/encoder_mixed_ep5.pth" 
if not os.path.exists(ENCODER_PATH): 
    # Fallback to whatever you have locally
    ENCODER_PATH = "./models/encoder_final_mixed.pth"

class MultiStepDataset(Dataset):
    def __init__(self):
        # We load both datasets to understand Grass + Road physics
        self.files = glob.glob("./data/*.npz") + glob.glob("./data_race/*.npz")
        print(f"Loading Episodes from {len(self.files)} files...")
        
        self.episodes = [] # Store each episode separately to avoid boundary crossing
        self.valid_indices = [] # Map (episode_idx, start_frame)
        
        total_frames = 0
        
        for f in tqdm(self.files, desc="Indexing Episodes"):
            try:
                with np.load(f) as arr:
                    if 'states' in arr: o, a = arr['states'], arr['actions']
                    elif 'obs' in arr: o, a = arr['obs'], arr['action']
                    else: continue
                    
                    # Sanity checks
                    if len(o) != len(a): 
                        min_l = min(len(o), len(a))
                        o = o[:min_l]; a = a[:min_l]
                    
                    # Need at least sequence length to be useful
                    if len(o) < SEQUENCE_LEN: continue

                    # Auto-Resize if needed
                    if o.shape[1] != 64:
                        o = np.array([cv2.resize(img, (64, 64)) for img in o])

                    # Store episode
                    # Transpose now to save time later: N H W C -> N C H W
                    o = np.transpose(o, (0, 3, 1, 2))
                    
                    # Save normalized float version to save GPU/CPU bandwidth during training
                    # (Optional: keep uint8 to save RAM if you are tight)
                    self.episodes.append({
                        'obs': o, # uint8
                        'act': a  # float32
                    })
                    
                    # Register valid start indices for this episode
                    # We can start anywhere from 0 to (Length - Sequence_Len)
                    ep_idx = len(self.episodes) - 1
                    num_valid_starts = len(o) - SEQUENCE_LEN + 1
                    for t in range(num_valid_starts):
                        self.valid_indices.append((ep_idx, t))
                        
                    total_frames += len(o)
            except: pass
            
        print(f"Loaded {len(self.episodes)} episodes.")
        print(f"Total Frames: {total_frames:,}")
        print(f"Total Training Sequences: {len(self.valid_indices):,}")

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_idx, start_t = self.valid_indices[idx]
        data = self.episodes[ep_idx]
        
        # Slicing the window
        end_t = start_t + SEQUENCE_LEN
        
        # Get Window
        obs_seq = data['obs'][start_t : end_t]      # Shape: [SEQ_LEN, 3, 64, 64]
        act_seq = data['act'][start_t : end_t - 1]  # Shape: [SEQ_LEN-1, Action_Dim]
        
        # Convert to Tensor & Normalize
        obs_seq = torch.from_numpy(obs_seq).float() / 255.0
        act_seq = torch.from_numpy(act_seq).float()
        
        return obs_seq, act_seq

def train():
    print(f"🚀 Training Multi-Step Predictor (Horizon={PRED_HORIZON}) on {DEVICE}")
    
    # 1. Load Frozen Encoder
    encoder = TinyEncoder().to(DEVICE)
    print(f"📥 Loading Encoder: {ENCODER_PATH}")
    # Force load on GPU to handle map location issues
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    # 2. Setup Predictor
    predictor = Predictor().to(DEVICE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # 3. Dataset
    # Check RAM usage here. If you crash, reduce file count in glob inside dataset
    dataset = MultiStepDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=8, pin_memory=True, drop_last=True)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        predictor.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for obs_seq, act_seq in pbar:
            # obs_seq shape: [B, T+1, C, H, W]
            # act_seq shape: [B, T, Action_Dim]
            
            # Reshape for Encoder: Flatten Time and Batch dimensions
            # We encode ALL frames at once to be efficient
            B, T_plus_1, C, H, W = obs_seq.shape
            
            obs_flat = obs_seq.view(-1, C, H, W).to(DEVICE, non_blocking=True)
            act_seq = act_seq.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # 1. Get Ground Truth Latents (Z)
                with torch.no_grad():
                    z_flat = encoder(obs_flat)
                    z_seq = z_flat.view(B, T_plus_1, -1) # [B, T+1, Latent_Dim]
                
                # 2. Unroll Predictions (The Chain Reaction)
                loss = 0
                
                # Start with the true current state
                z_current = z_seq[:, 0, :] 
                
                for t in range(PRED_HORIZON):
                    # Action to take at this step
                    action = act_seq[:, t, :]
                    
                    # Target is the NEXT true state
                    z_target = z_seq[:, t+1, :]
                    
                    # Predict
                    z_next_pred = predictor(z_current, action)
                    
                    # Calculate Loss for this step
                    loss += criterion(z_next_pred, z_target)
                    
                    # CRITICAL: For the next step, use our PREDICTION, not the ground truth.
                    # This is "Autoregressive Training". It forces the model to 
                    # fix its own drift.
                    z_current = z_next_pred

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item() / PRED_HORIZON:.6f}") # Avg loss per step

        # Save Checkpoint
        if (epoch+1) % 5 == 0:
            torch.save(predictor.state_dict(), f"models/predictor_multistep_ep{epoch+1}.pth")

    torch.save(predictor.state_dict(), "models/predictor_multistep_final.pth")
    print("✅ Predictor Training Complete.")

if __name__ == "__main__":
    train()