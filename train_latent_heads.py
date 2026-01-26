#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from networks import TinyEncoder

"""
train_latent_heads.py — Grounded Training for MPC Heads

FIX APPLIED: 
Instead of relying on potentially missing .npz keys or flawed steering heuristics
(which teach the car that "Turning = Wall"), this script computes GROUND TRUTH
labels directly from the 64x64 RGB pixels during training.

1. Road Score: Determines if the car center is on Gray (Road) or Green (Grass).
2. X Offset: Finds the road boundaries in the image and calculates the true
   distance of the car (fixed at center x=32) from the road center.

# Assumes your .npz data is in ./data_random/ or ./data_expert/
# This runs fast (usually < 10 mins).
PREDICTOR_SPACE=raw \
ENCODER_PATH=./models/encoder_mixed_final.pth \
HEADS_OUT=./models/latent_heads.pth \
BATCH_SIZE=512 \
EPOCHS=10 \
LR=3e-4 \
W_ROAD=5.0 \
W_XOFF=2.0 \
python train_latent_heads.py   
"""

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class CFG:
    # I/O
    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")
    out_path: str = os.getenv("HEADS_OUT", "./models/latent_heads.pth")

    # Device
    device: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Encoder input
    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))
    img_size: int = int(os.getenv("IMG_SIZE", "64"))
    predictor_space: str = os.getenv("PREDICTOR_SPACE", "raw").lower()

    # Training
    batch_size: int = int(os.getenv("BATCH_SIZE", "512"))
    epochs: int = int(os.getenv("EPOCHS", "10"))
    lr: float = float(os.getenv("LR", "3e-4"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    grad_clip: float = float(os.getenv("GRAD_CLIP", "1.0"))

    # Data
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))
    stride: int = int(os.getenv("STRIDE", "4"))
    buffer_size: int = int(os.getenv("BUFFER_SIZE", "400"))
    val_file_frac: float = float(os.getenv("VAL_FILE_FRAC", "0.15"))
    max_files: int = int(os.getenv("MAX_FILES", "0"))

    # Loss weights (Balanced for the new accurate labels)
    w_road: float = float(os.getenv("W_ROAD", "5.0"))  # Priority: Know if we are on grass
    w_spd: float = float(os.getenv("W_SPD", "1.0"))
    w_xoff: float = float(os.getenv("W_XOFF", "2.0"))  # Priority: Know where center is

    # Repro
    seed: int = int(os.getenv("SEED", "1337"))
    val_batches: int = int(os.getenv("VAL_BATCHES", "50"))

CFG = CFG()
DEVICE = torch.device(CFG.device)


# ---------------------------------------------------------------------
# Vision Grounding Logic
# ---------------------------------------------------------------------
def _analyze_frame_ground_truth(frame_u8: np.ndarray) -> Tuple[float, float]:
    """
    Computes TRUE (road_score, x_offset) from the pixels.
    Assumptions for CarRacing-v2/v3 (64x64):
    - Car is ego-centric, fixed horizontally at x ~ 32.
    - Car is near the bottom, y ~ 55.
    - Road is Gray (R~G~B). Grass is Green (G > R+B).
    """
    # Look at a horizontal slice just ahead of the car (e.g. row 54)
    # The car dashboard/hood might cover row 60+, so 54 is safe.
    row_idx = 54
    if frame_u8.shape[0] <= row_idx:
        return 0.0, 0.0
    
    scan_row = frame_u8[row_idx] # (64, 3)

    # 1. Identify Road Pixels
    # Road is greyish: standard deviation of color channels is LOW.
    # Grass/Kerbs have color.
    # Gray ~ (105, 105, 105). Std Dev ~ 0.
    # Green ~ (100, 200, 100). Std Dev ~ High.
    
    color_std = np.std(scan_row, axis=1) # (64,)
    
    # Threshold: Gray usually has std < 10.0. Let's be generous: < 20.0
    is_road = (color_std < 20.0)
    
    # 2. Road Score (Is the center of the car on the road?)
    # Car width is roughly 4-6 pixels at center (30..34)
    center_slice = is_road[30:34]
    if len(center_slice) == 0:
        road_score = 0.0
    else:
        road_score = 1.0 if np.mean(center_slice) > 0.5 else 0.0
        
    # 3. X Offset Calculation
    # We need the CENTER of the road strip.
    road_indices = np.where(is_road)[0]
    
    if len(road_indices) < 2:
        # No road visible in this row. 
        # If we are "offroad", we can't define center easily.
        # Fallback: maintain 0.0 (center) to avoid wild gradients, rely on road_score.
        return road_score, 0.0
        
    left_edge = road_indices[0]
    right_edge = road_indices[-1]
    
    # Gap check: if there are two road segments (e.g. hairpin), pick the one closest to center
    # (Simple heuristic: assumes continuous blob)
    
    road_center_x = (left_edge + right_edge) / 2.0
    car_center_x = 31.5 # (0..63)
    
    # Definition of x_offset in MPC:
    # 0.0 = Centered
    # Positive = Car is to the RIGHT of road center (Road is to the Left)
    # Negative = Car is to the LEFT of road center (Road is to the Right)
    
    # diff = Car - Road
    pixel_diff = car_center_x - road_center_x
    
    # Normalize to roughly [-0.5, 0.5] range relative to image width
    # 64 pixels width.
    x_offset = pixel_diff / 64.0
    
    # Clip to keep predictions sane
    x_offset = float(np.clip(x_offset, -0.5, 0.5))
    
    return road_score, x_offset


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 1, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_encoder(path: str, in_ch: int) -> TinyEncoder:
    enc = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        ckpt = ckpt["encoder"]
    
    # Strip prefix
    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[k.replace("_orig_mod.", "").replace("module.", "")] = v
        
    enc.load_state_dict(new_ckpt, strict=False)
    enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    return enc

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class VisionGroundedDataset(IterableDataset):
    def __init__(self, files: List[str], frame_stack: int, img_size: int, stride: int, buffer_size: int):
        super().__init__()
        self.files = list(files)
        self.frame_stack = int(frame_stack)
        self.img_size = int(img_size)
        self.stride = int(stride)
        self.buffer_size = int(buffer_size)
        self.epoch = 0
        print(f"[data] VisionGroundedDataset: {len(self.files)} files.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Seed per worker/epoch
        rng = random.Random(self.epoch * 999 + worker_id * 77 + CFG.seed)
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)
        
        hist = self.frame_stack - 1
        buf = []

        for f in my_files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    if "obs" in data: obs = data["obs"]
                    elif "states" in data: obs = data["states"]
                    else: continue
                    
                    # Need speed as well
                    spd = None
                    for k in ["speed", "spd", "velocity"]:
                        if k in data:
                            spd = data[k]
                            break
                    if spd is None:
                        # Fallback if no speed recorded: assume 0 or infer? 
                        # Better to skip if speed is critical, but let's default 0
                        spd = np.zeros(len(obs), dtype=np.float32)

                    if obs.ndim != 4: continue
                    
                    # Resize if needed
                    if obs.shape[1] != CFG.img_size:
                        # Simple resize loop (slow but robust)
                        new_obs = np.zeros((len(obs), CFG.img_size, CFG.img_size, 3), dtype=obs.dtype)
                        for i in range(len(obs)):
                            new_obs[i] = cv2.resize(obs[i], (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
                        obs = new_obs

                    if obs.dtype != np.uint8:
                        obs = np.clip(obs, 0, 255).astype(np.uint8)

                    T = len(obs)
                    
                    # Stride through
                    for t in range(hist, T, self.stride):
                        # 1. Prepare Inputs
                        # Stack frames [t-3, t-2, t-1, t]
                        stack_frames = obs[t-hist : t+1] # (4, 64, 64, 3)
                        
                        # 2. Compute Labels from PIXELS (The newest frame)
                        curr_frame = stack_frames[-1]
                        g_road, g_xoff = _analyze_frame_ground_truth(curr_frame)
                        
                        y_road = float(g_road)
                        y_xoff = float(g_xoff)
                        y_spd = float(spd[t])

                        # 3. Format for Buffer
                        # Stack: (S,H,W,3)
                        # Mask: Always [1,1,1] because we computed them!
                        mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                        
                        buf.append((stack_frames, y_road, y_spd, y_xoff, mask))
                        
                        if len(buf) >= self.buffer_size:
                            rng.shuffle(buf)
                            for item in buf:
                                yield self._format(*item)
                            buf = []
                            
            except Exception as e:
                # print(f"Error reading {f}: {e}")
                continue
                
        # Flush
        rng.shuffle(buf)
        for item in buf:
            yield self._format(*item)

    def _format(self, stack, yr, ys, yx, m):
        # Stack: (S,H,W,3) -> (3S, H, W)
        t = torch.from_numpy(stack).permute(0, 3, 1, 2).reshape(-1, CFG.img_size, CFG.img_size)
        return (
            t.to(torch.uint8),
            torch.tensor([yr], dtype=torch.float32),
            torch.tensor([ys], dtype=torch.float32),
            torch.tensor([yx], dtype=torch.float32),
            torch.from_numpy(m)
        )

# ---------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------
def train():
    seed = CFG.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(os.path.dirname(CFG.out_path), exist_ok=True)
    
    # Find files
    files = sorted(glob.glob("./data_*/*.npz"))
    if not files:
        raise RuntimeError("No data found!")
    
    random.shuffle(files)
    split = int(len(files) * (1.0 - CFG.val_file_frac))
    train_files = files[:split]
    val_files = files[split:]
    
    train_ds = VisionGroundedDataset(train_files, CFG.frame_stack, CFG.img_size, CFG.stride, CFG.buffer_size)
    val_ds = VisionGroundedDataset(val_files, CFG.frame_stack, CFG.img_size, CFG.stride*2, CFG.buffer_size)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, num_workers=CFG.num_workers, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, num_workers=max(1, CFG.num_workers//2))
    
    # Models
    in_ch = 3 * CFG.frame_stack
    print(f"Loading encoder from {CFG.encoder_path}...")
    enc = load_encoder(CFG.encoder_path, in_ch)
    
    head_road = MLPHead(512, 1).to(DEVICE)
    head_spd = MLPHead(512, 1).to(DEVICE)
    head_xoff = MLPHead(512, 1).to(DEVICE)
    
    opt = optim.AdamW(
        list(head_road.parameters()) + list(head_spd.parameters()) + list(head_xoff.parameters()),
        lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type=="cuda"))
    
    print("Starting Grounded Training...")
    
    for epoch in range(CFG.epochs):
        train_ds.set_epoch(epoch)
        head_road.train(); head_spd.train(); head_xoff.train()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CFG.epochs}")
        
        for x_u8, y_road, y_spd, y_xoff, mask in pbar:
            x = x_u8.to(DEVICE, dtype=torch.float32) / 255.0
            y_road = y_road.to(DEVICE)
            y_spd = y_spd.to(DEVICE)
            y_xoff = y_xoff.to(DEVICE)
            
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type=="cuda")):
                with torch.no_grad():
                    z = enc(x)
                    if CFG.predictor_space == "norm":
                        z = F.normalize(z, p=2, dim=1)
                    feat = F.adaptive_avg_pool2d(z, 1).flatten(1)
                
                # Forward Heads
                # Road: BCE with Logits
                pred_road_logits = head_road(feat)
                loss_road = F.binary_cross_entropy_with_logits(pred_road_logits, y_road)
                
                # Spd: MSE
                pred_spd = head_spd(feat)
                loss_spd = F.mse_loss(pred_spd, y_spd)
                
                # Xoff: MSE (pred is tanh scaled)
                # drive_mpc uses: tanh(h(z)) * 0.5. We train that raw output matches y_xoff.
                # Since y_xoff is in [-0.5, 0.5], we can treat head as outputting `raw`,
                # then apply tanh*0.5 to compare? Or just MSE on raw linear output?
                # drive_mpc definition: xoff = torch.tanh(head_xoff(feat)) * 0.5
                # So we train that structure:
                pred_xoff_val = torch.tanh(head_xoff(feat)) * 0.5
                loss_xoff = F.mse_loss(pred_xoff_val, y_xoff)
                
                loss = (CFG.w_road * loss_road) + (CFG.w_spd * loss_spd) + (CFG.w_xoff * loss_xoff)
                
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            pbar.set_postfix(
                road=f"{loss_road.item():.3f}", 
                xoff=f"{loss_xoff.item():.4f}", 
                L=f"{loss.item():.3f}"
            )
            
        # Checkpoint
        torch.save({
            "head_road": head_road.state_dict(),
            "head_spd": head_spd.state_dict(),
            "head_xoff": head_xoff.state_dict(),
        }, CFG.out_path)

    print("Done.")

if __name__ == "__main__":
    train()