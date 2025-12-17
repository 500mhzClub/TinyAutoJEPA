#!/usr/bin/env python3
"""
VICReg (ResNet18) training script tuned for ROCm/RDNA:
- Uses DataLoader + pinned memory + non_blocking H2D
- Reorders torchvision v2 pipeline (keep uint8 for resize; float32 only after)
- Optional AMP (fp16) for large speedups when FP32 is slow on ROCm
- Avoids hard-coding BATCH_SIZE inside the loss (uses actual batch size)
- Produces channels_last tensors directly from the augmentor to avoid extra copies

Notes:
- If you still see /opt/amdgpu/share/libdrm/amdgpu.ids warnings, that's a system packaging/path issue
  and not necessarily the cause of slowness, but it is worth fixing separately.
"""

import os
import glob
import time
import numpy as np
from tqdm import tqdm

import torch
# --- CRITICAL FIX: CACHE KERNELS ---
# This forces PyTorch to find the best algorithm ONCE and re-use it.
# Essential for RDNA cards when kernel databases are missing.
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchvision import models
from torchvision.transforms import v2


# --------------------
# CONFIG
# --------------------
DATA_DIR = "data"
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 3e-4

DEVICE = "cuda"  # ROCm uses the "cuda" device string in PyTorch
NUM_WORKERS = 4  # increase if you have CPU headroom
PERSISTENT_WORKERS = True
PIN_MEMORY = True

USE_AMP = True           # HIGH impact if you're FP32-bound
AMP_DTYPE = torch.float16  # try float16 first; bf16 support varies by stack

USE_CHANNELS_LAST = True
ANTIALIAS = True         # try False if aug is slow
PRINT_TIMINGS = True     # prints per-iteration timing breakdown for quick triage
TIMING_ITERS = 30        # only print timings for the first N iterations each epoch


# --------------------
# VICREG LOSS
# --------------------
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    # flatten then drop diag elements efficiently
    x = x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    return x

def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    # Similarity loss
    sim_loss = F.mse_loss(z1, z2)

    # Variance loss (use unbiased=False for speed/stability)
    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-4)
    std_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    # Covariance loss
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    b = z1.size(0)
    denom = max(b - 1, 1)

    cov_z1 = (z1.T @ z1) / denom
    cov_z2 = (z2.T @ z2) / denom

    d = z1.size(1)
    cov_loss = off_diagonal(cov_z1).pow(2).sum() / d + off_diagonal(cov_z2).pow(2).sum() / d

    return (25.0 * sim_loss) + (25.0 * std_loss) + (1.0 * cov_loss)


# --------------------
# AUGMENTATION
# --------------------
class GPUAugment(nn.Module):
    """
    Key perf rule: keep uint8 for resize/crop when possible; convert to float after.
    Also returns channels_last if enabled to avoid repeated conversions downstream.
    """
    def __init__(self, antialias: bool = True, channels_last: bool = True):
        super().__init__()
        self.channels_last = channels_last

        self.transforms = nn.Sequential(
            # If your data is already uint8 [0..255], this is a no-op.
            # If it's float, this will scale to uint8 only if you pass scale=True on ToDtype(torch.uint8).
            # But ToDtype(..., scale=True) expects float in [0..1] or uint8 in [0..255].
            v2.ToDtype(torch.uint8, scale=True),

            v2.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=antialias),
            v2.RandomHorizontalFlip(p=0.5),

            # Convert to float only after resize/crop
            v2.ToDtype(torch.float32, scale=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, H, W, C). Convert to NCHW for torchvision.
        x = x.permute(0, 3, 1, 2)
        x = self.transforms(x)

        if self.channels_last:
            # Produce NHWC memory layout for downstream conv perf (where it helps)
            x = x.contiguous(memory_format=torch.channels_last)
        return x


# --------------------
# MODEL
# --------------------
class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # drop fc

        self.projector = nn.Sequential(
            nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
        )

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        z = self.projector(h)
        return h, z


# --------------------
# DATA LOADING
# --------------------
def load_all_data(data_dir: str) -> torch.Tensor:
    print(f"Loading data from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    np_images = []

    for f in tqdm(files, desc="Reading Disk"):
        try:
            np_images.append(np.load(f)["states"])
        except Exception:
            pass

    if not np_images:
        raise RuntimeError(f"No usable .npz files found in {data_dir}")

    np_images = np.concatenate(np_images, axis=0)
    t = torch.from_numpy(np_images)
    return t


# --------------------
# TIMING UTILS
# --------------------
def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    os.makedirs("models", exist_ok=True)

    # Backend knobs (harmless if ignored on your build)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Device sanity
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False — ROCm not active?"
    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
    print("using device:", DEVICE)

    # Load to CPU RAM
    all_data = load_all_data(DATA_DIR)

    # Use DataLoader to get shuffle + pinned-memory batches
    ds = TensorDataset(all_data)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and (NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    # Model + augmentor
    model = VICRegModel().to(DEVICE)
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    augmentor = GPUAugment(antialias=ANTIALIAS, channels_last=USE_CHANNELS_LAST).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    print(f"Starting training on {DEVICE} | AMP={USE_AMP}({AMP_DTYPE}) | channels_last={USE_CHANNELS_LAST} | antialias={ANTIALIAS}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, (batch,) in pbar:
            # batch comes as a CPU tensor; move to GPU
            if PRINT_TIMINGS and step < TIMING_ITERS:
                t0 = sync_time()

            batch = batch.to(DEVICE, non_blocking=True)

            if PRINT_TIMINGS and step < TIMING_ITERS:
                t1 = sync_time()

            # Augment on GPU (no grad)
            with torch.no_grad():
                v1 = augmentor(batch)
                v2 = augmentor(batch)

            if PRINT_TIMINGS and step < TIMING_ITERS:
                t2 = sync_time()

            optimizer.zero_grad(set_to_none=True)

            # Forward + loss under AMP
            with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
                _, z1 = model(v1)
                _, z2 = model(v2)

            if PRINT_TIMINGS and step < TIMING_ITERS:
                t3 = sync_time()

            # Compute VICReg loss in fp32 for stability (cheap vs the matmuls themselves)
            loss = vicreg_loss(z1.float(), z2.float())

            if PRINT_TIMINGS and step < TIMING_ITERS:
                t4 = sync_time()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if PRINT_TIMINGS and step < TIMING_ITERS:
                t5 = sync_time()

            total_loss += float(loss.item())

            if PRINT_TIMINGS and step < TIMING_ITERS:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "H2D_ms": f"{(t1 - t0) * 1000:.1f}",
                    "aug_ms": f"{(t2 - t1) * 1000:.1f}",
                    "fwd_ms": f"{(t3 - t2) * 1000:.1f}",
                    "loss_ms": f"{(t4 - t3) * 1000:.1f}",
                    "bwd+opt_ms": f"{(t5 - t4) * 1000:.1f}",
                })
            else:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg:.4f}")

        # Optional checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, f"models/vicreg_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    main()
