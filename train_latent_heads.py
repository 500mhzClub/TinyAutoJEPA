#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple

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
train_latent_heads_lookahead_proper.py

Goal: train MPC heads that provide EARLY turn-in signal.

Key fixes vs a naive “single scanline” xoff label:
- Road/X-offset labels are computed from a FAR (lookahead) band of rows, not only near-field.
- Road mask uses HSV + low-saturation grey heuristic with grass rejection.
- X-offset uses the road segment closest to car center (handles split segments better).
- Speed labels are NORMALIZED by SPD_SCALE and trained with Huber (SmoothL1) to avoid dominating loss.

This produces a checkpoint compatible with drive_mpc.py:
  {"head_road": ..., "head_spd": ..., "head_xoff": ..., "meta": {...}}

Recommended run:
PREDICTOR_SPACE=raw \
ENCODER_PATH=./models/encoder_mixed_final.pth \
HEADS_OUT=./models/latent_heads_lookahead.pth \
BATCH_SIZE=512 EPOCHS=10 LR=3e-4 \
Y_FAR0=36 Y_FAR1=46 \
Y_NEAR0=50 Y_NEAR1=58 \
SPD_SCALE=80 W_ROAD=5.0 W_XOFF=3.0 W_SPD=0.1 \
python train_latent_heads_lookahead_proper.py
"""

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class CFG:
    # I/O
    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")
    out_path: str = os.getenv("HEADS_OUT", "./models/latent_heads_lookahead.pth")

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
    data_glob: str = os.getenv("DATA_GLOB", "./data_*/*.npz")
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))
    stride: int = int(os.getenv("STRIDE", "4"))
    buffer_size: int = int(os.getenv("BUFFER_SIZE", "400"))
    val_file_frac: float = float(os.getenv("VAL_FILE_FRAC", "0.15"))
    max_files: int = int(os.getenv("MAX_FILES", "0"))

    # Loss weights
    w_road: float = float(os.getenv("W_ROAD", "5.0"))
    w_spd: float = float(os.getenv("W_SPD", "0.1"))
    w_xoff: float = float(os.getenv("W_XOFF", "3.0"))

    # Lookahead label geometry
    y_near0: int = int(os.getenv("Y_NEAR0", "50"))
    y_near1: int = int(os.getenv("Y_NEAR1", "58"))
    y_far0: int = int(os.getenv("Y_FAR0", "36"))
    y_far1: int = int(os.getenv("Y_FAR1", "46"))

    # ROI around car center for road confidence
    center_x0: int = int(os.getenv("CENTER_X0", "28"))
    center_x1: int = int(os.getenv("CENTER_X1", "36"))

    # Speed normalization
    spd_scale: float = float(os.getenv("SPD_SCALE", "80.0"))  # divide recorded speed by this

    # Sanity
    xoff_clip: float = float(os.getenv("XOFF_CLIP", "0.5"))
    min_road_pixels_row: int = int(os.getenv("MIN_ROAD_PIXELS_ROW", "8"))

    # Repro / validation
    seed: int = int(os.getenv("SEED", "1337"))
    val_batches: int = int(os.getenv("VAL_BATCHES", "50"))

CFG = CFG()
DEVICE = torch.device(CFG.device)


# ---------------------------------------------------------------------
# Vision grounding: road mask + lookahead xoff
# ---------------------------------------------------------------------
def _road_mask(frame_u8: np.ndarray) -> np.ndarray:
    """
    Boolean road mask for 64x64 RGB.
    Road ~ grey (low saturation), mid value; reject green grass.
    """
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)

    road = (s < 70) & (v > 35) & (v < 220)
    grassish = (h > 35) & (h < 95) & (s > 80) & (v > 50)
    road = road & (~grassish)
    return road


def _row_road_center(is_road_row: np.ndarray, car_center_x: float = 31.5, min_pix: int = 8) -> Tuple[bool, float]:
    idx = np.where(is_road_row)[0]
    if idx.size < min_pix:
        return False, car_center_x

    splits = np.where(np.diff(idx) > 1)[0]
    segments = []
    start = 0
    for s in splits:
        segments.append(idx[start:s + 1])
        start = s + 1
    segments.append(idx[start:])

    best_center = car_center_x
    best_dist = 1e9
    for seg in segments:
        if seg.size < min_pix:
            continue
        c = 0.5 * (float(seg[0]) + float(seg[-1]))
        d = abs(c - car_center_x)
        if d < best_dist:
            best_dist = d
            best_center = c

    return best_dist < 1e9, best_center


def analyze_ground_truth(frame_u8: np.ndarray) -> Tuple[float, float]:
    """
    Returns (road_confidence, xoff_lookahead) from pixels.
    road_confidence in [0,1], xoff in [-0.5,0.5].
    """
    if frame_u8.shape[0] != CFG.img_size or frame_u8.shape[1] != CFG.img_size:
        frame_u8 = cv2.resize(frame_u8, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)

    road = _road_mask(frame_u8)

    # Continuous road confidence: center patch in NEAR band
    y0, y1 = max(0, CFG.y_near0), min(CFG.img_size, CFG.y_near1)
    x0, x1 = max(0, CFG.center_x0), min(CFG.img_size, CFG.center_x1)
    patch = road[y0:y1, x0:x1]
    road_conf = float(np.mean(patch)) if patch.size else 0.0

    # Lookahead xoff: FAR band average road center
    car_center_x = 31.5
    yf0, yf1 = max(0, CFG.y_far0), min(CFG.img_size, CFG.y_far1)
    centers = []
    for y in range(yf0, yf1):
        ok, c = _row_road_center(road[y], car_center_x=car_center_x, min_pix=CFG.min_road_pixels_row)
        if ok:
            centers.append(c)

    if len(centers) < 2:
        return road_conf, 0.0

    road_center_x = float(np.mean(centers))
    pixel_diff = car_center_x - road_center_x  # + => car right of road center
    xoff = float(pixel_diff / float(CFG.img_size))
    xoff = float(np.clip(xoff, -CFG.xoff_clip, CFG.xoff_clip))
    return road_conf, xoff


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

    new_ckpt = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt.items()}
    enc.load_state_dict(new_ckpt, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
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

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        rng = random.Random(self.epoch * 999 + worker_id * 77 + CFG.seed)
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)

        hist = self.frame_stack - 1
        buf = []

        for f in my_files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    if "obs" in data:
                        obs = data["obs"]
                    elif "states" in data:
                        obs = data["states"]
                    else:
                        continue

                    spd = None
                    for k in ["speed", "spd", "velocity"]:
                        if k in data:
                            spd = data[k]
                            break
                    if spd is None:
                        spd = np.zeros(len(obs), dtype=np.float32)

                    if obs.ndim != 4:
                        continue

                    # Ensure 64x64 uint8 RGB
                    if obs.shape[1] != CFG.img_size:
                        new_obs = np.zeros((len(obs), CFG.img_size, CFG.img_size, 3), dtype=np.uint8)
                        for i in range(len(obs)):
                            new_obs[i] = cv2.resize(obs[i], (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
                        obs = new_obs
                    else:
                        if obs.dtype != np.uint8:
                            obs = np.clip(obs, 0, 255).astype(np.uint8)

                    T = len(obs)
                    for t in range(hist, T, self.stride):
                        stack_frames = obs[t - hist:t + 1]  # (S,64,64,3)
                        curr = stack_frames[-1]
                        y_road, y_xoff = analyze_ground_truth(curr)

                        # Normalize speed label to keep loss scale sane
                        y_spd = float(spd[t]) / float(CFG.spd_scale)
                        y_spd = float(np.clip(y_spd, 0.0, 2.0))

                        buf.append((stack_frames, y_road, y_spd, y_xoff))
                        if len(buf) >= self.buffer_size:
                            rng.shuffle(buf)
                            for item in buf:
                                yield self._format(*item)
                            buf = []
            except Exception:
                continue

        rng.shuffle(buf)
        for item in buf:
            yield self._format(*item)

    def _format(self, stack: np.ndarray, yr: float, ys: float, yx: float):
        t = torch.from_numpy(stack).permute(0, 3, 1, 2).reshape(-1, CFG.img_size, CFG.img_size)
        return (
            t.to(torch.uint8),
            torch.tensor([yr], dtype=torch.float32),
            torch.tensor([ys], dtype=torch.float32),
            torch.tensor([yx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
def train():
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    os.makedirs(os.path.dirname(CFG.out_path), exist_ok=True)

    files = sorted(glob.glob(CFG.data_glob))
    if not files:
        raise RuntimeError(f"No data found for DATA_GLOB={CFG.data_glob}")

    if CFG.max_files > 0:
        random.shuffle(files)
        files = files[:CFG.max_files]

    random.shuffle(files)
    split = int(len(files) * (1.0 - CFG.val_file_frac))
    train_files = files[:split]
    val_files = files[split:]

    train_ds = VisionGroundedDataset(train_files, CFG.frame_stack, CFG.img_size, CFG.stride, CFG.buffer_size)
    val_ds = VisionGroundedDataset(val_files, CFG.frame_stack, CFG.img_size, max(1, CFG.stride * 2), CFG.buffer_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        persistent_workers=False,  # iterable dataset + set_epoch
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        num_workers=max(1, CFG.num_workers // 2),
        persistent_workers=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    in_ch = 3 * CFG.frame_stack
    print(f"Loading encoder from {CFG.encoder_path} ...")
    enc = load_encoder(CFG.encoder_path, in_ch)

    head_road = MLPHead(512, 1).to(DEVICE)
    head_spd = MLPHead(512, 1).to(DEVICE)
    head_xoff = MLPHead(512, 1).to(DEVICE)

    opt = optim.AdamW(
        list(head_road.parameters()) + list(head_spd.parameters()) + list(head_xoff.parameters()),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    def eval_val() -> Tuple[float, float, float]:
        head_road.eval(); head_spd.eval(); head_xoff.eval()
        acc = []
        with torch.no_grad():
            for i, (x_u8, y_road, y_spd, y_xoff) in enumerate(val_loader):
                if i >= CFG.val_batches:
                    break
                x = x_u8.to(DEVICE, dtype=torch.float32) / 255.0
                y_road = y_road.to(DEVICE)
                y_spd = y_spd.to(DEVICE)
                y_xoff = y_xoff.to(DEVICE)

                z = enc(x)
                if CFG.predictor_space == "norm":
                    z = F.normalize(z, p=2, dim=1)
                feat = F.adaptive_avg_pool2d(z, 1).flatten(1)

                road = torch.sigmoid(head_road(feat))
                spd = head_spd(feat)
                xoff = torch.tanh(head_xoff(feat)) * 0.5

                l_road = F.mse_loss(road, y_road)
                l_spd = F.smooth_l1_loss(spd, y_spd, beta=0.10)  # Huber on normalized speed
                l_xoff = F.mse_loss(xoff, y_xoff)

                total = CFG.w_road*l_road + CFG.w_spd*l_spd + CFG.w_xoff*l_xoff
                acc.append((l_road.item(), l_spd.item(), l_xoff.item(), total.item()))
        if not acc:
            return 0.0, 0.0, 0.0
        arr = np.array(acc, dtype=np.float32)
        return float(arr[:,0].mean()), float(arr[:,2].mean()), float(arr[:,3].mean())

    print("Starting grounded head training (lookahead xoff + normalized speed)...")
    print(f"[cfg] SPD_SCALE={CFG.spd_scale} Y_FAR={CFG.y_far0}:{CFG.y_far1} Y_NEAR={CFG.y_near0}:{CFG.y_near1}")

    for epoch in range(CFG.epochs):
        train_ds.set_epoch(epoch)
        head_road.train(); head_spd.train(); head_xoff.train()

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CFG.epochs}")
        for x_u8, y_road, y_spd, y_xoff in pbar:
            x = x_u8.to(DEVICE, dtype=torch.float32) / 255.0
            y_road = y_road.to(DEVICE)
            y_spd = y_spd.to(DEVICE)
            y_xoff = y_xoff.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                with torch.no_grad():
                    z = enc(x)
                    if CFG.predictor_space == "norm":
                        z = F.normalize(z, p=2, dim=1)
                    feat = F.adaptive_avg_pool2d(z, 1).flatten(1)

                pred_road = torch.sigmoid(head_road(feat))
                pred_spd = head_spd(feat)
                pred_xoff = torch.tanh(head_xoff(feat)) * 0.5

                loss_road = F.mse_loss(pred_road, y_road)
                loss_spd = F.smooth_l1_loss(pred_spd, y_spd, beta=0.10)
                loss_xoff = F.mse_loss(pred_xoff, y_xoff)

                loss = CFG.w_road*loss_road + CFG.w_spd*loss_spd + CFG.w_xoff*loss_xoff

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(head_road.parameters()) + list(head_spd.parameters()) + list(head_xoff.parameters()),
                CFG.grad_clip,
            )
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(L=f"{loss.item():.3f}", road=f"{loss_road.item():.4f}", spd=f"{loss_spd.item():.4f}", xoff=f"{loss_xoff.item():.5f}")

        vroad, vxoff, vtot = eval_val()
        print(f"[val] road_mse={vroad:.5f} xoff_mse={vxoff:.6f} total={vtot:.3f}")

        torch.save(
            {
                "head_road": head_road.state_dict(),
                "head_spd": head_spd.state_dict(),
                "head_xoff": head_xoff.state_dict(),
                "meta": {
                    "predictor_space": CFG.predictor_space,
                    "frame_stack": CFG.frame_stack,
                    "img_size": CFG.img_size,
                    "y_far": [CFG.y_far0, CFG.y_far1],
                    "y_near": [CFG.y_near0, CFG.y_near1],
                    "spd_scale": CFG.spd_scale,
                },
            },
            CFG.out_path,
        )

    print(f"Done. Saved heads to: {CFG.out_path}")


if __name__ == "__main__":
    train()
