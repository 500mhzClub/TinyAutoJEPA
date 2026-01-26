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


# ---------------------------------------------------------------------
# Example commands
# ---------------------------------------------------------------------
# Baseline (recommended; aligns with MPC heads_eval ranges)
# PREDICTOR_SPACE=raw \
# DEVICE=cuda \
# ENCODER_PATH=./models/encoder_mixed_final.pth \
# FRAME_STACK=4 IMG_SIZE=64 \
# BATCH_SIZE=512 NUM_WORKERS=8 EPOCHS=10 LR=3e-4 \
# STRIDE=4 BUFFER_SIZE=400 \
# VAL_FILE_FRAC=0.15 \
# python train_heads.py
#
# If your MPC uses normed BCHW latents:
# PREDICTOR_SPACE=norm \
# ... \
# python train_heads.py


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

    # MUST match MPC / predictor training convention
    predictor_space: str = os.getenv("PREDICTOR_SPACE", "raw").lower()  # raw | norm

    # Training
    batch_size: int = int(os.getenv("BATCH_SIZE", "512"))
    epochs: int = int(os.getenv("EPOCHS", "10"))
    lr: float = float(os.getenv("LR", "3e-4"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    grad_clip: float = float(os.getenv("GRAD_CLIP", "0.0"))

    # Data
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))
    stride: int = int(os.getenv("STRIDE", "4"))
    buffer_size: int = int(os.getenv("BUFFER_SIZE", "400"))
    val_file_frac: float = float(os.getenv("VAL_FILE_FRAC", "0.15"))
    max_files: int = int(os.getenv("MAX_FILES", "0"))  # 0 => all

    # Sampling / labels
    label_horizon: int = int(os.getenv("LABEL_HORIZON", "8"))  # for fallback heuristics over actions
    steer_index: int = int(os.getenv("STEER_INDEX", "0"))

    # Loss weights
    w_road: float = float(os.getenv("W_ROAD", "1.0"))
    w_spd: float = float(os.getenv("W_SPD", "1.0"))
    w_xoff: float = float(os.getenv("W_XOFF", "1.0"))

    # Road floor label thresholding fallback
    road_offroad_key_bias: float = float(os.getenv("ROAD_OFFROAD_KEY_BIAS", "0.0"))

    # Repro
    seed: int = int(os.getenv("SEED", "1337"))

    # Validation
    val_batches: int = int(os.getenv("VAL_BATCHES", "50"))

CFG = CFG()
assert CFG.predictor_space in ("raw", "norm"), "PREDICTOR_SPACE must be 'raw' or 'norm'"
DEVICE = torch.device(CFG.device)


# ---------------------------------------------------------------------
# Small heads (must match drive_mpc.py)
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


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_worker_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cv2.setNumThreads(0)


def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        out[k.replace("_orig_mod.", "").replace("module.", "")] = v
    return out


def _pick_obs(npz: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "states" in npz:
        return npz["states"]
    if "obs" in npz:
        return npz["obs"]
    if "observations" in npz:
        return npz["observations"]
    return None


def _pick_actions(npz: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "actions" in npz:
        return npz["actions"]
    if "action" in npz:
        return npz["action"]
    if "act" in npz:
        return npz["act"]
    return None


def _pick_speed(npz: np.lib.npyio.NpzFile, T_obs: int) -> np.ndarray:
    # primary: explicit per-step speed
    for k in ("speed", "spd", "v", "vel", "velocity"):
        if k in npz:
            a = npz[k]
            if a.ndim == 1:
                a = a.astype(np.float32, copy=False)
                if a.shape[0] >= T_obs:
                    return a[:T_obs]
                out = np.zeros((T_obs,), dtype=np.float32)
                out[: a.shape[0]] = a
                return out
    return np.zeros((T_obs,), dtype=np.float32)


def _pick_road(npz: np.lib.npyio.NpzFile, T_obs: int) -> Optional[np.ndarray]:
    """
    Returns float32 road_score in [0,1] if found.
    Accepts a variety of plausible keys from save_metrics pipelines.
    """
    # direct probability/score
    for k in ("road", "road_score", "on_road", "roadness", "is_road"):
        if k in npz:
            a = npz[k]
            if a.ndim == 1:
                a = a.astype(np.float32, copy=False)
                if a.max() > 1.5:  # sometimes 0/255
                    a = a / 255.0
                a = np.clip(a, 0.0, 1.0)
                if a.shape[0] >= T_obs:
                    return a[:T_obs]
                out = np.zeros((T_obs,), dtype=np.float32)
                out[: a.shape[0]] = a
                return out

    # boolean-ish offroad flags -> road = 1 - offroad
    for k in ("off_road", "offroad", "is_offroad"):
        if k in npz:
            a = npz[k]
            if a.ndim == 1:
                a = a.astype(np.float32, copy=False)
                if a.shape[0] >= T_obs:
                    a = a[:T_obs]
                else:
                    out = np.zeros((T_obs,), dtype=np.float32)
                    out[: a.shape[0]] = a
                    a = out
                # bias hook (default 0.0; you can nudge if your flag is inverted/noisy)
                road = 1.0 - np.clip(a + CFG.road_offroad_key_bias, 0.0, 1.0)
                return np.clip(road, 0.0, 1.0)

    return None


def _pick_xoff(npz: np.lib.npyio.NpzFile, T_obs: int) -> Optional[np.ndarray]:
    """
    Returns float32 x_offset in [-0.5,+0.5] if found.
    Accepts common keys: x_offset, cte, track_pos, center_offset.
    If source is outside range, it is squashed into [-0.5,+0.5] conservatively.
    """
    for k in ("x_offset", "xoff", "center_offset", "track_pos", "cte", "cross_track_error"):
        if k in npz:
            a = npz[k]
            if a.ndim == 1:
                a = a.astype(np.float32, copy=False)
                if a.shape[0] >= T_obs:
                    a = a[:T_obs]
                else:
                    out = np.zeros((T_obs,), dtype=np.float32)
                    out[: a.shape[0]] = a
                    a = out

                # normalize to [-0.5,+0.5]
                # - if already in range, keep
                # - else squash using tanh to avoid huge outliers
                if np.nanmax(np.abs(a)) <= 0.6:
                    return np.clip(a, -0.5, 0.5)
                # common: cte might be ~[-2,+2] (or bigger); squash
                return (0.5 * np.tanh(a)).astype(np.float32, copy=False)

    return None


def _resize_obs_if_needed(obs: np.ndarray, img_size: int) -> np.ndarray:
    if obs.shape[1] == img_size and obs.shape[2] == img_size:
        return obs
    out = np.empty((obs.shape[0], img_size, img_size, 3), dtype=obs.dtype)
    for i in range(obs.shape[0]):
        out[i] = cv2.resize(obs[i], (img_size, img_size), interpolation=cv2.INTER_AREA)
    return out


def _make_stack_u8(obs: np.ndarray, idx: int, stack: int) -> np.ndarray:
    """
    obs: (T,H,W,3) uint8
    idx: current time index (must satisfy idx-(stack-1) >= 0)
    returns: (stack,H,W,3) uint8
    """
    s = stack - 1
    sl = obs[(idx - s) : (idx + 1)]
    return sl


def norm_bchw(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, p=2, dim=1)


def pool_feat(z_bchw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(z_bchw, 1).flatten(1)  # (B,512)


# ---------------------------------------------------------------------
# Dataset (streaming, sharded)
# ---------------------------------------------------------------------
class ShardedHeadsDataset(IterableDataset):
    """
    Yields:
      x_u8:   (3*stack, H, W) uint8
      y_road: (1,) float32 in [0,1]
      y_spd:  (1,) float32 (raw units as stored)
      y_xoff: (1,) float32 in [-0.5,+0.5]
      mask:   (3,) float32 where 1 indicates label is valid for [road, spd, xoff]
    """
    def __init__(self, files: List[str], frame_stack: int, img_size: int, stride: int, buffer_size: int):
        super().__init__()
        self.files = list(files)
        self.frame_stack = int(frame_stack)
        self.img_size = int(img_size)
        self.stride = int(stride)
        self.buffer_size = int(buffer_size)
        self.epoch = 0
        print(f"[data] files={len(self.files)} stack={self.frame_stack} stride={self.stride} buffer={self.buffer_size}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        _set_worker_env()
        wi = get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        rng = random.Random(self.epoch * 10007 + worker_id * 1009 + CFG.seed)
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)

        hist = self.frame_stack - 1
        buf: List[Tuple[np.ndarray, float, float, float, np.ndarray]] = []

        for f in my_files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    obs = _pick_obs(data)
                    act = _pick_actions(data)

                    if obs is None or obs.ndim != 4:
                        continue

                    obs = _resize_obs_if_needed(obs, self.img_size)
                    if obs.dtype != np.uint8:
                        obs = np.clip(obs, 0, 255).astype(np.uint8, copy=False)

                    T_obs = int(obs.shape[0])
                    spd = _pick_speed(data, T_obs)               # always returns (T_obs,)
                    road = _pick_road(data, T_obs)               # optional
                    xoff = _pick_xoff(data, T_obs)               # optional

                    # Conservative alignment for actions (if used for fallbacks)
                    if act is not None:
                        act = act.astype(np.float32, copy=False)
                        if act.shape[0] > T_obs - 1:
                            act = act[: max(0, T_obs - 1)]

                    # Iterate valid indices for stacking
                    for t in range(hist, T_obs, self.stride):
                        # Build frame stack
                        stack = _make_stack_u8(obs, t, self.frame_stack)  # (S,H,W,3)
                        # Labels (prefer true metrics; fallback heuristics otherwise)
                        m = np.zeros((3,), dtype=np.float32)

                        # road label
                        if road is not None:
                            y_road = float(road[t])
                            m[0] = 1.0
                        else:
                            # fallback heuristic: if you have nothing, do not pretend.
                            # keep invalid; training will ignore via mask.
                            y_road = 0.0

                        # speed label (always valid if speed key existed; else zeros but still usable)
                        y_spd = float(spd[t])
                        m[1] = 1.0

                        # x_offset label
                        if xoff is not None:
                            y_xoff = float(xoff[t])
                            m[2] = 1.0
                        else:
                            # fallback: weak proxy from steering (NOT ideal, but better than training nothing if you lack xoff)
                            # Note: sign convention may vary; override by disabling XOFF loss if this is wrong.
                            if act is not None and t < act.shape[0]:
                                h = min(CFG.label_horizon, act.shape[0] - t)
                                steer = float(act[t : t + h, CFG.steer_index].mean())
                                y_xoff = float(np.clip(-0.5 * steer, -0.5, 0.5))
                                m[2] = 1.0
                            else:
                                y_xoff = 0.0

                        buf.append((stack, y_road, y_spd, y_xoff, m))
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

    def _format(
        self,
        stack_u8: np.ndarray,
        y_road: float,
        y_spd: float,
        y_xoff: float,
        mask: np.ndarray,
    ):
        # stack_u8: (S,H,W,3) -> (3S,H,W) uint8
        t = torch.from_numpy(stack_u8).permute(0, 3, 1, 2).contiguous()  # (S,3,H,W)
        x_u8 = t.permute(1, 0, 2, 3).reshape(3 * stack_u8.shape[0], stack_u8.shape[1], stack_u8.shape[2]).to(torch.uint8)

        y_road_t = torch.tensor([y_road], dtype=torch.float32)
        y_spd_t = torch.tensor([y_spd], dtype=torch.float32)
        y_xoff_t = torch.tensor([y_xoff], dtype=torch.float32)
        m_t = torch.from_numpy(mask).to(torch.float32)  # (3,)

        return x_u8, y_road_t, y_spd_t, y_xoff_t, m_t


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_encoder(path: str, in_ch: int) -> TinyEncoder:
    enc = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        ckpt = ckpt["encoder"]
    if isinstance(ckpt, dict):
        ckpt = _strip_prefixes(ckpt)
    missing, unexpected = enc.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[enc] load: missing={len(missing)} unexpected={len(unexpected)}")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


# ---------------------------------------------------------------------
# Losses consistent with drive_mpc.py heads_eval()
# ---------------------------------------------------------------------
def road_loss_from_logits(logits: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    # drive_mpc uses sigmoid(head_road(feat))
    # training with BCEWithLogits aligns precisely.
    return F.binary_cross_entropy_with_logits(logits, y01)


def xoff_loss_from_raw(raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # drive_mpc uses tanh(head_xoff(feat)) * 0.5
    pred = torch.tanh(raw) * 0.5
    return F.mse_loss(pred, y)


def spd_loss(raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # drive_mpc uses head_spd(feat) directly
    return F.mse_loss(raw, y)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
@torch.no_grad()
def validate(enc: TinyEncoder, head_road: nn.Module, head_spd: nn.Module, head_xoff: nn.Module, loader: DataLoader) -> None:
    enc.eval()
    head_road.eval()
    head_spd.eval()
    head_xoff.eval()

    tot = {"road": 0.0, "spd": 0.0, "xoff": 0.0}
    cnt = {"road": 0, "spd": 0, "xoff": 0}

    for i, (x_u8, y_road, y_spd, y_xoff, mask) in enumerate(loader):
        if i >= CFG.val_batches:
            break

        x = x_u8.to(DEVICE, non_blocking=True).to(torch.float32) / 255.0  # (B,12,64,64)
        y_road = y_road.to(DEVICE, non_blocking=True)
        y_spd = y_spd.to(DEVICE, non_blocking=True)
        y_xoff = y_xoff.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)  # (B,3)

        z = enc(x)  # (B,512,8,8)
        if CFG.predictor_space == "norm":
            z = norm_bchw(z)
        feat = pool_feat(z)  # (B,512)

        lr = head_road(feat)
        ls = head_spd(feat)
        lx = head_xoff(feat)

        m0 = (mask[:, 0] > 0.5)
        m1 = (mask[:, 1] > 0.5)
        m2 = (mask[:, 2] > 0.5)

        if m0.any():
            tot["road"] += road_loss_from_logits(lr[m0], y_road[m0]).item()
            cnt["road"] += 1
        if m1.any():
            tot["spd"] += spd_loss(ls[m1], y_spd[m1]).item()
            cnt["spd"] += 1
        if m2.any():
            tot["xoff"] += xoff_loss_from_raw(lx[m2], y_xoff[m2]).item()
            cnt["xoff"] += 1

    def _avg(k: str) -> float:
        return tot[k] / max(1, cnt[k])

    print(f"[val] road_bce={_avg('road'):.4f} spd_mse={_avg('spd'):.4f} xoff_mse={_avg('xoff'):.4f}")


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
def train() -> None:
    seed_all(CFG.seed)
    os.makedirs(os.path.dirname(CFG.out_path), exist_ok=True)

    files = sorted(glob.glob("./data_*/*.npz"))
    if CFG.max_files and CFG.max_files > 0:
        files = files[: CFG.max_files]
    if not files:
        raise RuntimeError("No .npz files found under ./data_*/*.npz")

    rng = random.Random(CFG.seed)
    rng.shuffle(files)
    split = int(len(files) * (1.0 - CFG.val_file_frac))
    train_files = files[:split]
    val_files = files[split:]
    print(f"[files] train={len(train_files)} val={len(val_files)} total={len(files)}")

    train_ds = ShardedHeadsDataset(
        files=train_files,
        frame_stack=CFG.frame_stack,
        img_size=CFG.img_size,
        stride=CFG.stride,
        buffer_size=CFG.buffer_size,
    )
    val_ds = ShardedHeadsDataset(
        files=val_files,
        frame_stack=CFG.frame_stack,
        img_size=CFG.img_size,
        stride=max(1, CFG.stride * 2),
        buffer_size=CFG.buffer_size,
    )

    pin = (DEVICE.type == "cuda")
    # Keep persistent_workers=False so set_epoch() affects RNG (workers restart each epoch).
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        num_workers=max(0, CFG.num_workers // 2),
        pin_memory=pin,
        persistent_workers=False,
    )

    in_ch = 3 * CFG.frame_stack
    print(f"[enc] loading {CFG.encoder_path} (in_ch={in_ch})")
    enc = load_encoder(CFG.encoder_path, in_ch=in_ch)

    head_road = MLPHead(512, 1).to(DEVICE)
    head_spd = MLPHead(512, 1).to(DEVICE)
    head_xoff = MLPHead(512, 1).to(DEVICE)

    params = list(head_road.parameters()) + list(head_spd.parameters()) + list(head_xoff.parameters())
    opt = optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    print(f"[train] DEVICE={DEVICE} PREDICTOR_SPACE={CFG.predictor_space} stack={CFG.frame_stack}")
    print(f"[train] losses: w_road={CFG.w_road} w_spd={CFG.w_spd} w_xoff={CFG.w_xoff}")
    print(f"[train] NOTE: road/xoff labels train only if present (or fallback enabled). Masks gate each loss term.")

    for epoch in range(CFG.epochs):
        train_ds.set_epoch(epoch)
        head_road.train()
        head_spd.train()
        head_xoff.train()

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CFG.epochs}")
        for x_u8, y_road, y_spd, y_xoff, mask in pbar:
            x = x_u8.to(DEVICE, non_blocking=True).to(torch.float32) / 255.0
            y_road = y_road.to(DEVICE, non_blocking=True)
            y_spd = y_spd.to(DEVICE, non_blocking=True)
            y_xoff = y_xoff.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)  # (B,3)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                with torch.no_grad():
                    z = enc(x)  # (B,512,8,8)
                    if CFG.predictor_space == "norm":
                        z = norm_bchw(z)
                    feat = pool_feat(z)  # (B,512)

                lr = head_road(feat)   # (B,1)
                ls = head_spd(feat)    # (B,1)
                lx = head_xoff(feat)   # (B,1)

                loss = torch.zeros((), device=DEVICE, dtype=torch.float32)

                m0 = (mask[:, 0] > 0.5)
                m1 = (mask[:, 1] > 0.5)
                m2 = (mask[:, 2] > 0.5)

                l_road = torch.tensor(0.0, device=DEVICE)
                l_spd = torch.tensor(0.0, device=DEVICE)
                l_xoff = torch.tensor(0.0, device=DEVICE)

                if CFG.w_road > 0 and m0.any():
                    l_road = road_loss_from_logits(lr[m0], y_road[m0])
                    loss = loss + CFG.w_road * l_road
                if CFG.w_spd > 0 and m1.any():
                    l_spd = spd_loss(ls[m1], y_spd[m1])
                    loss = loss + CFG.w_spd * l_spd
                if CFG.w_xoff > 0 and m2.any():
                    l_xoff = xoff_loss_from_raw(lx[m2], y_xoff[m2])
                    loss = loss + CFG.w_xoff * l_xoff

            scaler.scale(loss).backward()

            if CFG.grad_clip and CFG.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, CFG.grad_clip)

            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                road=f"{float(l_road.item()):.4f}",
                spd=f"{float(l_spd.item()):.4f}",
                xoff=f"{float(l_xoff.item()):.4f}",
            )

        validate(enc, head_road, head_spd, head_xoff, val_loader)

        # Save each epoch (small models)
        torch.save(
            {
                "head_road": head_road.state_dict(),
                "head_spd": head_spd.state_dict(),
                "head_xoff": head_xoff.state_dict(),
                "cfg": {
                    "frame_stack": CFG.frame_stack,
                    "img_size": CFG.img_size,
                    "predictor_space": CFG.predictor_space,
                    "w_road": CFG.w_road,
                    "w_spd": CFG.w_spd,
                    "w_xoff": CFG.w_xoff,
                },
            },
            CFG.out_path,
        )

    print(f"[done] saved heads to {CFG.out_path}")


if __name__ == "__main__":
    train()
