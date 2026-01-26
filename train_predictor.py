#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterator, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision.utils import save_image
from tqdm import tqdm

from networks import TinyEncoder, Predictor, TinyDecoder

# ==========================================
# FRESH START COMMAND (FUTURE USE)
# Strategy: Start fast (5e-4) -> Decay to (1e-6)
# ==========================================
# PREDICTOR_SPACE=raw \
# DEVICE=cuda \
# ENCODER_PATH=./models/encoder_mixed_final.pth \
# FRAME_STACK=4 IMG_SIZE=64 \
# PRED_HORIZON=15 \
# BATCH_SIZE=256 NUM_WORKERS=8 \
# EPOCHS=50 \
# LR=5e-4 MIN_LR=1e-6 \
# STRIDE=4 BUFFER_SIZE=500 VAL_FILE_FRAC=0.20 MAX_FILES=0 \
# STEER_THR=0.25 AXIS_HORIZON=10 \
# W_COS=0.05 W_DELTA=0.25 W_AXIS=0.10 \
# DELTA_BATCHES=120 \
# python train_predictor.py

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    batch_size: int = int(os.getenv("BATCH_SIZE", "256"))
    epochs: int = int(os.getenv("EPOCHS", "50"))
    lr: float = float(os.getenv("LR", "5e-4"))
    min_lr: float = float(os.getenv("MIN_LR", "1e-6"))  # End point for scheduler
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))

    device: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))

    pred_horizon: int = int(os.getenv("PRED_HORIZON", "15"))
    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))  # must match encoder training
    img_size: int = int(os.getenv("IMG_SIZE", "64"))

    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")
    decoder_path: str = os.getenv("DECODER_PATH", "./models/decoder_final.pth")

    # IMPORTANT: keep aligned with MPC
    # raw  => predictor learns raw BCHW latents (recommended if MPC uses raw)
    # norm => predictor learns per-spatial L2-normalized latents
    predictor_space: str = os.getenv("PREDICTOR_SPACE", "raw").lower()  # raw | norm

    # Loss weights
    w_cos: float = float(os.getenv("W_COS", "0.05"))
    w_delta: float = float(os.getenv("W_DELTA", "0.25"))
    w_axis: float = float(os.getenv("W_AXIS", "0.10"))
    axis_margin: float = float(os.getenv("AXIS_MARGIN", "0.20"))

    # Steer-axis estimation / labeling
    steer_index: int = int(os.getenv("STEER_INDEX", "0"))
    steer_thr: float = float(os.getenv("STEER_THR", "0.25"))
    axis_horizon: int = int(os.getenv("AXIS_HORIZON", "10"))

    # Scheduled teacher forcing
    teacher_start: float = float(os.getenv("TEACHER_START", "0.70"))
    teacher_end: float = float(os.getenv("TEACHER_END", "0.15"))
    teacher_decay_frac: float = float(os.getenv("TEACHER_DECAY_FRAC", "0.70"))

    # Dataset / sampling
    stride: int = int(os.getenv("STRIDE", "4"))
    buffer_size: int = int(os.getenv("BUFFER_SIZE", "500"))
    val_file_frac: float = float(os.getenv("VAL_FILE_FRAC", "0.20"))
    max_files: int = int(os.getenv("MAX_FILES", "0"))  # 0 => all

    # Delta estimation
    delta_batches: int = int(os.getenv("DELTA_BATCHES", "120"))

    # Misc
    seed: int = int(os.getenv("SEED", "1337"))
    grad_clip: float = float(os.getenv("GRAD_CLIP", "0.0"))  # 0 disables


CFG = CFG()
assert CFG.predictor_space in ("raw", "norm"), "PREDICTOR_SPACE must be 'raw' or 'norm'"

DEVICE = torch.device(CFG.device)
PRED_HORIZON = CFG.pred_horizon
SEQ_T = PRED_HORIZON + 1  # number of obs stacks per sample


# -----------------------------
# Utilities
# -----------------------------
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


def _pick_obs(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "states" in data:
        return data["states"]
    if "obs" in data:
        return data["obs"]
    if "observations" in data:
        return data["observations"]
    return None


def _pick_act(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "actions" in data:
        return data["actions"]
    if "action" in data:
        return data["action"]
    if "act" in data:
        return data["act"]
    return None


def _pick_speed(data: np.lib.npyio.NpzFile, T_obs: int) -> np.ndarray:
    if "speed" in data:
        spd = data["speed"]
        if spd.ndim == 1:
            spd = spd.astype(np.float32, copy=False)
            if spd.shape[0] >= T_obs:
                return spd[:T_obs]
            out = np.zeros((T_obs,), dtype=np.float32)
            out[: spd.shape[0]] = spd
            return out
    return np.zeros((T_obs,), dtype=np.float32)


def _resize_obs_if_needed(obs: np.ndarray, img_size: int) -> np.ndarray:
    if obs.shape[1] == img_size and obs.shape[2] == img_size:
        return obs
    out = np.empty((obs.shape[0], img_size, img_size, 3), dtype=obs.dtype)
    for i in range(obs.shape[0]):
        out[i] = cv2.resize(obs[i], (img_size, img_size), interpolation=cv2.INTER_AREA)
    return out


def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        out[k.replace("_orig_mod.", "").replace("module.", "")] = v
    return out


def pool_z(z: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(z, 1).flatten(1)  # (B,512)


def norm_latents(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 4:
        return F.normalize(z, p=2, dim=1)
    if z.dim() == 5:
        return F.normalize(z, p=2, dim=2)
    raise ValueError(f"Unexpected latent rank: {z.dim()} (expected 4 or 5)")


def scheduled_teacher_prob(epoch: int, total_epochs: int) -> float:
    frac = min(1.0, epoch / max(1, int(total_epochs * CFG.teacher_decay_frac)))
    p = CFG.teacher_start + frac * (CFG.teacher_end - CFG.teacher_start)
    return float(max(min(p, 1.0), 0.0))


# -----------------------------
# Dataset
# -----------------------------
class ShardedSeqDataset(IterableDataset):
    def __init__(
        self,
        files: List[str],
        pred_horizon: int,
        frame_stack: int,
        img_size: int,
        stride: int,
        buffer_size: int,
    ):
        super().__init__()
        self.files = list(files)
        self.pred_horizon = int(pred_horizon)
        self.frame_stack = int(frame_stack)
        self.img_size = int(img_size)
        self.stride = int(stride)
        self.buffer_size = int(buffer_size)
        self.epoch = 0
        print(f"[data] files={len(self.files)} H={self.pred_horizon} stack={self.frame_stack} stride={self.stride}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        
    def count_samples(self) -> int:
        total = 0
        H = self.pred_horizon
        T_need = H + 1
        hist = self.frame_stack - 1
        
        print(f"[data] Pre-scanning {len(self.files)} files for ETA...")
        
        for f in tqdm(self.files, desc="Scanning len"):
            try:
                with np.load(f, mmap_mode="r") as data:
                    if "obs" in data: T_obs = data["obs"].shape[0]
                    elif "states" in data: T_obs = data["states"].shape[0]
                    else: continue
                        
                    if "actions" in data: T_act = data["actions"].shape[0]
                    elif "action" in data: T_act = data["action"].shape[0]
                    elif "act" in data: T_act = data["act"].shape[0]
                    else: continue
                    
                    if T_act > T_obs - 1:
                        T_act = T_obs - 1
                    
                    if T_obs < (hist + T_need + 1): continue
                    if T_act < (hist + (T_need - 1) + 1): continue
                    
                    max_t0_obs = (T_obs - 1) - H
                    max_t0_act = (T_act - 1) - (H - 1)
                    max_t0 = min(max_t0_obs, max_t0_act)
                    
                    if max_t0 < hist: continue
                    
                    start = hist
                    stop = max_t0 + 1
                    count = (stop - start + self.stride - 1) // self.stride
                    total += max(0, count)
            except Exception:
                continue
                
        print(f"[data] Total valid samples: {total}")
        return total

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        _set_worker_env()
        wi = get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        rng = random.Random(self.epoch * 10007 + worker_id * 1009 + 999)
        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)

        H = self.pred_horizon
        T_need = H + 1
        hist = self.frame_stack - 1

        buf = []
        for f in my_files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    obs = _pick_obs(data)
                    act = _pick_act(data)
                    if obs is None or act is None:
                        continue

                    obs = _resize_obs_if_needed(obs, self.img_size)
                    spd = _pick_speed(data, int(obs.shape[0]))

                    if obs.dtype != np.uint8:
                        obs = np.clip(obs, 0, 255).astype(np.uint8, copy=False)
                    act = act.astype(np.float32, copy=False)
                    spd = spd.astype(np.float32, copy=False)

                    if obs.ndim != 4 or obs.shape[1] != self.img_size or obs.shape[3] != 3:
                        continue

                    T_obs = int(obs.shape[0])
                    T_act = int(act.shape[0])

                    if T_act > T_obs - 1:
                        act = act[: max(0, T_obs - 1)]
                        T_act = int(act.shape[0])

                    if T_obs < (hist + T_need + 1): continue
                    if T_act < (hist + (T_need - 1) + 1): continue

                    max_t0_obs = (T_obs - 1) - H
                    max_t0_act = (T_act - 1) - (H - 1)
                    max_t0 = min(max_t0_obs, max_t0_act)
                    if max_t0 < hist: continue

                    for t0 in range(hist, max_t0 + 1, self.stride):
                        stack_seq = []
                        spd_seq = []
                        act_seq = []

                        for k in range(T_need):
                            idx = t0 + k
                            s_idx = list(range(idx - hist, idx + 1))
                            stack_seq.append(obs[s_idx])
                            spd_seq.append(spd[idx])
                            if k < T_need - 1:
                                act_seq.append(act[idx])

                        buf.append((stack_seq, act_seq, spd_seq))
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

    def _format(self, stack_list, act_list, spd_list):
        stack_np = np.asarray(stack_list)
        T, S, H, W, C = stack_np.shape

        t_stack = torch.from_numpy(stack_np).to(torch.float32).div_(255.0)
        t_stack = t_stack.permute(0, 1, 4, 2, 3).reshape(T, 3 * S, H, W)

        t_act = torch.from_numpy(np.asarray(act_list)).to(torch.float32)
        t_spd = torch.from_numpy(np.asarray(spd_list)).to(torch.float32)

        return t_stack, t_act, t_spd


# -----------------------------
# Models
# -----------------------------
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

def load_decoder(path: str) -> TinyDecoder:
    dec = TinyDecoder(latent_channels=512).to(DEVICE)
    print(f"[dec] loading {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    if isinstance(ckpt, dict):
        ckpt = _strip_prefixes(ckpt)
    missing, unexpected = dec.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[dec] load: missing={len(missing)} unexpected={len(unexpected)}")
    dec.eval()
    for p in dec.parameters(): p.requires_grad = False
    return dec


# -----------------------------
# Turn-axis delta estimation
# -----------------------------
@torch.no_grad()
def estimate_turn_delta(encoder: TinyEncoder, loader: DataLoader) -> torch.Tensor:
    encoder.eval()
    sum_l = torch.zeros(512, device=DEVICE)
    sum_r = torch.zeros(512, device=DEVICE)
    n_l = 0
    n_r = 0

    batches = 0
    for obs, act, spd in loader:
        if batches >= CFG.delta_batches:
            break

        obs = obs.to(DEVICE, non_blocking=True)
        act = act.to(DEVICE, non_blocking=True)
        B, T, C, H, W = obs.shape

        z_all = encoder(obs.view(-1, C, H, W)).view(B, T, 512, 8, 8)
        if CFG.predictor_space == "norm":
            z_all = norm_latents(z_all)

        z0 = pool_z(z_all[:, 0])

        h = min(CFG.axis_horizon, act.shape[1])
        steer = act[:, :h, CFG.steer_index].mean(dim=1)
        left = steer <= -CFG.steer_thr
        right = steer >= CFG.steer_thr

        if left.any():
            sum_l += z0[left].sum(dim=0)
            n_l += int(left.sum().item())
        if right.any():
            sum_r += z0[right].sum(dim=0)
            n_r += int(right.sum().item())

        batches += 1

    if n_l < 128 or n_r < 128:
        print(f"[delta] WARNING: low counts left={n_l} right={n_r}")

    mu_l = sum_l / max(1, n_l)
    mu_r = sum_r / max(1, n_r)
    delta = (mu_r - mu_l)
    delta = delta / (delta.norm() + 1e-8)
    print(f"[delta] estimated with left={n_l} right={n_r}, ||delta||=1.0")
    return delta


# -----------------------------
# Validation / Vis
# -----------------------------
@torch.no_grad()
def validate(encoder: TinyEncoder, predictor: Predictor, loader: DataLoader) -> None:
    encoder.eval()
    predictor.eval()

    total = 0.0
    n = 0
    for i, (obs, act, spd) in enumerate(loader):
        if i >= 30: break

        obs = obs.to(DEVICE, non_blocking=True)
        act = act.to(DEVICE, non_blocking=True)
        spd = spd.to(DEVICE, non_blocking=True)

        B, T, C, H, W = obs.shape
        z_all = encoder(obs.view(-1, C, H, W)).view(B, T, 512, 8, 8)
        if CFG.predictor_space == "norm":
            z_all = norm_latents(z_all)

        z_in = z_all[:, 0]
        loss = 0.0
        for t in range(PRED_HORIZON):
            z_tgt = z_all[:, t + 1]
            z_pred = predictor(z_in, act[:, t], spd[:, t])
            if CFG.predictor_space == "norm":
                z_pred = norm_latents(z_pred)
            loss += F.mse_loss(z_pred, z_tgt).item()
            z_in = z_pred

        total += loss / PRED_HORIZON
        n += 1

    print(f"  >>> Val rollout MSE: {total / max(1, n):.6f}")
    predictor.train()

@torch.no_grad()
def save_dream_rollout(encoder: TinyEncoder, predictor: Predictor, decoder: TinyDecoder, loader: DataLoader, epoch: int):
    encoder.eval()
    predictor.eval()
    decoder.eval()
    
    try:
        obs, act, spd = next(iter(loader))
    except StopIteration:
        return

    x_curr = obs[0, 0].unsqueeze(0).to(DEVICE)
    z_curr = encoder(x_curr)
    if CFG.predictor_space == "norm":
        z_curr = norm_latents(z_curr)
        
    dreams = []
    ground_truths = []
    
    horizon = min(20, act.shape[1])
    for t in range(horizon):
        img = decoder(z_curr)
        dreams.append(img.cpu().squeeze(0))
        
        gt_stack = obs[0, t+1]
        gt_frame = gt_stack[-3:]
        ground_truths.append(gt_frame.cpu())
        
        a_t = act[0, t].unsqueeze(0).to(DEVICE)
        s_t = spd[0, t].unsqueeze(0).to(DEVICE)
        
        z_curr = predictor(z_curr, a_t, s_t)
        if CFG.predictor_space == "norm":
            z_curr = norm_latents(z_curr)
            
    strip_gt = torch.cat(ground_truths, dim=2)
    strip_dream = torch.cat(dreams, dim=2)
    viz = torch.cat([strip_gt, strip_dream], dim=1)
    
    os.makedirs("viz_predictor", exist_ok=True)
    save_image(viz, f"viz_predictor/epoch_{epoch:03d}.png")
    print(f"[viz] Saved dream rollout to viz_predictor/epoch_{epoch:03d}.png")


# -----------------------------
# Training
# -----------------------------
def train() -> None:
    seed_all(CFG.seed)
    os.makedirs("models", exist_ok=True)

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

    train_ds = ShardedSeqDataset(
        files=train_files,
        pred_horizon=PRED_HORIZON,
        frame_stack=CFG.frame_stack,
        img_size=CFG.img_size,
        stride=CFG.stride,
        buffer_size=CFG.buffer_size,
    )
    val_ds = ShardedSeqDataset(
        files=val_files,
        pred_horizon=PRED_HORIZON,
        frame_stack=CFG.frame_stack,
        img_size=CFG.img_size,
        stride=CFG.stride,
        buffer_size=CFG.buffer_size,
    )
    
    n_train_samples = train_ds.count_samples()
    total_train_batches = n_train_samples // CFG.batch_size
    print(f"[train] Total batches per epoch: {total_train_batches}")

    pin = (DEVICE.type == "cuda")
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
    print(f"[enc] loading {CFG.encoder_path}")
    encoder = load_encoder(CFG.encoder_path, in_ch=in_ch)

    print(f"[dec] loading {CFG.decoder_path}")
    decoder = load_decoder(CFG.decoder_path)

    predictor = Predictor(action_dim=3, features=512).to(DEVICE)
    optimizer = optim.AdamW(predictor.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # =========================================================================
    # RESUME LOGIC (Correctly placed)
    # =========================================================================
    resume_path = os.getenv("RESUME_PATH", "")
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"[train] 🟢 RESUMING from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        
        if isinstance(ckpt, dict) and "predictor" in ckpt:
            state_dict = ckpt["predictor"]
        else:
            state_dict = ckpt
        
        missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[train] Resume warning: missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[train] Weights loaded successfully.")
        
        # Determine start epoch from filename if possible (e.g. predictor_ep10.pth)
        try:
            base = os.path.basename(resume_path)
            # Extracts '10' from 'predictor_ep10.pth'
            ep_str = base.split("_ep")[-1].split(".")[0]
            start_epoch = int(ep_str)
            print(f"[train] Detected start epoch from filename: {start_epoch}")
        except Exception:
            print("[train] Could not parse epoch from filename, starting at 0")
    # =========================================================================

    # =========================================================================
    # SCHEDULER: Linear decay from LR -> MIN_LR
    # =========================================================================
    # We create a lambda that scales the LR from 1.0 down to (min_lr / lr)
    # We account for start_epoch so the decay is correct relative to remaining time
    def lr_lambda(current_step):
        # Overall progress from 0 to 1
        progress = current_step / CFG.epochs
        # Linear interpolation
        # factor = 1.0 - progress * (1.0 - min_lr/lr)
        # But clearer:
        target_lr = CFG.lr + (CFG.min_lr - CFG.lr) * progress
        return target_lr / CFG.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Fast-forward scheduler if resuming
    if start_epoch > 0:
        print(f"[train] Fast-forwarding scheduler to epoch {start_epoch}")
        for _ in range(start_epoch):
            scheduler.step()

    print(f"[train] PREDICTOR_SPACE={CFG.predictor_space}")
    print(f"[train] LR Schedule: {CFG.lr} -> {CFG.min_lr} over {CFG.epochs} epochs")

    print("[delta] estimating turn axis delta from train loader...")
    delta = estimate_turn_delta(encoder, train_loader).to(dtype=torch.float32, device=DEVICE)

    print("[train] starting...")
    for epoch in range(start_epoch, CFG.epochs):
        train_ds.set_epoch(epoch)
        predictor.train()
        p_teacher = scheduled_teacher_prob(epoch, CFG.epochs)
        
        current_lr = scheduler.get_last_lr()[0]

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CFG.epochs} (lr={current_lr:.2e}, teacher={p_teacher:.2f})", total=total_train_batches)
        for obs, act, spd in pbar:
            obs = obs.to(DEVICE, non_blocking=True)
            act = act.to(DEVICE, non_blocking=True)
            spd = spd.to(DEVICE, non_blocking=True)
            B, T, C, H, W = obs.shape

            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                with torch.no_grad():
                    z_all = encoder(obs.view(-1, C, H, W)).view(B, T, 512, 8, 8)
                    if CFG.predictor_space == "norm":
                        z_all = norm_latents(z_all)

                loss = 0.0
                z_in = z_all[:, 0]

                for t in range(PRED_HORIZON):
                    z_tgt = z_all[:, t + 1]
                    z_pred = predictor(z_in, act[:, t], spd[:, t])
                    if CFG.predictor_space == "norm":
                        z_pred = norm_latents(z_pred)

                    mse = F.mse_loss(z_pred, z_tgt)
                    cos = 1.0 - F.cosine_similarity(z_pred, z_tgt, dim=1).mean()
                    
                    z_in_pool = pool_z(z_in)
                    dz_pred = pool_z(z_pred) - z_in_pool
                    dz_true = pool_z(z_tgt) - z_in_pool
                    dloss = F.mse_loss(dz_pred, dz_true)

                    axis_loss = 0.0
                    hwin = min(CFG.axis_horizon, act.shape[1] - t)
                    if hwin > 0 and CFG.w_axis > 0:
                        steer_w = act[:, t:t + hwin, CFG.steer_index].mean(dim=1)
                        y = torch.zeros_like(steer_w)
                        y = torch.where(steer_w >= CFG.steer_thr, torch.ones_like(y), y)
                        y = torch.where(steer_w <= -CFG.steer_thr, -torch.ones_like(y), y)
                        m = (y != 0)
                        if m.any():
                            proj = (dz_pred[m] * delta).sum(dim=1)
                            axis_loss = F.relu(CFG.axis_margin - (y[m] * proj)).mean()

                    step_loss = mse + (CFG.w_cos * cos) + (CFG.w_delta * dloss) + (CFG.w_axis * axis_loss)
                    loss = loss + step_loss

                    if random.random() < p_teacher:
                        z_in = z_tgt
                    else:
                        z_in = z_pred

                loss = loss / PRED_HORIZON

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if CFG.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Step the scheduler at end of epoch
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            ckpt_state = f"models/predictor_ep{epoch+1}.pth"
            ckpt_meta = f"models/predictor_ep{epoch+1}_meta.pth"
            torch.save(predictor.state_dict(), ckpt_state)
            torch.save(
                {
                    "predictor": predictor.state_dict(),
                    "delta": delta.detach().cpu(),
                    "cfg": {
                        "pred_horizon": CFG.pred_horizon,
                        "frame_stack": CFG.frame_stack,
                        "predictor_space": CFG.predictor_space,
                        "w_cos": CFG.w_cos,
                        "w_delta": CFG.w_delta,
                        "w_axis": CFG.w_axis,
                        "axis_margin": CFG.axis_margin,
                        "steer_thr": CFG.steer_thr,
                        "axis_horizon": CFG.axis_horizon,
                    },
                },
                ckpt_meta,
            )
            print(f"[ckpt] saved {ckpt_state}")
            validate(encoder, predictor, val_loader)
            save_dream_rollout(encoder, predictor, decoder, val_loader, epoch+1)

    torch.save(predictor.state_dict(), "models/predictor_final.pth")
    print("[train] done.")

if __name__ == "__main__":
    train()