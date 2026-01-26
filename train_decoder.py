#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict
from contextlib import nullcontext

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision.utils import save_image
from tqdm import tqdm

from networks import TinyEncoder, TinyDecoder

# Prevent OpenCV from spawning threads inside workers (critical for stalls)
cv2.setNumThreads(0)


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    # Data
    data_glob: str = os.getenv("DATA_GLOB", "./data_*/*.npz")
    img_size: int = int(os.getenv("IMG_SIZE", "64"))
    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))  # MUST match encoder training

    # Train
    batch_size: int = int(os.getenv("BATCH_SIZE", "1024"))
    epochs: int = int(os.getenv("EPOCHS", "30"))
    lr: float = float(os.getenv("LR", "1e-3"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    seed: int = int(os.getenv("SEED", "1337"))
    grad_clip: float = float(os.getenv("GRAD_CLIP", "0.0"))  # 0 disables

    # AMP
    amp: bool = os.getenv("AMP", "1") == "1"

    # DataLoader perf/safety
    pin_memory: bool = os.getenv("PIN_MEMORY", "1") == "1"
    persistent_workers: bool = os.getenv("PERSISTENT_WORKERS", "0") == "1"
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))
    timeout_s: int = int(os.getenv("DATALOADER_TIMEOUT", "0"))  # 0 disables
    drop_last: bool = os.getenv("DROP_LAST", "0") == "1"

    # I/O
    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")

    # Use optimizer-state resume by default (as requested)
    resume_path: str = os.getenv("DECODER_RESUME", "./models/decoder_ckpt_latest.pt")

    out_final: str = os.getenv("DECODER_OUT", "./models/decoder_final.pth")
    visuals_dir: str = os.getenv("VISUALS_DIR", "./visuals")
    save_every_epochs: int = int(os.getenv("SAVE_EVERY_EPOCHS", "1"))
    save_visuals_every_epochs: int = int(os.getenv("SAVE_VIS_EVERY_EPOCHS", "1"))
    keep_epoch_ckpts: int = int(os.getenv("KEEP_EPOCH_CKPTS", "5"))  # keep last K decoder_ckpt_ep*.pt

CFG = CFG()
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


# -----------------------------
# NPZ helpers
# -----------------------------
def _pick_obs(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "obs" in data:
        return data["obs"]
    if "states" in data:
        return data["states"]
    if "observations" in data:
        return data["observations"]
    return None


def _strip_prefixes(sd: Dict) -> Dict:
    out = {}
    for k, v in sd.items():
        if isinstance(k, str):
            k = k.replace("_orig_mod.", "").replace("module.", "")
        out[k] = v
    return out


def worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    s = CFG.seed + worker_id * 1009
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _resize_if_needed(obs: np.ndarray, img_size: int) -> np.ndarray:
    if obs.shape[1] == img_size and obs.shape[2] == img_size:
        return obs
    out = np.empty((obs.shape[0], img_size, img_size, 3), dtype=obs.dtype)
    for i in range(obs.shape[0]):
        out[i] = cv2.resize(obs[i], (img_size, img_size), interpolation=cv2.INTER_AREA)
    return out


def _safe_makedirs(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _epoch_from_name(path: str) -> int:
    base = os.path.basename(path)
    # decoder_ckpt_ep{E}.pt
    for token in ("_ep", "ep"):
        if token in base:
            try:
                s = base.split(token, 1)[1]
                digits = ""
                for ch in s:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    return int(digits)
            except Exception:
                pass
    return -1


# -----------------------------
# Dataset: yields (x_stack, y_img)
# -----------------------------
class StreamingStackedReconDataset(IterableDataset):
    """
    For each sample at time t:
      x_stack: (3*frame_stack, H, W) float in [0,1], frames [t-hist ... t]
      y_img:   (3, H, W) float in [0,1], target frame at time t (the LAST frame in the stack)
    """
    def __init__(self, files: List[str], frame_stack: int, img_size: int, seed: int):
        super().__init__()
        self.files = list(files)
        self.frame_stack = int(frame_stack)
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.epoch = 0
        self.hist = self.frame_stack - 1

        if not self.files:
            raise FileNotFoundError("No .npz files found for decoder dataset.")

        # Precompute epoch sample count so tqdm can show % + ETA.
        total = 0
        valid = 0
        for f in self.files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    obs = _pick_obs(data)
                    if obs is None or obs.ndim != 4:
                        continue
                    T = int(obs.shape[0])
                    total += max(0, T - self.hist)
                    valid += 1
            except Exception:
                continue

        self._num_samples = int(total)
        print(f"[dec-data] files={len(self.files)} valid={valid} frame_stack={self.frame_stack} img={self.img_size}")
        print(f"[dec-data] epoch_samples={self._num_samples} (t from hist..T-1 across all files)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self._num_samples)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        wi = get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        # If persistent_workers=True, workers won't see updated self.epoch across epochs.
        # Decoder training is fine even if ordering repeats; leave this deterministic.
        epoch_for_seed = int(self.epoch) if not CFG.persistent_workers else 0
        rng = random.Random(self.seed + epoch_for_seed * 10007 + worker_id * 1009)

        my_files = self.files[worker_id::num_workers]
        rng.shuffle(my_files)

        hist = self.hist

        for f in my_files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    obs = _pick_obs(data)
                    if obs is None or obs.ndim != 4:
                        continue

                    if obs.shape[1] != self.img_size or obs.shape[2] != self.img_size:
                        obs = _resize_if_needed(obs, self.img_size)

                    if obs.dtype != np.uint8:
                        obs = np.clip(obs, 0, 255).astype(np.uint8, copy=False)

                    T = int(obs.shape[0])
                    if T <= hist:
                        continue

                    idxs = list(range(hist, T))
                    rng.shuffle(idxs)

                    for t in idxs:
                        stack = obs[t - hist : t + 1]  # (S,H,W,3)

                        # x_stack: (3S,H,W) in 0..1
                        x = torch.from_numpy(stack).to(torch.float32).div_(255.0)  # (S,H,W,3)
                        x = x.permute(0, 3, 1, 2).reshape(3 * self.frame_stack, self.img_size, self.img_size)

                        # y_img: last frame in stack
                        y = torch.from_numpy(obs[t]).to(torch.float32).div_(255.0)  # (H,W,3)
                        y = y.permute(2, 0, 1)  # (3,H,W)

                        yield x, y
            except Exception:
                continue


# -----------------------------
# Resume / checkpoint helpers
# -----------------------------
def _load_decoder_ckpt(decoder: nn.Module, optimizer: optim.Optimizer, path: str) -> int:
    """
    Returns start_epoch (0-based index to continue training).
    Expects ckpt dict:
      { "epoch": <completed_epochs_1_based>, "decoder": state_dict, "optimizer": state_dict }
    Also supports raw decoder state_dict (.pth) as a fallback (start_epoch=0).
    """
    if not path or not os.path.exists(path):
        return 0

    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "decoder" in ckpt:
        decoder.load_state_dict(_strip_prefixes(ckpt["decoder"]), strict=False)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[dec] WARNING: could not load optimizer state ({e})")
        epoch_done = int(ckpt.get("epoch", 0))  # completed epochs (1-based)
        print(f"[dec] Resumed decoder+optimizer from {path} (epoch_done={epoch_done})")
        return max(0, epoch_done)
    else:
        if isinstance(ckpt, dict):
            ckpt = _strip_prefixes(ckpt)
        decoder.load_state_dict(ckpt, strict=False)
        print(f"[dec] Resumed decoder weights only from {path} (optimizer not restored)")
        return 0


def _save_decoder_ckpt(path: str, decoder: nn.Module, optimizer: optim.Optimizer, epoch_done: int) -> None:
    torch.save(
        {
            "epoch": int(epoch_done),  # completed epochs (1-based)
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _prune_old_epoch_ckpts(keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    ckpts = sorted(glob.glob("./models/decoder_ckpt_ep*.pt"), key=_epoch_from_name)
    if len(ckpts) <= keep_last_k:
        return
    for p in ckpts[:-keep_last_k]:
        try:
            os.remove(p)
        except Exception:
            pass


# -----------------------------
# Train
# -----------------------------
def train() -> None:
    if not os.path.exists(CFG.encoder_path):
        raise FileNotFoundError(f"Encoder not found at {CFG.encoder_path}")

    _safe_makedirs(os.path.dirname(CFG.out_final))
    _safe_makedirs(CFG.visuals_dir)
    _safe_makedirs("./models")

    # Optional: speed up conv selection
    try:
        if DEVICE.type == "cuda" and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    files = sorted(glob.glob(CFG.data_glob))
    if not files:
        raise FileNotFoundError(f"No npz found under {CFG.data_glob}")

    ds = StreamingStackedReconDataset(
        files=files,
        frame_stack=CFG.frame_stack,
        img_size=CFG.img_size,
        seed=CFG.seed,
    )

    # Progress-bar totals
    epoch_samples = len(ds)
    steps_per_epoch = math.ceil(epoch_samples / max(1, CFG.batch_size))
    if CFG.drop_last:
        steps_per_epoch = epoch_samples // max(1, CFG.batch_size)

    # DataLoader
    dl_kwargs = dict(
        dataset=ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory and (DEVICE.type == "cuda"),
        persistent_workers=CFG.persistent_workers and (CFG.num_workers > 0),
        worker_init_fn=worker_init_fn if (CFG.num_workers > 0) else None,
        drop_last=CFG.drop_last,
    )
    if CFG.num_workers > 0:
        dl_kwargs["prefetch_factor"] = CFG.prefetch_factor
    if CFG.timeout_s and CFG.timeout_s > 0:
        dl_kwargs["timeout"] = CFG.timeout_s

    loader = DataLoader(**dl_kwargs)

    # Load frozen encoder (MUST be in_ch=3*frame_stack)
    in_ch = 3 * CFG.frame_stack
    encoder = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE)
    enc_sd = torch.load(CFG.encoder_path, map_location=DEVICE)
    if isinstance(enc_sd, dict) and "encoder" in enc_sd:
        enc_sd = enc_sd["encoder"]
    if isinstance(enc_sd, dict):
        enc_sd = _strip_prefixes(enc_sd)
    encoder.load_state_dict(enc_sd, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Decoder + optimizer
    decoder = TinyDecoder(latent_channels=512).to(DEVICE)
    opt = optim.AdamW(decoder.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    crit = nn.MSELoss()

    # Resume with optimizer state (as requested)
    start_epoch = 0
    if CFG.resume_path and os.path.exists(CFG.resume_path):
        # start_epoch here is "epochs already completed" (1-based stored), used as range(start_epoch, ...)
        start_epoch = _load_decoder_ckpt(decoder, opt, CFG.resume_path)

    amp_enabled = bool(CFG.amp and (DEVICE.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    autocast_ctx = (
        torch.autocast(device_type="cuda", enabled=amp_enabled)
        if DEVICE.type == "cuda"
        else nullcontext()
    )

    print("--- DECODER TRAIN CONFIG ---")
    print(f"device         : {DEVICE}")
    print(f"encoder_path   : {CFG.encoder_path}")
    print(f"in_ch          : {in_ch} (frame_stack={CFG.frame_stack})")
    print(f"batch_size     : {CFG.batch_size}")
    print(f"epochs         : {CFG.epochs}")
    print(f"amp            : {amp_enabled}")
    print(f"epoch_samples  : {epoch_samples}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"num_workers    : {CFG.num_workers} (persistent_workers={CFG.persistent_workers})")
    print(f"prefetch_factor: {CFG.prefetch_factor if CFG.num_workers>0 else 'n/a'}")
    print(f"timeout_s      : {CFG.timeout_s}")
    print(f"drop_last      : {CFG.drop_last}")
    if start_epoch:
        # start_epoch is completed epochs (1-based), so next epoch printed is start_epoch+1
        print(f"resume         : {CFG.resume_path} (continuing at epoch {start_epoch+1}/{CFG.epochs})")
    print("---------------------------")

    # start_epoch is "completed epochs" (1-based in ckpt), so loop epoch = start_epoch .. CFG.epochs-1
    for epoch in range(start_epoch, CFG.epochs):
        ds.set_epoch(epoch)
        decoder.train()

        pbar = tqdm(
            loader,
            total=steps_per_epoch,
            desc=f"Dec Ep {epoch+1}/{CFG.epochs}",
            dynamic_ncols=True,
            mininterval=0.5,
        )

        last_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        for x_stack, y_img in pbar:
            x_stack = x_stack.to(DEVICE, non_blocking=True)  # (B,12,64,64)
            y_img = y_img.to(DEVICE, non_blocking=True)      # (B,3,64,64)

            opt.zero_grad(set_to_none=True)

            with autocast_ctx:
                with torch.no_grad():
                    z = encoder(x_stack)                     # (B,512,8,8)
                recon = decoder(z)                           # (B,3,64,64)
                loss = crit(recon, y_img)

            scaler.scale(loss).backward()

            if CFG.grad_clip and CFG.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), CFG.grad_clip)

            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            last_batch = (x_stack.detach(), y_img.detach(), recon.detach())

        # Visuals (target vs recon)
        if last_batch is not None and ((epoch + 1) % CFG.save_visuals_every_epochs == 0):
            _, y, r = last_batch
            comp = torch.cat([y[:8], r[:8]], dim=0).clamp(0, 1)
            save_image(comp, os.path.join(CFG.visuals_dir, f"decoder_ep{epoch+1:03d}.png"), nrow=8)

        # Checkpoint
        if (epoch + 1) % CFG.save_every_epochs == 0:
            # Weights-only (compat)
            torch.save(decoder.state_dict(), f"./models/decoder_ep{epoch+1}.pth")
            torch.save(decoder.state_dict(), "./models/decoder_latest.pth")

            # Full resume (decoder + optimizer + epoch_done)
            _save_decoder_ckpt("./models/decoder_ckpt_latest.pt", decoder, opt, epoch_done=epoch + 1)
            _save_decoder_ckpt(f"./models/decoder_ckpt_ep{epoch+1}.pt", decoder, opt, epoch_done=epoch + 1)
            _prune_old_epoch_ckpts(CFG.keep_epoch_ckpts)

    torch.save(decoder.state_dict(), CFG.out_final)
    print(f"[done] Saved decoder -> {CFG.out_final}")


if __name__ == "__main__":
    train()
