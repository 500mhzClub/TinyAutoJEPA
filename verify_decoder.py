#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision.utils import save_image
from tqdm import tqdm

cv2.setNumThreads(0)

from networks import TinyEncoder, TinyDecoder


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    # Data
    data_glob: str = os.getenv("DATA_GLOB", "./data_*/*.npz")
    img_size: int = int(os.getenv("IMG_SIZE", "64"))
    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))

    # Train
    batch_size: int = int(os.getenv("BATCH_SIZE", "1024"))
    epochs: int = int(os.getenv("EPOCHS", "30"))
    lr: float = float(os.getenv("LR", "1e-3"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    seed: int = int(os.getenv("SEED", "42"))
    drop_last: bool = os.getenv("DROP_LAST", "1") == "1"

    # Paths
    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")
    decoder_latest: str = os.getenv("DECODER_LATEST", "./models/decoder_latest.pth")
    out_dir_models: str = os.getenv("MODEL_DIR", "./models")
    out_dir_visuals: str = os.getenv("VIS_DIR", "./visuals")
    final_decoder_path: str = os.getenv("DECODER_FINAL", "./models/decoder_final.pth")

    # AMP
    amp: bool = os.getenv("AMP", "1") == "1"


CFG = CFG()
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


# -----------------------------
# Helpers
# -----------------------------
def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        k2 = k.replace("_orig_mod.", "").replace("module.", "")
        out[k2] = v
    return out


def _pick_obs(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "obs" in data:
        return data["obs"]
    if "states" in data:
        return data["states"]
    if "observations" in data:
        return data["observations"]
    return None


def _resize64(obs: np.ndarray, img_size: int) -> np.ndarray:
    if obs.shape[1] == img_size and obs.shape[2] == img_size:
        return obs
    out = np.empty((obs.shape[0], img_size, img_size, 3), dtype=obs.dtype)
    for i in range(obs.shape[0]):
        out[i] = cv2.resize(obs[i], (img_size, img_size), interpolation=cv2.INTER_AREA)
    return out


def _seed_worker_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _stack_u8(obs_u8: np.ndarray, idx: int, stack: int) -> np.ndarray:
    # returns (stack, H, W, 3) uint8, frames idx-stack+1 .. idx
    return obs_u8[idx - (stack - 1) : idx + 1]


def _stack_to_chw_float(stack_u8: np.ndarray) -> torch.Tensor:
    # stack_u8: (S,H,W,3) -> (12,H,W) float 0..1
    t = torch.from_numpy(stack_u8).to(torch.float32).div_(255.0)  # (S,H,W,3)
    t = t.permute(0, 3, 1, 2).contiguous()                        # (S,3,H,W)
    t = t.permute(1, 0, 2, 3).reshape(3 * stack_u8.shape[0], stack_u8.shape[1], stack_u8.shape[2])
    return t


def _img_to_chw_float(img_u8: np.ndarray) -> torch.Tensor:
    # (H,W,3) -> (3,H,W) float 0..1
    t = torch.from_numpy(img_u8).to(torch.float32).div_(255.0)
    return t.permute(2, 0, 1).contiguous()


# -----------------------------
# Dataset
# -----------------------------
class StreamingStackToFrameDataset(IterableDataset):
    """
    Yields:
      stack_chw:  (12,64,64) float32 0..1  (frames t-3..t)
      target_chw: (3,64,64)  float32 0..1  (frame t)
    """
    def __init__(self, files, img_size: int, frame_stack: int, seed: int):
        super().__init__()
        self.files = list(files)
        self.img_size = int(img_size)
        self.frame_stack = int(frame_stack)
        self.seed = int(seed)
        self.epoch = 0
        if not self.files:
            raise FileNotFoundError("No .npz files found for decoder dataset.")
        print(f"[decoder_ds] files={len(self.files)} img={self.img_size} stack={self.frame_stack}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        _seed_worker_env()
        wi = get_worker_info()
        wid = wi.id if wi else 0
        wnum = wi.num_workers if wi else 1

        rng = random.Random(self.seed + self.epoch * 10007 + wid * 997)
        my_files = self.files[wid::wnum]
        rng.shuffle(my_files)

        for f in my_files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    obs = _pick_obs(data)
                    if obs is None:
                        continue
                    obs = _resize64(obs, self.img_size)

                    if obs.dtype != np.uint8:
                        obs = np.clip(obs, 0, 255).astype(np.uint8, copy=False)

                    T = int(obs.shape[0])
                    if T < self.frame_stack:
                        continue

                    # sample indices >= stack-1
                    idxs = list(range(self.frame_stack - 1, T))
                    rng.shuffle(idxs)

                    for t in idxs:
                        stack = _stack_u8(obs, t, self.frame_stack)
                        target = obs[t]
                        yield _stack_to_chw_float(stack), _img_to_chw_float(target)

            except Exception:
                continue


# -----------------------------
# Model loading
# -----------------------------
def load_frozen_encoder(path: str, in_ch: int) -> TinyEncoder:
    enc = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        ckpt = ckpt["encoder"]
    if isinstance(ckpt, dict):
        ckpt = _strip_prefixes(ckpt)
    enc.load_state_dict(ckpt, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


# -----------------------------
# Train
# -----------------------------
def main() -> None:
    os.makedirs(CFG.out_dir_models, exist_ok=True)
    os.makedirs(CFG.out_dir_visuals, exist_ok=True)

    if not os.path.exists(CFG.encoder_path):
        raise FileNotFoundError(f"Encoder not found: {CFG.encoder_path}")

    files = sorted(glob.glob(CFG.data_glob))
    if not files:
        raise FileNotFoundError(f"No files matched DATA_GLOB={CFG.data_glob}")

    # Models
    in_ch = 3 * CFG.frame_stack
    print(f"[load] encoder={CFG.encoder_path} in_ch={in_ch} device={DEVICE}")
    encoder = load_frozen_encoder(CFG.encoder_path, in_ch=in_ch)

    decoder = TinyDecoder(latent_channels=512).to(DEVICE)
    if os.path.exists(CFG.decoder_latest):
        print(f"[resume] decoder from {CFG.decoder_latest}")
        sd = torch.load(CFG.decoder_latest, map_location=DEVICE)
        if isinstance(sd, dict) and "decoder" in sd:
            sd = sd["decoder"]
        if isinstance(sd, dict):
            sd = _strip_prefixes(sd)
        decoder.load_state_dict(sd, strict=False)

    opt = optim.AdamW(decoder.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    crit = nn.MSELoss()

    use_amp = bool(CFG.amp and DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Data
    ds = StreamingStackToFrameDataset(files, CFG.img_size, CFG.frame_stack, CFG.seed)
    dl = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False,  # important since we call ds.set_epoch()
        drop_last=CFG.drop_last,
    )

    print(f"[train] batch={CFG.batch_size} epochs={CFG.epochs} workers={CFG.num_workers} amp={use_amp}")

    for epoch in range(CFG.epochs):
        ds.set_epoch(epoch)
        decoder.train()

        pbar = tqdm(dl, desc=f"Decoder Ep {epoch+1}/{CFG.epochs}")
        last_target = None
        last_recon = None

        for stack_chw, target_chw in pbar:
            # stack_chw: (B,12,64,64), target_chw: (B,3,64,64)
            stack_chw = stack_chw.to(DEVICE, non_blocking=True)
            target_chw = target_chw.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                with torch.no_grad():
                    z = encoder(stack_chw)       # (B,512,8,8)
                recon = decoder(z)               # (B,3,64,64)
                loss = crit(recon, target_chw)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if random.random() < 0.01:
                last_target = target_chw.detach()
                last_recon = recon.detach()

        # visuals
        if last_target is not None and last_recon is not None:
            comparison = torch.cat([last_target[:8], last_recon[:8]], dim=0)
            out_png = os.path.join(CFG.out_dir_visuals, f"decoder_ep{epoch+1:03d}.png")
            save_image(comparison, out_png, nrow=8)
            print(f"[viz] saved {out_png}")

        # checkpoints
        ep_path = os.path.join(CFG.out_dir_models, f"decoder_ep{epoch+1}.pth")
        torch.save(decoder.state_dict(), ep_path)
        torch.save(decoder.state_dict(), CFG.decoder_latest)
        print(f"[ckpt] saved {ep_path} and {CFG.decoder_latest}")

    torch.save(decoder.state_dict(), CFG.final_decoder_path)
    print(f"[done] saved final decoder -> {CFG.final_decoder_path}")


if __name__ == "__main__":
    main()
