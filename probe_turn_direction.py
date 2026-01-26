from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from networks import TinyEncoder

# ENCODER_PATH=./models/encoder_mixed_final.pth \
# DEVICE=cpu \
# FRAME_STACK=4 \
# HORIZON=10 \
# STEER_THR=0.25 \
# BATCH_SIZE=64 \
# NUM_WORKERS=4 \
# EPOCHS=3 \
# LR=3e-3 \
# MAX_FILES=800 \
# MAX_SAMPLES=200000 \
# python probe_turn_direction.py


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    data_random: str = os.getenv("DATA_RANDOM", "./data_random")
    data_expert: str = os.getenv("DATA_EXPERT", "./data_expert")
    data_recover: str = os.getenv("DATA_RECOVER", "./data_recover")

    # Encoder checkpoint
    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")

    # Frame stacking
    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))

    # Probe training settings
    batch_size: int = int(os.getenv("BATCH_SIZE", "512"))
    epochs: int = int(os.getenv("EPOCHS", "3"))
    lr: float = float(os.getenv("LR", "3e-3"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))

    # Label parameters (Tuned for cleaner signal)
    # Horizon 10 steps (~0.5s) gives a clear "future intent"
    horizon: int = int(os.getenv("HORIZON", "10"))
    steer_index: int = int(os.getenv("STEER_INDEX", "0"))
    # Threshold 0.15 ignores minor lane-keeping adjustments
    steer_thr: float = float(os.getenv("STEER_THR", "0.15"))

    # Sampling
    max_files: int = int(os.getenv("MAX_FILES", "800"))
    max_samples: int = int(os.getenv("MAX_SAMPLES", "200000"))

    seed: int = int(os.getenv("SEED", "1337"))
    
    # Device (Defaults to CPU to protect training run)
    device: str = os.getenv("DEVICE", "cpu")


CFG = CFG()
DEVICE = torch.device(CFG.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _list_npz(dir_path: str) -> List[str]:
    if not dir_path or not os.path.exists(dir_path):
        return []
    return sorted(glob.glob(os.path.join(dir_path, "*.npz")))


def _pick_key(d: np.lib.npyio.NpzFile, preferred: List[str]) -> Optional[str]:
    keys = list(d.keys())
    for k in preferred:
        if k in d:
            return k
    return keys[0] if keys else None


def _extract_actions(npz: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    for k in ["action", "actions", "act", "u", "steer", "steering"]:
        if k in npz:
            a = npz[k]
            if a.ndim == 1:
                return a[:, None]
            return a
    return None


class TurnProbeDataset(Dataset):
    def __init__(self, files: List[str], max_samples: int = None):
        super().__init__()
        self.samples: List[Tuple[str, int]] = []
        
        limit = max_samples if max_samples is not None else CFG.max_samples
        total_added = 0
        
        for f in files:
            try:
                with np.load(f, mmap_mode="r") as npz:
                    obs_key = _pick_key(npz, ["obs", "observations", "states"])
                    if obs_key is None: continue
                    obs = npz[obs_key]
                    
                    # Strict shape check
                    if (obs.ndim != 4 or 
                        obs.shape[1] != 64 or 
                        obs.shape[2] != 64 or 
                        obs.shape[3] != 3):
                        continue

                    actions = _extract_actions(npz)
                    if actions is None: continue

                    # Initial Sync: Handle standard off-by-one from collection
                    if obs.shape[0] == actions.shape[0] + 1:
                        obs = obs[:-1]
                    
                    if actions.shape[0] != obs.shape[0]:
                        continue

                    T = obs.shape[0]
                    
                    # Indexing logic: Ensure room for forward stack AND forward label
                    min_i = 0
                    max_i = T - max(CFG.frame_stack, CFG.horizon)

                    if max_i < min_i:
                        continue

                    # Random subsample from this file
                    valid_range = np.arange(min_i, max_i + 1)
                    take = min(400, len(valid_range))
                    picks = np.random.choice(valid_range, size=take, replace=False)

                    for i in picks:
                        self.samples.append((f, int(i)))
                        total_added += 1
                        if total_added >= limit:
                            break
            except Exception:
                continue

            if total_added >= limit:
                break
        
        print(f"[probe] Dataset built: {len(self.samples)} samples from {len(files)} files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        f, i = self.samples[idx]
        with np.load(f, mmap_mode="r") as npz:
            obs_key = _pick_key(npz, ["obs", "observations", "states"])
            obs = npz[obs_key]
            actions = _extract_actions(npz)
            
            # --- ROBUST ALIGNMENT (Belt-and-Braces) ---
            if actions is None:
                 # Should not happen given __init__, but safe guard
                 return None, None

            # 1. Handle off-by-one scenarios
            if obs.shape[0] == actions.shape[0] + 1:
                obs = obs[:-1]
            elif actions.shape[0] == obs.shape[0] + 1:
                actions = actions[:-1]

            # 2. Force strict length equality (truncating to min)
            T = min(obs.shape[0], actions.shape[0])
            obs = obs[:T]
            actions = actions[:T]
            
            # 3. Check if index is still valid after potential extra truncation
            if i + max(CFG.frame_stack, CFG.horizon) >= T:
                # This sample is effectively corrupted/edge-case
                return None, torch.tensor(-1, dtype=torch.int64)
            # ------------------------------------------

            # Build Input: Stack of frames starting at i
            # Returns (3*stack, 64, 64)
            if CFG.frame_stack <= 1:
                fr = obs[i]
                fr = torch.from_numpy(fr).permute(2, 0, 1).contiguous()
            else:
                block = obs[i : i + CFG.frame_stack]
                t = torch.from_numpy(block).permute(0, 3, 1, 2).contiguous()
                fr = t.permute(1, 0, 2, 3).reshape(3 * CFG.frame_stack, 64, 64)

            if fr.dtype != torch.uint8:
                fr = fr.clamp(0, 255).to(torch.uint8)

            # Label: Future steer from i to i+horizon
            a = actions
            if a.ndim > 1:
                steer = a[:, CFG.steer_index]
            else:
                steer = a

            # Correct window: starts at i
            w = steer[i : i + CFG.horizon].astype(np.float32)
            
            m = float(np.mean(w))
            if abs(m) < CFG.steer_thr:
                y = -1
            else:
                y = 0 if m < 0 else 1

            return fr, torch.tensor(y, dtype=torch.int64)


def collate_drop_invalid(batch):
    xs, ys = [], []
    for x, y in batch:
        if x is not None and int(y.item()) != -1:
            xs.append(x)
            ys.append(y)
    if not xs: return None, None
    return torch.stack(xs), torch.stack(ys)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def encoder_embed(encoder: nn.Module, x_u8: torch.Tensor) -> torch.Tensor:
    # Use float32 to avoid CPU half-precision crashes
    x = x_u8.to(device=DEVICE, dtype=torch.float32) / 255.0
    z = encoder(x)
    z = F.adaptive_avg_pool2d(z, 1).flatten(1)
    return z


def main():
    seed_everything(CFG.seed)

    # 1. Gather files
    files = _list_npz(CFG.data_random) + _list_npz(CFG.data_expert) + _list_npz(CFG.data_recover)
    if not files:
        files = sorted(glob.glob("./data_*/*.npz"))
    if not files:
        raise RuntimeError("No .npz files found.")
    
    # 2. Split Train/Val (File-level split prevents leak)
    random.shuffle(files)
    files = files[: CFG.max_files]
    
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    print(f"[probe] Train files: {len(train_files)} | Val files: {len(val_files)}")

    ds_train = TurnProbeDataset(train_files, max_samples=int(CFG.max_samples * 0.8))
    ds_val   = TurnProbeDataset(val_files, max_samples=int(CFG.max_samples * 0.2))

    # Conditional pinning
    use_pin = (DEVICE.type == "cuda")
    
    dl_train = DataLoader(ds_train, batch_size=CFG.batch_size, shuffle=True, 
                          num_workers=CFG.num_workers, collate_fn=collate_drop_invalid,
                          pin_memory=use_pin, persistent_workers=(CFG.num_workers > 0))
                          
    dl_val   = DataLoader(ds_val, batch_size=CFG.batch_size, shuffle=False, 
                          num_workers=CFG.num_workers, collate_fn=collate_drop_invalid,
                          pin_memory=use_pin, persistent_workers=(CFG.num_workers > 0))

    # 3. Load Encoder (Robustly)
    in_ch = 3 * max(1, CFG.frame_stack)
    encoder = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE)
    
    print(f"[probe] Loading encoder from {CFG.encoder_path}...")
    ckpt = torch.load(CFG.encoder_path, map_location=DEVICE)
    
    # Handle both full checkpoint dict and direct state_dict
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        print("[probe] Detected full training checkpoint, extracting 'encoder' key.")
        state_dict = ckpt["encoder"]
    else:
        print("[probe] Assuming direct state_dict.")
        state_dict = ckpt

    # --- FIX: STRIP COMPILE/DDP PREFIXES ---
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove torch.compile prefix
        k = k.replace("_orig_mod.", "")
        # Remove DataParallel prefix (if present)
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    # ---------------------------------------
        
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    print(f"[probe] Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # FAIL HARD IF WEIGHTS MISSING
    if len(missing) > 10:
        raise RuntimeError(f"Too many missing keys ({len(missing)}). Weights did NOT load correctly! Check prefixes.")
    
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    # 4. Train Probe
    probe = LinearProbe(in_dim=512, num_classes=2).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=CFG.lr)

    for ep in range(CFG.epochs):
        # Train Loop
        probe.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(dl_train, desc=f"Ep {ep+1} Train")
        for x, y in pbar:
            if x is None: continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            with torch.no_grad():
                feat = encoder_embed(encoder, x)
            
            logits = probe(feat)
            loss = F.cross_entropy(logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Count-based accuracy accumulation
            batch_correct = (logits.argmax(1) == y).sum().item()
            batch_total = y.numel()
            
            train_correct += batch_correct
            train_total += batch_total
            
            # Show running accuracy
            run_acc = train_correct / max(1, train_total)
            pbar.set_postfix(acc=f"{run_acc:.3f}", loss=f"{loss.item():.3f}")
            
        # Val Loop
        probe.eval()
        val_correct, val_total = 0, 0
        for x, y in dl_val:
            if x is None: continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                feat = encoder_embed(encoder, x)
                logits = probe(feat)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.numel()
        
        tr_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        print(f"Ep {ep+1} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("[probe] Done.")

if __name__ == "__main__":
    main()