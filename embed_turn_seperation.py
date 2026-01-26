#!/usr/bin/env python3
"""
embed_turn_seperation.py  (spelling preserved for backward-compat)

Turn embedding separation diagnostics WITHOUT training a probe.

What it does
------------
1) Builds a balanced left/right dataset from your recorded episodes.
   - Uses obs frames + action steer over a future horizon to label "left" vs "right".
   - Trims off-by-one obs/action mismatches (your collector saves T obs and T-1 actions).

2) Embeds stacked frames with your frozen encoder (TinyEncoder).
3) Computes simple separation metrics:
   - Raw cosine between class means (often ~1.0 due to shared scene content)
   - L2 distance between class means
   - Within-class mean radius
   - Separation ratio = dist / avg_radius
   - Nearest-centroid val accuracy (cosine + L2)
   - NEW: d-prime along the discriminative direction (train)
   - NEW: AUC on val using 1D projection onto the discriminative direction (no sklearn)

Important note about "centered cosine"
-------------------------------------
If you force a perfectly balanced dataset and compute mu_all on that same set,
then (mu_left - mu_all) and (mu_right - mu_all) are exact negatives by construction,
so their cosine is ~ -1.0 and is not a meaningful separation metric.

Env vars
--------
ENCODER_PATH   path to .pth weights or full ckpt with "encoder" key
DEVICE         cpu (default) or cuda:0
BATCH_SIZE     embedding batch size (default 64)
NUM_WORKERS    dataloader workers (default 4)
FRAME_STACK    must match encoder training (default 4)
HORIZON        label horizon in steps (default 10)
STEER_THR      ignore near-straight; higher = cleaner labels (default 0.25)
MAX_FILES      cap number of episode files considered (default 800)
TRAIN_SAMPLES  total balanced train samples (default 40000)
VAL_SAMPLES    total balanced val samples (default 10000)
SEED           RNG seed (default 1337)


Embedding separation
ENCODER_PATH=./models/encoder_mixed_final.pth \
DEVICE=cpu \
FRAME_STACK=4 \
HORIZON=10 \
STEER_THR=0.25 \
BATCH_SIZE=64 \
python embed_turn_seperation.py


"""

from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from networks import TinyEncoder


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    data_random: str = os.getenv("DATA_RANDOM", "./data_random")
    data_expert: str = os.getenv("DATA_EXPERT", "./data_expert")
    data_recover: str = os.getenv("DATA_RECOVER", "./data_recover")

    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")
    device: str = os.getenv("DEVICE", "cpu")

    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))

    # label definition
    horizon: int = int(os.getenv("HORIZON", "10"))
    steer_index: int = int(os.getenv("STEER_INDEX", "0"))
    steer_thr: float = float(os.getenv("STEER_THR", "0.25"))

    # sampling
    max_files: int = int(os.getenv("MAX_FILES", "800"))
    train_samples: int = int(os.getenv("TRAIN_SAMPLES", "40000"))  # total = left+right
    val_samples: int = int(os.getenv("VAL_SAMPLES", "10000"))      # total = left+right
    seed: int = int(os.getenv("SEED", "1337"))

    # embedding
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))


CFG = CFG()
DEVICE = torch.device(CFG.device)


# -----------------------------
# Utils
# -----------------------------
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


def _pick_key(npz: np.lib.npyio.NpzFile, preferred: List[str]) -> Optional[str]:
    keys = list(npz.keys())
    for k in preferred:
        if k in npz:
            return k
    return keys[0] if keys else None


def _extract_actions(npz: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    # Your collector uses "action"
    for k in ["action", "actions", "act", "u", "steer", "steering"]:
        if k in npz:
            a = npz[k]
            if a.ndim == 1:
                return a[:, None]
            return a
    return None


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k2 = k.replace("_orig_mod.", "").replace("module.", "")
        out[k2] = v
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


# -----------------------------
# Dataset: build balanced labeled indices
# -----------------------------
class TurnIndexDataset(Dataset):
    """
    Holds (file, idx, label) triples.
    Returns stacked frames as uint8 tensor and label int64.
    """
    def __init__(self, triples: List[Tuple[str, int, int]]):
        super().__init__()
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, j: int):
        f, i, y = self.triples[j]
        with np.load(f, mmap_mode="r") as npz:
            obs_key = _pick_key(npz, ["obs", "observations", "states"])
            if obs_key is None:
                return None, None

            obs = npz[obs_key]  # (T,64,64,3) uint8
            actions = _extract_actions(npz)
            if actions is None:
                return None, None

            # Robust alignment (belt-and-braces)
            if obs.shape[0] == actions.shape[0] + 1:
                obs = obs[:-1]
            elif actions.shape[0] == obs.shape[0] + 1:
                actions = actions[:-1]

            T = min(obs.shape[0], actions.shape[0])
            obs = obs[:T]
            actions = actions[:T]

            # Bound check
            need = max(CFG.frame_stack, CFG.horizon)
            if i + need > T:
                return None, None

            # Build forward stack: obs[i : i+stack]
            if CFG.frame_stack <= 1:
                fr = obs[i]  # (64,64,3)
                t = torch.from_numpy(fr).permute(2, 0, 1).contiguous()  # (3,64,64)
            else:
                block = obs[i:i + CFG.frame_stack]  # (S,64,64,3)
                tt = torch.from_numpy(block).permute(0, 3, 1, 2).contiguous()  # (S,3,64,64)
                t = tt.permute(1, 0, 2, 3).reshape(3 * CFG.frame_stack, 64, 64)  # (3S,64,64)

            if t.dtype != torch.uint8:
                t = t.clamp(0, 255).to(torch.uint8)

            return t, torch.tensor(y, dtype=torch.int64)


def collate_drop_none(batch):
    xs, ys = [], []
    for x, y in batch:
        if x is None:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_balanced_triples(files: List[str], total: int, *, seed: int) -> List[Tuple[str, int, int]]:
    """
    Scan files and produce balanced (left,right) samples.
    total = left+right.
    """
    rng = np.random.RandomState(seed)
    want_per = total // 2
    left: List[Tuple[str, int, int]] = []
    right: List[Tuple[str, int, int]] = []

    # We will sample up to ~400 candidates per file then label them.
    for f in files:
        if len(left) >= want_per and len(right) >= want_per:
            break

        try:
            with np.load(f, mmap_mode="r") as npz:
                obs_key = _pick_key(npz, ["obs", "observations", "states"])
                if obs_key is None:
                    continue
                obs = npz[obs_key]
                actions = _extract_actions(npz)
                if actions is None:
                    continue

                # shape check
                if obs.ndim != 4 or obs.shape[1] != 64 or obs.shape[2] != 64 or obs.shape[3] != 3:
                    continue

                # align lengths
                if obs.shape[0] == actions.shape[0] + 1:
                    obs = obs[:-1]
                elif actions.shape[0] == obs.shape[0] + 1:
                    actions = actions[:-1]

                T = min(obs.shape[0], actions.shape[0])
                if T <= max(CFG.frame_stack, CFG.horizon) + 1:
                    continue

                # valid start indices for forward stack + forward horizon
                max_i = T - max(CFG.frame_stack, CFG.horizon)
                if max_i <= 0:
                    continue

                valid = np.arange(0, max_i + 1)
                take = min(400, len(valid))
                picks = rng.choice(valid, size=take, replace=False)

                # steer vector
                if actions.ndim > 1:
                    steer = actions[:, CFG.steer_index]
                else:
                    steer = actions

                for i in picks:
                    # label uses mean steer over horizon starting at i
                    w = steer[i:i + CFG.horizon].astype(np.float32)
                    m = float(np.mean(w))
                    if abs(m) < float(CFG.steer_thr):
                        continue
                    y = 0 if m < 0 else 1
                    if y == 0 and len(left) < want_per:
                        left.append((f, int(i), 0))
                    elif y == 1 and len(right) < want_per:
                        right.append((f, int(i), 1))

                    if len(left) >= want_per and len(right) >= want_per:
                        break
        except Exception:
            continue

    n = min(len(left), len(right), want_per)
    left = left[:n]
    right = right[:n]
    triples = left + right
    rng.shuffle(triples)
    return triples


# -----------------------------
# Encoder embedding
# -----------------------------
@torch.no_grad()
def embed_batches(encoder: torch.nn.Module, dl: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for x, y in tqdm(dl, desc="[sep] embedding", leave=False):
        if x is None:
            continue
        # CPU-safe: float32
        x = x.to(device=DEVICE, dtype=torch.float32) / 255.0
        z = encoder(x)                       # (B,512,8,8)
        z = F.adaptive_avg_pool2d(z, 1)      # (B,512,1,1)
        z = z.flatten(1)                     # (B,512)
        feats.append(z.detach().cpu().numpy().astype(np.float32))
        labels.append(y.detach().cpu().numpy().astype(np.int64))

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)
    Y = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
    return X, Y


def nearest_centroid_acc(X: np.ndarray, y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, metric: str) -> float:
    if X.shape[0] == 0:
        return 0.0
    if metric == "l2":
        d0 = np.linalg.norm(X - mu0[None, :], axis=1)
        d1 = np.linalg.norm(X - mu1[None, :], axis=1)
        pred = (d1 < d0).astype(np.int64)
        return float((pred == y).mean())
    if metric == "cos":
        # normalize
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        m0 = mu0 / (np.linalg.norm(mu0) + 1e-12)
        m1 = mu1 / (np.linalg.norm(mu1) + 1e-12)
        s0 = (Xn @ m0).astype(np.float32)
        s1 = (Xn @ m1).astype(np.float32)
        pred = (s1 > s0).astype(np.int64)
        return float((pred == y).mean())
    raise ValueError("metric must be 'l2' or 'cos'")


def auc_from_scores(scores: np.ndarray, labels01: np.ndarray) -> float:
    """
    AUC = P(score_pos > score_neg) + 0.5 P(ties)
    Computed via ranking with tie averaging, no sklearn.
    labels01: 0/1
    """
    scores = scores.astype(np.float64)
    labels01 = labels01.astype(np.int64)
    n = scores.shape[0]
    if n == 0:
        return 0.5

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    pos = (labels01 == 1)
    n_pos = int(pos.sum())
    n_neg = int(n_pos and (n - n_pos) or (n - n_pos))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_ranks_sum = float(ranks[pos].sum())
    auc = (pos_ranks_sum - n_pos * (n_pos - 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def main() -> None:
    seed_everything(CFG.seed)

    # gather files
    files = _list_npz(CFG.data_random) + _list_npz(CFG.data_expert) + _list_npz(CFG.data_recover)
    if not files:
        files = sorted(glob.glob("./data_*/*.npz"))
    if not files:
        raise RuntimeError("No .npz files found (check DATA_RANDOM/DATA_EXPERT/DATA_RECOVER).")

    random.shuffle(files)
    files = files[: CFG.max_files]

    split = int(0.8 * len(files))
    train_files = files[:split]
    val_files = files[split:]

    print(f"[sep] Files: train={len(train_files)} val={len(val_files)}")
    print(f"[sep] Using: frame_stack={CFG.frame_stack} horizon={CFG.horizon} steer_thr={CFG.steer_thr}")

    # build balanced indices
    train_triples = build_balanced_triples(train_files, CFG.train_samples, seed=CFG.seed + 1)
    val_triples = build_balanced_triples(val_files, CFG.val_samples, seed=CFG.seed + 2)

    def _count_lr(triples):
        y = np.array([t[2] for t in triples], dtype=np.int64) if triples else np.zeros((0,), dtype=np.int64)
        return int((y == 0).sum()), int((y == 1).sum())

    tl, tr = _count_lr(train_triples)
    vl, vr = _count_lr(val_triples)
    print(f"[sep] Train samples: {len(train_triples)} (left={tl} right={tr})")
    print(f"[sep] Val   samples: {len(val_triples)} (left={vl} right={vr})")

    if len(train_triples) < 1000 or len(val_triples) < 200:
        raise RuntimeError(
            "Not enough labeled samples. Increase MAX_FILES, lower STEER_THR, or lower HORIZON."
        )

    # dataloaders
    use_pin = (DEVICE.type == "cuda")
    ds_tr = TurnIndexDataset(train_triples)
    ds_va = TurnIndexDataset(val_triples)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=use_pin,
        persistent_workers=(CFG.num_workers > 0),
        collate_fn=collate_drop_none,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=use_pin,
        persistent_workers=(CFG.num_workers > 0),
        collate_fn=collate_drop_none,
        drop_last=False,
    )

    # load encoder
    in_ch = 3 * max(1, CFG.frame_stack)
    encoder = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE)

    ckpt = torch.load(CFG.encoder_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        state = ckpt["encoder"]
    else:
        state = ckpt
    state = _strip_prefixes(state)

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"[sep] Weights loaded. Missing={len(missing)} Unexpected={len(unexpected)}")
    if len(missing) > 0:
        # fail hard: for this script you want exact match
        raise RuntimeError(f"Encoder load incomplete: missing={missing[:10]} ... (total {len(missing)})")
    if len(unexpected) > 0:
        raise RuntimeError(f"Encoder load unexpected keys: {unexpected[:10]} ... (total {len(unexpected)})")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # embed
    Xtr, ytr = embed_batches(encoder, dl_tr)
    Xva, yva = embed_batches(encoder, dl_va)

    # compute metrics
    L = Xtr[ytr == 0]
    R = Xtr[ytr == 1]
    muL = L.mean(axis=0)
    muR = R.mean(axis=0)

    cos_raw = _cosine(muL, muR)
    dist_l2 = float(np.linalg.norm(muL - muR))
    rL = float(np.linalg.norm(L - muL[None, :], axis=1).mean())
    rR = float(np.linalg.norm(R - muR[None, :], axis=1).mean())
    r_avg = 0.5 * (rL + rR)
    sep_ratio = dist_l2 / max(1e-12, r_avg)

    # nearest centroid acc on val
    acc_cos = nearest_centroid_acc(Xva, yva, muL, muR, metric="cos")
    acc_l2 = nearest_centroid_acc(Xva, yva, muL, muR, metric="l2")

    # discriminative direction metrics
    delta = (muR - muL).astype(np.float32)
    delta_norm = delta / max(1e-12, float(np.linalg.norm(delta)))

    s_tr = (Xtr @ delta_norm).astype(np.float32)
    s_va = (Xva @ delta_norm).astype(np.float32)

    sL = s_tr[ytr == 0]
    sR = s_tr[ytr == 1]
    mL, mR = float(sL.mean()), float(sR.mean())
    vL, vR = float(sL.var() + 1e-12), float(sR.var() + 1e-12)
    dprime = (mR - mL) / float(np.sqrt(0.5 * (vL + vR)))

    auc = auc_from_scores(s_va, yva)

    print("\n=== TURN EMBEDDING SEPARATION (NO PROBE TRAINING) ===")
    print(f"Mean cosine(mu_left, mu_right):           {cos_raw:.4f}   (lower is better)")
    print(f"L2 distance ||mu_left - mu_right||:       {dist_l2:.4f} (higher is better)")
    print(f"Within-class mean radius: left={rL:.4f} right={rR:.4f}")
    print(f"Separation ratio (dist / avg radius):     {sep_ratio:.4f} (higher is better)")
    print(f"Val nearest-centroid accuracy (cosine):   {acc_cos:.4f}")
    print(f"Val nearest-centroid accuracy (L2):       {acc_l2:.4f}")
    print(f"d-prime along delta (train):              {dprime:.4f} (higher is better)")
    print(f"AUC on val via delta projection:          {auc:.4f} (0.5 chance, 1.0 perfect)")

    print("\nInterpretation:")
    print("- Raw mean cosine near 1.0 is common because shared scene content dominates the mean direction.")
    print("- Centroid acc ~0.50 suggests ambiguous; >0.65 suggests more robust turn semantics (for this simple rule).")
    print("- d-prime/AUC quantify separability along the optimal mean-difference axis; if these are low,")
    print("  then turn info exists but is weak relative to nuisance variance (track position/speed/background).")


if __name__ == "__main__":
    main()
