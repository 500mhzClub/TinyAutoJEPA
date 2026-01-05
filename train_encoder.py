import os
import re
import json
import glob
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from tqdm import tqdm

from networks import TinyEncoder, Projector
from vicreg import vicreg_loss


# ----------------------------
# Configuration (env-overridable)
# ----------------------------

@dataclass
class Config:
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))
    epochs: int = int(os.getenv("EPOCHS", "30"))
    lr: float = float(os.getenv("LR", "3e-4"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.05"))

    num_workers: int = int(os.getenv("NUM_WORKERS", "16"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "4"))
    persistent_workers: bool = os.getenv("PERSISTENT_WORKERS", "1") == "1"

    # Cap work per epoch to iterate faster (0 = full epoch)
    max_steps_per_epoch: int = int(os.getenv("MAX_STEPS_PER_EPOCH", "0"))

    validate_every: int = int(os.getenv("VALIDATE_EVERY", "5"))
    save_every: int = int(os.getenv("SAVE_EVERY", "5"))

    seed: int = int(os.getenv("SEED", "1337"))

    device: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset dirs
    data_random: str = os.getenv("DATA_RANDOM", "./data")
    data_race: str = os.getenv("DATA_RACE", "./data_race")
    data_recovery: str = os.getenv("DATA_RECOVERY", "./data_recovery")
    data_edge: str = os.getenv("DATA_EDGE", "./data_edge")

    # Cache scan results
    cache_scan: bool = os.getenv("CACHE_SCAN", "1") == "1"
    cache_path: str = os.getenv("CACHE_PATH", "./.cache/dataset_index.json")

    # Model save paths
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    ckpt_prefix: str = os.getenv("CKPT_PREFIX", "encoder_mixed_ckpt_ep")
    weights_prefix: str = os.getenv("WEIGHTS_PREFIX", "encoder_mixed_ep")

CFG = Config()
DEVICE = torch.device(CFG.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset
# ----------------------------

def _list_npz(dir_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dir_path, "*.npz")))


def _file_sig(path: str) -> Tuple[int, int]:
    """Return (mtime_ns, size_bytes) to detect changes."""
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _load_cache(cache_path: str) -> Dict:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache_path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, cache_path)


def _count_frames_npz(path: str) -> int:
    # Note: npz is zipped; mmap_mode does not actually memory-map compressed arrays.
    with np.load(path, allow_pickle=False) as d:
        if "states" in d:
            return int(d["states"].shape[0])
        if "obs" in d:
            return int(d["obs"].shape[0])
    return 0


class BalancedMixedDataset(IterableDataset):
    """
    File-balanced mix across available datasets.
    - If a dataset has fewer files (e.g., 28 recovery chunks), it is cycled up to the max.
    - Yields (aug1(img), aug2(img)) pairs for VICReg.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.random_files = _list_npz(cfg.data_random)
        self.race_files = _list_npz(cfg.data_race)
        self.recovery_files = _list_npz(cfg.data_recovery)
        self.edge_files = _list_npz(cfg.data_edge)

        if not (self.random_files or self.race_files or self.recovery_files or self.edge_files):
            raise RuntimeError("No .npz files found in any data directories.")

        max_files = max(
            len(self.random_files),
            len(self.race_files),
            len(self.recovery_files),
            len(self.edge_files),
        )

        self.balanced_files: List[str] = []
        for i in range(max_files):
            if self.random_files:
                self.balanced_files.append(self.random_files[i % len(self.random_files)])
            if self.race_files:
                self.balanced_files.append(self.race_files[i % len(self.race_files)])
            if self.recovery_files:
                self.balanced_files.append(self.recovery_files[i % len(self.recovery_files)])
            if self.edge_files:
                self.balanced_files.append(self.edge_files[i % len(self.edge_files)])

        print(f"Balanced Dataset: {len(self.balanced_files)} files.")

        self.total_frames = self._scan_total_frames()

        # Transform: keep directionality (no horizontal flip)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=5),
        ])

    def _scan_total_frames(self) -> int:
        if not self.cfg.cache_scan:
            return self._scan_total_frames_no_cache()

        cache = _load_cache(self.cfg.cache_path)
        cached = cache.get("files", {})
        updated = {}
        total = 0

        scan_iter = tqdm(self.balanced_files, desc="Scanning Dataset")
        for f in scan_iter:
            try:
                sig = _file_sig(f)
                key = os.path.abspath(f)
                entry = cached.get(key)
                if entry and tuple(entry.get("sig", ())) == sig:
                    frames = int(entry.get("frames", 0))
                else:
                    frames = _count_frames_npz(f)
                updated[key] = {"sig": list(sig), "frames": frames}
                total += frames
            except Exception:
                # Skip unreadable files (you already deleted corrupt chunks; this is a safety net)
                continue

        cache_out = {"files": updated, "generated_at": time.time()}
        _save_cache(self.cfg.cache_path, cache_out)

        print(f"Total Frames: {total:,}")
        return total

    def _scan_total_frames_no_cache(self) -> int:
        total = 0
        for f in tqdm(self.balanced_files, desc="Scanning Dataset"):
            try:
                total += _count_frames_npz(f)
            except Exception:
                continue
        print(f"Total Frames: {total:,}")
        return total

    def __len__(self) -> int:
        return self.total_frames

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1

        # Deterministic-but-distinct shuffle per worker
        rng = random.Random(self.cfg.seed + worker_id)
        my_files = self.balanced_files[worker_id::num_workers]
        rng.shuffle(my_files)

        for f in my_files:
            try:
                with np.load(f, allow_pickle=False) as data:
                    raw = data["states"] if "states" in data else data["obs"]

                # If any legacy 96x96 slips in, resize once per file (slow, but rare)
                if raw.shape[1] != 64:
                    raw = np.array([cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA) for img in raw])

                # Shuffle indices for this shard
                idxs = np.random.permutation(len(raw))
                for idx in idxs:
                    img = torch.from_numpy(raw[idx]).float().div_(255.0)  # HWC float32 in [0,1]
                    img = img.permute(2, 0, 1)  # CHW
                    yield self.transform(img), self.transform(img)

            except Exception:
                continue


# ----------------------------
# Validation
# ----------------------------

@torch.no_grad()
def validate_encoder(encoder, projector, val_loader, epoch: int):
    encoder.eval()
    projector.eval()

    all_embeddings = []
    for i, (x1, _) in enumerate(val_loader):
        if i >= 10:
            break
        x1 = x1.to(DEVICE, non_blocking=True)
        z = encoder(x1)
        all_embeddings.append(z.detach().cpu())

    if not all_embeddings:
        print(f"\n[Validation Epoch {epoch}] No validation batches available.")
        encoder.train()
        projector.train()
        return

    z_all = torch.cat(all_embeddings, dim=0)
    std_per_dim = z_all.std(dim=0)
    dead_dims = int((std_per_dim < 0.01).sum().item())

    print(f"\n[Validation Epoch {epoch}]")
    print(f"  Dead dimensions: {dead_dims}/512")
    print(f"  Avg Std: {std_per_dim.mean().item():.4f}")

    encoder.train()
    projector.train()


# ----------------------------
# Checkpointing
# ----------------------------

def _find_latest_ckpt(model_dir: str, ckpt_prefix: str) -> str | None:
    ckpts = glob.glob(os.path.join(model_dir, f"{ckpt_prefix}*.pt"))
    if not ckpts:
        return None

    def ep_num(path: str) -> int:
        m = re.search(r"ep(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(ckpts, key=ep_num)


def save_checkpoint(path: str, epoch: int, encoder, projector, optimizer, scheduler, scaler):
    tmp = path + ".tmp"
    payload = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": CFG.__dict__,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, encoder, projector, optimizer, scheduler, scaler) -> int:
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0))


# ----------------------------
# Train
# ----------------------------

def train():
    os.makedirs(CFG.model_dir, exist_ok=True)
    seed_everything(CFG.seed)

    # perf knobs
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    dataset = BalancedMixedDataset(CFG)

    pin = (DEVICE.type == "cuda")

    # DataLoader kwargs conditioned on workers
    dl_kwargs = dict(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=pin,
    )
    if CFG.num_workers > 0:
        dl_kwargs["prefetch_factor"] = CFG.prefetch_factor
        dl_kwargs["persistent_workers"] = CFG.persistent_workers

    dataloader = DataLoader(dataset, **dl_kwargs)

    # Small validation stream; keep workers low so it doesn’t contend too much
    val_loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=min(2, CFG.num_workers),
        pin_memory=pin,
    )

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-5)

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume if checkpoint exists
    start_epoch = 0
    latest = _find_latest_ckpt(CFG.model_dir, CFG.ckpt_prefix)
    if latest:
        print(f"Resuming from checkpoint: {latest}")
        start_epoch = load_checkpoint(latest, encoder, projector, optimizer, scheduler, scaler)
        print(f"Resumed at epoch {start_epoch} (next epoch will be {start_epoch + 1})")

    encoder.train()
    projector.train()

    steps_per_full_epoch = max(1, dataset.total_frames // CFG.batch_size)
    steps_this_epoch = CFG.max_steps_per_epoch if CFG.max_steps_per_epoch > 0 else steps_per_full_epoch

    for epoch in range(start_epoch, CFG.epochs):
        pbar = tqdm(dataloader, total=steps_this_epoch, desc=f"Epoch {epoch+1}/{CFG.epochs}")

        for step, (x1, x2) in enumerate(pbar):
            if CFG.max_steps_per_epoch > 0 and step >= CFG.max_steps_per_epoch:
                break

            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = vicreg_loss(projector(encoder(x1)), projector(encoder(x2)))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = vicreg_loss(projector(encoder(x1)), projector(encoder(x2)))
                loss.backward()
                optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        # Validation + save
        epoch_num = epoch + 1

        if (epoch_num % CFG.validate_every) == 0:
            validate_encoder(encoder, projector, val_loader, epoch_num)

        if (epoch_num % CFG.save_every) == 0:
            # Full checkpoint (resume-ready)
            ckpt_path = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}{epoch_num}.pt")
            save_checkpoint(ckpt_path, epoch_num, encoder, projector, optimizer, scheduler, scaler)

            # Convenience: encoder weights only (drop-in for downstream)
            weights_path = os.path.join(CFG.model_dir, f"{CFG.weights_prefix}{epoch_num}.pth")
            torch.save(encoder.state_dict(), weights_path)

    # Final saves
    final_ckpt = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}final.pt")
    save_checkpoint(final_ckpt, CFG.epochs, encoder, projector, optimizer, scheduler, scaler)

    final_weights = os.path.join(CFG.model_dir, "encoder_mixed_final.pth")
    torch.save(encoder.state_dict(), final_weights)


if __name__ == "__main__":
    train()
