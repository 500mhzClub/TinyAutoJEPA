import os
import re
import json
import glob
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ----------------------------
# ROCm/MIOpen: set stable defaults EARLY (before torch import)
# ----------------------------

def _mkdir_700(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except Exception:
        pass

def _rocm_miopen_early_setup() -> None:
    """
    Env-controlled MIOpen stability settings. These must be applied before torch import.

    Recommended env usage:
      MIOPEN_STABLE=1
      MIOPEN_CACHE_TAG=gpu1
      MIOPEN_DISABLE_FIND_DB=1
      MIOPEN_FORCE_CONV_IMMED_FALLBACK=1
    """
    if os.getenv("MIOPEN_STABLE", "0") != "1":
        return

    tag = os.getenv("MIOPEN_CACHE_TAG", "gpu0")

    user_db = os.getenv("MIOPEN_USER_DB_PATH", str(Path.home() / ".config" / f"miopen_{tag}"))
    cache_dir = os.getenv("MIOPEN_CUSTOM_CACHE_DIR", str(Path.home() / ".cache" / f"miopen_{tag}"))

    os.environ["MIOPEN_USER_DB_PATH"] = user_db
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = cache_dir

    _mkdir_700(user_db)
    _mkdir_700(cache_dir)

    # Disable FindDb to avoid SQLite DB failures (your sqlite_db.cpp:227 crash).
    if os.getenv("MIOPEN_DISABLE_FIND_DB", "0") == "1":
        os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

    # Force convolution immediate fallback to avoid lengthy "9 dim space" searches.
    if os.getenv("MIOPEN_FORCE_CONV_IMMED_FALLBACK", "0") == "1":
        os.environ["MIOPEN_DEBUG_CONV_IMMED_FALLBACK"] = "1"
        os.environ["MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK"] = "1"

_rocm_miopen_early_setup()

# ----------------------------
# Imports (after early env setup)
# ----------------------------
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
    # Training
    batch_size: int = int(os.getenv("BATCH_SIZE", "48"))
    accum_steps: int = int(os.getenv("ACCUM_STEPS", "1"))
    epochs: int = int(os.getenv("EPOCHS", "30"))
    lr: float = float(os.getenv("LR", "3e-4"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.05"))

    # DataLoader stability
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))
    persistent_workers: bool = os.getenv("PERSISTENT_WORKERS", "0") == "1"

    # Cap steps per epoch for faster iteration (0 = full epoch)
    max_steps_per_epoch: int = int(os.getenv("MAX_STEPS_PER_EPOCH", "0"))

    # Validation/checkpoint cadence
    validate_every_epochs: int = int(os.getenv("VALIDATE_EVERY_EPOCHS", "5"))
    save_every_epochs: int = int(os.getenv("SAVE_EVERY_EPOCHS", "1"))
    ckpt_every_steps: int = int(os.getenv("CKPT_EVERY_STEPS", "2000"))  # 0 disables step ckpt

    # Repro
    seed: int = int(os.getenv("SEED", "1337"))

    # Device
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
    ckpt_prefix: str = os.getenv("CKPT_PREFIX", "encoder_ckpt_ep")
    weights_prefix: str = os.getenv("WEIGHTS_PREFIX", "encoder_ep")

CFG = Config()
DEVICE = torch.device(CFG.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset helpers
# ----------------------------

def _list_npz(dir_path: str) -> List[str]:
    if not dir_path:
        return []
    return sorted(glob.glob(os.path.join(dir_path, "*.npz")))

def _file_sig(path: str) -> Tuple[int, int]:
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
    with np.load(path, allow_pickle=False) as d:
        if "states" in d:
            return int(d["states"].shape[0])
        if "obs" in d:
            return int(d["obs"].shape[0])
    return 0


class BalancedMixedDataset(IterableDataset):
    """
    File-balanced mix across datasets. Yields (aug1(img), aug2(img)) for VICReg.
    Expected .npz keys: 'states' (preferred) or 'obs', with shape [N, H, W, C].
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

        # IMPORTANT: keep directionality (no horizontal flip) for driving
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=5),
        ])

    def _scan_total_frames(self) -> int:
        if not self.cfg.cache_scan:
            total = self._scan_total_frames_no_cache()
            print(f"Total Frames: {total:,}")
            return total

        cache = _load_cache(self.cfg.cache_path)
        cached = cache.get("files", {})
        updated = {}
        total = 0

        scan_iter = tqdm(self.balanced_files, desc="Scanning Dataset")
        for f in scan_iter:
            key = os.path.abspath(f)
            try:
                sig = _file_sig(f)
                entry = cached.get(key)
                if entry and tuple(entry.get("sig", ())) == sig:
                    frames = int(entry.get("frames", 0))
                else:
                    frames = _count_frames_npz(f)
                updated[key] = {"sig": list(sig), "frames": frames}
                total += frames
            except Exception:
                # Skip corrupt files, continue scanning
                continue

        _save_cache(self.cfg.cache_path, {"files": updated, "generated_at": time.time()})
        print(f"Total Frames: {total:,}")
        return total

    def _scan_total_frames_no_cache(self) -> int:
        total = 0
        for f in tqdm(self.balanced_files, desc="Scanning Dataset"):
            try:
                total += _count_frames_npz(f)
            except Exception:
                continue
        return total

    def __len__(self) -> int:
        return self.total_frames

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1

        rng = random.Random(self.cfg.seed + worker_id)
        my_files = self.balanced_files[worker_id::num_workers]
        rng.shuffle(my_files)

        np_rng = np.random.default_rng(self.cfg.seed + worker_id)

        for f in my_files:
            try:
                with np.load(f, allow_pickle=False) as data:
                    raw = data["states"] if "states" in data else data["obs"]

                # Ensure 64x64 RGB
                if raw.shape[1] != 64 or raw.shape[2] != 64:
                    raw = np.array(
                        [cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA) for img in raw],
                        dtype=raw.dtype,
                    )

                idxs = np_rng.permutation(len(raw))
                for idx in idxs:
                    img = torch.from_numpy(raw[idx]).float().div_(255.0)  # [H,W,C]
                    img = img.permute(2, 0, 1)  # [C,H,W]
                    yield self.transform(img), self.transform(img)

            except Exception:
                # Skip corrupt file
                continue


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
    print(f"  Dead dimensions: {dead_dims}/{z_all.shape[1]}")
    print(f"  Avg Std: {std_per_dim.mean().item():.4f}")

    encoder.train()
    projector.train()


def _find_latest_ckpt(model_dir: str, ckpt_prefix: str) -> Optional[str]:
    ckpts = glob.glob(os.path.join(model_dir, f"{ckpt_prefix}*.pt"))
    if not ckpts:
        return None

    def ep_num(path: str) -> int:
        m = re.search(r"ep(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(ckpts, key=ep_num)


def save_checkpoint(path: str, epoch: int, global_step: int, encoder, projector, optimizer, scheduler, scaler):
    tmp = path + ".tmp"
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": CFG.__dict__,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, encoder, projector, optimizer, scheduler, scaler) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    return epoch, global_step


def train():
    os.makedirs(CFG.model_dir, exist_ok=True)
    seed_everything(CFG.seed)

    # Sanity prints
    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    if torch.cuda.is_available():
        print(f"visible_devices={torch.cuda.device_count()}")
        print(f"device0={torch.cuda.get_device_name(0)}")

    # ROCm: avoid benchmark heuristics which can amplify tuning behavior across shapes
    is_rocm = getattr(torch.version, "hip", None) is not None
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = False if is_rocm else True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if CFG.accum_steps < 1:
        raise ValueError("ACCUM_STEPS must be >= 1")

    dataset = BalancedMixedDataset(CFG)
    pin = (DEVICE.type == "cuda")

    dl_kwargs = dict(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=pin,
    )
    if CFG.num_workers > 0:
        dl_kwargs["prefetch_factor"] = CFG.prefetch_factor
        dl_kwargs["persistent_workers"] = CFG.persistent_workers

    dataloader = DataLoader(dataset, **dl_kwargs)

    # Small validation loader to monitor collapse
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

    start_epoch = 0
    global_step = 0

    latest = _find_latest_ckpt(CFG.model_dir, CFG.ckpt_prefix)
    if latest:
        print(f"Resuming from checkpoint: {latest}")
        start_epoch, global_step = load_checkpoint(latest, encoder, projector, optimizer, scheduler, scaler)
        print(f"Resumed at epoch {start_epoch} (next epoch will be {start_epoch + 1}), global_step={global_step}")

    encoder.train()
    projector.train()

    steps_per_full_epoch = max(1, dataset.total_frames // CFG.batch_size)
    steps_this_epoch = CFG.max_steps_per_epoch if CFG.max_steps_per_epoch > 0 else steps_per_full_epoch

    for epoch in range(start_epoch, CFG.epochs):
        pbar = tqdm(dataloader, total=steps_this_epoch, desc=f"Epoch {epoch+1}/{CFG.epochs}")

        # We step optimizer every accum_steps micro-batches.
        optimizer.zero_grad(set_to_none=True)

        for step, (x1, x2) in enumerate(pbar):
            if CFG.max_steps_per_epoch > 0 and step >= CFG.max_steps_per_epoch:
                break

            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = vicreg_loss(projector(encoder(x1)), projector(encoder(x2)))
                    loss = loss / CFG.accum_steps
                scaler.scale(loss).backward()
            else:
                loss = vicreg_loss(projector(encoder(x1)), projector(encoder(x2)))
                loss = loss / CFG.accum_steps
                loss.backward()

            do_step = ((step + 1) % CFG.accum_steps == 0)
            if do_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # Throughput reporting: images/sec uses micro-batch size
            it_s = pbar.format_dict.get("rate", None)
            imgs_s = (it_s * CFG.batch_size) if (it_s is not None) else None

            postfix = {
                "loss": f"{(loss.item() * CFG.accum_steps):.4f}",  # unscaled loss for readability
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                "eff_bs": f"{CFG.batch_size * CFG.accum_steps}",
            }
            if imgs_s is not None:
                postfix["img/s"] = f"{imgs_s:.0f}"

            pbar.set_postfix(postfix)

            # Step-based checkpointing (safer for long runs)
            if CFG.ckpt_every_steps > 0 and (global_step % CFG.ckpt_every_steps == 0):
                ckpt_path = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}{epoch+1}_step{global_step}.pt")
                save_checkpoint(ckpt_path, epoch + 1, global_step, encoder, projector, optimizer, scheduler, scaler)

        # If epoch ended mid-accumulation, flush remaining grads
        if (step + 1) % CFG.accum_steps != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        epoch_num = epoch + 1

        if CFG.validate_every_epochs > 0 and (epoch_num % CFG.validate_every_epochs) == 0:
            validate_encoder(encoder, projector, val_loader, epoch_num)

        if CFG.save_every_epochs > 0 and (epoch_num % CFG.save_every_epochs) == 0:
            ckpt_path = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}{epoch_num}.pt")
            save_checkpoint(ckpt_path, epoch_num, global_step, encoder, projector, optimizer, scheduler, scaler)

            weights_path = os.path.join(CFG.model_dir, f"{CFG.weights_prefix}{epoch_num}.pth")
            torch.save(encoder.state_dict(), weights_path)

    final_ckpt = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}final.pt")
    save_checkpoint(final_ckpt, CFG.epochs, global_step, encoder, projector, optimizer, scheduler, scaler)

    final_weights = os.path.join(CFG.model_dir, "encoder_final.pth")
    torch.save(encoder.state_dict(), final_weights)


if __name__ == "__main__":
    train()
