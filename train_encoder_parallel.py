#!/usr/bin/env python3
import os
import re
import json
import glob
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import nullcontext, ExitStack

# ----------------------------
# ROCm/MIOpen early setup (must run before torch import)
# ----------------------------

def _mkdir_700(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except Exception:
        pass

def _early_miopen_setup() -> None:
    """
    Stability-oriented MIOpen settings.
    - Uses per-local-rank cache dirs by default to avoid DB/cache contention in DDP.
    - Optional: disable FindDb (SQLite) and force immediate fallback to avoid long finds.
    """
    if os.getenv("MIOPEN_STABLE", "0") != "1":
        return

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # Default per-rank tag to avoid contention when running 2 processes
    tag = os.getenv("MIOPEN_CACHE_TAG", f"gpu{local_rank}")

    user_db = os.getenv("MIOPEN_USER_DB_PATH", str(Path.home() / ".config" / f"miopen_{tag}"))
    cache_dir = os.getenv("MIOPEN_CUSTOM_CACHE_DIR", str(Path.home() / ".cache" / f"miopen_{tag}"))

    os.environ["MIOPEN_USER_DB_PATH"] = user_db
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = cache_dir
    _mkdir_700(user_db)
    _mkdir_700(cache_dir)

    if os.getenv("MIOPEN_DISABLE_FIND_DB", "0") == "1":
        os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

    if os.getenv("MIOPEN_FORCE_CONV_IMMED_FALLBACK", "0") == "1":
        os.environ["MIOPEN_DEBUG_CONV_IMMED_FALLBACK"] = "1"
        os.environ["MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK"] = "1"

_early_miopen_setup()

# ----------------------------
# Imports (after early env setup)
# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

print("DEBUG: Imports finished, entering main...", flush=True)

# ----------------------------
# Distributed helpers
# ----------------------------

def _is_distributed() -> bool:
    return int(os.getenv("WORLD_SIZE", "1")) > 1

def _ddp_setup() -> Tuple[int, int, int]:
    """
    Returns (rank, local_rank, world_size).
    If WORLD_SIZE>1, initializes process group and sets correct CUDA device.
    """
    if not _is_distributed():
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # ROCm: backend is still "nccl" (maps to RCCL under the hood)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def _ddp_cleanup() -> None:
    if _is_distributed():
        dist.destroy_process_group()


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Config:
    # Training
    batch_size: int = int(os.getenv("BATCH_SIZE", "48"))          # micro-batch per GPU
    accum_steps: int = int(os.getenv("ACCUM_STEPS", "3"))         # gradient accumulation
    epochs: int = int(os.getenv("EPOCHS", "28"))
    lr: float = float(os.getenv("LR", "3e-4"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.05"))

    # Loader
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))
    persistent_workers: bool = os.getenv("PERSISTENT_WORKERS", "0") == "1"

    # Cap steps per epoch (per-rank steps)
    max_steps_per_epoch: int = int(os.getenv("MAX_STEPS_PER_EPOCH", "10000"))

    # Logging / validation / saving
    validate_every_epochs: int = int(os.getenv("VALIDATE_EVERY_EPOCHS", "3"))
    save_every_epochs: int = int(os.getenv("SAVE_EVERY_EPOCHS", "1"))
    ckpt_every_steps: int = int(os.getenv("CKPT_EVERY_STEPS", "5000"))  # 0 disables step ckpt

    seed: int = int(os.getenv("SEED", "1337"))

    # Dataset dirs
    data_random: str = os.getenv("DATA_RANDOM", "./data")
    data_race: str = os.getenv("DATA_RACE", "./data_race")
    data_recovery: str = os.getenv("DATA_RECOVERY", "./data_recovery")
    data_edge: str = os.getenv("DATA_EDGE", "./data_edge")

    # Cache scan results
    cache_scan: bool = os.getenv("CACHE_SCAN", "1") == "1"
    cache_path: str = os.getenv("CACHE_PATH", "./.cache/dataset_index.json")

    # Outputs
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    ckpt_prefix: str = os.getenv("CKPT_PREFIX", "encoder_ckpt_ep")
    weights_prefix: str = os.getenv("WEIGHTS_PREFIX", "encoder_ep")

CFG = Config()


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

    DDP sharding:
      - First shard files by rank/world_size: files[rank::world_size]
      - Then shard by dataloader worker within each rank: files[worker_id::num_workers]
    """

    def __init__(self, cfg: Config, rank: int, world_size: int, cache_path: str):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.cache_path = cache_path

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

        # Shard by rank for DDP
        self.rank_files = self.balanced_files[self.rank::self.world_size]

        if self.rank == 0:
            print(f"Balanced Dataset: {len(self.balanced_files)} files total.")
            if self.world_size > 1:
                print(f"DDP: each rank sees ~{len(self.rank_files)} files.")

        self.total_frames = self._scan_total_frames()

        # Keep directionality (no horizontal flip)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=5),
        ])

    def _scan_total_frames(self) -> int:
        # Scan *all* files for a consistent total count (fine for progress/estimates).
        # Use per-rank cache file to avoid concurrent writes.
        files_to_scan = self.balanced_files

        if not self.cfg.cache_scan:
            total = 0
            it = tqdm(files_to_scan, desc="Scanning Dataset", disable=(self.rank != 0))
            for f in it:
                try:
                    total += _count_frames_npz(f)
                except Exception:
                    continue
            if self.rank == 0:
                print(f"Total Frames: {total:,}")
            return total

        cache = _load_cache(self.cache_path)
        cached = cache.get("files", {})
        updated = {}
        total = 0

        it = tqdm(files_to_scan, desc="Scanning Dataset", disable=(self.rank != 0))
        for f in it:
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
                continue

        _save_cache(self.cache_path, {"files": updated, "generated_at": time.time()})
        if self.rank == 0:
            print(f"Total Frames: {total:,}")
        return total

    def __len__(self) -> int:
        return self.total_frames

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1

        # Ensure different shuffles per-rank and per-worker
        base_seed = self.cfg.seed + (self.rank * 10_000) + worker_id
        rng = random.Random(base_seed)
        np_rng = np.random.default_rng(base_seed)

        my_files = self.rank_files[worker_id::num_workers]
        rng.shuffle(my_files)

        for f in my_files:
            try:
                with np.load(f, allow_pickle=False) as data:
                    raw = data["states"] if "states" in data else data["obs"]

                if raw.shape[1] != 64 or raw.shape[2] != 64:
                    raw = np.array(
                        [cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA) for img in raw],
                        dtype=raw.dtype,
                    )

                idxs = np_rng.permutation(len(raw))
                for idx in idxs:
                    img = torch.from_numpy(raw[idx]).float().div_(255.0)
                    img = img.permute(2, 0, 1)  # [C,H,W]
                    yield self.transform(img), self.transform(img)

            except Exception:
                continue


@torch.no_grad()
def validate_encoder(encoder: nn.Module, projector: nn.Module, val_loader, device: torch.device, epoch: int):
    encoder.eval()
    projector.eval()

    all_embeddings = []
    for i, (x1, _) in enumerate(val_loader):
        if i >= 10:
            break
        x1 = x1.to(device, non_blocking=True)
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


def save_checkpoint(path: str, epoch: int, global_step: int,
                    encoder: nn.Module, projector: nn.Module,
                    optimizer, scheduler, scaler):
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


def load_checkpoint(path: str, encoder: nn.Module, projector: nn.Module, optimizer, scheduler, scaler,
                    map_location: torch.device) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=map_location)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    return epoch, global_step


def main():
    rank, local_rank, world_size = _ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rank-aware cache file to avoid concurrent writes
    cache_path = CFG.cache_path
    if world_size > 1:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}.rank{rank}{ext or '.json'}"

    # Seeds
    seed_everything(CFG.seed + rank * 1000)

    if rank == 0:
        print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
        if torch.cuda.is_available():
            print(f"visible_devices={torch.cuda.device_count()}")
            print(f"device0={torch.cuda.get_device_name(0)}")
        print(f"DDP world_size={world_size} (rank={rank}, local_rank={local_rank})" if world_size > 1 else "Single GPU run")

    # ROCm: reduce heuristic benchmarking
    is_rocm = getattr(torch.version, "hip", None) is not None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False if is_rocm else True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if CFG.accum_steps < 1:
        raise ValueError("ACCUM_STEPS must be >= 1")

    os.makedirs(CFG.model_dir, exist_ok=True)

    dataset = BalancedMixedDataset(CFG, rank=rank, world_size=world_size, cache_path=cache_path)
    pin = (device.type == "cuda")

    dl_kwargs = dict(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=pin,
    )
    if CFG.num_workers > 0:
        dl_kwargs["prefetch_factor"] = CFG.prefetch_factor
        dl_kwargs["persistent_workers"] = CFG.persistent_workers

    dataloader = DataLoader(dataset, **dl_kwargs)

    # Validation loader (only used by rank0)
    val_loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=min(2, CFG.num_workers),
        pin_memory=pin,
    )

    encoder = TinyEncoder().to(device)
    projector = Projector().to(device)

    # Wrap in DDP if needed
    if world_size > 1:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank], output_device=local_rank)
        projector = torch.nn.parallel.DistributedDataParallel(projector, device_ids=[local_rank], output_device=local_rank)

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-5)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch = 0
    global_step = 0

    # Resume only from rank0-discovered checkpoint; then broadcast state (simple: load on all ranks)
    latest = _find_latest_ckpt(CFG.model_dir, CFG.ckpt_prefix)
    if latest:
        if rank == 0:
            print(f"Resuming from checkpoint: {latest}")
        start_epoch, global_step = load_checkpoint(latest,
                                                   encoder.module if world_size > 1 else encoder,
                                                   projector.module if world_size > 1 else projector,
                                                   optimizer, scheduler, scaler,
                                                   map_location=device)
        if rank == 0:
            print(f"Resumed at epoch {start_epoch} (next epoch {start_epoch + 1}), global_step={global_step}")

    steps_this_epoch = CFG.max_steps_per_epoch if CFG.max_steps_per_epoch > 0 else max(1, len(dataset) // CFG.batch_size)

    # Only rank0 shows progress
    pbar_disable = (rank != 0)

    for epoch in range(start_epoch, CFG.epochs):
        encoder.train()
        projector.train()

        pbar = tqdm(dataloader, total=steps_this_epoch, desc=f"Epoch {epoch+1}/{CFG.epochs}", disable=pbar_disable)

        optimizer.zero_grad(set_to_none=True)

        for step, (x1, x2) in enumerate(pbar):
            if CFG.max_steps_per_epoch > 0 and step >= CFG.max_steps_per_epoch:
                break

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            sync_now = ((step + 1) % CFG.accum_steps == 0)

            # On accumulation steps, avoid gradient all-reduce to reduce overhead
            if world_size > 1 and not sync_now:
                ctx_enc = encoder.no_sync()
                ctx_proj = projector.no_sync()
            else:
                ctx_enc = nullcontext()
                ctx_proj = nullcontext()

            with ExitStack() as stack:
                stack.enter_context(ctx_enc)
                stack.enter_context(ctx_proj)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        z1 = projector(encoder(x1))
                        z2 = projector(encoder(x2))
                        loss = vicreg_loss(z1, z2) / CFG.accum_steps
                    scaler.scale(loss).backward()
                else:
                    z1 = projector(encoder(x1))
                    z2 = projector(encoder(x2))
                    loss = vicreg_loss(z1, z2) / CFG.accum_steps
                    loss.backward()

            if sync_now:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if rank == 0:
                it_s = pbar.format_dict.get("rate", None)
                imgs_s = (it_s * CFG.batch_size) if it_s is not None else None
                postfix = {
                    "loss": f"{(loss.item() * CFG.accum_steps):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                    "eff_bs": f"{CFG.batch_size * CFG.accum_steps * world_size}",
                }
                if imgs_s is not None:
                    postfix["img/s (per-rank)"] = f"{imgs_s:.0f}"
                pbar.set_postfix(postfix)

            # Step-based checkpointing: only rank0 writes
            if rank == 0 and CFG.ckpt_every_steps > 0 and (global_step % CFG.ckpt_every_steps == 0):
                enc_to_save = encoder.module if world_size > 1 else encoder
                proj_to_save = projector.module if world_size > 1 else projector
                ckpt_path = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}{epoch+1}_step{global_step}.pt")
                save_checkpoint(ckpt_path, epoch + 1, global_step, enc_to_save, proj_to_save, optimizer, scheduler, scaler)

        # Flush if we stopped mid-accumulation
        if (step + 1) % CFG.accum_steps != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        epoch_num = epoch + 1

        # Validation + epoch checkpoints: rank0 only
        if rank == 0:
            enc_mod = encoder.module if world_size > 1 else encoder
            proj_mod = projector.module if world_size > 1 else projector

            if CFG.validate_every_epochs > 0 and (epoch_num % CFG.validate_every_epochs) == 0:
                validate_encoder(enc_mod, proj_mod, val_loader, device, epoch_num)

            if CFG.save_every_epochs > 0 and (epoch_num % CFG.save_every_epochs) == 0:
                ckpt_path = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}{epoch_num}.pt")
                save_checkpoint(ckpt_path, epoch_num, global_step, enc_mod, proj_mod, optimizer, scheduler, scaler)

                weights_path = os.path.join(CFG.model_dir, f"{CFG.weights_prefix}{epoch_num}.pth")
                torch.save(enc_mod.state_dict(), weights_path)

    if rank == 0:
        enc_mod = encoder.module if world_size > 1 else encoder
        proj_mod = projector.module if world_size > 1 else projector
        final_ckpt = os.path.join(CFG.model_dir, f"{CFG.ckpt_prefix}final.pt")
        save_checkpoint(final_ckpt, CFG.epochs, global_step, enc_mod, proj_mod, optimizer, scheduler, scaler)
        torch.save(enc_mod.state_dict(), os.path.join(CFG.model_dir, "encoder_final.pth"))

    _ddp_cleanup()


if __name__ == "__main__":
    main()
