import os
import re
import glob
import math
import random
import sys
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from tqdm import tqdm
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

@dataclass
class CFG:
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))
    epochs: int = int(os.getenv("EPOCHS", "30"))
    lr: float = float(os.getenv("LR", "3e-4"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.05"))
    
    # DataLoader
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))
    persistent_workers: bool = os.getenv("PERSISTENT_WORKERS", "0") == "1"
    
    # Dataset dirs
    data_random: str = os.getenv("DATA_RANDOM", "./data")
    data_race: str = os.getenv("DATA_RACE", "./data_race")
    data_recovery: str = os.getenv("DATA_RECOVERY", "./data_recovery")
    data_edge: str = os.getenv("DATA_EDGE", "./data_edge")
    
    # Saving/validation cadence
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    save_every_epochs: int = int(os.getenv("SAVE_EVERY_EPOCHS", "5"))
    validate_every_epochs: int = int(os.getenv("VALIDATE_EVERY_EPOCHS", "5"))
    max_epoch_ckpts: int = int(os.getenv("MAX_EPOCH_CKPTS", "20"))  
    
    # Resume
    resume: bool = os.getenv("RESUME", "1") == "1"
    resume_full_if_avail: bool = os.getenv("RESUME_FULL_IF_AVAIL", "1") == "1"
    warm_start_encoder: str = os.getenv("WARM_START_ENCODER", "")  # path to .pth encoder weights (fresh opt/sched)
    val_num_batches: int = int(os.getenv("VAL_NUM_BATCHES", "20"))
    dead_std_thr: float = float(os.getenv("DEAD_STD_THR", "0.01"))
    seed: int = int(os.getenv("SEED", "1337"))
    device: str = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

CFG = CFG()
DEVICE = torch.device(CFG.device)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int) -> None:

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    s = CFG.seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

# Running Average
class RunningAverage:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Dataset
def _list_npz(dir_path: str) -> List[str]:
    if not dir_path:
        return []
    return sorted(glob.glob(os.path.join(dir_path, "*.npz")))

def _count_frames_npz(path: str) -> int:
    try:
        # mmap_mode='r' avoids loading data just to check shape
        with np.load(path, mmap_mode="r", allow_pickle=False) as d:
            if "states" in d:
                return int(d["states"].shape[0])
            if "obs" in d:
                return int(d["obs"].shape[0])
    except Exception:
        pass
    return 0

class BalancedMixedDataset(IterableDataset):
    """
    Balanced 4-way mix (random/race/recovery/edge) by file count.
    Yields (aug(img), aug(img)) for VICReg.
    """
    def __init__(self):
        super().__init__()
        self.random_files = _list_npz(CFG.data_random)
        self.race_files = _list_npz(CFG.data_race)
        self.recovery_files = _list_npz(CFG.data_recovery)
        self.edge_files = _list_npz(CFG.data_edge)

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
        
        self.total_frames = 0
        # Fast scan
        for f in tqdm(self.balanced_files, desc="Scanning Dataset"):
            self.total_frames += _count_frames_npz(f)
        print(f"Total Frames: {self.total_frames:,}")

        self.epoch = 0

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomRotation(degrees=3),
        ])

    def set_epoch(self, epoch: int):
        """Update the current epoch so the next iterator uses a fresh seed."""
        self.epoch = epoch

    def __iter__(self):
        cv2.setNumThreads(0)
        
        info = get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        
        current_seed = CFG.seed + worker_id + (self.epoch * 10000)
        
        rng = random.Random(current_seed)
        
        np_rng = np.random.default_rng(current_seed)

        torch.manual_seed(current_seed)
        
        my_files = self.balanced_files[worker_id::num_workers]
        my_files = list(my_files)
        
        rng.shuffle(my_files)
        
        for f in my_files:
            try:
                with np.load(f, allow_pickle=False) as data:
                    raw = data["states"] if "states" in data else data["obs"]
                
                # Shuffle frames
                idxs = np_rng.permutation(len(raw))
                
                for idx in idxs:
                    img_np = raw[idx] # Grab single frame
                    
                    # Lazy Resize: Only resize this specific frame if needed
                    if img_np.shape[0] != 64 or img_np.shape[1] != 64:
                        img_np = cv2.resize(img_np, (64, 64), interpolation=cv2.INTER_AREA)

                    img = torch.from_numpy(img_np).float().div_(255.0)  # [H,W,C]
                    img = img.permute(2, 0, 1)  # [C,H,W]
                    yield self.transform(img), self.transform(img)
                
                # [OPTIMIZATION] Explicit delete to help GC
                del raw

            except Exception:
                continue
        
        if sys.platform == "linux":
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    def __len__(self):
        return self.total_frames

# Validation
@torch.no_grad()
def validate_encoder(encoder, projector, val_loader, epoch: int) -> None:
    encoder.eval()
    projector.eval()
    
    zs_enc = []
    zs_prj = []
    
    for i, (x1, _) in enumerate(val_loader):
        if i >= CFG.val_num_batches:
            break
        x1 = x1.to(DEVICE, non_blocking=True)
        z = encoder(x1)
        p = projector(z)
        zs_enc.append(z.detach().float().cpu())
        zs_prj.append(p.detach().float().cpu())
        
    if not zs_enc:
        print(f"\n[Validation Epoch {epoch}] No validation batches available.")
        encoder.train()
        projector.train()
        return

    z_enc = torch.cat(zs_enc, dim=0)
    z_prj = torch.cat(zs_prj, dim=0)

    def report(name: str, z: torch.Tensor) -> None:
        thr = CFG.dead_std_thr
        std_per_dim = z.std(dim=0)
        dead_abs = int((std_per_dim < thr).sum().item())
        avg_std = float(std_per_dim.mean().item())
        
        norms = z.norm(dim=1)
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        
        z_n = z / (norms.unsqueeze(1) + 1e-8)
        std_per_dim_n = z_n.std(dim=0)
        dead_norm = int((std_per_dim_n < thr).sum().item())
        avg_std_n = float(std_per_dim_n.mean().item())
        
        print(f"  [{name}]")
        print(f"    ||z|| mean/std: {mean_norm:.4f} / {std_norm:.4f}")
        print(f"    Avg Std (raw): {avg_std:.4f}  Dead@{thr:g}: {dead_abs}/{z.shape[1]}")
        print(f"    Avg Std (L2):  {avg_std_n:.4f}  Dead@{thr:g}: {dead_norm}/{z.shape[1]}")

    print(f"\n[Validation Epoch {epoch}] batches={CFG.val_num_batches} samples={z_enc.shape[0]}")
    report("ENCODER", z_enc)
    report("PROJECTOR", z_prj)
    
    encoder.train()
    projector.train()

# Checkpoint helpers
def _epoch_from_name(path: str) -> int:
    m = re.search(r"ep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

def _latest_by_epoch(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=_epoch_from_name)

def save_full_ckpt(path: str, epoch: int, global_step: int, encoder, projector, optimizer, scheduler, scaler) -> None:
    tmp = path + ".tmp"
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": {
            "batch_size": CFG.batch_size,
            "epochs": CFG.epochs,
            "lr": CFG.lr,
            "weight_decay": CFG.weight_decay,
        },
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)

def load_full_ckpt(path: str, encoder, projector, optimizer, scheduler, scaler) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))

def _prune_old(pattern: str, keep: int) -> None:
    if keep <= 0:
        return
    files = glob.glob(pattern)
    if len(files) <= keep:
        return
    files_sorted = sorted(files, key=_epoch_from_name)
    for f in files_sorted[:-keep]:
        try:
            os.remove(f)
        except Exception:
            pass

# Train
def train() -> None:
    os.makedirs(CFG.model_dir, exist_ok=True)
    seed_everything(CFG.seed)

    if CFG.persistent_workers:
        raise RuntimeError(
            "PERSISTENT_WORKERS=1 is not supported with this epoch-based shuffling logic. "
            "Please set PERSISTENT_WORKERS=0 in your environment."
        )

    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    if torch.cuda.is_available():
        print(f"visible_devices={torch.cuda.device_count()}")
        print(f"device0={torch.cuda.get_device_name(0)}")
        
    dataset = BalancedMixedDataset()
    
    steps_per_epoch = max(1, math.ceil(dataset.total_frames / CFG.batch_size))
    
    dl_kwargs = dict(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=worker_init_fn if CFG.num_workers > 0 else None,
    )
    if CFG.num_workers > 0:
        dl_kwargs["prefetch_factor"] = CFG.prefetch_factor
        dl_kwargs["persistent_workers"] = False # Enforce disabled
        
    dataloader = DataLoader(dataset, **dl_kwargs)
    
    val_loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=min(2, CFG.num_workers),
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=worker_init_fn if min(2, CFG.num_workers) > 0 else None,
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
    
    if CFG.warm_start_encoder:
        print(f"Warm-start encoder from: {CFG.warm_start_encoder}")
        encoder.load_state_dict(torch.load(CFG.warm_start_encoder, map_location=DEVICE))
        print("Warm-start enabled: starting with fresh optimizer/scheduler.")
    
    elif CFG.resume:
        full_ckpt = _latest_by_epoch(os.path.join(CFG.model_dir, "encoder_mixed_ckpt_ep*.pt"))
        enc_only = _latest_by_epoch(os.path.join(CFG.model_dir, "encoder_mixed_ep*.pth"))
        
        if CFG.resume_full_if_avail and full_ckpt is not None:
            print(f"Resuming FULL state from {full_ckpt}")
            start_epoch, global_step = load_full_ckpt(full_ckpt, encoder, projector, optimizer, scheduler, scaler)
        elif enc_only is not None:
            print(f"Resuming ENCODER ONLY from {enc_only}")
            encoder.load_state_dict(torch.load(enc_only, map_location=DEVICE))
            start_epoch = _epoch_from_name(enc_only)
            for _ in range(start_epoch):
                scheduler.step()

    encoder.train()
    projector.train()

    for epoch in range(start_epoch, CFG.epochs):
        dataset.set_epoch(epoch)
        
        if epoch == start_epoch:
             print(f"[DEBUG] epoch={epoch} dataset.epoch={dataset.epoch}")

        loss_avg = RunningAverage()

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{CFG.epochs}")
        it = iter(dataloader)
        
        for step in range(steps_per_epoch):
            try:
                x1, x2 = next(it)
            except StopIteration:
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
            
            loss_val = loss.item()
            loss_avg.update(loss_val, n=x1.size(0))
            
            global_step += 1
            
            pbar.set_postfix(loss=f"{loss_avg.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            pbar.update(1)
            
        pbar.close()
        
    
        print(f"Epoch {epoch+1} finished. Avg Loss: {loss_avg.avg:.4f}")
        
        scheduler.step()
        
        epoch_num = epoch + 1
        
        if CFG.save_every_epochs > 0 and (epoch_num % CFG.save_every_epochs == 0):
            if CFG.validate_every_epochs > 0 and (epoch_num % CFG.validate_every_epochs == 0):
                validate_encoder(encoder, projector, val_loader, epoch_num)
            
            enc_path = os.path.join(CFG.model_dir, f"encoder_mixed_ep{epoch_num}.pth")
            torch.save(encoder.state_dict(), enc_path)
            
            ckpt_path = os.path.join(CFG.model_dir, f"encoder_mixed_ckpt_ep{epoch_num}.pt")
            save_full_ckpt(ckpt_path, epoch_num, global_step, encoder, projector, optimizer, scheduler, scaler)
            
            if CFG.max_epoch_ckpts > 0:
                _prune_old(os.path.join(CFG.model_dir, "encoder_mixed_ep*.pth"), CFG.max_epoch_ckpts)
                _prune_old(os.path.join(CFG.model_dir, "encoder_mixed_ckpt_ep*.pt"), CFG.max_epoch_ckpts)
                
    # Final save
    torch.save(encoder.state_dict(), os.path.join(CFG.model_dir, "encoder_mixed_final.pth"))
    save_full_ckpt(
        os.path.join(CFG.model_dir, "encoder_mixed_ckpt_final.pt"),
        CFG.epochs,
        global_step,
        encoder,
        projector,
        optimizer,
        scheduler,
        scaler,
    )

if __name__ == "__main__":
    train()