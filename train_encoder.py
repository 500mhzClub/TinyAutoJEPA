import os
import glob
import re
import random
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

@dataclass
class CFG:
    # --- R9700 OPTIMIZED DEFAULTS ---
    # Batch Size 2048 is the "Sweet Spot" (~26GB VRAM usage, safe from crashes)
    batch_size: int = int(os.getenv("BATCH_SIZE", "2048"))
    
    # LR scaled for Batch 2048 (0.0015 * 4 = 0.006)
    lr: float = float(os.getenv("LR", "0.006"))
    
    epochs: int = int(os.getenv("EPOCHS", "30"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.05"))
    
    # Workers: 8 is optimal with Shared Memory
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "4"))
    
    # Directories
    data_random: str = os.getenv("DATA_RANDOM", "./data_random")
    data_expert: str = os.getenv("DATA_EXPERT", "./data_expert")
    data_recover: str = os.getenv("DATA_RECOVER", "./data_recover")
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    
    # Checkpointing & Validation
    save_every_epochs: int = 1
    validate_every_epochs: int = 5
    max_epoch_ckpts: int = 5
    resume: bool = True
    val_num_batches: int = 20
    dead_std_thr: float = 0.01
    seed: int = 1337
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

CFG = CFG()
DEVICE = torch.device(CFG.device)

# --- UTILS ---
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def worker_init_fn(worker_id: int):
    cv2.setNumThreads(0)
    s = CFG.seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

# --- FAST DATASET (SHARED MEMORY) ---
def _list_npy(dir_path: str) -> List[str]:
    if not dir_path or not os.path.exists(dir_path): return []
    return sorted(glob.glob(os.path.join(dir_path, "*.npy")))

class FastRAMDataset(Dataset):
    def __init__(self):
        super().__init__()
        files = _list_npy(CFG.data_random) + _list_npy(CFG.data_expert) + _list_npy(CFG.data_recover)
        if not files: raise RuntimeError("No .npy files found!")
        
        print(f"Loading {len(files)} files into Shared Memory Tensor...")
        data_list = []
        for f in tqdm(files, desc="Reading"):
            try: data_list.append(np.load(f))
            except: pass
        if not data_list: raise RuntimeError("Failed to load data!")

        # Concatenate and move to shared memory
        np_data = np.concatenate(data_list, axis=0)
        self.data = torch.from_numpy(np_data)
        self.data.share_memory_() # Zero-copy for workers
        
        print(f"RAM Load Complete. Size: {self.data.nbytes/1024**3:.2f} GB")
        del data_list, np_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- GPU AUGMENT ---
class GPUAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True), 
            v2.Lambda(lambda x: x.permute(0, 3, 1, 2)), # NHWC -> NCHW
            v2.RandomResizedCrop(64, scale=(0.85, 1.0), antialias=True),
            v2.ColorJitter(0.3, 0.3, 0.2, 0.1),
            v2.RandomHorizontalFlip(p=0.5),
        ])
    def forward(self, x): return self.transforms(x)

# --- VALIDATION ---
@torch.no_grad()
def validate(encoder, projector, gpu_aug, val_loader, epoch):
    encoder.eval()
    projector.eval()
    zs = []
    print(f"\n[Validation Epoch {epoch}]")
    for i, batch_raw in enumerate(val_loader):
        if i >= CFG.val_num_batches: break
        raw = batch_raw.to(DEVICE, non_blocking=True)
        x = gpu_aug(raw) 
        z = encoder(x)
        zs.append(z)
    if not zs: return
    z = torch.cat(zs, dim=0).float()
    
    # Calculate feature standard deviation to detect collapse
    std = z.std(dim=0)
    dead = (std < CFG.dead_std_thr).sum().item()
    avg_std = std.mean().item()
    
    print(f"  [ENCODER] Avg Std: {avg_std:.4f} | Dead Neurons: {dead}/{z.shape[1]}")
    encoder.train()
    projector.train()

# --- CHECKPOINTING HELPERS ---
def _epoch_from_name(path: str) -> int:
    m = re.search(r"ep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

def _latest_by_epoch(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files: return None
    return max(files, key=_epoch_from_name)

def save_full_ckpt(path: str, epoch: int, encoder, projector, optimizer, scheduler, scaler):
    tmp = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, tmp)
    os.replace(tmp, path)

def load_full_ckpt(path: str, encoder, projector, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0))

def _prune_old(pattern: str, keep: int):
    if keep <= 0: return
    files = sorted(glob.glob(pattern), key=_epoch_from_name)
    for f in files[:-keep]:
        try: os.remove(f)
        except: pass

# --- TRAINING LOOP ---
def train():
    os.makedirs(CFG.model_dir, exist_ok=True)
    seed_everything(CFG.seed)

    # CRITICAL: Enable CuDNN Benchmark for RDNA 4 performance
    torch.backends.cudnn.benchmark = True 
    
    print("--- Initializing Shared Memory Dataset ---")
    dataset = FastRAMDataset()
    dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True, 
                            num_workers=CFG.num_workers, pin_memory=True, 
                            prefetch_factor=CFG.prefetch_factor, 
                            worker_init_fn=worker_init_fn, persistent_workers=True)
    
    print("--- Initializing Optimized Architecture ---")
    # memory_format=torch.channels_last gives significant speedup on AMD
    encoder = TinyEncoder().to(DEVICE, memory_format=torch.channels_last)
    projector = Projector().to(DEVICE, memory_format=torch.channels_last)
    gpu_aug = GPUAugment().to(DEVICE)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), 
                           lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Resume Logic
    start_epoch = 0
    if CFG.resume:
        full_ckpt = _latest_by_epoch(os.path.join(CFG.model_dir, "encoder_mixed_ckpt_ep*.pt"))
        if full_ckpt:
            print(f"Resuming from {full_ckpt}")
            start_epoch = load_full_ckpt(full_ckpt, encoder, projector, optimizer, scheduler, scaler)

    print(f"--- STARTING TRAINING (Batch: {CFG.batch_size}, LR: {CFG.lr}) ---")
    for epoch in range(start_epoch, CFG.epochs):
        encoder.train()
        projector.train()
        
        pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{CFG.epochs}")
        for step, batch_raw in enumerate(pbar):
            # 1. Non-blocking transfer
            raw = batch_raw.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast("cuda"):
                # 2. Fast GPU Augment
                x1 = gpu_aug(raw)
                x2 = gpu_aug(raw)
                
                # 3. Forward
                z1, z2 = encoder(x1), encoder(x2)
                p1, p2 = projector(z1), projector(z2)
                loss = vicreg_loss(p1, p2)

            # 4. Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if step % 25 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()
        
        # Save & Validate
        epoch_num = epoch + 1
        if epoch_num % CFG.save_every_epochs == 0:
            torch.save(encoder.state_dict(), os.path.join(CFG.model_dir, f"encoder_mixed_ep{epoch_num}.pth"))
            ckpt_path = os.path.join(CFG.model_dir, f"encoder_mixed_ckpt_ep{epoch_num}.pt")
            save_full_ckpt(ckpt_path, epoch_num, encoder, projector, optimizer, scheduler, scaler)
            _prune_old(os.path.join(CFG.model_dir, "encoder_mixed_ckpt_ep*.pt"), CFG.max_epoch_ckpts)
            _prune_old(os.path.join(CFG.model_dir, "encoder_mixed_ep*.pth"), CFG.max_epoch_ckpts)

            if epoch_num % CFG.validate_every_epochs == 0:
                validate(encoder, projector, gpu_aug, dataloader, epoch_num)

    torch.save(encoder.state_dict(), os.path.join(CFG.model_dir, "encoder_mixed_final.pth"))
    print("Training Complete.")

if __name__ == "__main__":
    train()