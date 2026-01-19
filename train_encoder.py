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
    batch_size: int = int(os.getenv("BATCH_SIZE", "2048"))
    lr: float = float(os.getenv("LR", "0.006"))
    epochs: int = int(os.getenv("EPOCHS", "30"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.05"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "4"))
    
    data_random: str = os.getenv("DATA_RANDOM", "./data_random")
    data_expert: str = os.getenv("DATA_EXPERT", "./data_expert")
    data_recover: str = os.getenv("DATA_RECOVER", "./data_recover")
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    
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

def _list_npy(dir_path: str) -> List[str]:
    if not dir_path or not os.path.exists(dir_path): return []
    return sorted(glob.glob(os.path.join(dir_path, "*.npy")))

class FastRAMDataset(Dataset):
    def __init__(self):
        super().__init__()
        files = _list_npy(CFG.data_random) + _list_npy(CFG.data_expert) + _list_npy(CFG.data_recover)
        if not files: raise RuntimeError("No .npy files found!")
        
        print(f"Loading {len(files)} files into Shared Memory...")
        data_list = []
        for f in tqdm(files, desc="Reading"):
            try: data_list.append(np.load(f))
            except: pass
        
        if not data_list: raise RuntimeError("Failed to load data!")
        np_data = np.concatenate(data_list, axis=0)
        
        # Store as ByteTensor in Shared Memory
        self.data = torch.from_numpy(np_data)
        self.data.share_memory_()
        self.total_len = len(self.data)
        
        print(f"RAM Load Complete. {self.total_len} frames.")
        del data_list, np_data

    def __len__(self): return self.total_len

    def __getitem__(self, idx):
        # Stack 4 frames: t, t-1, t-2, t-3
        # If we hit index 0, replicate it
        indices = []
        for i in range(4):
            indices.append(max(0, idx - i))
        
        # We need to reverse because range(4) gives [0, 1, 2, 3] -> [t, t-1, t-2, t-3]
        # Standard stack order is usually Old -> New. Let's do Old -> New.
        indices = indices[::-1] # Now [t-3, t-2, t-1, t]
        
        # Gather frames
        frames = self.data[indices] # (4, 64, 64, 3)
        
        # Reshape to (12, 64, 64) for Conv2d
        # (4, H, W, 3) -> (4, 3, H, W) -> (12, H, W)
        frames = frames.permute(0, 3, 1, 2).reshape(12, 64, 64)
        return frames

# --- GPU AUGMENT ---
class GPUAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Note: We apply geometry transforms to the whole stack to keep alignment
        self.geom = v2.Compose([
            v2.ToDtype(torch.float32, scale=True), 
            v2.RandomResizedCrop(64, scale=(0.85, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
        ])
        # Color jitter is tricky on stacks. For simplicity in this robust version,
        # we apply it to the whole block or skip. Let's skip heavy color jitter 
        # on stacks to prevent "flicker" confusion, or apply same jitter to all.
        # Here we just do geometry + normalization.
        
    def forward(self, x):
        # x: (B, 12, 64, 64)
        return self.geom(x)

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
        z = encoder(x) # (B, 512, 8, 8)
        zs.append(z)
        
    if not zs: return
    z_spatial = torch.cat(zs, dim=0).float()
    z_flat = z_spatial.mean(dim=[2, 3]) 
    std = z_flat.std(dim=0)
    dead = (std < CFG.dead_std_thr).sum().item()
    avg_std = std.mean().item()
    
    print(f"  [ENCODER] Avg Std: {avg_std:.4f} | Dead Neurons: {dead}/{z_spatial.shape[1]}")
    encoder.train()
    projector.train()

def _epoch_from_name(path: str) -> int:
    m = re.search(r"ep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1

def _latest_by_epoch(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files: return None
    return max(files, key=_epoch_from_name)

def train():
    os.makedirs(CFG.model_dir, exist_ok=True)
    seed_everything(CFG.seed)
    torch.backends.cudnn.benchmark = True 
    
    dataset = FastRAMDataset()
    dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True, 
                            num_workers=CFG.num_workers, pin_memory=True, 
                            prefetch_factor=CFG.prefetch_factor, 
                            worker_init_fn=worker_init_fn, persistent_workers=True)
    
    encoder = TinyEncoder().to(DEVICE, memory_format=torch.channels_last)
    projector = Projector().to(DEVICE, memory_format=torch.channels_last)
    gpu_aug = GPUAugment().to(DEVICE)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), 
                           lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    if CFG.resume:
        full_ckpt = _latest_by_epoch(os.path.join(CFG.model_dir, "encoder_mixed_ckpt_ep*.pt"))
        if full_ckpt:
            # Note: This will fail if dimensions mismatch (re-training required)
            try:
                ckpt = torch.load(full_ckpt, map_location=DEVICE)
                encoder.load_state_dict(ckpt["encoder"])
                projector.load_state_dict(ckpt["projector"])
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt["epoch"]
                print(f"Resumed from {full_ckpt}")
            except Exception as e:
                print(f"Could not resume (Arch changed?): {e}")

    print(f"--- STARTING TRAINING (Batch: {CFG.batch_size}) ---")
    for epoch in range(start_epoch, CFG.epochs):
        encoder.train()
        projector.train()
        
        pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{CFG.epochs}")
        for step, batch_raw in enumerate(pbar):
            raw = batch_raw.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast("cuda"):
                x1 = gpu_aug(raw)
                x2 = gpu_aug(raw)
                
                z1, z2 = encoder(x1), encoder(x2)
                p1, p2 = projector(z1), projector(z2)
                loss = vicreg_loss(p1, p2)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if step % 25 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        
        epoch_num = epoch + 1
        if epoch_num % CFG.save_every_epochs == 0:
            torch.save(encoder.state_dict(), os.path.join(CFG.model_dir, f"encoder_mixed_ep{epoch_num}.pth"))
            torch.save({
                "epoch": epoch_num,
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(CFG.model_dir, f"encoder_mixed_ckpt_ep{epoch_num}.pt"))
            
            if epoch_num % CFG.validate_every_epochs == 0:
                validate(encoder, projector, gpu_aug, dataloader, epoch_num)

    torch.save(encoder.state_dict(), os.path.join(CFG.model_dir, "encoder_mixed_final.pth"))

if __name__ == "__main__":
    train()