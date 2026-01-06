import os
import re
import glob
import math
import random

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from tqdm import tqdm

from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- CONFIGURATION (env-overridable, but defaults match your old script) ---
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "128"))
EPOCHS        = int(os.getenv("EPOCHS", "30"))
LR            = float(os.getenv("LR", "3e-4"))
NUM_WORKERS   = int(os.getenv("NUM_WORKERS", "8"))
WEIGHT_DECAY  = float(os.getenv("WEIGHT_DECAY", "0.05"))
MODEL_DIR     = os.getenv("MODEL_DIR", "./models")

# Save cadence: matches your old behaviour (save every 5 epochs)
SAVE_EVERY_EPOCHS      = int(os.getenv("SAVE_EVERY_EPOCHS", "5"))          # encoder weights + full ckpt
VALIDATE_EVERY_EPOCHS  = int(os.getenv("VALIDATE_EVERY_EPOCHS", "5"))      # validate when saving

# Resume controls
RESUME                = os.getenv("RESUME", "1") == "1"
RESUME_FULL_IF_AVAIL  = os.getenv("RESUME_FULL_IF_AVAIL", "1") == "1"

DEVICE = torch.device(os.getenv("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")


class BalancedMixedDataset(IterableDataset):
    def __init__(self):
        self.random_files = sorted(glob.glob("./data/*.npz"))
        self.race_files = sorted(glob.glob("./data_race/*.npz"))
        self.recovery_files = sorted(glob.glob("./data_recovery/*.npz"))
        self.edge_files = sorted(glob.glob("./data_edge/*.npz"))

        # Create balanced 4-way mix
        max_files = max(
            len(self.random_files),
            len(self.race_files),
            len(self.recovery_files),
            len(self.edge_files),
        )

        self.balanced_files = []
        for i in range(max_files):
            if self.random_files:   self.balanced_files.append(self.random_files[i % len(self.random_files)])
            if self.race_files:     self.balanced_files.append(self.race_files[i % len(self.race_files)])
            if self.recovery_files: self.balanced_files.append(self.recovery_files[i % len(self.recovery_files)])
            if self.edge_files:     self.balanced_files.append(self.edge_files[i % len(self.edge_files)])

        print(f"Balanced Dataset: {len(self.balanced_files)} files.")

        # Calculate total frames (kept same style, just made it a bit safer)
        self.total_frames = 0
        for f in tqdm(self.balanced_files, desc="Scanning Dataset"):
            try:
                with np.load(f, mmap_mode="r", allow_pickle=False) as d:
                    if "states" in d:
                        self.total_frames += d["states"].shape[0]
                    elif "obs" in d:
                        self.total_frames += d["obs"].shape[0]
            except Exception:
                pass
        print(f"Total Frames: {self.total_frames:,}")

        # TRANSFORM: same as your old script
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomRotation(degrees=3),
        ])

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            my_files = self.balanced_files[worker_info.id::worker_info.num_workers]
        else:
            my_files = list(self.balanced_files)

        random.shuffle(my_files)

        for f in my_files:
            try:
                with np.load(f, allow_pickle=False) as data:
                    raw = data["states"] if "states" in data else data["obs"]

                if raw.shape[1] != 64 or raw.shape[2] != 64:
                    raw = np.array([cv2.resize(i, (64, 64), interpolation=cv2.INTER_AREA) for i in raw])

                indices = np.random.permutation(len(raw))
                for idx in indices:
                    img = torch.from_numpy(raw[idx]).float() / 255.0
                    img = img.permute(2, 0, 1)  # HWC -> CHW
                    yield self.transform(img), self.transform(img)

            except Exception:
                continue

    def __len__(self):
        return self.total_frames


@torch.no_grad()
def validate_encoder(encoder, val_loader, epoch: int):
    encoder.eval()
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
        return

    z_all = torch.cat(all_embeddings, dim=0)
    std_per_dim = z_all.std(dim=0)
    dead_dims = int((std_per_dim < 0.01).sum().item())

    print(f"\n[Validation Epoch {epoch}]")
    print(f"  Dead dimensions: {dead_dims}/512")
    print(f"  Avg Std: {std_per_dim.mean().item():.4f}")

    encoder.train()


def _latest_epoch_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    def ep_num(p: str) -> int:
        m = re.search(r"ep(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(files, key=ep_num)


def save_full_ckpt(path: str, epoch: int, global_step: int, encoder, projector, optimizer, scheduler, scaler):
    tmp = path + ".tmp"
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "meta": {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_full_ckpt(path: str, encoder, projector, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = BalancedMixedDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    # Validation loader (kept close to your original)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    start_epoch = 0
    global_step = 0

    if RESUME:
        # Prefer full-state resume if available (prevents projector randomisation)
        full_ckpt = _latest_epoch_file(os.path.join(MODEL_DIR, "encoder_mixed_ckpt_ep*.pt"))
        enc_only = _latest_epoch_file(os.path.join(MODEL_DIR, "encoder_mixed_ep*.pth"))

        if RESUME_FULL_IF_AVAIL and full_ckpt is not None:
            print(f"Resuming FULL state from {full_ckpt}")
            start_epoch, global_step = load_full_ckpt(full_ckpt, encoder, projector, optimizer, scheduler, scaler)
            # start_epoch is the epoch number already completed
        elif enc_only is not None:
            # Legacy behaviour (your old script)
            print(f"Resuming ENCODER ONLY from {enc_only}")
            encoder.load_state_dict(torch.load(enc_only, map_location=DEVICE))
            start_epoch = int(re.search(r"ep(\d+)", enc_only).group(1))
            for _ in range(start_epoch):
                scheduler.step()

    encoder.train()
    projector.train()

    # FIX: use ceil to avoid tqdm "extra iteration" confusion
    steps = max(1, math.ceil(dataset.total_frames / BATCH_SIZE))

    for epoch in range(start_epoch, EPOCHS):
        pbar = tqdm(dataloader, total=steps, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for x1, x2 in pbar:
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    loss = vicreg_loss(projector(encoder(x1)), projector(encoder(x2)))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = vicreg_loss(projector(encoder(x1)), projector(encoder(x2)))
                loss.backward()
                optimizer.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        # Save/validate every 5 epochs (as before)
        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
            if (epoch + 1) % VALIDATE_EVERY_EPOCHS == 0:
                validate_encoder(encoder, val_loader, epoch + 1)

            enc_path = os.path.join(MODEL_DIR, f"encoder_mixed_ep{epoch+1}.pth")
            torch.save(encoder.state_dict(), enc_path)

            ckpt_path = os.path.join(MODEL_DIR, f"encoder_mixed_ckpt_ep{epoch+1}.pt")
            save_full_ckpt(ckpt_path, epoch + 1, global_step, encoder, projector, optimizer, scheduler, scaler)

    torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "encoder_mixed_final.pth"))


if __name__ == "__main__":
    train()
