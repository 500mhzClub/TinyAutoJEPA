from __future__ import annotations

import os
import glob
import re
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from networks import TinyEncoder, Projector
from vicreg import vicreg_loss


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    # Data
    data_random: str = os.getenv("DATA_RANDOM", "./data_random")
    data_expert: str = os.getenv("DATA_EXPERT", "./data_expert")
    data_recover: str = os.getenv("DATA_RECOVER", "./data_recover")

    # Training
    batch_size: int = int(os.getenv("BATCH_SIZE", "2048"))
    accum_steps: int = int(os.getenv("ACCUM_STEPS", "1"))
    epochs: int = int(os.getenv("EPOCHS", "30"))
    lr: float = float(os.getenv("LR", "6e-3"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "5e-4"))

    # Dataloader
    num_workers: int = int(os.getenv("NUM_WORKERS", "16"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))
    drop_last: bool = os.getenv("DROP_LAST", "1") == "1"

    # Frame stacking (temporal context)
    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))  # 4 -> 12ch if RGB

    # AMP
    amp: bool = os.getenv("AMP", "1") == "1"
    amp_dtype: str = os.getenv("AMP_DTYPE", "fp16").lower()  # fp16 | bf16
    # Compute VICReg variance/covariance stats in FP32 for stability under AMP
    loss_fp32: bool = os.getenv("LOSS_FP32", "1") == "1"

    # Augmentations
    aug_mode: str = os.getenv("AUG_MODE", "crop").lower()  # crop | rrc
    aug_dtype: str = os.getenv("AUG_DTYPE", "fp16").lower()  # fp16 | fp32 | bf16
    aug_antialias: bool = os.getenv("AUG_ANTIALIAS", "0") == "1"
    aug_scale_min: float = float(os.getenv("AUG_SCALE_MIN", "0.95"))
    second_view_fast: bool = os.getenv("SECOND_VIEW_FAST", "1") == "1"

    # IMPORTANT for turn semantics: horizontal flip changes left/right meaning.
    # Default OFF. If you ever enable it, this script COUPLES flip across both views.
    hflip_prob: float = float(os.getenv("HFLIP_PROB", "0.0"))

    # Layout
    channels_last: bool = os.getenv("CHANNELS_LAST", "1") == "1"

    # Compile
    compile: bool = os.getenv("COMPILE", "0") == "1"
    compile_backend: str = os.getenv("COMPILE_BACKEND", "inductor")
    compile_mode: str = os.getenv("COMPILE_MODE", "max-autotune")
    compile_dynamic: bool = os.getenv("COMPILE_DYNAMIC", "0") == "1"
    compile_fullgraph: bool = os.getenv("COMPILE_FULLGRAPH", "0") == "1"
    compile_cudagraphs: bool = os.getenv("COMPILE_CUDAGRAPHS", "0") == "1"
    compile_threads: int = int(os.getenv("COMPILE_THREADS", "1"))
    async_compile: bool = os.getenv("ASYNC_COMPILE", "0") == "1"

    # Checkpointing / validation
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    resume: bool = os.getenv("RESUME", "1") == "1"
    resume_load_optimizer: bool = os.getenv("RESUME_LOAD_OPTIMIZER", "1") == "1"
    save_every_epochs: int = int(os.getenv("SAVE_EVERY_EPOCHS", "1"))
    validate_every_epochs: int = int(os.getenv("VALIDATE_EVERY_EPOCHS", "5"))
    sanity_every_epochs: int = int(os.getenv("SANITY_EVERY_EPOCHS", "1"))  # set 0 to disable
    val_num_batches: int = int(os.getenv("VAL_NUM_BATCHES", "20"))
    dead_std_thr: float = float(os.getenv("DEAD_STD_THR", "0.01"))
    max_epoch_ckpts: int = int(os.getenv("MAX_EPOCH_CKPTS", "5"))

    # Repro
    seed: int = int(os.getenv("SEED", "1337"))

    # Device
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


CFG = CFG()
DEVICE = torch.device(CFG.device)


# -----------------------------
# Utils
# -----------------------------
def _dtype(name: str) -> torch.dtype:
    n = (name or "").lower().strip()
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    # prevent OpenCV from spawning its own threads in each worker
    cv2.setNumThreads(0)
    s = CFG.seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _list_npz(dir_path: str) -> List[str]:
    if not dir_path or not os.path.exists(dir_path):
        return []
    return sorted(glob.glob(os.path.join(dir_path, "*.npz")))


# -----------------------------
# Dataset: RAM-backed frame store
# -----------------------------
class FastRAMDataset(Dataset):
    """
    Loads all RGB frames into a shared-memory uint8 tensor: (N, 3, 64, 64).
    Assumes inputs are already 64x64.
    """
    def __init__(self):
        super().__init__()
        files = _list_npz(CFG.data_random) + _list_npz(CFG.data_expert) + _list_npz(CFG.data_recover)
        if not files:
            files = sorted(glob.glob("./data_*/*.npz"))
        if not files:
            raise RuntimeError("No .npz files found. Set DATA_RANDOM/DATA_EXPERT/DATA_RECOVER.")

        print(f"Scanning {len(files)} files to calculate size...")
        total_frames = 0
        valid_files: List[str] = []

        for f in tqdm(files, desc="Scanning"):
            try:
                with np.load(f, mmap_mode="r") as data:
                    key = "obs" if "obs" in data else ("states" if "states" in data else None)
                    if key is None:
                        keys = list(data.keys())
                        if not keys:
                            continue
                        key = keys[0]
                    arr = data[key]
                    if arr.ndim != 4:
                        continue
                    total_frames += int(arr.shape[0])
                    valid_files.append(f)
            except Exception:
                continue

        print(f"Total Frames to Load: {total_frames}")

        self.frames = torch.empty((total_frames, 3, 64, 64), dtype=torch.uint8)

        start = 0
        print("Allocating Shared Memory Tensor...")
        print("Loading and Permuting data into RAM...")
        for f in tqdm(valid_files, desc="Filling RAM"):
            try:
                with np.load(f) as data:
                    key = "obs" if "obs" in data else ("states" if "states" in data else list(data.keys())[0])
                    arr = data[key]  # (T,H,W,C)

                    if arr.shape[1] != 64 or arr.shape[2] != 64:
                        raise ValueError(f"{f}: expected 64x64, got {arr.shape[1]}x{arr.shape[2]}")

                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)

                    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (T,3,64,64)
                    end = start + t.shape[0]
                    self.frames[start:end].copy_(t)
                    start = end
            except Exception as e:
                print(f"Skipping file {f}: {e}")

        self.frames.share_memory_()
        self.n = int(self.frames.shape[0])
        print(f"RAM Load Complete. Usage: ~{self.frames.nbytes/1024**3:.2f} GB")

    def __len__(self) -> int:
        if CFG.frame_stack <= 1:
            return self.n
        return max(0, self.n - (CFG.frame_stack - 1))

    def __getitem__(self, idx: int) -> int:
        # returning python int reduces per-sample tensor overhead
        return int(idx)


# -----------------------------
# torch.compile / Inductor safety knobs
# -----------------------------
def _configure_inductor() -> None:
    if not CFG.compile:
        return
    try:
        import torch._inductor.config as ic
        if hasattr(ic, "compile_threads"):
            ic.compile_threads = int(max(1, CFG.compile_threads))
        if hasattr(ic, "async_compile"):
            ic.async_compile = bool(CFG.async_compile)

        if hasattr(ic, "triton") and hasattr(ic.triton, "cudagraphs"):
            ic.triton.cudagraphs = bool(CFG.compile_cudagraphs)
        if hasattr(ic, "triton") and hasattr(ic.triton, "cudagraph_trees") and not CFG.compile_cudagraphs:
            ic.triton.cudagraph_trees = False
    except Exception:
        pass


def _effective_compile_mode() -> str:
    if CFG.compile_mode == "max-autotune" and not CFG.compile_cudagraphs:
        return "max-autotune-no-cudagraphs"
    return CFG.compile_mode


def _try_compile(module: torch.nn.Module, name: str) -> torch.nn.Module:
    if not CFG.compile:
        return module
    if not hasattr(torch, "compile"):
        print(f"[COMPILE] torch.compile not available; leaving {name} eager.")
        return module
    mode = _effective_compile_mode()
    try:
        out = torch.compile(
            module,
            backend=CFG.compile_backend,
            mode=mode,
            dynamic=CFG.compile_dynamic,
            fullgraph=CFG.compile_fullgraph,
        )
        print(
            f"[COMPILE] compiled {name} (backend={CFG.compile_backend}, mode={mode}, "
            f"dynamic={CFG.compile_dynamic}, fullgraph={CFG.compile_fullgraph}, cudagraphs={CFG.compile_cudagraphs})"
        )
        return out
    except Exception as e:
        print(f"[COMPILE] failed to compile {name}; running eager. Error: {e}")
        return module


# -----------------------------
# GPU augmentations
# -----------------------------
def _random_crop_params_rrc(
    b: int, h: int, w: int, scale_min: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = min(h, w)
    scale = torch.empty((b,), device=device).uniform_(scale_min, 1.0)
    s = torch.clamp((scale * base).to(torch.int64), min=1, max=base)

    max_y = (h - s).clamp(min=0)
    max_x = (w - s).clamp(min=0)

    y0 = (torch.rand((b,), device=device) * (max_y.to(torch.float32) + 1.0)).floor().to(torch.int64)
    x0 = (torch.rand((b,), device=device) * (max_x.to(torch.float32) + 1.0)).floor().to(torch.int64)
    y1 = y0 + s
    x1 = x0 + s
    return y0, x0, y1, x1


def _apply_crop_resize(
    x: torch.Tensor,
    y0: torch.Tensor,
    x0: torch.Tensor,
    y1: torch.Tensor,
    x1: torch.Tensor,
    out_hw: int,
    antialias: bool,
) -> torch.Tensor:
    b = x.size(0)
    crops = [x[i : i + 1, :, y0[i] : y1[i], x0[i] : x1[i]] for i in range(b)]
    x_c = torch.cat(crops, dim=0)
    return F.interpolate(x_c, size=(out_hw, out_hw), mode="bilinear", align_corners=False, antialias=antialias)


def _color_jitter(x: torch.Tensor) -> torch.Tensor:
    # expects 0..1 float
    b = 0.2
    c = 0.2
    s = 0.2

    # brightness
    x = x + (torch.rand((x.size(0), 1, 1, 1), device=x.device) * 2 - 1) * b

    # contrast
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * (1 + (torch.rand((x.size(0), 1, 1, 1), device=x.device) * 2 - 1) * c) + mean

    # saturation
    gray = x.mean(dim=1, keepdim=True)
    sat = 1 + (torch.rand((x.size(0), 1, 1, 1), device=x.device) * 2 - 1) * s
    x = x * sat + gray * (1.0 - sat)

    return x.clamp(0.0, 1.0)


def _fixed_random_crop_64_from_padded72(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    x is (B,C,72,72) after reflect pad(4). Sample a true random 64x64 window per-sample.
    Returns (cropped, (y0, x0)).
    """
    b = x.size(0)
    max_off = 72 - 64  # 8
    y0 = torch.randint(0, max_off + 1, (b,), device=x.device)
    x0 = torch.randint(0, max_off + 1, (b,), device=x.device)
    crops = [x[i : i + 1, :, y0[i] : y0[i] + 64, x0[i] : x0[i] + 64] for i in range(b)]
    return torch.cat(crops, dim=0), (y0, x0)


def _apply_fixed_crop_64_from_padded72(x: torch.Tensor, y0: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    b = x.size(0)
    crops = [x[i : i + 1, :, y0[i] : y0[i] + 64, x0[i] : x0[i] + 64] for i in range(b)]
    return torch.cat(crops, dim=0)


def _apply_hflip_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: (B,) bool
    if mask is None or mask.numel() == 0 or not mask.any().item():
        return x
    x_flipped = x.clone()
    x_flipped[mask] = torch.flip(x_flipped[mask], dims=[3])
    return x_flipped


def gpu_augment(
    raw_u8: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    mode: str,
    antialias: bool,
    scale_min: float,
    hflip_prob: float,
    reuse_rrc: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    reuse_fixed: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    reuse_hflip_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    raw_u8: (B, C, 64, 64) uint8 or float in 0..1
    Returns: (augmented_float, aug_state)
      aug_state holds crop params + hflip_mask so view2 can reuse them.
    """
    if raw_u8.dtype == torch.uint8:
        x = raw_u8.to(dtype=out_dtype) / 255.0
    else:
        x = raw_u8.to(dtype=out_dtype)

    b, _, h, w = x.shape
    assert h == 64 and w == 64, f"Expected 64x64 input, got {h}x{w}"

    st: Dict[str, torch.Tensor] = {}

    if mode == "rrc":
        crop = _random_crop_params_rrc(b, h, w, scale_min, x.device) if reuse_rrc is None else reuse_rrc
        x = _apply_crop_resize(x, *crop, out_hw=64, antialias=antialias)
        st["rrc_y0"], st["rrc_x0"], st["rrc_y1"], st["rrc_x1"] = crop
    else:
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")  # -> (B,C,72,72)
        if reuse_fixed is None:
            x, (y0, x0) = _fixed_random_crop_64_from_padded72(x)
        else:
            y0, x0 = reuse_fixed
            x = _apply_fixed_crop_64_from_padded72(x, y0, x0)
        st["crop_y0"] = y0
        st["crop_x0"] = x0

    # HFLIP: per-sample mask, and COUPLED across both views by default.
    if hflip_prob > 0.0:
        if reuse_hflip_mask is None:
            hmask = (torch.rand((b,), device=x.device) < float(hflip_prob))
        else:
            hmask = reuse_hflip_mask.to(device=x.device, dtype=torch.bool)
        x = _apply_hflip_mask(x, hmask)
        st["hflip_mask"] = hmask.to(torch.uint8)
    else:
        st["hflip_mask"] = torch.zeros((b,), device=x.device, dtype=torch.uint8)

    # new colour jitter each call (even if crop reused)
    x = _color_jitter(x)
    return x, st


def build_frame_stack(frames_u8: torch.Tensor, idx: torch.Tensor, stack: int) -> torch.Tensor:
    """
    frames_u8: (N,3,64,64) uint8
    idx: (B,) indices in [0, N-stack]
    Returns (B, 3*stack, 64, 64) uint8
    """
    if stack <= 1:
        return frames_u8[idx]
    offsets = torch.arange(stack, device=idx.device).view(1, stack)
    gather_idx = idx.view(-1, 1) + offsets
    f = frames_u8[gather_idx]  # (B,stack,3,64,64)
    return f.permute(0, 2, 1, 3, 4).reshape(idx.size(0), 3 * stack, 64, 64)


# -----------------------------
# Optimizer: correct AdamW param groups
# -----------------------------
def build_adamw_param_groups(
    encoder: torch.nn.Module,
    projector: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    AdamW with proper decay/no-decay split:
      - decay: conv/linear weight tensors (ndim >= 2)
      - no_decay: biases + norm scale/bias (typically ndim == 1)
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for p in list(encoder.parameters()) + list(projector.parameters()):
        if not p.requires_grad:
            continue
        if p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)

    return optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


# -----------------------------
# Checkpoint helpers
# -----------------------------
def _epoch_from_name(path: str) -> int:
    m = re.search(r"ep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _latest_by_epoch(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=_epoch_from_name)


@torch.no_grad()
def validate(encoder: torch.nn.Module, frames_cpu: torch.Tensor, batch_size: int, autocast_ctx) -> None:
    """
    Health check:
      - pool encoder output (B,512,8,8) -> (B,512)
      - compute per-dim std
      - count dims with std < threshold
    """
    encoder.eval()
    zs: List[torch.Tensor] = []
    n = int(frames_cpu.shape[0])
    start = 0

    for _ in range(int(CFG.val_num_batches)):
        if start + batch_size + (CFG.frame_stack - 1) >= n:
            break

        idx = torch.arange(start, start + batch_size, dtype=torch.int64)  # CPU
        raw = build_frame_stack(frames_cpu, idx, CFG.frame_stack)         # CPU uint8
        raw = raw.to(DEVICE, non_blocking=True)                           # GPU

        with autocast_ctx:
            x1, _ = gpu_augment(
                raw,
                out_dtype=_dtype(CFG.aug_dtype),
                mode=CFG.aug_mode,
                antialias=CFG.aug_antialias,
                scale_min=CFG.aug_scale_min,
                hflip_prob=CFG.hflip_prob,
            )
            if CFG.channels_last:
                x1 = x1.contiguous(memory_format=torch.channels_last)

            z = encoder(x1)  # (B,512,8,8)
            z = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)  # (B,512)

        zs.append(z.detach().float().cpu())
        start += batch_size

    if zs:
        zcat = torch.cat(zs, dim=0)  # (N,512)
        std = zcat.std(dim=0)
        dead = int((std < float(CFG.dead_std_thr)).sum().item())
        print(f"  [ENCODER] Avg Std: {std.mean().item():.4f} | Dead Dims: {dead}/{zcat.shape[1]}")

    encoder.train()


# -----------------------------
# Train
# -----------------------------
def train() -> None:
    os.makedirs(CFG.model_dir, exist_ok=True)
    seed_everything(CFG.seed)

    if CFG.accum_steps < 1:
        raise ValueError("ACCUM_STEPS must be >= 1")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    dataset = FastRAMDataset()

    dl_kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=(CFG.num_workers > 0),
        drop_last=CFG.drop_last,
    )
    if CFG.num_workers > 0:
        dl_kwargs["prefetch_factor"] = CFG.prefetch_factor

    dataloader = DataLoader(**dl_kwargs)

    in_ch = 3 * max(1, CFG.frame_stack)
    encoder = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE)
    projector = Projector(in_dim=512, hid_dim=2048, out_dim=512).to(DEVICE)

    if CFG.channels_last:
        encoder = encoder.to(memory_format=torch.channels_last)

    amp_enabled = bool(CFG.amp and DEVICE.type == "cuda")
    amp_dtype = _dtype(CFG.amp_dtype) if amp_enabled else torch.float32
    aug_dtype = _dtype(CFG.aug_dtype)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)

    _configure_inductor()
    encoder = _try_compile(encoder, "encoder")
    projector = _try_compile(projector, "projector")

    optimizer = build_adamw_param_groups(encoder, projector, lr=CFG.lr, weight_decay=CFG.weight_decay)

    # Scheduler MUST be created before resume, and its state must be restored.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6, last_epoch=-1)

    use_scaler = amp_enabled and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 0  # this is "epochs already completed" (1-based stored in ckpt)
    if CFG.resume:
        ckpt_path = _latest_by_epoch(os.path.join(CFG.model_dir, "encoder_mixed_ckpt_ep*.pt"))
        if ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                encoder.load_state_dict(ckpt["encoder"])
                projector.load_state_dict(ckpt["projector"])
                start_epoch = int(ckpt.get("epoch", 0))

                if CFG.resume_load_optimizer and "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])

                # restore scaler (AMP) if present
                if "scaler" in ckpt:
                    try:
                        scaler.load_state_dict(ckpt["scaler"])
                    except Exception:
                        pass

                # restore scheduler if present; else reconstruct to the correct epoch position
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                else:
                    # old checkpoints: recreate scheduler so lr continues at the right point
                    # after completing `start_epoch` epochs, scheduler.last_epoch should be start_epoch-1
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=CFG.epochs, eta_min=1e-6, last_epoch=start_epoch - 1
                    )

                # Ensure env overrides win (esp if you changed LR/WD)
                # NOTE: if you want strict continuity, remove the lr override below.
                # for pg in optimizer.param_groups:
                #     pg["lr"] = CFG.lr
                if len(optimizer.param_groups) == 2:
                    optimizer.param_groups[0]["weight_decay"] = CFG.weight_decay
                    optimizer.param_groups[1]["weight_decay"] = 0.0

                print(f"Resumed from {ckpt_path} (epoch={start_epoch}, load_optimizer={CFG.resume_load_optimizer})")
            except Exception as e:
                print(f"Could not resume from {ckpt_path}: {e}")

    effective_batch = CFG.batch_size * CFG.accum_steps

    print("--- CONFIG ---")
    print(f"device              : {DEVICE}")
    print(f"batch_size          : {CFG.batch_size}")
    print(f"accum_steps         : {CFG.accum_steps}")
    print(f"effective_batch     : {effective_batch}")
    print(f"drop_last           : {CFG.drop_last}")
    print(f"frame_stack         : {CFG.frame_stack} (in_ch={in_ch})")
    print(f"amp_enabled         : {amp_enabled}")
    print(f"amp_dtype           : {str(amp_dtype).replace('torch.', '')}")
    print(f"loss_fp32           : {CFG.loss_fp32}")
    print(f"aug_mode            : {CFG.aug_mode}")
    print(f"aug_antialias       : {CFG.aug_antialias}")
    print(f"aug_scale_min       : {CFG.aug_scale_min}")
    print(f"aug_dtype           : {str(aug_dtype).replace('torch.', '')}")
    print(f"second_view_fast    : {CFG.second_view_fast}")
    print(f"hflip_prob          : {CFG.hflip_prob} (coupled across views)")
    print(f"channels_last       : {CFG.channels_last}")
    print(f"compile             : {CFG.compile}")
    if CFG.compile:
        print(f"compile_backend     : {CFG.compile_backend}")
        print(f"compile_mode        : {_effective_compile_mode()} (requested: {CFG.compile_mode})")
        print(f"compile_dynamic     : {CFG.compile_dynamic}")
        print(f"compile_fullgraph   : {CFG.compile_fullgraph}")
        print(f"compile_cudagraphs  : {CFG.compile_cudagraphs}")
        print(f"compile_threads     : {CFG.compile_threads}")
        print(f"async_compile       : {CFG.async_compile}")
    print(f"sanity_every_epochs : {CFG.sanity_every_epochs}")
    print("----------------")

    frames_cpu = dataset.frames
    print(f"--- STARTING TRAINING (MicroBatch: {CFG.batch_size}, Effective: {effective_batch}) ---")

    # Loop expects start_epoch to be "completed epochs" (1-based in ckpt).
    # If ckpt epoch=11, we start at epoch index 11 -> printing Ep 12/30.
    for epoch_idx in range(start_epoch, CFG.epochs):
        encoder.train()
        projector.train()

        optimizer.zero_grad(set_to_none=True)
        micro_since_step = 0

        pbar = tqdm(dataloader, desc=f"Ep {epoch_idx+1}/{CFG.epochs}")
        for micro_idx, idx in enumerate(pbar):
            idx = torch.as_tensor(idx, dtype=torch.int64)  # batch of indices

            raw_u8 = build_frame_stack(frames_cpu, idx, CFG.frame_stack)
            raw_u8 = raw_u8.to(DEVICE, non_blocking=True)

            # View 1
            x1, st1 = gpu_augment(
                raw_u8,
                out_dtype=aug_dtype,
                mode=CFG.aug_mode,
                antialias=CFG.aug_antialias,
                scale_min=CFG.aug_scale_min,
                hflip_prob=CFG.hflip_prob,
            )

            # View 2: reuse crop + reuse flip-mask if requested (to preserve left/right semantics)
            reuse_rrc = None
            reuse_fixed = None
            if CFG.second_view_fast:
                if CFG.aug_mode == "rrc":
                    reuse_rrc = (st1["rrc_y0"], st1["rrc_x0"], st1["rrc_y1"], st1["rrc_x1"])
                else:
                    reuse_fixed = (st1["crop_y0"], st1["crop_x0"])

            reuse_flip = st1["hflip_mask"].to(torch.bool) if CFG.hflip_prob > 0.0 else None

            x2, _ = gpu_augment(
                raw_u8,
                out_dtype=aug_dtype,
                mode=("crop" if CFG.second_view_fast else CFG.aug_mode),
                antialias=(False if CFG.second_view_fast else CFG.aug_antialias),
                scale_min=CFG.aug_scale_min,
                hflip_prob=CFG.hflip_prob,
                reuse_rrc=reuse_rrc,
                reuse_fixed=reuse_fixed,
                reuse_hflip_mask=reuse_flip,
            )

            if CFG.channels_last:
                x1 = x1.contiguous(memory_format=torch.channels_last)
                x2 = x2.contiguous(memory_format=torch.channels_last)

            with autocast_ctx:
                z1 = encoder(x1)
                z2 = encoder(x2)
                p1 = projector(z1)
                p2 = projector(z2)

                loss, metrics = vicreg_loss(p1, p2, stats_fp32=CFG.loss_fp32)

            loss_scaled = loss / CFG.accum_steps

            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            micro_since_step += 1

            if micro_since_step >= CFG.accum_steps:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_since_step = 0

            if (micro_idx % 25) == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    repr=f"{metrics['repr'].item():.4f}",
                    std=f"{metrics['std'].item():.4f}",
                    cov=f"{metrics['cov'].item():.4f}",
                    lr=f"{lr_now:.2e}",
                )

        # Flush any remainder
        if micro_since_step > 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Scheduler AFTER epoch optimization (correct order)
        scheduler.step()

        epoch_num = epoch_idx + 1  # 1-based human epoch number

        # Sanity stats (optional)
        if CFG.sanity_every_epochs > 0 and (epoch_num % CFG.sanity_every_epochs == 0):
            print(f"\n[Sanity Stats Epoch {epoch_num}]")
            validate(encoder, frames_cpu, batch_size=min(512, CFG.batch_size), autocast_ctx=autocast_ctx)

        # Save
        if epoch_num % CFG.save_every_epochs == 0:
            torch.save(encoder.state_dict(), os.path.join(CFG.model_dir, f"encoder_mixed_ep{epoch_num}.pth"))
            torch.save(
                {
                    "epoch": epoch_num,
                    "encoder": encoder.state_dict(),
                    "projector": projector.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else None,
                },
                os.path.join(CFG.model_dir, f"encoder_mixed_ckpt_ep{epoch_num}.pt"),
            )

            # Keep last K
            try:
                ckpts = sorted(glob.glob(os.path.join(CFG.model_dir, "encoder_mixed_ckpt_ep*.pt")), key=_epoch_from_name)
                for c in ckpts[:-CFG.max_epoch_ckpts]:
                    os.remove(c)
                wts = sorted(glob.glob(os.path.join(CFG.model_dir, "encoder_mixed_ep*.pth")), key=_epoch_from_name)
                for w in wts[:-CFG.max_epoch_ckpts]:
                    os.remove(w)
            except Exception:
                pass

        # Validation cadence
        if epoch_num % CFG.validate_every_epochs == 0:
            print(f"\n[Validation Epoch {epoch_num}]")
            validate(encoder, frames_cpu, batch_size=min(512, CFG.batch_size), autocast_ctx=autocast_ctx)

    torch.save(encoder.state_dict(), os.path.join(CFG.model_dir, "encoder_mixed_final.pth"))
    print("Training Complete.")


if __name__ == "__main__":
    train()
