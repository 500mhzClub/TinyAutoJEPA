#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

cv2.setNumThreads(0)

from networks import TinyEncoder, TinyDecoder, Predictor


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    encoder_path: str = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")
    decoder_path: str = os.getenv("DECODER_PATH", "./models/decoder_final.pth")
    predictor_path: str = os.getenv("PRED_PATH", "./models/predictor_final.pth")

    data_glob: str = os.getenv("DATA_GLOB", "./data_expert/*.npz")
    out_dir: str = os.getenv("OUT_DIR", "./visuals")

    img_size: int = int(os.getenv("IMG_SIZE", "64"))
    frame_stack: int = int(os.getenv("FRAME_STACK", "4"))

    horizon: int = int(os.getenv("HORIZON", "10"))
    num_seqs: int = int(os.getenv("NUM_SEQS", "5"))
    warmup: int = int(os.getenv("WARMUP", "10"))  # avoid early zoom-in

    predictor_space: str = os.getenv("PREDICTOR_SPACE", "raw").lower()  # raw | norm

    seed: int = int(os.getenv("SEED", "123"))

CFG = CFG()
assert CFG.predictor_space in ("raw", "norm"), "PREDICTOR_SPACE must be raw|norm"
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


# -----------------------------
# Helpers
# -----------------------------
def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        out[k.replace("_orig_mod.", "").replace("module.", "")] = v
    return out


def _pick_obs(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "obs" in data:
        return data["obs"]
    if "states" in data:
        return data["states"]
    if "observations" in data:
        return data["observations"]
    return None


def _pick_act(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "action" in data:
        return data["action"]
    if "actions" in data:
        return data["actions"]
    if "act" in data:
        return data["act"]
    return None


def _pick_speed(data: np.lib.npyio.NpzFile, T: int) -> np.ndarray:
    if "speed" in data:
        spd = data["speed"]
        if spd.ndim == 1:
            spd = spd.astype(np.float32)
            if spd.shape[0] >= T:
                return spd[:T]
            out = np.zeros((T,), dtype=np.float32)
            out[: spd.shape[0]] = spd
            return out
    return np.zeros((T,), dtype=np.float32)


def _resize_obs(obs: np.ndarray, img_size: int) -> np.ndarray:
    if obs.shape[1] == img_size and obs.shape[2] == img_size:
        return obs
    out = np.empty((obs.shape[0], img_size, img_size, 3), dtype=obs.dtype)
    for i in range(obs.shape[0]):
        out[i] = cv2.resize(obs[i], (img_size, img_size), interpolation=cv2.INTER_AREA)
    return out


def _stack_u8(obs_u8: np.ndarray, t: int, stack: int) -> np.ndarray:
    # frames (t-stack+1 .. t)
    return obs_u8[t - (stack - 1) : t + 1]


def _stack_to_tensor_float(stack_u8: np.ndarray) -> torch.Tensor:
    # (S,H,W,3) -> (1, 3S, H, W) float 0..1
    t = torch.from_numpy(stack_u8).to(torch.float32).div_(255.0)  # (S,H,W,3)
    t = t.permute(0, 3, 1, 2).contiguous()                        # (S,3,H,W)
    t = t.permute(1, 0, 2, 3).reshape(3 * stack_u8.shape[0], stack_u8.shape[1], stack_u8.shape[2])
    return t.unsqueeze(0)


def _img_to_tensor_float(img_u8: np.ndarray) -> torch.Tensor:
    # (H,W,3) -> (1,3,H,W) float 0..1
    t = torch.from_numpy(img_u8).to(torch.float32).div_(255.0)
    t = t.permute(2, 0, 1).contiguous()
    return t.unsqueeze(0)


def _norm_bchw(z: torch.Tensor) -> torch.Tensor:
    # per-spatial unit norm along channel dim
    return F.normalize(z, p=2, dim=1)


def _pred_step(pred: torch.nn.Module, z: torch.Tensor, a: torch.Tensor, spd: torch.Tensor) -> torch.Tensor:
    # a: (B,3), spd: (B,) or (B,1)
    try:
        return pred(z, a, spd)
    except TypeError:
        return pred(z, a)


def load_models() -> Tuple[TinyEncoder, TinyDecoder, Predictor]:
    # Encoder
    in_ch = 3 * CFG.frame_stack
    enc = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE).eval()
    ckpt = torch.load(CFG.encoder_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        ckpt = ckpt["encoder"]
    if isinstance(ckpt, dict):
        ckpt = _strip_prefixes(ckpt)
    enc.load_state_dict(ckpt, strict=False)

    # Decoder
    dec = TinyDecoder(latent_channels=512).to(DEVICE).eval()
    dsd = torch.load(CFG.decoder_path, map_location=DEVICE)
    if isinstance(dsd, dict) and "decoder" in dsd:
        dsd = dsd["decoder"]
    if isinstance(dsd, dict):
        dsd = _strip_prefixes(dsd)
    dec.load_state_dict(dsd, strict=False)

    # Predictor (support raw state_dict OR dict checkpoint)
    pred = Predictor(action_dim=3, features=512).to(DEVICE).eval()
    psd = torch.load(CFG.predictor_path, map_location=DEVICE)
    if isinstance(psd, dict) and "predictor" in psd:
        psd = psd["predictor"]
    if isinstance(psd, dict):
        psd = _strip_prefixes(psd)
    pred.load_state_dict(psd, strict=False)

    print(f"[load] device={DEVICE} stack={CFG.frame_stack} horizon={CFG.horizon} PREDICTOR_SPACE={CFG.predictor_space}")
    return enc, dec, pred


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    os.makedirs(CFG.out_dir, exist_ok=True)

    files = sorted(glob.glob(CFG.data_glob))
    if not files:
        raise FileNotFoundError(f"No files matched DATA_GLOB={CFG.data_glob}")

    enc, dec, pred = load_models()

    for seq_idx in range(CFG.num_seqs):
        f = random.choice(files)

        with np.load(f, mmap_mode="r") as data:
            obs = _pick_obs(data)
            act = _pick_act(data)
            if obs is None or act is None:
                continue

            obs = _resize_obs(obs, CFG.img_size)
            if obs.dtype != np.uint8:
                obs = np.clip(obs, 0, 255).astype(np.uint8, copy=False)

            act = act.astype(np.float32, copy=False)

            T_obs = int(obs.shape[0])
            T_act = int(act.shape[0])
            if T_act > T_obs - 1:
                act = act[: max(0, T_obs - 1)]
                T_act = int(act.shape[0])

            spd = _pick_speed(data, T_obs)  # (T_obs,)

        # Need: initial stack ends at start_t, and actions start_t..start_t+H-1 must exist
        min_start = max(CFG.frame_stack - 1, CFG.warmup)
        max_start = min(T_obs - 1 - CFG.horizon, T_act - CFG.horizon)
        if max_start <= min_start:
            print(f"[skip] too short: {os.path.basename(f)} T_obs={T_obs} T_act={T_act}")
            continue

        start_t = random.randint(min_start, max_start)

        # Build initial stack (frames start_t-3 .. start_t)
        stack0 = _stack_u8(obs, start_t, CFG.frame_stack)
        x0 = _stack_to_tensor_float(stack0).to(DEVICE)  # (1,12,64,64)

        # Real sequence frames: t0..tH
        real_imgs: List[torch.Tensor] = []
        for k in range(CFG.horizon + 1):
            real_imgs.append(_img_to_tensor_float(obs[start_t + k]))

        with torch.no_grad():
            z = enc(x0)  # (1,512,8,8)
            if CFG.predictor_space == "norm":
                z = _norm_bchw(z)

            # Dream sequence (decoded from z rollout)
            dream_imgs: List[torch.Tensor] = []
            dream_imgs.append(torch.clamp(dec(z), 0, 1))  # recon at t0 from encoder latent

            for t in range(CFG.horizon):
                a = torch.from_numpy(act[start_t + t]).to(DEVICE).unsqueeze(0)  # (1,3)

                # speed aligned to obs[t]; use speed[start_t+t]
                s = torch.tensor([float(spd[start_t + t])], device=DEVICE, dtype=torch.float32)  # (1,)
                z = _pred_step(pred, z, a, s)
                if CFG.predictor_space == "norm":
                    z = _norm_bchw(z)

                dream_imgs.append(torch.clamp(dec(z), 0, 1))

        # Grid: top row real, bottom row dream
        # Concatenate width-wise across time
        row_real = torch.cat([im.to(DEVICE) for im in real_imgs], dim=3)   # (1,3,H,W*(H+1))
        row_dream = torch.cat([im.to(DEVICE) for im in dream_imgs], dim=3) # (1,3,H,W*(H+1))
        grid = torch.cat([row_real, row_dream], dim=2)                     # (1,3,2H,W*(H+1))

        out = os.path.join(CFG.out_dir, f"dream_verify_{seq_idx+1:02d}_t{start_t:05d}.png")
        save_image(grid.cpu(), out)
        print(f"[saved] {out} | file={os.path.basename(f)} start_t={start_t}")

    print("[done]")


if __name__ == "__main__":
    main()
