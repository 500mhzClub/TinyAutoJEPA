#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from networks import TinyEncoder, Predictor, TinyDecoder

# -----------------------------
# Config
# -----------------------------
NUM_DREAMS = int(os.getenv("NUM_DREAMS", "5"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "15"))
GROUNDING_INTERVAL = int(os.getenv("GROUNDING_INTERVAL", "1000"))
FRAME_STACK = int(os.getenv("FRAME_STACK", "4")) 

PREDICTOR_SPACE = os.getenv("PREDICTOR_SPACE", "raw").lower()
assert PREDICTOR_SPACE in ("raw", "norm")

DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

ENCODER_PATH = os.getenv("ENCODER_PATH", "./models/encoder_mixed_final.pth")
PREDICTOR_PATH = os.getenv("PREDICTOR_PATH", "./models/predictor_final.pth")
DECODER_PATH = os.getenv("DECODER_PATH", "./models/decoder_final.pth")

DATA_GLOB_1 = os.getenv("DATA_GLOB_1", "./data_expert/*.npz")

OUT_DIR = os.getenv("OUT_DIR", ".")
FPS = int(os.getenv("FPS", "20"))

# -----------------------------
# Helpers
# -----------------------------
def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        out[k] = v
    return out

def _unwrap_state_dict(ckpt: Any, *, prefer_keys: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in prefer_keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint did not contain a state_dict dict.")
    return _strip_prefixes(ckpt)

def load_state(model: torch.nn.Module, path: str, *, prefer_keys: Tuple[str, ...], strict: bool) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=DEVICE)
    sd = _unwrap_state_dict(ckpt, prefer_keys=prefer_keys)
    try:
        model.load_state_dict(sd, strict=strict)
    except Exception as e:
        if strict: raise e
        print(f"[warn] {e}")
    print(f"Loaded {model.__class__.__name__} from {path}")

def _pick_obs(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "states" in data: return data["states"]
    if "obs" in data: return data["obs"]
    if "observations" in data: return data["observations"]
    return None

def _pick_actions(data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "actions" in data: return data["actions"]
    if "action" in data: return data["action"]
    if "act" in data: return data["act"]
    return None

def _pick_speed(data: np.lib.npyio.NpzFile, idx: int) -> float:
    for k in ("speed", "spd", "v", "velocity"):
        if k in data:
            arr = data[k]
            try:
                if idx < len(arr):
                    return float(arr[idx])
            except Exception:
                pass
    return 0.0

def _rgb64_u8(img: np.ndarray) -> np.ndarray:
    if img.shape[0] != 64 or img.shape[1] != 64:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def get_stacked_input_u8(obs: np.ndarray, idx: int) -> torch.Tensor:
    """
    Returns uint8 tensor (1, 3*FRAME_STACK, 64, 64) in RGBRGB order.
    """
    frames = []
    for i in range(FRAME_STACK - 1, -1, -1):
        safe = max(0, idx - i)
        frames.append(_rgb64_u8(obs[safe]))  # (64,64,3) u8

    stack = np.stack(frames, axis=0)  # (S,64,64,3)
    
    # --- FIX: Match train_predictor.py / train_decoder.py ordering ---
    # Convert to (S, 3, 64, 64) -> flatten to (S*3, 64, 64)
    # This creates interleaved: Frame1_RGB, Frame2_RGB...
    t = torch.from_numpy(stack).permute(0, 3, 1, 2).contiguous() 
    t = t.reshape(3 * FRAME_STACK, 64, 64)
    
    return t.unsqueeze(0).to(torch.uint8)

def norm_bchw(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, p=2, dim=1)

# -----------------------------
# Load models
# -----------------------------
def load_models():
    print(f"--- Loading Models on {DEVICE} ---")
    in_ch = 3 * FRAME_STACK

    encoder = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE).eval()
    load_state(encoder, ENCODER_PATH, prefer_keys=("encoder", "model", "state_dict"), strict=False)

    predictor = Predictor(action_dim=3, features=512).to(DEVICE).eval()
    load_state(predictor, PREDICTOR_PATH, prefer_keys=("predictor", "model", "state_dict"), strict=True)

    decoder = TinyDecoder(latent_channels=512).to(DEVICE).eval()
    load_state(decoder, DECODER_PATH, prefer_keys=("decoder", "model", "state_dict"), strict=False)

    return encoder, predictor, decoder

# -----------------------------
# Main dream loop
# -----------------------------
@torch.no_grad()
def generate_dream_batch():
    encoder, predictor, decoder = load_models()

    files = glob.glob(DATA_GLOB_1)
    if not files: files = glob.glob(DATA_GLOB_2)
    if not files: return

    print(f"Found {len(files)} sequence files. Generating {NUM_DREAMS} dreams...")
    os.makedirs(OUT_DIR, exist_ok=True)

    for i in range(1, NUM_DREAMS + 1):
        filename = random.choice(files)
        try:
            with np.load(filename) as data:
                obs = _pick_obs(data)
                actions = _pick_actions(data)
                if obs is None or actions is None: continue
                if len(obs) < SEQ_LEN + 20: continue

                start_idx = random.randint(FRAME_STACK + 5, len(obs) - SEQ_LEN - 10)
                print(f"Dream {i}: {os.path.basename(filename)} | start={start_idx}")

                x0_u8 = get_stacked_input_u8(obs, start_idx).to(DEVICE)
                x0 = x0_u8.to(dtype=torch.float32) / 255.0
                z = encoder(x0)
                if PREDICTOR_SPACE == "norm": z = norm_bchw(z)

                video_frames = []

                for t in range(SEQ_LEN):
                    if (t > 0) and (GROUNDING_INTERVAL > 0) and (t % GROUNDING_INTERVAL == 0):
                        xt_u8 = get_stacked_input_u8(obs, start_idx + t).to(DEVICE)
                        xt = xt_u8.to(dtype=torch.float32) / 255.0
                        z = encoder(xt)
                        if PREDICTOR_SPACE == "norm": z = norm_bchw(z)
                    elif t > 0:
                        prev_idx = start_idx + t - 1
                        act = actions[prev_idx].reshape(-1)
                        act_t = torch.from_numpy(act).to(DEVICE).float().unsqueeze(0)
                        
                        spd = _pick_speed(data, prev_idx)
                        spd_t = torch.tensor([spd], dtype=torch.float32, device=DEVICE)

                        z = predictor(z, act_t, spd_t)
                        if PREDICTOR_SPACE == "norm": z = norm_bchw(z)

                    recon = decoder(z).clamp(0, 1)
                    d_img = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    
                    r_img = _rgb64_u8(obs[start_idx + t])
                    r_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
                    d_bgr = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)

                    scale = 4
                    r_big = cv2.resize(r_bgr, (64*scale, 64*scale), interpolation=0)
                    d_big = cv2.resize(d_bgr, (64*scale, 64*scale), interpolation=0)

                    if t == 0:
                        cv2.putText(d_big, "CONTEXT", (10, 30), 0, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(d_big, f"PRED +{t}", (10, 30), 0, 0.8, (0, 255, 255), 2)

                    combined = np.vstack([r_big, d_big])
                    video_frames.append(combined)

            out_name = os.path.join(OUT_DIR, f"dream_pure_{i}.avi")
            h, w, _ = video_frames[0].shape
            out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"DIVX"), FPS, (w, h))
            for f in video_frames: out.write(f)
            out.release()
            print(f"--> Saved {out_name}")

        except Exception as e:
            print(f"Skipping file due to error: {e}")

if __name__ == "__main__":
    generate_dream_batch()