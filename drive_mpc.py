#!/usr/bin/env python3
from __future__ import annotations

import os
import cv2
import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, Tuple, Optional, List

"""
drive_mpc.py — Calibrated & Grounded Latent MPC

UPDATES:
- Added X_GAIN and X_BIAS to correct left-hugging tendency.
- Increased ROAD_FLOOR to 0.80 to stop "curb surfing".
- Visualization now draws the "Target Center" to help you debug alignment.

PREDICTOR_SPACE=raw \
MODEL_PATH_ENC=./models/encoder_mixed_final.pth \
MODEL_PATH_PRED=./models/predictor_final.pth \
MODEL_PATH_HEADS=./models/latent_heads.pth \
MODEL_PATH_DEC=./models/decoder_final.pth \
HORIZON=15 \
NUM_TENTACLES=512 \
W_WALL=500.0 \
W_OFFROAD=5.0 \
W_CENTER=10.0 \
W_SPEED=0.8 \
ROAD_FLOOR=0.80 \
K_PRIOR=0.5 \
X_GAIN=2.0 \
X_BIAS=-0.05 \
python drive_mpc.py
"""

# -----------------------------
# Paths / Device
# -----------------------------
MODEL_PATH_ENC   = os.getenv("MODEL_PATH_ENC",   "./models/encoder_mixed_final.pth")
MODEL_PATH_PRED  = os.getenv("MODEL_PATH_PRED",  "./models/predictor_final.pth")
MODEL_PATH_DEC   = os.getenv("MODEL_PATH_DEC",   "./models/decoder_final.pth")   
MODEL_PATH_HEADS = os.getenv("MODEL_PATH_HEADS", "./models/latent_heads.pth")    

DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
FRAME_STACK = int(os.getenv("FRAME_STACK", "4"))
PREDICTOR_SPACE = os.getenv("PREDICTOR_SPACE", "raw").lower()

# -----------------------------
# MPC parameters
# -----------------------------
HORIZON = int(os.getenv("HORIZON", "15"))
NUM_TENTACLES = int(os.getenv("NUM_TENTACLES", "512"))

STEER_STD    = float(os.getenv("STEER_STD", "0.5"))
STEER_DSTD   = float(os.getenv("STEER_DSTD", "0.20"))
STEER_SMOOTH = float(os.getenv("STEER_SMOOTH", "0.15"))

GAS_BASE = float(os.getenv("GAS_BASE", "0.78"))
GAS_DROP = float(os.getenv("GAS_DROP", "0.45"))
MIN_GAS  = float(os.getenv("MIN_GAS", "0.22"))

STEER_MAG_PEN  = float(os.getenv("STEER_MAG_PEN", "0.01"))
STEER_JERK_PEN = float(os.getenv("STEER_JERK_PEN", "0.01"))

W_CENTER  = float(os.getenv("W_CENTER", "10.0"))  # Strong centering
W_OFFROAD = float(os.getenv("W_OFFROAD", "5.0"))
ROAD_FLOOR = float(os.getenv("ROAD_FLOOR", "0.80")) # 80% confidence required
W_SPEED = float(os.getenv("W_SPEED", "0.8"))
W_WALL = float(os.getenv("W_WALL", "500.0")) 

K_PRIOR = float(os.getenv("K_PRIOR", "0.5"))

# --- CALIBRATION ---
# X_GAIN: Amplifies the offset. If model says -0.1 but reality is -0.3, Gain=3.0 fixes it.
# X_BIAS: Shifts the target. If car hugs left, add NEGATIVE bias to force it RIGHT.
#         (Based on observation: Left = Negative Xoff. To make 0 target appear "Right", we subtract).
X_GAIN = float(os.getenv("X_GAIN", "2.0")) 
X_BIAS = float(os.getenv("X_BIAS", "-0.05"))

DECODE_STEPS = os.getenv("DECODE_STEPS", "2,5,8")
DECODE_STEPS = [int(x) for x in DECODE_STEPS.split(",") if x.strip()]
DECODE_STEPS = [s for s in DECODE_STEPS if 1 <= s <= HORIZON]

DEBUG_PRINT = int(os.getenv("DEBUG_PRINT", "1")) == 1
RECORD = int(os.getenv("RECORD", "0")) == 1
VIDEO_DIR = os.getenv("VIDEO_DIR", "videos")
VIDEO_OUT = os.getenv("VIDEO_OUT", "mpc_run.mp4")
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
VIDEO_FOURCC = os.getenv("VIDEO_FOURCC", "mp4v")

# -----------------------------
# Imports
# -----------------------------
from networks import TinyEncoder, Predictor, TinyDecoder

def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        out[k] = v
    return out

def _l2norm_bchw(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, p=2, dim=1)

def pool_feat(z_bchw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(z_bchw, 1).flatten(1)

def capture_window(env) -> Optional[np.ndarray]:
    try:
        surf = env.unwrapped.screen
        if surf is not None:
            frame_t = pygame.surfarray.array3d(surf)
            frame = frame_t.transpose(1, 0, 2)
            return np.ascontiguousarray(frame, dtype=np.uint8)
    except: pass
    try:
        fr = env.render()
        if fr is None: return None
        return np.ascontiguousarray(fr, dtype=np.uint8)
    except: return None

def _rgb64_u8(frame_rgb: np.ndarray) -> np.ndarray:
    img64 = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    if img64.dtype != np.uint8: img64 = np.clip(img64, 0, 255).astype(np.uint8)
    return img64

def _stack_to_tensor_u8(stack: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(stack).permute(0, 3, 1, 2).contiguous() 
    fr = t.reshape(3 * stack.shape[0], 64, 64)
    return fr.unsqueeze(0).to(torch.uint8)

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.no_grad()
def heads_eval(head_road, head_spd, head_xoff, feat512):
    road = torch.sigmoid(head_road(feat512))
    spd  = head_spd(feat512)
    # Raw output + Tanh scaling from training
    xoff_raw = torch.tanh(head_xoff(feat512)) * 0.5
    
    # --- CALIBRATION INJECTION ---
    # 1. Gain: Amplify signal (If model underestimates offset)
    # 2. Bias: Shift equilibrium (If model consistently hugs Left)
    xoff_calib = (xoff_raw * X_GAIN) + X_BIAS
    
    return road, spd, xoff_calib

@torch.no_grad()
def encoder_latent(enc: nn.Module, x_u8: torch.Tensor) -> torch.Tensor:
    x = x_u8.to(device=DEVICE, dtype=torch.float32) / 255.0
    z = enc(x)
    if PREDICTOR_SPACE == "norm": z = _l2norm_bchw(z)
    return z

def _pred_step(pred: nn.Module, z_bchw: torch.Tensor, action: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    try: return pred(z_bchw, action, speed)
    except TypeError: return pred(z_bchw, action)

def _make_actions(horizon: int, n: int, steer_center: float) -> torch.Tensor:
    n_commit = max(int(n * 0.15), 1)
    n_rand = n - n_commit
    
    # Biased sampling around the 'steer_center' (heuristic from xoff)
    offsets = torch.tensor([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5], device=DEVICE)
    
    if n_commit < len(offsets):
        idx = torch.randperm(len(offsets), device=DEVICE)[:n_commit]
        s_vals = torch.clamp(steer_center + offsets[idx], -1.0, 1.0)
    else:
        s_vals = (torch.rand(n_commit, device=DEVICE) * 2 - 1)
    
    steer_commit = s_vals[:, None].repeat(1, horizon)
    
    s0 = torch.randn(n_rand, device=DEVICE) * STEER_STD + steer_center
    s0 = torch.clamp(s0, -1.0, 1.0)
    ds = torch.randn(n_rand, horizon, device=DEVICE) * STEER_DSTD

    steer = torch.zeros(n_rand, horizon, device=DEVICE)
    steer[:, 0] = s0
    for t in range(1, horizon):
        proposed = torch.clamp(steer[:, t - 1] + ds[:, t], -1.0, 1.0)
        steer[:, t] = STEER_SMOOTH * steer[:, t - 1] + (1.0 - STEER_SMOOTH) * proposed

    steer_all = torch.cat([steer_commit, steer], dim=0)
    
    abs_s = torch.abs(steer_all)
    gas = GAS_BASE - GAS_DROP * abs_s
    gas = torch.clamp(gas, MIN_GAS, 1.0)
    brake = torch.zeros_like(gas)
    
    return torch.stack([steer_all, gas, brake], dim=-1).to(torch.float32)

def load_models():
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    for p in (MODEL_PATH_ENC, MODEL_PATH_PRED, MODEL_PATH_HEADS):
        if not os.path.exists(p): raise FileNotFoundError(p)

    in_ch = 3 * FRAME_STACK
    enc = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE).eval()
    enc_ckpt = torch.load(MODEL_PATH_ENC, map_location=DEVICE)
    if isinstance(enc_ckpt, dict) and "encoder" in enc_ckpt: enc_ckpt = enc_ckpt["encoder"]
    enc.load_state_dict(_strip_prefixes(enc_ckpt), strict=False)

    pred = Predictor().to(DEVICE).eval()
    pred_ckpt = torch.load(MODEL_PATH_PRED, map_location=DEVICE)
    if isinstance(pred_ckpt, dict) and "predictor" in pred_ckpt: pred_ckpt = pred_ckpt["predictor"]
    pred.load_state_dict(_strip_prefixes(pred_ckpt), strict=False)

    dec = None
    if MODEL_PATH_DEC and os.path.exists(MODEL_PATH_DEC):
        dec = TinyDecoder().to(DEVICE).eval()
        dec.load_state_dict(torch.load(MODEL_PATH_DEC, map_location=DEVICE), strict=False)

    heads_ckpt = torch.load(MODEL_PATH_HEADS, map_location=DEVICE)
    head_road = MLPHead(512, 1).to(DEVICE).eval()
    head_spd  = MLPHead(512, 1).to(DEVICE).eval()
    head_xoff = MLPHead(512, 1).to(DEVICE).eval()
    head_road.load_state_dict(_strip_prefixes(heads_ckpt["head_road"]), strict=True)
    head_spd.load_state_dict(_strip_prefixes(heads_ckpt["head_spd"]), strict=True)
    head_xoff.load_state_dict(_strip_prefixes(heads_ckpt["head_xoff"]), strict=True)
    
    return enc, pred, dec, head_road, head_spd, head_xoff

def _ensure_video_writer(path: str, fps: int, fourcc: str, frame_wh: Tuple[int, int]) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cc = cv2.VideoWriter_fourcc(*fourcc)
    vw = cv2.VideoWriter(path, cc, float(fps), frame_wh)
    return vw

def _tensor01_to_bgr_u8(t: torch.Tensor, w: int, h: int) -> np.ndarray:
    x = torch.clamp(t, 0, 1).detach().cpu()
    if x.dim() == 4: x = x.squeeze(0)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0).astype(np.uint8)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(x)

@torch.no_grad()
def mpc_step(enc, pred, dec, head_road, head_spd, head_xoff, stack_u8: np.ndarray):
    x_u8 = _stack_to_tensor_u8(stack_u8).to(DEVICE)
    z0_bchw = encoder_latent(enc, x_u8)
    feat0 = pool_feat(z0_bchw)
    
    road0, spd0, xoff0 = heads_eval(head_road, head_spd, head_xoff, feat0)
    xoff_val = xoff0.item()

    # Prior: K_PRIOR steers us back to 0.
    steer_bias = -K_PRIOR * xoff_val 
    steer_bias = float(np.clip(steer_bias, -0.5, 0.5))

    actions = _make_actions(HORIZON, NUM_TENTACLES, steer_bias) 

    steer = actions[:, :, 0]
    mag_pen  = STEER_MAG_PEN  * torch.mean(torch.abs(steer), dim=1)
    jerk_pen = STEER_JERK_PEN * torch.mean(torch.abs(steer[:, 1:] - steer[:, :-1]), dim=1)
    
    z = z0_bchw.repeat(NUM_TENTACLES, 1, 1, 1)
    cost = torch.zeros(NUM_TENTACLES, device=DEVICE)
    dreams: List[Tuple[int, torch.Tensor]] = []
    decode_set = set(DECODE_STEPS)

    for t in range(1, HORIZON + 1):
        feat_t = pool_feat(z)
        road_t, spd_t, xoff_t = heads_eval(head_road, head_spd, head_xoff, feat_t)
        
        # Center Cost: Penalize deviation from 0.0
        # (Since we applied Bias in heads_eval, 0.0 is now the 'Corrected Center')
        dist = torch.abs(xoff_t.squeeze(1))
        cost += (W_CENTER * dist) 
        
        # Wall Cost
        in_danger = torch.relu(dist - 0.30) 
        cost += (W_WALL * (in_danger ** 2))

        # Offroad Cost
        is_grass = torch.relu(ROAD_FLOOR - road_t.squeeze(1))
        cost += (W_OFFROAD * is_grass)
        
        # Speed Reward
        cost -= (W_SPEED * spd_t.squeeze(1))

        a_t = actions[:, t - 1, :]
        z = _pred_step(pred, z, a_t, spd_t.squeeze(1))
        if PREDICTOR_SPACE == "norm": z = _l2norm_bchw(z)

        if dec is not None and t in decode_set:
            dreams.append((t, z))

    score = -(cost + mag_pen + jerk_pen)
    best = torch.argmax(score).item()
    a0 = actions[best, 0].detach().cpu().numpy()

    recon = None
    if dec is not None:
        try: recon = torch.clamp(dec(z0_bchw), 0, 1)
        except: pass

    dream_imgs = []
    if dec is not None and dreams:
        for (t, z_batch) in dreams:
            try:
                z_best = z_batch[best:best+1] 
                dream = torch.clamp(dec(z_best), 0, 1)
                dream_imgs.append((t, dream))
            except: pass

    dbg = {
        "xoff0": xoff_val,
        "road0": float(road0.item()),
        "spd0": float(spd0.item()),
        "best_cost": float(cost[best].item())
    }
    return a0, dbg, recon, dream_imgs

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    enc, pred, dec, head_road, head_spd, head_xoff = load_models()
    vw: Optional[cv2.VideoWriter] = None
    out_path = os.path.join(VIDEO_DIR, VIDEO_OUT)

    env.reset()
    env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    buf = deque(maxlen=FRAME_STACK)
    for _ in range(FRAME_STACK):
        fr = capture_window(env)
        if fr is not None: buf.append(_rgb64_u8(fr))
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    
    for _ in range(20): 
        env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        fr = capture_window(env)
        if fr is not None: buf.append(_rgb64_u8(fr))

    step_i = 0
    last_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE.type == "cuda" else torch.no_grad()

    try:
        while True:
            frame = capture_window(env)
            if frame is None:
                env.step(last_action)
                continue
            
            buf.append(_rgb64_u8(frame))
            if len(buf) < FRAME_STACK:
                env.step(last_action)
                continue
            
            stack = np.stack(list(buf), axis=0)
            with amp_ctx:
                a0, dbg, recon, dreams = mpc_step(enc, pred, dec, head_road, head_spd, head_xoff, stack)

            last_action = a0.astype(np.float32)
            _, _, terminated, truncated, _ = env.step(last_action)
            step_i += 1

            if DEBUG_PRINT and (step_i % 10 == 0):
                print(f"step={step_i:05d} | steer={last_action[0]:+.2f} xoff={dbg['xoff0']:+.2f} road={dbg['road0']:.2f}")

            # --- HUD ---
            vis_real = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vis_real = cv2.resize(vis_real, (400, 300), interpolation=cv2.INTER_AREA)
            
            # Gauge: X Offset with Calibration Markers
            cx = int(200 + (dbg['xoff0'] * 400))
            # Draw Center
            cv2.line(vis_real, (200, 260), (200, 290), (100, 100, 100), 1) 
            # Draw Target Equilibrium (which is 0.0 post-bias, so visually centered)
            cv2.circle(vis_real, (cx, 275), 6, (0, 0, 255) if abs(dbg['xoff0'])>0.3 else (0, 255, 0), -1)
            cv2.putText(vis_real, f"X:{dbg['xoff0']:.2f}", (cx+10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            # Road Confidence
            rw = int(dbg['road0'] * 100)
            col = (0, 255, 0) if dbg['road0'] > ROAD_FLOOR else (0, 0, 255)
            cv2.rectangle(vis_real, (10, 10), (10+rw, 25), col, -1)
            cv2.putText(vis_real, f"ROAD", (120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            panels = [vis_real]
            if recon is not None:
                r = _tensor01_to_bgr_u8(recon, 400, 300)
                panels.append(r)
            if dreams:
                t, dream = dreams[-1] 
                d = _tensor01_to_bgr_u8(dream, 400, 300)
                cv2.putText(d, f"T+{t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                panels.append(d)
            
            vis = np.hstack(panels)
            cv2.imshow("MPC Grounded", vis)
            
            if RECORD:
                if vw is None: vw = _ensure_video_writer(out_path, VIDEO_FPS, VIDEO_FOURCC, (vis.shape[1], vis.shape[0]))
                vw.write(vis)
                
            if (cv2.waitKey(1) == 27) or terminated or truncated:
                if terminated or truncated: 
                    print("Resetting...")
                    env.reset()
                    buf.clear()
                    for _ in range(FRAME_STACK):
                        fr = capture_window(env)
                        if fr is not None: buf.append(_rgb64_u8(fr))
                        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    finally:
        try: env.close()
        except: pass
        cv2.destroyAllWindows()
        if vw is not None: vw.release()

if __name__ == "__main__":
    main()