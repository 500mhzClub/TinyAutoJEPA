#!/usr/bin/env python3
from __future__ import annotations


"""
drive_mpc.py — Single Run & Video Recording
- Stops immediately upon crash or finish.
- Saves 'run_mpc.mp4'.
- Config: "Best Yet" (Low Tentacles, High Stability).
"""

import os
import cv2
import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

from networks import TinyEncoder, Predictor, TinyDecoder


# -----------------------------
# Env helpers
# -----------------------------
def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)).strip())

def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)).strip())

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


# -----------------------------
# Paths / Device
# -----------------------------
MODEL_PATH_ENC   = _env_str("MODEL_PATH_ENC",   "./models/encoder_mixed_final.pth")
MODEL_PATH_PRED  = _env_str("MODEL_PATH_PRED",  "./models/predictor_final.pth")
MODEL_PATH_DEC   = _env_str("MODEL_PATH_DEC",   "./models/decoder_final.pth")
MODEL_PATH_HEADS = _env_str("MODEL_PATH_HEADS", "./models/latent_heads_lookahead.pth")

DEVICE = torch.device(_env_str("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
FRAME_STACK = _env_int("FRAME_STACK", 4)
PREDICTOR_SPACE = _env_str("PREDICTOR_SPACE", "raw").lower()

SPD_SCALE_HEAD = _env_float("SPD_SCALE_HEAD", 80.0)


# -----------------------------
# MPC / Sampling parameters ("Best Yet")
# -----------------------------
HORIZON = _env_int("HORIZON", 12)
NUM_TENTACLES = _env_int("NUM_TENTACLES", 6)        # Low latency
TEMPLATE_FRACTION = _env_float("TEMPLATE_FRACTION", 1.0) # Smooth curves only

MPPI_TEMP = _env_float("MPPI_TEMP", 2.5)

STEER_STD    = _env_float("STEER_STD", 0.30)
STEER_DSTD   = _env_float("STEER_DSTD", 0.10)
STEER_SMOOTH = _env_float("STEER_SMOOTH", 0.55)

GAS_BASE = _env_float("GAS_BASE", 0.60)
GAS_DROP = _env_float("GAS_DROP", 0.55)
MIN_GAS  = _env_float("MIN_GAS", 0.18)

BRAKE_THRESH = _env_float("BRAKE_THRESH", 0.55)
BRAKE_GAIN   = _env_float("BRAKE_GAIN", 1.6)
BRAKE_MAX    = _env_float("BRAKE_MAX", 0.80)
GAS_WHEN_BRAKE = _env_float("GAS_WHEN_BRAKE", 0.12)

ACTION_BLEND  = _env_float("ACTION_BLEND", 0.20)
ACTION_REPEAT = _env_int("ACTION_REPEAT", 1)

STEER_MAG_PEN  = _env_float("STEER_MAG_PEN", 2.5)
STEER_JERK_PEN = _env_float("STEER_JERK_PEN", 0.25)
STEER_FLIP_PEN = _env_float("STEER_FLIP_PEN", 0.25)
GAS_JERK_PEN   = _env_float("GAS_JERK_PEN", 0.05)
BRAKE_JERK_PEN = _env_float("BRAKE_JERK_PEN", 0.05)

W_CENTER   = _env_float("W_CENTER", 6.0)
EARLY_CENTER = _env_float("EARLY_CENTER", 1.0)
W_WALL     = _env_float("W_WALL", 5000.0)
W_OFFROAD  = _env_float("W_OFFROAD", 100.0)
ROAD_FLOOR = _env_float("ROAD_FLOOR", 0.60)
WALL_DIST  = _env_float("WALL_DIST", 0.25)

W_SPEED        = _env_float("W_SPEED", 0.00)
W_SPEED_OFF    = _env_float("W_SPEED_OFF", 2.0)
SPEED_DIST     = _env_float("SPEED_DIST", 0.22)
SPEED_ROAD_EPS = _env_float("SPEED_ROAD_EPS", 0.05)

CENTER_BIAS = _env_float("CENTER_BIAS", 0.0)
INVERT_PREDICTOR = _env_bool("INVERT_PREDICTOR", "0")
K_PRIOR = _env_float("K_PRIOR", 0.8)

EMA_BETA  = _env_float("EMA_BETA", 0.85)

STRAIGHT_XOFF = _env_float("STRAIGHT_XOFF", 0.02)
STRAIGHT_ROAD = _env_float("STRAIGHT_ROAD", 0.70)
W_STRAIGHT_STEER = _env_float("W_STRAIGHT_STEER", 6.0)
STEER_GUARD_BLEND = _env_float("STEER_GUARD_BLEND", 0.20)

CENTER_EASE_XOFF = _env_float("CENTER_EASE_XOFF", 0.10)
CENTER_EASE_ROAD = _env_float("CENTER_EASE_ROAD", 0.55)
CENTER_EASE_T    = _env_int("CENTER_EASE_T", 3)
W_CENTER_EASE    = _env_float("W_CENTER_EASE", 6.0)
CENTER_EASE_BLEND = _env_float("CENTER_EASE_BLEND", 0.55)
CENTER_EASE_PRIOR_SCALE = _env_float("CENTER_EASE_PRIOR_SCALE", 0.80)
CENTER_EASE_REQUIRE_TOWARD = _env_bool("CENTER_EASE_REQUIRE_TOWARD", "0")

XOFF_DEADBAND = _env_float("XOFF_DEADBAND", 0.02)

STARTUP_NO_MPC_STEPS = _env_int("STARTUP_NO_MPC_STEPS", 60)
STARTUP_ACTION = _env_str("STARTUP_ACTION", "0.0,0.55,0.0")
RESET_STACK_AFTER_STARTUP = _env_bool("RESET_STACK_AFTER_STARTUP", "1")

W_ROAD_DROP = _env_float("W_ROAD_DROP", 30.0)
SPEED_TARGET = _env_float("SPEED_TARGET", 45.0)
W_SPEED_CAP  = _env_float("W_SPEED_CAP", 0.08)

# Probe
PROBE_XOFF_T = _env_int("PROBE_XOFF_T", 6)
PROBE_MIN_T = _env_int("PROBE_MIN_T", 6)
PROBE_XOFF_GAIN = _env_float("PROBE_XOFF_GAIN", 5.0)
PROBE_XOFF_CAP = _env_float("PROBE_XOFF_CAP", 0.35)
PROBE_FF_BLEND = _env_float("PROBE_FF_BLEND", 0.35)

PROBE_XOFF_MAX = _env_float("PROBE_XOFF_MAX", 0.03)
PROBE_ROAD_MIN = _env_float("PROBE_ROAD_MIN", 0.95)
PROBE_STEER_CAP = _env_float("PROBE_STEER_CAP", 0.25)
PROBE_CURV_CLAMP = _env_float("PROBE_CURV_CLAMP", 0.40)

PROBE_DECODE_T = _env_int("PROBE_DECODE_T", 8)
VIS_PRED_T = _env_int("VIS_PRED_T", 8)  # visualization: decode t+VIS_PRED_T (does not affect control)
PROBE_CURVE_THRESH = _env_float("PROBE_CURVE_THRESH", 0.10)
PROBE_STEER_GAIN = _env_float("PROBE_STEER_GAIN", 0.60)
PROBE_SHOW = _env_bool("PROBE_SHOW", "0")

PROBE_GAS = _env_float("PROBE_GAS", 0.60)
PROBE_ROAD_THRESH = _env_float("PROBE_ROAD_THRESH", 0.88)

PROBE_STEER_THRESH = _env_float("PROBE_STEER_THRESH", 0.90)
PROBE_BRAKE_THRESH = _env_float("PROBE_BRAKE_THRESH", 0.35)

CAUTION_GAS_MIN = _env_float("CAUTION_GAS_MIN", 0.18)
CAUTION_BRAKE_MAXADD = _env_float("CAUTION_BRAKE_MAXADD", 0.60)

BRAKE_GLOBAL_MAX = _env_float("BRAKE_GLOBAL_MAX", 0.75)

# Recovery
RECOVERY_ROAD = _env_float("RECOVERY_ROAD", 0.65)
RECOVERY_STICKY_STEPS = _env_int("RECOVERY_STICKY_STEPS", 10)
RECOVERY_K = _env_float("RECOVERY_K", 1.0)
RECOVERY_STEER_BLEND = _env_float("RECOVERY_STEER_BLEND", 0.25)
RECOVERY_GAS_MIN = _env_float("RECOVERY_GAS_MIN", 0.30)
RECOVERY_BRAKE_CAP = _env_float("RECOVERY_BRAKE_CAP", 0.15)

OFFROAD_GAS_MIN = _env_float("OFFROAD_GAS_MIN", 0.35)
OFFROAD_BRAKE_CAP = _env_float("OFFROAD_BRAKE_CAP", 0.10)

SPD_LIMIT = _env_float("SPD_LIMIT", 50.0)
GAS_MAX_HIGHSPD = _env_float("GAS_MAX_HIGHSPD", 0.45)
BRAKE_MIN_HIGHSPD = _env_float("BRAKE_MIN_HIGHSPD", 0.08)

DECODE_STEPS = _env_str("DECODE_STEPS", "2,5,8")
DECODE_STEPS = [int(x) for x in DECODE_STEPS.split(",") if x.strip()]
DECODE_STEPS = [s for s in DECODE_STEPS if 1 <= s <= HORIZON]

DEBUG_PRINT = _env_bool("DEBUG_PRINT", "1")
DEBUG_EVERY = _env_int("DEBUG_EVERY", 20)
USE_CV2_GUI = _env_bool("USE_CV2_GUI", "1")

# Offroad Confirmation
OFFROAD_CONFIRM_NOW   = _env_float("OFFROAD_CONFIRM_NOW",   0.55)
OFFROAD_CONFIRM_PRED  = _env_float("OFFROAD_CONFIRM_PRED",  0.30)
OFFROAD_CONFIRM_STEPS = _env_int("OFFROAD_CONFIRM_STEPS",   2)
OFFROAD_CONFIRM_K     = _env_int("OFFROAD_CONFIRM_K",       4)
OFFROAD_CONFIRM_XOFF  = _env_float("OFFROAD_CONFIRM_XOFF",  0.22)

W_OFFROAD_SOFT_SCALE    = _env_float("W_OFFROAD_SOFT_SCALE",    0.15)
W_ROAD_DROP_SOFT_SCALE  = _env_float("W_ROAD_DROP_SOFT_SCALE",  0.25)
W_SPEED_OFF_SOFT_SCALE  = _env_float("W_SPEED_OFF_SOFT_SCALE",  0.35)
SOFT_CENTER_BOOST       = _env_float("SOFT_CENTER_BOOST",       1.8)

BRAKE_UNCONF_MAX        = _env_float("BRAKE_UNCONF_MAX",        0.12)
CAUTION_BRAKE_UNCONF_MAXADD = _env_float("CAUTION_BRAKE_UNCONF_MAXADD", 0.10)


def _risk_confirmed_now(roadE: float, xoffE: float) -> bool:
    return (roadE < OFFROAD_CONFIRM_NOW) or (abs(xoffE) > OFFROAD_CONFIRM_XOFF)


# -----------------------------
# Model helpers
# -----------------------------
def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        out[k] = v
    return out

def _l2norm_bchw(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, p=2, dim=1)

def pool_feat(z_bchw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(z_bchw, 1).flatten(1)

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
    spd_norm = head_spd(feat512)
    spd_norm = torch.clamp(spd_norm, 0.0, 2.5)
    spd = spd_norm * float(SPD_SCALE_HEAD)
    xoff = torch.tanh(head_xoff(feat512)) * 0.5
    return road, spd, xoff

@torch.no_grad()
def encoder_latent(enc: nn.Module, x_u8: torch.Tensor) -> torch.Tensor:
    x = x_u8.to(device=DEVICE, dtype=torch.float32) / 255.0
    z = enc(x)
    if PREDICTOR_SPACE == "norm":
        z = _l2norm_bchw(z)
    return z

def _pred_step(pred: nn.Module, z_bchw: torch.Tensor, action: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    return pred(z_bchw, action, speed)


# -----------------------------
# Rendering / preprocessing
# -----------------------------
def capture_window(env) -> Optional[np.ndarray]:
    try:
        surf = env.unwrapped.screen
        if surf is not None:
            frame_t = pygame.surfarray.array3d(surf)
            frame = frame_t.transpose(1, 0, 2)
            return np.ascontiguousarray(frame, dtype=np.uint8)
    except Exception:
        pass
    try:
        fr = env.render()
        if fr is None:
            return None
        return np.ascontiguousarray(fr, dtype=np.uint8)
    except Exception:
        return None

def _rgb64_u8(frame_rgb: np.ndarray) -> np.ndarray:
    img64 = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    if img64.dtype != np.uint8:
        img64 = np.clip(img64, 0, 255).astype(np.uint8)
    return img64

def _stack_to_tensor_u8(stack_rgb_u8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(stack_rgb_u8).permute(0, 3, 1, 2).contiguous()
    fr = t.reshape(3 * stack_rgb_u8.shape[0], 64, 64)
    return fr.unsqueeze(0).to(torch.uint8)

def _tensor01_to_bgr_u8(t: torch.Tensor, w: int, h: int) -> np.ndarray:
    x = torch.clamp(t, 0, 1).detach().cpu()
    if x.dim() == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0).astype(np.uint8)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(x)


# -----------------------------
# Action sampling
# -----------------------------
def _make_actions(horizon: int, n: int, steer_center: float) -> torch.Tensor:
    n_templates = max(int(n * TEMPLATE_FRACTION), 1)
    n_templates = min(n_templates, n)
    n_rand = n - n_templates

    H = horizon
    t = torch.arange(H, device=DEVICE, dtype=torch.float32)

    def ramp(k: int) -> torch.Tensor:
        k = max(1, min(k, H))
        r = torch.clamp(t / float(k), 0.0, 1.0)
        return r * r

    ramps = [ramp(3), ramp(5), ramp(7)]
    ones = torch.ones(H, device=DEVICE)

    templates: List[torch.Tensor] = []
    # 1. Holds
    for a in (0.0, 0.15, 0.30):
        templates.append(torch.clamp(torch.tensor(steer_center, device=DEVICE) + a * ones, -1.0, 1.0))
        templates.append(torch.clamp(torch.tensor(steer_center, device=DEVICE) - a * ones, -1.0, 1.0))
    # 2. Ramps (Early)
    for r in ramps:
        for a in (0.35, 0.50, 0.65):
            templates.append(torch.clamp(torch.tensor(steer_center, device=DEVICE) + a * r, -1.0, 1.0))
            templates.append(torch.clamp(torch.tensor(steer_center, device=DEVICE) - a * r, -1.0, 1.0))
    # 3. Late Ramps
    late_r = torch.clamp((t - 4.0) / 5.0, 0.0, 1.0) ** 2
    for a in (0.50, 0.75):
        templates.append(torch.clamp(torch.tensor(steer_center, device=DEVICE) + a * late_r, -1.0, 1.0))
        templates.append(torch.clamp(torch.tensor(steer_center, device=DEVICE) - a * late_r, -1.0, 1.0))

    if len(templates) >= n_templates:
        idx = torch.randperm(len(templates), device=DEVICE)[:n_templates]
        steer_tmpl = torch.stack([templates[i] for i in idx], dim=0)
    else:
        reps = (n_templates + len(templates) - 1) // len(templates)
        steer_tmpl = torch.stack((templates * reps)[:n_templates], dim=0)

    if n_rand > 0:
        s0 = torch.randn(n_rand, device=DEVICE) * STEER_STD + float(steer_center)
        s0 = torch.clamp(s0, -1.0, 1.0)
        ds = torch.randn(n_rand, H, device=DEVICE) * STEER_DSTD
        steer_rw = torch.zeros(n_rand, H, device=DEVICE)
        steer_rw[:, 0] = s0
        for i in range(1, H):
            proposed = torch.clamp(steer_rw[:, i - 1] + ds[:, i], -1.0, 1.0)
            steer_rw[:, i] = STEER_SMOOTH * steer_rw[:, i - 1] + (1.0 - STEER_SMOOTH) * proposed
        steer_all = torch.cat([steer_tmpl, steer_rw], dim=0)
    else:
        steer_all = steer_tmpl

    abs_s = torch.abs(steer_all)
    gas = GAS_BASE - GAS_DROP * abs_s
    gas = torch.clamp(gas, MIN_GAS, 0.95)
    brake = torch.clamp((abs_s - BRAKE_THRESH) * BRAKE_GAIN, 0.0, BRAKE_MAX)
    gas = torch.minimum(gas, torch.clamp(1.0 - brake, GAS_WHEN_BRAKE, 1.0))

    return torch.stack([steer_all, gas, brake], dim=-1).to(torch.float32)


# -----------------------------
# Model loading
# -----------------------------
def load_models():
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    for p in (MODEL_PATH_ENC, MODEL_PATH_PRED, MODEL_PATH_HEADS):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    in_ch = 3 * FRAME_STACK
    enc = TinyEncoder(in_ch=in_ch, emb_dim=512).to(DEVICE).eval()
    enc_ckpt = torch.load(MODEL_PATH_ENC, map_location=DEVICE)
    if isinstance(enc_ckpt, dict):
        enc_ckpt = enc_ckpt.get("encoder", enc_ckpt.get("model", enc_ckpt))
    enc.load_state_dict(_strip_prefixes(enc_ckpt), strict=False)

    pred = Predictor(action_dim=3, features=512).to(DEVICE).eval()
    pred_ckpt = torch.load(MODEL_PATH_PRED, map_location=DEVICE)
    if isinstance(pred_ckpt, dict):
        pred_ckpt = pred_ckpt.get("predictor", pred_ckpt.get("model", pred_ckpt))
    pred.load_state_dict(_strip_prefixes(pred_ckpt), strict=False)

    dec = None
    if MODEL_PATH_DEC and os.path.exists(MODEL_PATH_DEC):
        dec = TinyDecoder(latent_channels=512).to(DEVICE).eval()
        dec_ckpt = torch.load(MODEL_PATH_DEC, map_location=DEVICE)
        if isinstance(dec_ckpt, dict):
            dec_ckpt = dec_ckpt.get("decoder", dec_ckpt.get("model", dec_ckpt))
        dec.load_state_dict(_strip_prefixes(dec_ckpt), strict=False)

    heads_ckpt = torch.load(MODEL_PATH_HEADS, map_location=DEVICE)
    head_road = MLPHead(512, 1).to(DEVICE).eval()
    head_spd  = MLPHead(512, 1).to(DEVICE).eval()
    head_xoff = MLPHead(512, 1).to(DEVICE).eval()
    head_road.load_state_dict(_strip_prefixes(heads_ckpt["head_road"]), strict=True)
    head_spd.load_state_dict(_strip_prefixes(heads_ckpt["head_spd"]), strict=True)
    head_xoff.load_state_dict(_strip_prefixes(heads_ckpt["head_xoff"]), strict=True)

    return enc, pred, dec, head_road, head_spd, head_xoff


# -----------------------------
# EMA state
# -----------------------------
@dataclass
class EmaState:
    road: float = 1.0
    spd:  float = 0.0
    xoff: float = 0.0
    prev_xoff: float = 0.0
    dxoff: float = 0.0
    init: bool = False

    def update(self, road: float, spd: float, xoff: float) -> None:
        if not self.init:
            self.road, self.spd, self.xoff = road, spd, xoff
            self.prev_xoff = xoff
            self.dxoff = 0.0
            self.init = True
            return
        old_xoff = self.xoff
        b = EMA_BETA
        self.road = b * self.road + (1.0 - b) * road
        self.spd  = b * self.spd  + (1.0 - b) * spd
        self.xoff = b * self.xoff + (1.0 - b) * xoff
        self.prev_xoff = old_xoff
        self.dxoff = self.xoff - old_xoff


# -----------------------------
# MPPI action
# -----------------------------
@torch.no_grad()
def mppi_action(actions_0: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    c = costs - torch.min(costs)
    w = torch.softmax(-c / max(MPPI_TEMP, 1e-6), dim=0)
    return (w[:, None] * actions_0).sum(dim=0)


def _deadband(x: float, db: float) -> float:
    return 0.0 if abs(x) < db else x


def _analyze_curve_from_rgb64(rgb_u8: np.ndarray) -> float:
    if rgb_u8 is None or rgb_u8.shape[0] != 64 or rgb_u8.shape[1] != 64:
        return 0.0
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)
    road = (S < 70) & (V > 35) & (V < 220)
    road = road & (S < 120)
    rows_bottom = [56, 58, 60]
    rows_mid = [40, 42, 44]
    def row_center(y_list):
        xs = []
        for y in y_list:
            x_idx = np.where(road[y])[0]
            if x_idx.size >= 6:
                xs.append(float(x_idx.mean()))
        if not xs: return None
        return float(np.mean(xs))
    xb = row_center(rows_bottom)
    xm = row_center(rows_mid)
    if xb is None or xm is None: return 0.0
    curvature = (xm - xb) / 32.0
    curvature = float(np.clip(curvature, -1.0, 1.0))
    if abs(curvature) < 0.03: curvature = 0.0
    return curvature


@torch.no_grad()
def probe_straight_future(enc, pred, dec, head_road, head_spd, head_xoff, z0: torch.Tensor) -> Tuple[float, float, float, Optional[torch.Tensor]]:
    z = z0.clone()
    min_road = 1.0
    feat = pool_feat(z)
    road_t, spd_t, _ = heads_eval(head_road, head_spd, head_xoff, feat)
    spd = spd_t.squeeze(1)
    a = torch.tensor([0.0, PROBE_GAS, 0.0], device=DEVICE, dtype=torch.float32).view(1, 3)

    probe_img = None
    curvature = 0.0
    probe_xoff = 0.0
    probe_road_at = 1.0

    for t_i in range(1, HORIZON + 1):
        feat = pool_feat(z)
        road_t, spd_t, xoff_t = heads_eval(head_road, head_spd, head_xoff, feat)
        if t_i == PROBE_XOFF_T:
            probe_xoff = float(xoff_t.item())
        r = float(road_t.item())
        min_road = min(min_road, r)
        spd = spd_t.squeeze(1)
        z = _pred_step(pred, z, a, spd)
        if PREDICTOR_SPACE == "norm":
            z = _l2norm_bchw(z)
        if dec is not None and (t_i == PROBE_DECODE_T):
            try:
                probe_img = torch.clamp(dec(z), 0, 1)
                rgb_u8 = (probe_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                curvature = _analyze_curve_from_rgb64(rgb_u8)
            except Exception:
                probe_img = None
                curvature = 0.0
    curvature = float(np.clip(curvature, -PROBE_CURV_CLAMP, PROBE_CURV_CLAMP))
    return float(min_road), float(curvature), float(probe_xoff), float(probe_road_at), probe_img




@torch.no_grad()
def rollout_for_viz(pred, head_road, head_spd, head_xoff, z0: torch.Tensor, actions_seq: torch.Tensor, steps: int) -> torch.Tensor:
    """Roll out a *single* latent trajectory for visualization only.

    - Uses per-step speed estimate from the heads as the predictor's speed input.
    - If steps > len(actions_seq), repeats the last action.
    """
    z = z0.clone()
    if steps <= 0:
        return z
    last_a = actions_seq[-1].clone()
    for t_i in range(1, steps + 1):
        feat = pool_feat(z)
        _, spd_t, _ = heads_eval(head_road, head_spd, head_xoff, feat)
        if t_i <= actions_seq.shape[0]:
            a_t = actions_seq[t_i - 1].clone()
        else:
            a_t = last_a.clone()
        if INVERT_PREDICTOR:
            a_t[0] = -a_t[0]
        z = _pred_step(pred, z, a_t.view(1, 3), spd_t.squeeze(1))
        if PREDICTOR_SPACE == "norm":
            z = _l2norm_bchw(z)
    return z

# -----------------------------
# MPC step
# -----------------------------
@torch.no_grad()
def mpc_step(enc, pred, dec, head_road, head_spd, head_xoff, stack_rgb_u8: np.ndarray, ema: EmaState):
    x_u8 = _stack_to_tensor_u8(stack_rgb_u8).to(DEVICE)
    z0 = encoder_latent(enc, x_u8)
    feat0 = pool_feat(z0)
    road0_t, spd0_t, xoff0_t = heads_eval(head_road, head_spd, head_xoff, feat0)

    road0 = float(road0_t.item())
    spd0  = float(spd0_t.item())
    xoff0 = float(xoff0_t.item())

    ema.update(road0, spd0, xoff0)

    probe_min_road, probe_curv, probe_xoff, probe_road_at, probe_img = probe_straight_future(enc, pred, dec, head_road, head_spd, head_xoff, z0)

    xoff_ema_db = _deadband(ema.xoff, XOFF_DEADBAND)
    steer_center = float(np.clip(-K_PRIOR * xoff_ema_db, -0.65, 0.65))

    ease_factor = 0.0
    if (ema.road >= CENTER_EASE_ROAD) and (abs(ema.xoff) <= CENTER_EASE_XOFF):
        ease_factor = 1.0 - (abs(ema.xoff) / max(CENTER_EASE_XOFF, 1e-6))
        if CENTER_EASE_REQUIRE_TOWARD and (ema.xoff * ema.dxoff > 0.0):
            ease_factor = 0.0
        ease_factor = float(np.clip(ease_factor, 0.0, 1.0))
        steer_center = float(steer_center * (1.0 - CENTER_EASE_PRIOR_SCALE * ease_factor))

    # Feedforward turn-in
    if (probe_min_road < PROBE_STEER_THRESH):
        ff = float(np.clip(-PROBE_XOFF_GAIN * probe_xoff, -PROBE_XOFF_CAP, PROBE_XOFF_CAP))
        steer_center = float(np.clip(steer_center + ff, -0.85, 0.85))

    actions = _make_actions(HORIZON, NUM_TENTACLES, steer_center)

    steer = actions[:, :, 0]
    gas   = actions[:, :, 1]
    brake = actions[:, :, 2]

    mag_pen  = STEER_MAG_PEN  * torch.mean(torch.abs(steer), dim=1)
    jerk_pen = STEER_JERK_PEN * torch.mean(torch.abs(steer[:, 1:] - steer[:, :-1]), dim=1)
    flip_pen = STEER_FLIP_PEN * torch.mean((steer[:, 1:] * steer[:, :-1] < 0).float(), dim=1)

    gas_jerk   = GAS_JERK_PEN   * torch.mean(torch.abs(gas[:, 1:] - gas[:, :-1]), dim=1)
    brake_jerk = BRAKE_JERK_PEN * torch.mean(torch.abs(brake[:, 1:] - brake[:, :-1]), dim=1)

    if (abs(ema.xoff) < STRAIGHT_XOFF) and (ema.road > STRAIGHT_ROAD):
        mag_pen = mag_pen + W_STRAIGHT_STEER * torch.abs(steer[:, 0])

    z = z0.repeat(NUM_TENTACLES, 1, 1, 1)
    cost = torch.zeros(NUM_TENTACLES, device=DEVICE)

    dreams: List[Tuple[int, torch.Tensor]] = []
    decode_set = set(DECODE_STEPS)

    confirmed = torch.full((NUM_TENTACLES,), _risk_confirmed_now(ema.road, ema.xoff), device=DEVICE, dtype=torch.bool)
    bad_run = torch.zeros(NUM_TENTACLES, device=DEVICE, dtype=torch.int32)

    prev_road = None
    for t in range(1, HORIZON + 1):
        feat_t = pool_feat(z)
        road_t, spd_t, xoff_t = heads_eval(head_road, head_spd, head_xoff, feat_t)

        road_v = road_t.squeeze(1)
        xoff_v = xoff_t.squeeze(1)

        if t <= max(OFFROAD_CONFIRM_K, 0):
            confirmed = confirmed | (torch.abs(xoff_v) > OFFROAD_CONFIRM_XOFF)
            bad = (road_v < OFFROAD_CONFIRM_PRED)
            bad_run = torch.where(bad, bad_run + 1, torch.zeros_like(bad_run))
            confirmed = confirmed | (bad_run >= int(OFFROAD_CONFIRM_STEPS))

        unconf = (~confirmed).float()
        w_center_t  = W_CENTER * (1.0 + (SOFT_CENTER_BOOST - 1.0) * unconf)
        w_offroad_t = W_OFFROAD * (1.0 - (1.0 - W_OFFROAD_SOFT_SCALE) * unconf)
        w_drop_t    = W_ROAD_DROP * (1.0 - (1.0 - W_ROAD_DROP_SOFT_SCALE) * unconf)
        w_spdoff_t  = W_SPEED_OFF * (1.0 - (1.0 - W_SPEED_OFF_SOFT_SCALE) * unconf)

        biased_xoff = xoff_v + CENTER_BIAS
        dist = torch.abs(biased_xoff)
        dist = torch.relu(dist - XOFF_DEADBAND)

        cost += w_center_t * dist * (1.0 + EARLY_CENTER * (float(HORIZON - t) / float(max(HORIZON, 1))))
        in_danger = torch.relu(dist - WALL_DIST)
        cost += W_WALL * (in_danger ** 2)

        off = torch.relu(ROAD_FLOOR - road_v)
        cost += w_offroad_t * off

        if prev_road is not None:
            road_drop = torch.relu(prev_road - road_v)
            cost += w_drop_t * road_drop
        prev_road = road_v

        spd_excess = torch.relu(spd_t.squeeze(1) - SPEED_TARGET)
        cost += W_SPEED_CAP * (spd_excess ** 2)

        road_gate = torch.clamp((road_v - (ROAD_FLOOR - SPEED_ROAD_EPS)) / (SPEED_ROAD_EPS + 1e-6), 0.0, 1.0)
        dist_gate = torch.clamp(1.0 - dist / (SPEED_DIST + 1e-6), 0.0, 1.0)
        good_gate = road_gate * dist_gate

        cost -= W_SPEED * spd_t.squeeze(1) * good_gate
        cost += w_spdoff_t * spd_t.squeeze(1) * off

        a_t = actions[:, t - 1, :].clone()
        if INVERT_PREDICTOR:
            a_t[:, 0] = -a_t[:, 0]
        z = _pred_step(pred, z, a_t, spd_t.squeeze(1))
        if PREDICTOR_SPACE == "norm":
            z = _l2norm_bchw(z)
        if dec is not None and t in decode_set:
            dreams.append((t, z))

    center_ease_pen = torch.zeros(NUM_TENTACLES, device=DEVICE)
    if (ease_factor > 0.0) and (CENTER_EASE_T > 0):
        tN = min(int(CENTER_EASE_T), HORIZON)
        center_ease_pen = (W_CENTER_EASE * float(ease_factor)) * torch.mean(torch.abs(steer[:, :tN]), dim=1)

    total_cost = cost + mag_pen + jerk_pen + flip_pen + gas_jerk + brake_jerk + center_ease_pen

    a0 = mppi_action(actions[:, 0, :], total_cost).detach().cpu().numpy().astype(np.float32)
    a0[0] = float(np.clip(a0[0], -1.0, 1.0))

    if ease_factor > 0.0:
        a0[0] = float((1.0 - CENTER_EASE_BLEND * ease_factor) * a0[0])
        cap = float(np.clip(1.0 - 0.55 * ease_factor, 0.35, 1.0))
        a0[0] = float(np.clip(a0[0], -cap, cap))

    if (probe_min_road < PROBE_STEER_THRESH):
        ff2 = float(np.clip(-PROBE_XOFF_GAIN * probe_xoff, -PROBE_XOFF_CAP, PROBE_XOFF_CAP))
        a0[0] = float(np.clip((1.0 - PROBE_FF_BLEND) * a0[0] + PROBE_FF_BLEND * ff2, -1.0, 1.0))

    a0[1] = float(np.clip(a0[1], 0.0, 1.0))
    a0[2] = float(np.clip(a0[2], 0.0, 1.0))

    if (abs(ema.xoff) < STRAIGHT_XOFF) and (ema.road > STRAIGHT_ROAD) and (probe_min_road >= PROBE_STEER_THRESH):
        a0[0] = (1.0 - STEER_GUARD_BLEND) * a0[0]

    if not _risk_confirmed_now(ema.road, ema.xoff):
        a0[2] = float(min(a0[2], BRAKE_UNCONF_MAX))

    recon = None
    if dec is not None:
        try: recon = torch.clamp(dec(z0), 0, 1)
        except Exception: recon = None

    dream_imgs = []
    if dec is not None and dreams:
        best = torch.argmin(total_cost).item()
        for (t, z_batch) in dreams:
            try:
                z_best = z_batch[best:best+1]
                dream = torch.clamp(dec(z_best), 0, 1)
                dream_imgs.append((t, dream))
            except Exception: pass

    viz_dream_img = None
    viz_dream_t = int(VIS_PRED_T)
    if dec is not None and viz_dream_t > 0:
        try:
            best = int(torch.argmin(total_cost).item())
            z_vis = rollout_for_viz(pred, head_road, head_spd, head_xoff, z0, actions[best], viz_dream_t)
            viz_dream_img = torch.clamp(dec(z_vis), 0, 1)
        except Exception:
            viz_dream_img = None


    dbg = {
        "steer_center": steer_center,
        "road0": road0,
        "spd0": spd0,
        "xoff0": xoff0,
        "roadE": ema.road,
        "spdE": ema.spd,
        "xoffE": ema.xoff,
        "dxoffE": ema.dxoff,
        "ease": float(ease_factor),
        "cmin": float(torch.min(total_cost).item()),
        "cmean": float(torch.mean(total_cost).item()),
        "probe_min_road": probe_min_road,
        "probe_curv": float(probe_curv),
        "probe_xoff": float(probe_xoff),
        "probe_road_at": float(probe_road_at),
        "confirmed_now": float(_risk_confirmed_now(ema.road, ema.xoff)),
    }
    return a0, dbg, recon, dream_imgs, probe_img, viz_dream_img, viz_dream_t


# -----------------------------
# Main loop
# -----------------------------
def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    enc, pred, dec, head_road, head_spd, head_xoff = load_models()
    obs, info = env.reset()
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    buf = deque(maxlen=FRAME_STACK)
    for _ in range(FRAME_STACK):
        fr = capture_window(env)
        if fr is not None:
            buf.append(_rgb64_u8(fr))
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    try:
        sa = [float(x) for x in STARTUP_ACTION.split(",")]
        if len(sa) != 3: raise ValueError
        startup_action = np.array(sa, dtype=np.float32)
    except Exception:
        startup_action = np.array([0.0, 0.55, 0.0], dtype=np.float32)

    use_amp = (DEVICE.type == "cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else None

    step_i = 0
    last_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ema = EmaState()
    recovery_left = 0
    startup_remaining = max(STARTUP_NO_MPC_STEPS, 0)
    probe_img = None

    # VIDEO WRITER SETUP
    video_writer = None
    video_path = "run_mpc.mp4"
    print(f"Recording run to {video_path} ...")

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

            if startup_remaining > 0:
                last_action = startup_action.copy()
                _, _, terminated, truncated, _ = env.step(last_action)
                step_i += 1
                startup_remaining -= 1
                if DEBUG_PRINT and (step_i % DEBUG_EVERY == 0):
                    print(f"step={step_i:05d} STARTUP hold action steer={last_action[0]:+.3f} gas={last_action[1]:.3f} brk={last_action[2]:.3f} remaining={startup_remaining}")
                
                # CRITICAL: If crash during startup, stop immediately
                if terminated or truncated:
                    print("Terminated during startup!")
                    break 
                continue

            stack = np.stack(list(buf), axis=0)
            if autocast_ctx is None:
                a0, dbg, recon, dreams, probe_img, viz_dream_img, viz_dream_t = mpc_step(enc, pred, dec, head_road, head_spd, head_xoff, stack, ema)
            else:
                with autocast_ctx:
                    a0, dbg, recon, dreams, probe_img, viz_dream_img, viz_dream_t = mpc_step(enc, pred, dec, head_road, head_spd, head_xoff, stack, ema)

            confirmed_now = bool(dbg.get("confirmed_now", 0.0) > 0.5)
            probe_min = float(dbg.get("probe_min_road", 1.0))
            if probe_min < PROBE_ROAD_THRESH:
                risk = float(np.clip((PROBE_ROAD_THRESH - probe_min) / max(PROBE_ROAD_THRESH, 1e-6), 0.0, 1.0))
                gas_target = (1.0 - risk) * a0[1] + risk * max(CAUTION_GAS_MIN, 0.0)
                a0[1] = float(np.clip(gas_target, 0.0, 1.0))
                max_add = CAUTION_BRAKE_MAXADD if confirmed_now else CAUTION_BRAKE_UNCONF_MAXADD
                a0[2] = float(np.clip(a0[2] + risk * max_add, 0.0, 1.0))

            if ema.init and (ema.spd > SPD_LIMIT) and (ema.road > 0.60) and (abs(ema.xoff) < STRAIGHT_XOFF):
                a0[1] = float(min(a0[1], GAS_MAX_HIGHSPD))
                a0[2] = float(max(a0[2], BRAKE_MIN_HIGHSPD))

            a0[2] = float(min(a0[2], BRAKE_GLOBAL_MAX))
            if not confirmed_now:
                a0[2] = float(min(a0[2], BRAKE_UNCONF_MAX))

            if ema.init and (ema.road < ROAD_FLOOR):
                a0[1] = float(max(a0[1], OFFROAD_GAS_MIN))
                a0[2] = float(min(a0[2], OFFROAD_BRAKE_CAP))

            if recovery_left > 0: recovery_left -= 1
            if ema.init and (ema.road < RECOVERY_ROAD):
                recovery_left = max(recovery_left, RECOVERY_STICKY_STEPS)

            if recovery_left > 0 and ema.init:
                rec_steer = float(np.clip(-RECOVERY_K * ema.xoff, -1.0, 1.0))
                a0[0] = float(np.clip((1.0 - RECOVERY_STEER_BLEND) * a0[0] + RECOVERY_STEER_BLEND * rec_steer, -1.0, 1.0))
                a0[1] = float(np.clip(max(a0[1], RECOVERY_GAS_MIN), 0.0, 1.0))
                a0[2] = float(np.clip(min(a0[2], RECOVERY_BRAKE_CAP), 0.0, 1.0))

            if ACTION_BLEND > 0.0:
                a0 = (ACTION_BLEND * last_action + (1.0 - ACTION_BLEND) * a0).astype(np.float32)
            last_action = a0.astype(np.float32)

            terminated = truncated = False
            for _ in range(max(ACTION_REPEAT, 1)):
                _, _, terminated, truncated, _ = env.step(last_action)
                step_i += 1
                if terminated or truncated: break

            if DEBUG_PRINT and (step_i % DEBUG_EVERY == 0):
                print(
                    f"step={step_i:05d} steer={last_action[0]:+.3f} gas={last_action[1]:.3f} brk={last_action[2]:.3f} | "
                    f"xoff={dbg['xoff0']:+.3f} road={dbg['road0']:.2f} spd={dbg['spd0']:.1f} || "
                    f"xoffE={dbg['xoffE']:+.3f} roadE={dbg['roadE']:.2f} spdE={dbg['spdE']:.1f} | "
                    f"probeMin={dbg.get('probe_min_road',1.0):.2f} pRoad@={dbg.get('probe_road_at',1.0):.2f} pxoff={dbg.get('probe_xoff',0.0):+.3f} curv={dbg.get('probe_curv',0.0):+.2f} rec={recovery_left:02d} conf={int(confirmed_now)} | "
                    f"sc={dbg['steer_center']:+.3f} cmin={dbg['cmin']:.2f} cmean={dbg['cmean']:.2f}"
                )

            # VIDEO & GUI
            if USE_CV2_GUI:
                vis_real = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                vis_real = cv2.resize(vis_real, (420, 315), interpolation=cv2.INTER_AREA)
                bias_px = int(CENTER_BIAS * 420)
                cv2.line(vis_real, (210 - bias_px, 290), (210 - bias_px, 315), (255, 255, 0), 2)
                x_curr = int(210 + (dbg["xoff0"] * 420))
                cv2.line(vis_real, (x_curr, 285), (x_curr, 315), (0, 255, 0), 4)
                w_left = int(210 - (WALL_DIST * 420) - bias_px)
                w_rght = int(210 + (WALL_DIST * 420) - bias_px)
                cv2.line(vis_real, (w_left, 290), (w_left, 315), (0, 0, 255), 2)
                cv2.line(vis_real, (w_rght, 290), (w_rght, 315), (0, 0, 255), 2)
                panels = [vis_real]
                if recon is not None: panels.append(_tensor01_to_bgr_u8(recon, 420, 315))
                if PROBE_SHOW and (probe_img is not None):
                    try:
                        p = _tensor01_to_bgr_u8(probe_img, 420, 315)
                        cv2.putText(p, f"PROBE t+{PROBE_DECODE_T} curv={dbg.get('probe_curv',0.0):+.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
                        panels.append(p)
                    except Exception: pass
                if viz_dream_img is not None:
                    d = _tensor01_to_bgr_u8(viz_dream_img, 420, 315)
                    cv2.putText(d, f"PRED t+{viz_dream_t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                    panels.append(d)
                elif dreams:
                    t, dream = dreams[0]
                    d = _tensor01_to_bgr_u8(dream, 420, 315)
                    cv2.putText(d, f"PRED t+{t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                    panels.append(d)
                vis = np.hstack(panels)
                
                # Visual Indicator for Recording
                cv2.circle(vis, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(vis, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.putText(vis, f"xoffE={dbg['xoffE']:+.2f} roadE={dbg['roadE']:.2f} spdE={dbg['spdE']:.1f} probeMin={dbg.get('probe_min_road',1.0):.2f} conf={int(confirmed_now)} | SPD_SCALE_HEAD={SPD_SCALE_HEAD}", (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(vis, f"STARTUP_NO_MPC_STEPS={STARTUP_NO_MPC_STEPS} N={NUM_TENTACLES} MPPI_TEMP={MPPI_TEMP}", (10, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                
                # Write to video
                if video_writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                if video_writer is not None:
                    video_writer.write(vis)

                cv2.imshow("Latent MPC (MPPI)", vis)
                if (cv2.waitKey(1) == 27) or terminated or truncated:
                    print("Run complete. Saving video and exiting...")
                    break
            else:
                # If no GUI, check termination
                if terminated or truncated:
                    print("Run complete (Headless). Exiting...")
                    break

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {video_path}")
        try: env.close()
        except Exception: pass
        if USE_CV2_GUI: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()