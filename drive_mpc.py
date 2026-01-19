#!/usr/bin/env python3
"""
drive_mpc.py — Prior-guided MPC with:
- decoder/predictor space handling (norm/raw)
- image-based road costs
- center-band offroad classifier + recovery
- NEW: conservative MPC penalties for "too fast into chicanes"
- NEW: real-speed target governor (reads env.unwrapped.car velocity)

This addresses the failure mode where the model predicts plausible frames but the real sim spins at high slip.
"""

import os
import cv2
import gymnasium as gym
import numpy as np
import pygame
import torch

from networks import TinyEncoder, Predictor, TinyDecoder


# -----------------------------
# Paths / Device
# -----------------------------
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED = "./models/predictor_final.pth"
MODEL_PATH_DEC = "./models/decoder_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# MPC / rollout
# -----------------------------
HORIZON = int(os.getenv("HORIZON", "14"))
NUM_TENTACLES = int(os.getenv("NUM_TENTACLES", "256"))

DECODE_STEPS = os.getenv("DECODE_STEPS", "2,4,7,10,14")
DECODE_STEPS = [int(x) for x in DECODE_STEPS.split(",") if x.strip()]
DECODE_STEPS = [s for s in DECODE_STEPS if 1 <= s <= HORIZON]
if not DECODE_STEPS:
    DECODE_STEPS = [HORIZON]

PREDICTOR_SPACE = os.getenv("PREDICTOR_SPACE", "norm").lower()
assert PREDICTOR_SPACE in ("raw", "norm")


# -----------------------------
# Control shaping / sampling
# -----------------------------
MOMENTUM = float(os.getenv("MOMENTUM", "0.0"))

STEER_STD = float(os.getenv("STEER_STD", "0.25"))
STEER_DSTD = float(os.getenv("STEER_DSTD", "0.20"))
STEER_SMOOTH = float(os.getenv("STEER_SMOOTH", "0.65"))

GAS_BASE = float(os.getenv("GAS_BASE", "0.70"))
GAS_DROP = float(os.getenv("GAS_DROP", "0.55"))
MIN_GAS = float(os.getenv("MIN_GAS", "0.18"))

STEER_MAG_PEN = float(os.getenv("STEER_MAG_PEN", "0.06"))
STEER_JERK_PEN = float(os.getenv("STEER_JERK_PEN", "0.05"))
STEER_FLIP_PEN = float(os.getenv("STEER_FLIP_PEN", "0.03"))
STEER_SAT_PEN = float(os.getenv("STEER_SAT_PEN", "0.25"))
STEER_SAT_THRESH = float(os.getenv("STEER_SAT_THRESH", "0.85"))

# NEW: conservative gas penalties (key for chicanes)
GAS_MEAN_PEN = float(os.getenv("GAS_MEAN_PEN", "0.28"))        # penalize high mean gas
GAS_TURN_PEN = float(os.getenv("GAS_TURN_PEN", "0.55"))        # penalize gas*|steer|
GAS_FLIP_PEN = float(os.getenv("GAS_FLIP_PEN", "0.80"))        # penalize gas during steer sign-flips

OVERSPEED_GAS_CAP = float(os.getenv("OVERSPEED_GAS_CAP", "0.45"))
FLIP_GAS_CAP = float(os.getenv("FLIP_GAS_CAP", "0.35"))


# -----------------------------
# Prior guidance
# -----------------------------
K_PRIOR = float(os.getenv("K_PRIOR", "3.2"))
PRIOR_W = float(os.getenv("PRIOR_W", "1.4"))
PRIOR_CLAMP = float(os.getenv("PRIOR_CLAMP", "0.75"))

CONF_MIN = float(os.getenv("CONF_MIN", "0.10"))
CONF_MAX = float(os.getenv("CONF_MAX", "0.28"))


# -----------------------------
# Image metrics
# -----------------------------
IMG_W_CENTER = float(os.getenv("IMG_W_CENTER", "5.0"))
IMG_W_GRASS = float(os.getenv("IMG_W_GRASS", "5.0"))
IMG_W_NOROAD = float(os.getenv("IMG_W_NOROAD", "3.0"))

ROI_PRIOR = (20, 58, 6, 58)
ROI_COST = (28, 62, 8, 56)

# Center-band classifier ROI
ROI_CLASS = (32, 62, 24, 40)

GRASS_BIAS = float(os.getenv("GRASS_BIAS", "0.12"))
GRASS_SCALE = float(os.getenv("GRASS_SCALE", "0.06"))

MIN_M = float(os.getenv("MIN_M", "0.10"))
MAX_M = float(os.getenv("MAX_M", "0.90"))
M_SOFT = float(os.getenv("M_SOFT", "0.05"))


# -----------------------------
# Recovery gating
# -----------------------------
OFF_ROAD_CLASS = float(os.getenv("OFF_ROAD_CLASS", "0.085"))
REC_EXIT_CLASS = float(os.getenv("REC_EXIT_CLASS", "0.200"))
REC_EXIT_COUNT = int(os.getenv("REC_EXIT_COUNT", "10"))


# -----------------------------
# Speed governor (actual sim speed)
# -----------------------------
# CarRacing uses Box2D units; speed magnitudes ~ 0..30-ish depending.
# Tune by printing v occasionally if needed.
V_BASE = float(os.getenv("V_BASE", "18.0"))            # target speed straight
V_DROP = float(os.getenv("V_DROP", "10.0"))            # reduce target by |steer|
V_FLIP_DROP = float(os.getenv("V_FLIP_DROP", "6.0"))   # extra reduction on chicane snap
V_MIN = float(os.getenv("V_MIN", "7.0"))               # minimum target speed
SPEED_BRAKE_K = float(os.getenv("SPEED_BRAKE_K", "0.09"))  # braking gain when v > v_target
SPEED_GAS_CAP = float(os.getenv("SPEED_GAS_CAP", "0.72"))  # cap gas even if MPC wants more


# -----------------------------
# Flip-brake / snap-oversteer control
# -----------------------------
FLIP_STEER_DELTA = float(os.getenv("FLIP_STEER_DELTA", "0.55"))
FLIP_BRAKE = float(os.getenv("FLIP_BRAKE", "0.45"))
FLIP_BRAKE_STEPS = int(os.getenv("FLIP_BRAKE_STEPS", "10"))


# -----------------------------
# Recovery state machine
# -----------------------------
REC_STABILIZE_STEPS = int(os.getenv("REC_STABILIZE_STEPS", "10"))
REC_SEARCH_STEPS = int(os.getenv("REC_SEARCH_STEPS", "160"))
REC_REJOIN_STEPS = int(os.getenv("REC_REJOIN_STEPS", "55"))

REC_SEARCH_STEER = float(os.getenv("REC_SEARCH_STEER", "0.55"))
REC_SEARCH_GAS = float(os.getenv("REC_SEARCH_GAS", "0.66"))
REC_SEARCH_BRAKE = float(os.getenv("REC_SEARCH_BRAKE", "0.00"))
REC_SEARCH_BOOST = float(os.getenv("REC_SEARCH_BOOST", "0.10"))

REC_REJOIN_K = float(os.getenv("REC_REJOIN_K", "2.0"))
REC_REJOIN_STEER_CLAMP = float(os.getenv("REC_REJOIN_STEER_CLAMP", "0.70"))
REC_REJOIN_GAS = float(os.getenv("REC_REJOIN_GAS", "0.72"))
REC_REJOIN_BRAKE = float(os.getenv("REC_REJOIN_BRAKE", "0.00"))

POST_REC_STEPS = int(os.getenv("POST_REC_STEPS", "40"))
POST_REC_GAS = float(os.getenv("POST_REC_GAS", "0.50"))
POST_REC_STEER_CLAMP = float(os.getenv("POST_REC_STEER_CLAMP", "0.55"))


# -----------------------------
# Visualization / debug
# -----------------------------
VIS_HORIZON = int(os.getenv("VIS_HORIZON", "1"))
DEBUG_PRINT = int(os.getenv("DEBUG_PRINT", "1")) == 1


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def load_models():
    for p in (MODEL_PATH_ENC, MODEL_PATH_PRED, MODEL_PATH_DEC):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))

    dec = TinyDecoder().to(DEVICE).eval()
    dec.load_state_dict(torch.load(MODEL_PATH_DEC, map_location=DEVICE))

    print(f"Loaded models on {DEVICE}")
    print(f"PREDICTOR_SPACE={PREDICTOR_SPACE} H={HORIZON} N={NUM_TENTACLES} DECODE_STEPS={DECODE_STEPS}")
    return enc, pred, dec


def capture_window(env):
    surf = env.unwrapped.screen
    if surf is None:
        return None
    frame_t = pygame.surfarray.array3d(surf)
    frame = frame_t.transpose(1, 0, 2)
    return np.ascontiguousarray(frame, dtype=np.uint8)


def get_speed(env) -> float:
    """Return speed magnitude if CarRacing internals are present; else 0."""
    try:
        car = env.unwrapped.car
        v = car.hull.linearVelocity
        return float(np.sqrt(v[0] * v[0] + v[1] * v[1]))
    except Exception:
        return 0.0


def _soft_road_stats(img64_rgb01: torch.Tensor, roi):
    y1, y2, x1, x2 = roi
    roi_t = img64_rgb01[:, :, y1:y2, x1:x2]
    r = roi_t[:, 0]
    g = roi_t[:, 1]
    b = roi_t[:, 2]
    m = (r + g + b) / 3.0

    green_index = g - 0.5 * (r + b)
    grassness = _sigmoid((green_index - GRASS_BIAS) / GRASS_SCALE)

    gate_lo = _sigmoid((m - MIN_M) / M_SOFT)
    gate_hi = _sigmoid((MAX_M - m) / M_SOFT)
    bright_gate = gate_lo * gate_hi

    road_w = (1.0 - grassness) * bright_gate
    grass_w = grassness * bright_gate

    road_mean = road_w.mean(dim=(1, 2))
    grass_mean = grass_w.mean(dim=(1, 2))

    _, h, w = road_w.shape
    xs = torch.linspace(0, 1, w, device=img64_rgb01.device)[None, None, :].repeat(1, h, 1)
    denom = road_w.sum(dim=(1, 2)).clamp(min=1e-6)
    x_mean = (road_w * xs).sum(dim=(1, 2)) / denom

    return float(x_mean.item()), float(road_mean.item()), float(grass_mean.item())


def _conf_from_class(class_road: float) -> float:
    denom = max(1e-6, (CONF_MAX - CONF_MIN))
    return float(np.clip((class_road - CONF_MIN) / denom, 0.0, 1.0))


def _compute_signals(img64_rgb01: torch.Tensor):
    x_mean, road_m, grass_m = _soft_road_stats(img64_rgb01, ROI_PRIOR)
    err = x_mean - 0.5

    _, class_road, class_grass = _soft_road_stats(img64_rgb01, ROI_CLASS)
    conf = _conf_from_class(class_road)

    steer_prior = float(np.clip(K_PRIOR * err, -PRIOR_CLAMP, PRIOR_CLAMP))
    steer_prior = float(np.clip(steer_prior * conf, -PRIOR_CLAMP, PRIOR_CLAMP))

    return steer_prior, err, x_mean, road_m, grass_m, class_road, class_grass, conf


def _make_actions(horizon, n, steer_center: float) -> torch.Tensor:
    n_commit = max(32, n // 6)
    n_rand = n - n_commit

    offsets = torch.tensor([-0.55, -0.35, -0.20, -0.10, 0.0, 0.10, 0.20, 0.35, 0.55], device=DEVICE)
    idx = torch.randint(0, offsets.numel(), (n_commit,), device=DEVICE)
    s_commit = torch.clamp(torch.tensor(steer_center, device=DEVICE) + offsets[idx], -1.0, 1.0)
    steer_commit = s_commit[:, None].repeat(1, horizon)

    s0 = torch.randn(n_rand, device=DEVICE) * STEER_STD + float(steer_center)
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
    gas = torch.clamp(gas, MIN_GAS, 0.90)

    brake = torch.zeros_like(gas)
    ramp = torch.clamp((abs_s - 0.55) / (1.0 - 0.55), 0.0, 1.0)
    brake = torch.clamp(0.20 * ramp, 0.0, 0.20)

    return torch.stack([steer_all, gas, brake], dim=-1)


def _image_cost_batch(decoded_bchw_01: torch.Tensor) -> torch.Tensor:
    y1, y2, x1, x2 = ROI_COST
    roi_t = decoded_bchw_01[:, :, y1:y2, x1:x2]
    r = roi_t[:, 0]
    g = roi_t[:, 1]
    b = roi_t[:, 2]
    m = (r + g + b) / 3.0

    green_index = g - 0.5 * (r + b)
    grassness = _sigmoid((green_index - GRASS_BIAS) / GRASS_SCALE)

    gate_lo = _sigmoid((m - MIN_M) / M_SOFT)
    gate_hi = _sigmoid((MAX_M - m) / M_SOFT)
    bright_gate = gate_lo * gate_hi

    road_w = (1.0 - grassness) * bright_gate
    grass_w = grassness * bright_gate

    road_mean = road_w.mean(dim=(1, 2))
    grass_mean = grass_w.mean(dim=(1, 2))

    B, _, h, w = roi_t.shape
    xs = torch.linspace(0, 1, w, device=decoded_bchw_01.device)[None, None, :].repeat(B, h, 1)
    denom = road_w.sum(dim=(1, 2)).clamp(min=1e-6)
    x_mean = (road_w * xs).sum(dim=(1, 2)) / denom
    center_offset = torch.abs(x_mean - 0.5)

    noroad = torch.relu(0.18 - road_mean)

    return (IMG_W_CENTER * center_offset) + (IMG_W_GRASS * grass_mean) + (IMG_W_NOROAD * noroad)


def mpc_policy(enc, pred, dec, frame):
    img64 = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    t_img = torch.from_numpy(img64).float().to(DEVICE).div(255.0).permute(2, 0, 1).unsqueeze(0)

    steer_prior, err, x_mean, road_m, grass_m, class_road, class_grass, conf = _compute_signals(t_img)

    with torch.no_grad():
        z_raw = enc(t_img)
        z_mag = z_raw.norm(dim=1, keepdim=True).clamp(min=1e-6)
        z0 = z_raw if PREDICTOR_SPACE == "raw" else _l2norm(z_raw)

    actions = _make_actions(HORIZON, NUM_TENTACLES, steer_prior)

    steer = actions[:, :, 0]
    gas = actions[:, :, 1]

    mag_pen = STEER_MAG_PEN * torch.mean(torch.abs(steer), dim=1)
    jerk_pen = STEER_JERK_PEN * torch.mean(torch.abs(steer[:, 1:] - steer[:, :-1]), dim=1)
    flip_pen = STEER_FLIP_PEN * torch.mean((steer[:, 1:] * steer[:, :-1] < 0).float(), dim=1)
    sat_pen = STEER_SAT_PEN * torch.relu(torch.abs(actions[:, 0, 0]) - STEER_SAT_THRESH)

    prior_pen = (PRIOR_W * conf) * torch.abs(actions[:, 0, 0] - float(steer_prior))

    # NEW: chicane / speed conservatism penalties
    gas_mean_pen = GAS_MEAN_PEN * torch.mean(gas, dim=1)
    gas_turn_pen = GAS_TURN_PEN * torch.mean(gas * torch.abs(steer), dim=1)
    steer_flip_mask = (steer[:, 1:] * steer[:, :-1] < 0).float()
    gas_flip_pen = GAS_FLIP_PEN * torch.mean(steer_flip_mask * gas[:, 1:], dim=1)

    z = z0.repeat(NUM_TENTACLES, 1)

    with torch.no_grad():
        img_cost_total = torch.zeros(NUM_TENTACLES, device=DEVICE)
        for t in range(1, HORIZON + 1):
            z = pred(z, actions[:, t - 1, :])
            if PREDICTOR_SPACE == "norm":
                z = _l2norm(z)

            if t in DECODE_STEPS:
                if PREDICTOR_SPACE == "raw":
                    z_dec = z
                else:
                    z_dec = _l2norm(z) * z_mag
                imgs = torch.clamp(dec(z_dec), 0, 1)
                img_cost_total += _image_cost_batch(imgs)

        score = -img_cost_total \
                - mag_pen - jerk_pen - flip_pen - sat_pen - prior_pen \
                - gas_mean_pen - gas_turn_pen - gas_flip_pen

        best = torch.argmax(score).item()
        best_action0 = actions[best, 0].detach().cpu().numpy()

        recon = dec(z_raw)

        z_vis = z0.clone()
        for _ in range(VIS_HORIZON):
            z_vis = pred(z_vis, actions[best, 0, :].unsqueeze(0))
            if PREDICTOR_SPACE == "norm":
                z_vis = _l2norm(z_vis)

        if PREDICTOR_SPACE == "raw":
            z_dec = z_vis
        else:
            z_dec = _l2norm(z_vis) * z_mag
        dream = dec(z_dec)

    dbg = dict(
        steer_prior=float(steer_prior),
        err=float(err),
        x_mean=float(x_mean),
        road_m=float(road_m),
        grass_m=float(grass_m),
        class_road=float(class_road),
        class_grass=float(class_grass),
        conf=float(conf),
    )
    return best_action0, float(score[best].item()), recon, dream, img64, dbg


def apply_real_speed_governor(env, action: np.ndarray, flip_brake_left: int) -> np.ndarray:
    """
    Uses actual env speed to keep you out of the spin regime.
    Target speed drops with |steer| and with flip-brake state.
    """
    steer = float(action[0])
    gas = float(action[1])
    brake = float(action[2])

    v = get_speed(env)

    v_target = V_BASE - V_DROP * abs(steer)
    if flip_brake_left > 0:
        v_target -= V_FLIP_DROP
    v_target = float(np.clip(v_target, V_MIN, V_BASE))

    if v > v_target:
        # brake proportional to overspeed
        brake = max(brake, float(np.clip(SPEED_BRAKE_K * (v - v_target), 0.0, 0.60)))
        gas = min(gas, OVERSPEED_GAS_CAP)
    else:
        gas = min(gas, SPEED_GAS_CAP)

    return np.array([steer, gas, brake], dtype=np.float32)


def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    enc, pred, dec = load_models()

    env.reset()
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    step_i = 0
    last_steer = 0.0

    # Recovery
    rec_mode = False
    rec_phase = 0
    rec_t = 0
    rec_turn_sign = 1.0
    best_class_seen = 0.0
    class_prev = 0.0
    exit_good = 0

    # Flip-brake
    flip_brake_left = 0

    # Post-recovery
    post_rec_left = 0

    # warmup + kickstart
    for _ in range(40):
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        capture_window(env)
    for _ in range(20):
        env.step(np.array([0.0, 0.7, 0.0], dtype=np.float32))
        capture_window(env)

    try:
        while True:
            frame = capture_window(env)
            if frame is None:
                continue

            a0, sc, recon, dream, tiny, dbg = mpc_policy(enc, pred, dec, frame)

            # Enter recovery
            if (not rec_mode) and (dbg["class_road"] < OFF_ROAD_CLASS):
                rec_mode = True
                rec_phase = 0
                rec_t = 0
                best_class_seen = dbg["class_road"]
                class_prev = dbg["class_road"]
                exit_good = 0

                if abs(last_steer) > 0.10:
                    rec_turn_sign = 1.0 if last_steer > 0 else -1.0
                elif abs(dbg["steer_prior"]) > 0.05:
                    rec_turn_sign = 1.0 if dbg["steer_prior"] > 0 else -1.0
                else:
                    rec_turn_sign = 1.0

            # Sticky exit
            if rec_mode:
                if dbg["class_road"] > REC_EXIT_CLASS:
                    exit_good += 1
                else:
                    exit_good = 0

                if exit_good >= REC_EXIT_COUNT:
                    rec_mode = False
                    rec_phase = 0
                    rec_t = 0
                    exit_good = 0
                    post_rec_left = POST_REC_STEPS

            # Choose action
            if rec_mode:
                rec_t += 1
                mode_str = "REC"

                if rec_phase == 0:
                    action = np.array([0.0, 0.0, 0.60], dtype=np.float32)
                    if rec_t >= REC_STABILIZE_STEPS:
                        rec_phase = 1
                        rec_t = 0

                elif rec_phase == 1:
                    improving = (dbg["class_road"] - class_prev)
                    boost = REC_SEARCH_BOOST if improving > 0.004 else 0.0
                    class_prev = dbg["class_road"]

                    steer = rec_turn_sign * REC_SEARCH_STEER
                    gas = float(np.clip(REC_SEARCH_GAS + boost, 0.0, 0.85))
                    brake = REC_SEARCH_BRAKE
                    action = np.array([steer, gas, brake], dtype=np.float32)

                    if dbg["class_road"] > best_class_seen + 0.01:
                        best_class_seen = dbg["class_road"]
                    if rec_t % 28 == 0 and dbg["class_road"] < best_class_seen + 0.005:
                        rec_turn_sign *= -1.0

                    if rec_t >= REC_SEARCH_STEPS:
                        rec_phase = 2
                        rec_t = 0

                else:
                    steer = float(np.clip(REC_REJOIN_K * dbg["err"], -REC_REJOIN_STEER_CLAMP, REC_REJOIN_STEER_CLAMP))
                    action = np.array([steer, REC_REJOIN_GAS, REC_REJOIN_BRAKE], dtype=np.float32)
                    if rec_t >= REC_REJOIN_STEPS:
                        rec_phase = 1
                        rec_t = 0

            else:
                mode_str = "MPC"
                steer_cmd = MOMENTUM * last_steer + (1.0 - MOMENTUM) * float(a0[0])
                steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))
                action = np.array([steer_cmd, float(a0[1]), float(a0[2])], dtype=np.float32)

                # Detect snap (chicane sign flip)
                if (np.sign(last_steer) != np.sign(action[0])) and (abs(action[0] - last_steer) > FLIP_STEER_DELTA):
                    flip_brake_left = FLIP_BRAKE_STEPS

                # Apply flip-brake pulse first (prevents immediate snap-oversteer)
                if flip_brake_left > 0:
                    action[2] = max(action[2], FLIP_BRAKE)
                    action[1] = min(action[1], FLIP_GAS_CAP)
                    flip_brake_left -= 1

                # Apply real-speed governor (key mismatch fix)
                action = apply_real_speed_governor(env, action, flip_brake_left)

                # Post-recovery ramp
                if post_rec_left > 0:
                    action[0] = float(np.clip(action[0], -POST_REC_STEER_CLAMP, POST_REC_STEER_CLAMP))
                    action[1] = min(action[1], POST_REC_GAS)
                    action[2] = min(action[2], 0.12)
                    post_rec_left -= 1

            last_steer = float(action[0])

            _, _, done, trunc, _ = env.step(action)

            step_i += 1
            if DEBUG_PRINT and step_i % 10 == 0:
                v = get_speed(env)
                extra = f" phase={rec_phase} t={rec_t:03d} exit_good={exit_good}" if rec_mode else ""
                print(
                    f"step={step_i:05d} mode={mode_str}{extra} v={v:5.2f} "
                    f"steer={action[0]:+.3f} gas={action[1]:.3f} brake={action[2]:.3f} score={sc:+.3f} | "
                    f"prior={dbg['steer_prior']:+.3f} conf={dbg['conf']:.2f} err={dbg['err']:+.3f} "
                    f"class_road={dbg['class_road']:.3f}"
                )

            # HUD panels
            vis_real = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vis_real = cv2.resize(vis_real, (400, 300))
            cv2.putText(vis_real, "LIVE GAME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            vis_input = cv2.cvtColor(tiny, cv2.COLOR_RGB2BGR)
            vis_input = cv2.resize(vis_input, (100, 100), interpolation=cv2.INTER_NEAREST)
            vis_real[200:300, 300:400] = vis_input

            r = torch.clamp(recon, 0, 1).squeeze().detach().cpu().permute(1, 2, 0).numpy()
            r = cv2.resize(r, (400, 300))
            r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
            cv2.putText(r, "What Net Sees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            d = torch.clamp(dream, 0, 1).squeeze().detach().cpu().permute(1, 2, 0).numpy()
            d = cv2.resize(d, (400, 300))
            d = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)
            cv2.putText(d, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            vis = np.hstack((vis_real, r, d))
            cv2.putText(vis, f"mode={mode_str} score={sc:.3f}", (10, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("World Model Pilot (conservative MPC + real-speed governor)", vis)

            if cv2.waitKey(1) == 27 or done or trunc:
                env.reset()
                last_steer = 0.0
                rec_mode = False
                rec_phase = 0
                rec_t = 0
                rec_turn_sign = 1.0
                best_class_seen = 0.0
                class_prev = 0.0
                exit_good = 0
                flip_brake_left = 0
                post_rec_left = 0

                for _ in range(40):
                    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    capture_window(env)
                for _ in range(20):
                    env.step(np.array([0.0, 0.7, 0.0], dtype=np.float32))
                    capture_window(env)

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
