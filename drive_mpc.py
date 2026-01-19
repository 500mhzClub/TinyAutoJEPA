import gymnasium as gym
import torch
import numpy as np
import cv2
import os
import pygame
from networks import TinyEncoder, Predictor, TinyDecoder

MODEL_PATH_ENC    = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED   = "./models/predictor_final.pth"
MODEL_PATH_DEC    = "./models/decoder_final.pth"
MODEL_PATH_MEMORY = "./models/memory_bank.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MPC
HORIZON       = int(os.getenv("HORIZON", "12"))
NUM_TENTACLES = int(os.getenv("NUM_TENTACLES", "256"))
TOP_K         = int(os.getenv("TOP_K", "20"))

# control shaping
MOMENTUM      = float(os.getenv("MOMENTUM", "0.15"))     # lower than before; reduces “late steering”
STEER_STD     = float(os.getenv("STEER_STD", "0.25"))    # initial steer spread
STEER_DSTD    = float(os.getenv("STEER_DSTD", "0.18"))   # per-step random-walk
STEER_SMOOTH  = float(os.getenv("STEER_SMOOTH", "0.75")) # higher => smoother

GAS_BASE      = float(os.getenv("GAS_BASE", "0.75"))
GAS_DROP      = float(os.getenv("GAS_DROP", "0.35"))
MIN_GAS       = float(os.getenv("MIN_GAS", "0.25"))

STEER_MAG_PEN = float(os.getenv("STEER_MAG_PEN", "0.10"))
STEER_JERK_PEN= float(os.getenv("STEER_JERK_PEN", "0.08"))

# predictor space
PREDICTOR_SPACE = os.getenv("PREDICTOR_SPACE", "norm").lower()
assert PREDICTOR_SPACE in ("raw", "norm")

# decoded-image road/grass penalty
USE_IMG_COST   = int(os.getenv("USE_IMG_COST", "1")) == 1
IMG_COST_W     = float(os.getenv("IMG_COST_W", "3.0"))
ROI            = (34, 62, 16, 48)  # y1,y2,x1,x2

VIS_HORIZON    = int(os.getenv("VIS_HORIZON", "1"))

def _l2norm(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)

def load_models():
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))

    dec = TinyDecoder().to(DEVICE).eval()
    dec.load_state_dict(torch.load(MODEL_PATH_DEC, map_location=DEVICE))

    bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    bank = _l2norm(bank)

    print(f"PREDICTOR_SPACE={PREDICTOR_SPACE} H={HORIZON} N={NUM_TENTACLES} img_cost={USE_IMG_COST}")
    return enc, pred, dec, bank

def capture_window(env):
    surf = env.unwrapped.screen
    if surf is None:
        return None
    frame_t = pygame.surfarray.array3d(surf)
    frame = frame_t.transpose(1, 0, 2)
    return np.ascontiguousarray(frame, dtype=np.uint8)

def _make_actions(horizon, n, last_steer):
    # smooth steer random-walk around last steer
    s0 = torch.randn(n, device=DEVICE) * STEER_STD + float(last_steer)
    s0 = torch.clamp(s0, -1.0, 1.0)

    ds = torch.randn(n, horizon, device=DEVICE) * STEER_DSTD
    steer = torch.zeros(n, horizon, device=DEVICE)
    steer[:, 0] = s0

    for t in range(1, horizon):
        proposed = torch.clamp(steer[:, t-1] + ds[:, t], -1.0, 1.0)
        steer[:, t] = STEER_SMOOTH * steer[:, t-1] + (1.0 - STEER_SMOOTH) * proposed

    # throttle: always forward, less on turns
    gas = GAS_BASE - GAS_DROP * torch.abs(steer)
    gas = torch.clamp(gas, MIN_GAS, 0.90)

    # brake disabled (stopping is almost always a losing local optimum here)
    brake = torch.zeros_like(gas)

    return torch.stack([steer, gas, brake], dim=-1)  # [n,h,3]

def _roi_grass_road_cost(decoded_bchw_01: torch.Tensor):
    """
    decoded_bchw_01: [B,3,64,64] 0..1
    returns cost[B] where higher is worse (grass ahead / no road)
    """
    y1, y2, x1, x2 = ROI
    roi = decoded_bchw_01[:, :, y1:y2, x1:x2]
    r = roi[:, 0]
    g = roi[:, 1]
    b = roi[:, 2]

    grass = (g > (r + 0.08)) & (g > (b + 0.08)) & (g > 0.20)
    m = (r + g + b) / 3.0
    road  = (torch.abs(r - g) < 0.07) & (torch.abs(g - b) < 0.07) & (m > 0.15) & (m < 0.75)

    grass_frac = grass.float().mean(dim=(1, 2))
    road_frac  = road.float().mean(dim=(1, 2))

    # penalize grass and penalize lack of road
    return grass_frac + (0.30 - road_frac).clamp(min=0.0)

def mpc_policy(enc, pred, dec, bank, frame, last_steer):
    img64 = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    t_img = torch.from_numpy(img64).float().to(DEVICE).div(255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        z_raw = enc(t_img)
        z_mag = z_raw.norm(dim=1, keepdim=True).clamp(min=1e-6)
        z0 = z_raw if PREDICTOR_SPACE == "raw" else _l2norm(z_raw)

    actions = _make_actions(HORIZON, NUM_TENTACLES, last_steer)

    z = z0.repeat(NUM_TENTACLES, 1)
    total = torch.zeros(NUM_TENTACLES, device=DEVICE)

    with torch.no_grad():
        for t in range(HORIZON):
            z = pred(z, actions[:, t, :])
            if PREDICTOR_SPACE == "norm":
                z = _l2norm(z)

            z_score = _l2norm(z)
            sim = torch.mm(z_score, bank.T)
            top, _ = torch.topk(sim, k=TOP_K, dim=1)
            total += top.mean(dim=1) * (0.93 ** t)

    # steering regularization
    steer = actions[:, :, 0]
    mag_pen  = STEER_MAG_PEN  * torch.mean(torch.abs(steer), dim=1)
    jerk_pen = STEER_JERK_PEN * torch.mean(torch.abs(steer[:, 1:] - steer[:, :-1]), dim=1)

    score = total - mag_pen - jerk_pen

    # decoded predicted end-state image cost
    if USE_IMG_COST:
        with torch.no_grad():
            z_end = z
            if PREDICTOR_SPACE == "raw":
                z_dec = z_end
            else:
                z_dec = _l2norm(z_end) * z_mag  # rescale for decoder
            end_imgs = dec(z_dec)
            img_cost = _roi_grass_road_cost(torch.clamp(end_imgs, 0, 1))
            score = score - IMG_COST_W * img_cost

    best = torch.argmax(score).item()
    best_action = actions[best, 0].detach().cpu().numpy()

    # visuals
    with torch.no_grad():
        recon = dec(z_raw)

        z_vis = z0.clone()
        for t in range(VIS_HORIZON):
            z_vis = pred(z_vis, actions[best, t, :].unsqueeze(0))
            if PREDICTOR_SPACE == "norm":
                z_vis = _l2norm(z_vis)

        if PREDICTOR_SPACE == "raw":
            z_dec = z_vis
        else:
            z_dec = _l2norm(z_vis) * z_mag
        dream = dec(z_dec)

    return best_action, float(score[best].item()), recon, dream, img64

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    enc, pred, dec, bank = load_models()

    env.reset()
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    last_steer = 0.0

    # warmup
    for _ in range(50):
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        capture_window(env)

    # kickstart
    for _ in range(25):
        env.step(np.array([0.0, 0.7, 0.0], dtype=np.float32))
        capture_window(env)

    try:
        while True:
            frame = capture_window(env)
            if frame is None:
                continue

            a, sc, recon, dream, tiny = mpc_policy(enc, pred, dec, bank, frame, last_steer)

            steer_cmd = MOMENTUM * last_steer + (1.0 - MOMENTUM) * float(a[0])
            last_steer = float(np.clip(steer_cmd, -1.0, 1.0))
            action = np.array([last_steer, float(a[1]), float(a[2])], dtype=np.float32)

            _, _, done, trunc, _ = env.step(action)

            # --- HUD ---
            vis_real = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vis_real = cv2.resize(vis_real, (400, 300))
            cv2.putText(vis_real, "LIVE GAME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            vis_input = cv2.cvtColor(tiny, cv2.COLOR_RGB2BGR)
            vis_input = cv2.resize(vis_input, (100, 100), interpolation=cv2.INTER_NEAREST)
            vis_real[200:300, 300:400] = vis_input

            r = torch.clamp(recon, 0, 1).squeeze().cpu().permute(1,2,0).numpy()
            r = cv2.resize(r, (400, 300))
            r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
            cv2.putText(r, "What Net Sees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            d = torch.clamp(dream, 0, 1).squeeze().cpu().permute(1,2,0).numpy()
            d = cv2.resize(d, (400, 300))
            d = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)
            cv2.putText(d, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            vis = np.hstack((vis_real, r, d))
            cv2.putText(vis, f"score={sc:.3f}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("World Model Pilot", vis)

            if cv2.waitKey(1) == 27 or done or trunc:
                env.reset()
                last_steer = 0.0
                for _ in range(50):
                    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    capture_window(env)
                for _ in range(25):
                    env.step(np.array([0.0, 0.7, 0.0], dtype=np.float32))
                    capture_window(env)

    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
