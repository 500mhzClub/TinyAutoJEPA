import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time
import warnings

# --- CONFIGURATION (override via env vars) ---
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(min(32, mp.cpu_count()))))
EPISODES_PER_WORKER = int(os.getenv("EPISODES_PER_WORKER", "30"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2500"))
DATA_DIR = os.getenv("DATA_DIR", "data_recovery")
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))

# Expert (same geometry fix: ry)
BASE_LOOKAHEAD = float(os.getenv("BASE_LOOKAHEAD", "4.0"))
LOOKAHEAD_SPEED_GAIN = float(os.getenv("LOOKAHEAD_SPEED_GAIN", "0.3"))
EXPERT_STEER_GAIN = float(os.getenv("EXPERT_STEER_GAIN", "6.0"))

TARGET_SPEED = float(os.getenv("TARGET_SPEED", "35.0"))

# Mistake injection
MISTAKE_PROB_PER_STEP = float(os.getenv("MISTAKE_PROB_PER_STEP", "0.04"))
MISTAKE_MIN_DUR = int(os.getenv("MISTAKE_MIN_DUR", "12"))
MISTAKE_MAX_DUR = int(os.getenv("MISTAKE_MAX_DUR", "25"))

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def get_car_and_track(env):
    u = env.unwrapped
    car = getattr(u, "car", None)
    track = getattr(u, "track", None)
    return car, track

def expert_action(env, prev_idx: int):
    car, track = get_car_and_track(env)
    if car is None or track is None or len(track) == 0:
        return 0.0, 0.0, prev_idx

    car_pos = np.array(car.hull.position, dtype=np.float32)
    car_vel = np.array(car.hull.linearVelocity, dtype=np.float32)
    speed = float(np.linalg.norm(car_vel))
    car_angle = float(car.hull.angle)

    track_coords = np.array([t[2:4] for t in track], dtype=np.float32)
    track_len = track_coords.shape[0]

    search_radius = min(80, track_len)
    idxs = (np.arange(prev_idx - search_radius, prev_idx + search_radius) % track_len).astype(int)
    dists = np.linalg.norm(track_coords[idxs] - car_pos[None, :], axis=1)
    closest_idx = int(idxs[int(np.argmin(dists))])

    L = BASE_LOOKAHEAD + speed * LOOKAHEAD_SPEED_GAIN

    target_idx = closest_idx
    for i in range(1, 80):
        idx = (closest_idx + i) % track_len
        if float(np.linalg.norm(track_coords[idx] - car_pos)) > L:
            target_idx = int(idx)
            break

    target_pt = track_coords[target_idx]
    dx = float(target_pt[0] - car_pos[0])
    dy = float(target_pt[1] - car_pos[1])

    rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
    ry = dx * math.sin(-car_angle) + dy * math.cos(-car_angle)

    curvature = (2.0 * ry) / (L * L)
    steer = float(np.clip(curvature * EXPERT_STEER_GAIN, -1.0, 1.0))

    return steer, speed, closest_idx

class RecoveryPolicy:
    def __init__(self):
        self.prev_idx = 0
        self.mistake_timer = 0
        self.mistake_duration = 0
        self.mistake_type = None
        self.late_buffer = []  # for "late turn" effect

    def reset(self):
        self.prev_idx = 0
        self.mistake_timer = 0
        self.mistake_duration = 0
        self.mistake_type = None
        self.late_buffer = []

    def _maybe_start_mistake(self):
        if self.mistake_timer != 0:
            return
        if np.random.rand() < MISTAKE_PROB_PER_STEP:
            self.mistake_timer = 1
            self.mistake_duration = np.random.randint(MISTAKE_MIN_DUR, MISTAKE_MAX_DUR + 1)
            # include braking-related mistakes explicitly
            self.mistake_type = np.random.choice(
                ["under", "over", "late", "panic_brake", "throttle_cut"],
                p=[0.30, 0.30, 0.15, 0.15, 0.10],
            )
            self.late_buffer = []

    def act(self, env):
        expert_steer, speed, self.prev_idx = expert_action(env, self.prev_idx)

        # default "expert" throttle
        if speed < TARGET_SPEED:
            gas = 0.6
            brake = 0.0
        elif speed > TARGET_SPEED + 8.0:
            gas = 0.0
            brake = 0.25
        else:
            gas = 0.25
            brake = 0.0

        self._maybe_start_mistake()

        steer = expert_steer

        if self.mistake_timer > 0:
            t = self.mistake_timer
            self.mistake_timer += 1

            if self.mistake_type == "under":
                steer = float(np.clip(expert_steer * 0.35, -1.0, 1.0))
                gas = 0.40
                brake = 0.0

            elif self.mistake_type == "over":
                steer = float(np.clip(expert_steer * 1.8, -1.0, 1.0))
                gas = 0.35
                brake = 0.0

            elif self.mistake_type == "late":
                # delay steering response for a few steps, then snap harder
                self.late_buffer.append(expert_steer)
                if t < 7 and len(self.late_buffer) >= 2:
                    steer = self.late_buffer[-2]  # lagged steer
                else:
                    steer = float(np.clip(expert_steer * 1.5, -1.0, 1.0))
                gas = 0.35
                brake = 0.0

            elif self.mistake_type == "panic_brake":
                # force a decel burst (creates real brake distribution)
                steer = expert_steer
                gas = 0.0
                # stronger brake if fast or turning
                base = 0.35 + 0.25 * (speed > TARGET_SPEED + 5.0) + 0.20 * (abs(expert_steer) > 0.4)
                brake = float(np.clip(base, 0.2, 0.9))

            elif self.mistake_type == "throttle_cut":
                steer = expert_steer
                gas = 0.0
                brake = 0.0

            if self.mistake_timer > self.mistake_duration:
                self.mistake_timer = 0
                self.mistake_type = None
                self.mistake_duration = 0
                self.late_buffer = []

        action = np.array([steer, gas, brake], dtype=np.float32)
        return action

def worker_func(worker_id: int) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

    seed = int(time.time()) + worker_id * 10000
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    policy = RecoveryPolicy()

    states, actions, next_states = [], [], []

    try:
        for ep in range(EPISODES_PER_WORKER):
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            policy.reset()

            for _ in range(10):
                obs, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                if terminated or truncated:
                    obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
                    policy.reset()

            for _ in range(MAX_STEPS):
                s = process_frame(obs)
                action = policy.act(env)
                obs2, _, terminated, truncated, _ = env.step(action)

                ns = process_frame(obs2)
                states.append(s)
                actions.append(action)
                next_states.append(ns)

                obs = obs2
                if terminated or truncated:
                    break
    finally:
        env.close()

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"recovery_chunk_{worker_id}.npz")
    np.savez_compressed(
        filename,
        states=np.asarray(states, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.uint8),
    )
    print(f"[Worker {worker_id}] wrote {len(states):,} frames -> {filename}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Recovery Data Collection (Expert + Mistakes) ===")
    print(f"workers={NUM_WORKERS} episodes/worker={EPISODES_PER_WORKER} max_steps={MAX_STEPS}")

    mp.set_start_method("spawn", force=True)
    with mp.Pool(NUM_WORKERS) as pool:
        pool.map(worker_func, range(NUM_WORKERS))
