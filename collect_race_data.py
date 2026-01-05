import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time
import warnings

# --- CONFIGURATION (override via env vars) ---
VISUAL_VERIFY = os.getenv("VISUAL_VERIFY", "0") == "1"  # set to 1 to watch single-process run
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(min(32, mp.cpu_count()))))
EPISODES_PER_WORKER = int(os.getenv("EPISODES_PER_WORKER", "20"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "3000"))
DATA_DIR = os.getenv("DATA_DIR", "data_race")
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))

# Pure pursuit parameters
BASE_LOOKAHEAD = float(os.getenv("BASE_LOOKAHEAD", "4.0"))
LOOKAHEAD_SPEED_GAIN = float(os.getenv("LOOKAHEAD_SPEED_GAIN", "0.3"))
STEER_GAIN = float(os.getenv("STEER_GAIN", "8.0"))

# Simple speed control
BASE_TARGET_SPEED = float(os.getenv("BASE_TARGET_SPEED", "40.0"))
STEER_SPEED_PENALTY = float(os.getenv("STEER_SPEED_PENALTY", "30.0"))
MIN_TARGET_SPEED = float(os.getenv("MIN_TARGET_SPEED", "15.0"))

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

def pure_pursuit_action(env, prev_idx: int):
    car, track = get_car_and_track(env)
    if car is None or track is None or len(track) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, prev_idx

    car_pos = np.array(car.hull.position, dtype=np.float32)
    car_vel = np.array(car.hull.linearVelocity, dtype=np.float32)
    speed = float(np.linalg.norm(car_vel))
    car_angle = float(car.hull.angle)

    track_coords = np.array([t[2:4] for t in track], dtype=np.float32)
    track_len = track_coords.shape[0]

    # find closest track point (local search around prev_idx helps stability)
    search_radius = min(80, track_len)
    idxs = (np.arange(prev_idx - search_radius, prev_idx + search_radius) % track_len).astype(int)
    dists = np.linalg.norm(track_coords[idxs] - car_pos[None, :], axis=1)
    closest_idx = int(idxs[int(np.argmin(dists))])

    # lookahead
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

    # rotate into car coordinates (forward=x, left=y)
    rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
    ry = dx * math.sin(-car_angle) + dy * math.cos(-car_angle)

    # pure pursuit curvature uses lateral offset (ry), not rx
    curvature = (2.0 * ry) / (L * L)
    steer = float(np.clip(curvature * STEER_GAIN, -1.0, 1.0))

    # speed control (heuristic): slow down when |steer| is high
    target_speed = max(MIN_TARGET_SPEED, BASE_TARGET_SPEED - STEER_SPEED_PENALTY * abs(steer))
    speed_error = target_speed - speed

    if speed_error > 2.0:
        gas = 0.6
        brake = 0.0
    elif speed_error < -5.0:
        gas = 0.0
        brake = float(np.clip(abs(speed_error) / 15.0, 0.1, 0.6))
    else:
        gas = 0.3
        brake = 0.0

    # emergency brake if very sharp steer at high speed
    if abs(steer) > 0.7 and speed > (target_speed + 10.0):
        gas = 0.0
        brake = max(brake, 0.4)

    action = np.array([steer, gas, brake], dtype=np.float32)
    return action, speed, closest_idx

def worker_func(worker_id: int) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

    seed = int(time.time()) + worker_id * 10000
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    states, actions, next_states = [], [], []

    try:
        prev_idx = 0
        for ep in range(EPISODES_PER_WORKER):
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            prev_idx = 0

            # warm-up a few frames so track/car are ready
            for _ in range(10):
                obs, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                if terminated or truncated:
                    obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))

            for _ in range(MAX_STEPS):
                s = process_frame(obs)
                action, _, prev_idx = pure_pursuit_action(env, prev_idx)
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
    filename = os.path.join(DATA_DIR, f"race_chunk_{worker_id}.npz")
    np.savez_compressed(
        filename,
        states=np.asarray(states, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.uint8),
    )
    print(f"[Worker {worker_id}] wrote {len(states):,} frames -> {filename}")

def visual_verify_run():
    env = gym.make("CarRacing-v3", render_mode="human", max_episode_steps=MAX_STEPS)
    prev_idx = 0
    obs, _ = env.reset()
    for _ in range(10):
        obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    step = 0
    try:
        while True:
            action, speed, prev_idx = pure_pursuit_action(env, prev_idx)
            obs, _, terminated, truncated, _ = env.step(action)
            if step % 20 == 0:
                print(f"step={step} speed={speed:5.1f} steer={action[0]:+.2f} gas={action[1]:.2f} brake={action[2]:.2f}")
            step += 1
            if terminated or truncated:
                obs, _ = env.reset()
                prev_idx = 0
    finally:
        env.close()

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    if VISUAL_VERIFY:
        visual_verify_run()
    else:
        print("=== Race Data Collection (Pure Pursuit Expert) ===")
        print(f"workers={NUM_WORKERS} episodes/worker={EPISODES_PER_WORKER} max_steps={MAX_STEPS}")

        mp.set_start_method("spawn", force=True)
        with mp.Pool(NUM_WORKERS) as pool:
            pool.map(worker_func, range(NUM_WORKERS))
