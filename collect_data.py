import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import time
import warnings

# --- CONFIGURATION ---
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(min(32, mp.cpu_count()))))
EPISODES_PER_WORKER = int(os.getenv("EPISODES_PER_WORKER", "80"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "600"))
DATA_DIR = os.getenv("DATA_DIR", "data")
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # CarRacing obs is ~96x96; crop top HUD-ish area a bit (keeps road + horizon)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def sample_random_action(prev_steer: float) -> tuple[np.ndarray, float]:
    # High-entropy steering with mild temporal smoothing (prevents pure white noise)
    noise = np.random.uniform(-1.0, 1.0)
    steer = np.clip(0.2 * prev_steer + 0.8 * noise, -1.0, 1.0)

    # Gas/Brake mode switching to avoid tiny simultaneous values
    mode = np.random.choice(["accelerate", "brake", "coast"], p=[0.60, 0.10, 0.30])
    if mode == "accelerate":
        gas = np.random.uniform(0.5, 1.0)
        brake = 0.0
    elif mode == "brake":
        gas = 0.0
        brake = np.random.uniform(0.2, 0.9)
    else:
        gas = np.random.uniform(0.0, 0.2)
        brake = 0.0

    return np.array([steer, gas, brake], dtype=np.float32), float(steer)

def worker_func(worker_id: int) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

    seed = int(time.time()) + worker_id * 10000
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)

    states, actions, next_states = [], [], []
    prev_steer = 0.0

    try:
        for ep in range(EPISODES_PER_WORKER):
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            prev_steer = 0.0

            for _ in range(MAX_STEPS):
                s = process_frame(obs)
                action, prev_steer = sample_random_action(prev_steer)

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
    filename = os.path.join(DATA_DIR, f"random_chunk_{worker_id}.npz")
    np.savez_compressed(
        filename,
        states=np.asarray(states, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.uint8),
    )
    print(f"[Worker {worker_id}] wrote {len(states):,} frames -> {filename}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Random Data Collection (High Entropy) ===")
    print(f"workers={NUM_WORKERS} episodes/worker={EPISODES_PER_WORKER} max_steps={MAX_STEPS}")

    mp.set_start_method("spawn", force=True)
    with mp.Pool(NUM_WORKERS) as pool:
        pool.map(worker_func, range(NUM_WORKERS))
