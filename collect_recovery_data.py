import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import time
import warnings
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(min(32, mp.cpu_count()))))
EPISODES_PER_WORKER = int(os.getenv("EPISODES_PER_WORKER", "80"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "600"))
DATA_DIR = os.getenv("DATA_DIR", "data_recover") # Folder for recovery data
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))
REPO_ID = "sb3/ppo-CarRacing-v2"
MODEL_FILENAME = "ppo-CarRacing-v2.zip"

# --- RECOVERY / SABOTAGE PARAMS ---
SABOTAGE_PROB = 0.01       # 1% chance per frame to start sabotaging
MIN_SABOTAGE_DUR = 5       # Min frames to hold the bad action
MAX_SABOTAGE_DUR = 20      # Max frames to hold the bad action

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def worker_func(worker_id: int, model_path: str) -> None:
    warnings.filterwarnings("ignore")
    model = PPO.load(model_path, device="cpu")
    
    seed = int(time.time()) + worker_id * 10000
    rng = np.random.RandomState(seed)
    
    try:
        env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    except:
        env = gym.make("CarRacing-v2", render_mode=None)

    states, actions, next_states = [], [], []

    try:
        for ep in range(EPISODES_PER_WORKER):
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            
            sabotage_timer = 0
            sabotage_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            for _ in range(MAX_STEPS):
                s = process_frame(obs)
                
                # --- LOGIC CONTROLLER ---
                if sabotage_timer > 0:
                    # We are currently in sabotage mode
                    action = sabotage_action
                    sabotage_timer -= 1
                else:
                    # Normal driving mode
                    # Check if we should start sabotaging
                    if rng.rand() < SABOTAGE_PROB:
                        sabotage_timer = rng.randint(MIN_SABOTAGE_DUR, MAX_SABOTAGE_DUR)
                        
                        # Generate a nasty action (Hard Left or Hard Right + Gas)
                        steer = -1.0 if rng.rand() < 0.5 else 1.0
                        gas = rng.uniform(0.4, 0.8) # Add gas so it flies off track
                        brake = 0.0
                        sabotage_action = np.array([steer, gas, brake], dtype=np.float32)
                        
                        action = sabotage_action
                    else:
                        # Let the expert drive
                        action, _ = model.predict(obs, deterministic=True)

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

    if len(states) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = os.path.join(DATA_DIR, f"recover_chunk_{worker_id}.npz")
        
        np.savez(
            filename,
            states=np.asarray(states, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.uint8),
        )
        print(f"[Worker {worker_id}] wrote {len(states):,} frames -> {filename}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"=== Recovery Data Collection (Expert + Sabotage) ===")
    print(f"workers={NUM_WORKERS} episodes/worker={EPISODES_PER_WORKER} max_steps={MAX_STEPS}")

    print(f"Downloading expert model from {REPO_ID}...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    
    mp.set_start_method("spawn", force=True)
    task_args = [(i, model_path) for i in range(NUM_WORKERS)]
    
    with mp.Pool(NUM_WORKERS) as pool:
        pool.starmap(worker_func, task_args)