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
DATA_DIR = os.getenv("DATA_DIR", "data_race")  # Folder for expert data
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))
REPO_ID = "sb3/ppo-CarRacing-v2"  # Reliable expert model
MODEL_FILENAME = "ppo-CarRacing-v2.zip"

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # CarRacing obs is ~96x96; crop top HUD-ish area a bit
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def worker_func(worker_id: int, model_path: str) -> None:
    warnings.filterwarnings("ignore")
    
    # Load model on CPU to avoid CUDA fork issues in multiprocessing
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
            # Reset
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            
            # Reset internal LSTM states if the model is recurrent (PPO default is not, but good practice)
            # The sb3/ppo-CarRacing-v2 model is standard MLP/CNN, so no LSTM state needed.
            
            for _ in range(MAX_STEPS):
                s = process_frame(obs)
                
                # GET EXPERT ACTION
                # We pass the raw obs (96x96) to the model, not the cropped 's'
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
        filename = os.path.join(DATA_DIR, f"race_chunk_{worker_id}.npz")
        
        np.savez(
            filename,
            states=np.asarray(states, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.uint8),
        )
        print(f"[Worker {worker_id}] wrote {len(states):,} frames -> {filename}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"=== Expert Data Collection (Perfect Driving) ===")
    print(f"workers={NUM_WORKERS} episodes/worker={EPISODES_PER_WORKER} max_steps={MAX_STEPS}")

    # 1. Download model once in main process
    print(f"Downloading expert model from {REPO_ID}...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    print(f"Model cached at {model_path}")

    # 2. Spawn workers
    mp.set_start_method("spawn", force=True)
    
    # We use starmap to pass arguments (worker_id, model_path)
    task_args = [(i, model_path) for i in range(NUM_WORKERS)]
    
    with mp.Pool(NUM_WORKERS) as pool:
        pool.starmap(worker_func, task_args)