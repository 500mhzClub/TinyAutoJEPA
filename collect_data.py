import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import time

# --- CONFIGURATION ---
NUM_WORKERS = 32
EPISODES_PER_WORKER = 80
MAX_STEPS = 600
DATA_DIR = "data"
IMG_SIZE = 64

def process_frame(frame):
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def get_action_entropy(env, previous_steering):
    """
    High-Entropy Random Policy:
    Designed to explore the edges of the physics engine.
    """
    # 1. Full steering range exploration
    # Sample widely [-1, 1]
    noise = np.random.uniform(-1.0, 1.0)
    
    # 2. Minimal persistence
    # 20% old, 80% new -> Allows rapid zig-zags
    new_steer = (0.2 * previous_steering) + (0.8 * noise)
    new_steer = np.clip(new_steer, -1.0, 1.0)

    # 3. Mode Switching (Gas vs Brake)
    # Prevents "gray zone" behavior where gas/brake are both 0.1
    mode = np.random.choice(['accelerate', 'brake', 'coast'], p=[0.6, 0.1, 0.3])
    
    if mode == 'accelerate':
        gas = np.random.uniform(0.5, 1.0) # Commit to gas
        brake = 0.0
    elif mode == 'brake':
        gas = 0.0
        brake = np.random.uniform(0.2, 0.9) # Commit to brake
    else:
        gas = 0.0
        brake = 0.0

    return np.array([new_steer, gas, brake], dtype=np.float32)

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 10000
    np.random.seed(seed)
    
    env = gym.make("CarRacing-v3", render_mode=None)
    
    states, actions, next_states = [], [], []

    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed + episode)
        
        # Skip zoom
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0], dtype=np.float32))
        
        # Random spawn: floor it to get moving
        if episode % 5 < 2:
            for _ in range(np.random.randint(50, 300)):
                obs, _, d, t, _ = env.step(np.array([0, 1.0, 0]))
                if d or t: break

        obs = process_frame(obs)
        current_steering = 0.0
        
        for step in range(MAX_STEPS):
            action = get_action_entropy(env, current_steering)
            current_steering = action[0]
            
            next_obs_raw, _, terminated, truncated, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            
            obs = next_obs
            if done: break

    filename = os.path.join(DATA_DIR, f"random_chunk_{worker_id}.npz")
    np.savez_compressed(filename, states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
    env.close()
    print(f"[Worker {worker_id}] Complete")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"=== Random Data Collection (High Entropy) ===")
    print(f"Goal: Steering Std > 0.40")
    
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(NUM_WORKERS)
    pool.map(worker_func, range(NUM_WORKERS))
    pool.close()
    pool.join()