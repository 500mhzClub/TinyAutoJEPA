import warnings
import os

# --- SILENCE WARNINGS ---
# Must be done before importing gymnasium/pygame
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import gymnasium as gym
import numpy as np
import multiprocessing as mp
import cv2
import time
from tqdm import tqdm  # Requires: pip install tqdm

# --- CONFIGURATION ---
NUM_WORKERS = 16
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

def get_speed(env):
    try:
        car = env.unwrapped.car if hasattr(env.unwrapped, 'car') else env.car
        if car is None: return 10.0
        velocity = car.hull.linearVelocity
        return np.sqrt(velocity[0]**2 + velocity[1]**2)
    except:
        return 10.0

def get_correlated_action(env, previous_steering):
    speed = get_speed(env)
    max_steer = max(0.25, 1.0 - (speed / 50.0))
    new_steering = np.random.uniform(-max_steer, max_steer)
    smoothed_steering = (0.8 * previous_steering) + (0.2 * new_steering)
    smoothed_steering = np.clip(smoothed_steering, -1.0, 1.0)
    
    target_speed = np.random.uniform(15, 35)
    if speed < target_speed - 3:
        gas = np.random.uniform(0.4, 0.8); brake = 0.0
    elif speed > target_speed + 3:
        gas = 0.0; brake = np.random.uniform(0.1, 0.4)
    else:
        gas = np.random.uniform(0.2, 0.5); brake = 0.0
        
    return np.array([smoothed_steering, gas, brake], dtype=np.float32)

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 10000
    # Capture warnings inside worker process too
    warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
    
    env = gym.make("CarRacing-v3", render_mode=None)
    states, actions, next_states = [], [], []
    
    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed + episode)
        for _ in range(50):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0], dtype=np.float32))
            
        if episode % 5 < 2:
            skip_steps = np.random.randint(100, 600)
            for _ in range(skip_steps):
                obs, _, done, trunc, _ = env.step(np.array([0, 0.5, 0]))
                if done or trunc:
                    obs, _ = env.reset(seed=seed + episode + 999)
                    for _ in range(50): env.step(np.array([0,0,0]))
                    break
                    
        obs = process_frame(obs)
        current_steering = 0.0
        
        for step in range(MAX_STEPS):
            action = get_correlated_action(env, current_steering)
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
    return worker_id # Return ID so main loop knows who finished

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    
    print(f"Launching {NUM_WORKERS} workers...")
    
    with mp.Pool(NUM_WORKERS) as pool:
        # Use tqdm to track workers as they finish
        results = list(tqdm(pool.imap_unordered(worker_func, range(NUM_WORKERS)), 
                           total=NUM_WORKERS, 
                           desc="Workers Completed",
                           unit="worker"))
    
    print("Done!")