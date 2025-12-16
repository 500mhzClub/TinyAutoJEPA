import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import time

# --- CONFIGURATION ---
# 5950X Sweet Spot: 20-24 threads. 32 might choke on Box2D math.
NUM_WORKERS = 1        
EPISODES_PER_WORKER = 1  # Total = 24 * 80 = 1920 episodes
MAX_STEPS = 600
DATA_DIR = "data"
IMG_SIZE = 64

def process_frame(frame):
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :] # Crop dashboard
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return frame.astype(np.uint8)

def get_correlated_action(env, previous_steering):
    """
    Improves data quality!
    Instead of pure random jitter, we smooth the steering.
    If we were turning left, we likely keep turning left.
    """
    # 1. Sample pure random action
    action = env.action_space.sample()
    
    # 2. Smooth the steering (Mix 80% old steering, 20% new random)
    # This creates "momentum" in the steering wheel so it drives curves.
    new_steering = action[0]
    smoothed_steering = (0.8 * previous_steering) + (0.2 * new_steering)
    action[0] = np.clip(smoothed_steering, -1.0, 1.0)
    
    # 3. Force Gas (keep moving!)
    action[1] = np.random.uniform(0.2, 1.0) 
    action[2] = 0.0 # Disable break to encourage speed
    
    return action

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 1000
    # try:
    env = gym.make("CarRacing-v2", render_mode=None)
    # except:
    #     return

    states, actions, next_states = [], [], []
    
    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed + episode)
        obs = process_frame(obs)
        
        # Zoom-in wait
        for _ in range(50):
            obs, _, _, _, _ = env.step([0, 0, 0])
            obs = process_frame(obs)

        current_steering = 0.0
        
        for step in range(MAX_STEPS):
            # Use the better driving logic
            action = get_correlated_action(env, current_steering)
            current_steering = action[0] # Update for next frame
            
            next_obs_raw, _, terminated, truncated, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)

            obs = next_obs
            if done: break
    
    # Save
    filename = os.path.join(DATA_DIR, f"chunk_{worker_id}.npz")
    np.savez_compressed(filename, states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
    env.close()
    print(f"Worker {worker_id} Done.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Launching {NUM_WORKERS} threads on 5950X...")
    mp.set_start_method("spawn") # Safer for Gym on Windows/Linux
    pool = mp.Pool(NUM_WORKERS)
    pool.map(worker_func, range(NUM_WORKERS))
    pool.close()
    pool.join()