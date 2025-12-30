import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time

# --- CONFIGURATION ---
NUM_WORKERS = 32         
EPISODES_PER_WORKER = 50 
MAX_STEPS = 1000         
DATA_DIR = "data_recovery"   
IMG_SIZE = 64
TARGET_SPEED = 20.0      

def process_frame(frame):
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :] 
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return frame.astype(np.uint8)

class RecoveryDriver:
    def __init__(self):
        self.prev_idx = 0
        self.has_reset = False
        self.noise_timer = 0
        self.noise_val = 0.0
        
    def get_action(self, env):
        # 1. Physics Access
        car = None
        track = None
        if hasattr(env.unwrapped, 'car'):
            car = env.unwrapped.car
            track = env.unwrapped.track
        elif hasattr(env, 'car'):
            car = env.car
            track = env.track
            
        if car is None: return np.array([0,0,0], dtype=np.float32)

        car_pos = np.array(car.hull.position)
        car_vel = np.array(car.hull.linearVelocity)
        speed = np.linalg.norm(car_vel)
        car_angle = car.hull.angle
        
        track_coords = np.array([t[2:4] for t in track])
        track_len = len(track_coords)

        # 2. Find Closest Track Point (Pure Pursuit Logic)
        if not self.has_reset:
            dists = np.linalg.norm(track_coords - car_pos, axis=1)
            closest_idx = np.argmin(dists)
            self.has_reset = True
        else:
            search_len = 30
            best_dist = float('inf')
            best_idx = self.prev_idx
            for i in range(search_len):
                idx = (self.prev_idx + i) % track_len
                dist = np.linalg.norm(track_coords[idx] - car_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            closest_idx = best_idx
        self.prev_idx = closest_idx

        # 3. Calculate Perfect Expert Steering
        L = 6.0 + (speed * 0.5) 
        target_idx = closest_idx
        for i in range(50): 
            idx = (closest_idx + i) % track_len
            if np.linalg.norm(track_coords[idx] - car_pos) > L:
                target_idx = idx
                break
        
        target_pt = track_coords[target_idx]
        dx = target_pt[0] - car_pos[0]
        dy = target_pt[1] - car_pos[1]
        rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
        curvature = (2.0 * rx) / (L * L)
        expert_steer = np.clip(curvature * 3.0, -1.0, 1.0) 

        # --- THE MAGIC SAUCE: INJECT NOISE ---
        # Every 100 steps, force the car to drift for 20 steps
        self.noise_timer += 1
        
        if self.noise_timer > 100:
            # Start a drift
            if self.noise_timer == 101:
                # Pick a direction AWAY from the turn (Understeer) or Random
                self.noise_val = np.random.uniform(-0.5, 0.5)
            
            # Override Expert
            final_steer = self.noise_val
            
            # End drift after 20 steps
            if self.noise_timer > 120:
                self.noise_timer = 0 # Reset, let Expert recover
        else:
            # Pure Expert Recovery
            final_steer = expert_steer

        # Speed Control
        gas = 0.4 if speed < TARGET_SPEED else 0.0
        
        return np.array([final_steer, gas, 0.0], dtype=np.float32)

    def reset(self):
        self.prev_idx = 0
        self.has_reset = False
        self.noise_timer = 0

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 10000
    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    driver = RecoveryDriver()
    
    states, actions = [], []
    
    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed + episode)
        driver.reset()
        obs = process_frame(obs)
        
        # Zoom in
        for _ in range(50): env.step(np.array([0, 0, 0]))
            
        for step in range(MAX_STEPS):
            try:
                action = driver.get_action(env)
            except:
                action = np.array([0,0,0], dtype=np.float32)

            next_obs_raw, _, terminated, truncated, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            
            states.append(obs)
            actions.append(action)
            obs = next_obs
            
            if terminated or truncated: break
            
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"recovery_chunk_{worker_id}.npz")
    np.savez_compressed(filename, states=np.array(states), actions=np.array(actions))
    env.close()
    print(f"Worker {worker_id} Done.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Launching {NUM_WORKERS} Drunk Drivers...")
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(NUM_WORKERS)
    pool.map(worker_func, range(NUM_WORKERS))
    pool.close()
    pool.join()