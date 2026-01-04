import warnings
import os

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import gymnasium as gym
import numpy as np
import multiprocessing as mp
import cv2
import math
import time
from tqdm import tqdm

NUM_WORKERS = 16
EPISODES_PER_WORKER = 50
MAX_STEPS = 1000
DATA_DIR = "data_recovery"
IMG_SIZE = 64
TARGET_SPEED = 20.0

def process_frame(frame):
    if frame is None: return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :] 
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

class RecoveryDriver:
    def __init__(self):
        self.prev_idx = 0; self.has_reset = False
        self.mistake_timer = 0; self.mistake_type = None; self.mistake_duration = 0

    def get_action(self, env):
        car = env.unwrapped.car if hasattr(env.unwrapped, 'car') else env.car
        track = env.unwrapped.track if hasattr(env.unwrapped, 'track') else env.track
        if car is None: return np.array([0,0,0], dtype=np.float32)

        car_pos = np.array(car.hull.position)
        car_vel = np.array(car.hull.linearVelocity)
        speed = np.linalg.norm(car_vel)
        car_angle = car.hull.angle
        track_coords = np.array([t[2:4] for t in track])
        
        if not self.has_reset:
            self.prev_idx = np.argmin(np.linalg.norm(track_coords - car_pos, axis=1))
            self.has_reset = True
        else:
            best_dist = float('inf')
            for i in range(30):
                idx = (self.prev_idx + i) % len(track_coords)
                d = np.linalg.norm(track_coords[idx] - car_pos)
                if d < best_dist: best_dist, self.prev_idx = d, idx

        L = 6.0 + (speed * 0.5)
        target_idx = self.prev_idx
        for i in range(50):
            idx = (self.prev_idx + i) % len(track_coords)
            if np.linalg.norm(track_coords[idx] - car_pos) > L: target_idx = idx; break
        
        target_pt = track_coords[target_idx]
        dx, dy = target_pt[0] - car_pos[0], target_pt[1] - car_pos[1]
        rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
        expert_steer = np.clip((2.0 * rx) / (L*L) * 3.0, -1.0, 1.0)
        
        if self.mistake_timer == 0:
            if np.random.rand() < 0.04: 
                self.mistake_timer = 1
                self.mistake_type = np.random.choice(['under', 'over', 'late'], p=[0.4, 0.4, 0.2])
                self.mistake_duration = np.random.randint(12, 25)
        
        if self.mistake_timer > 0:
            self.mistake_timer += 1
            if self.mistake_type == 'under': final_steer = expert_steer * 0.3
            elif self.mistake_type == 'over': final_steer = np.clip(expert_steer * 1.7, -1, 1)
            elif self.mistake_type == 'late': final_steer = expert_steer * 0.2 if self.mistake_timer < 8 else np.clip(expert_steer * 1.5, -1, 1)
            else: final_steer = expert_steer
            
            if self.mistake_timer > self.mistake_duration: self.mistake_timer = 0; self.mistake_type = None
            gas, brake = 0.4, 0.0
        else:
            final_steer = expert_steer
            gas = 0.5 if speed < TARGET_SPEED else 0.2
            brake = 0.0
        return np.array([final_steer, gas, brake], dtype=np.float32)

    def reset(self):
        self.prev_idx = 0; self.has_reset = False; self.mistake_timer = 0

def worker_func(worker_id):
    warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
    seed = int(time.time()) + worker_id * 10000
    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    driver = RecoveryDriver()
    states, actions, next_states = [], [], []
    
    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed+episode)
        driver.reset()
        for _ in range(50): env.step(np.array([0,0,0]))
        obs = process_frame(obs)
        
        for step in range(MAX_STEPS):
            try: action = driver.get_action(env)
            except: action = np.array([0,0,0], dtype=np.float32)
            
            next_obs_raw, _, term, trunc, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            states.append(obs); actions.append(action); next_states.append(next_obs)
            obs = next_obs
            if term or trunc: break

    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(DATA_DIR, f"recovery_chunk_{worker_id}.npz"), states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
    env.close()
    return worker_id

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    print(f"Launching {NUM_WORKERS} Recovery Drivers...")
    with mp.Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(worker_func, range(NUM_WORKERS)), total=NUM_WORKERS, desc="Workers Completed"))