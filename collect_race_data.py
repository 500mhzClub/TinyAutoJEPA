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

# --- CONFIGURATION ---
VISUAL_VERIFY = False    
NUM_WORKERS = 16
EPISODES_PER_WORKER = 20
MAX_STEPS = 3000
DATA_DIR = "data_race"
IMG_SIZE = 64

def process_frame(frame):
    if frame is None: return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :] 
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

class PurePursuitDriver:
    def __init__(self):
        self.prev_idx = 0
        self.has_reset = False
        
    def get_action(self, env):
        car = env.unwrapped.car if hasattr(env.unwrapped, 'car') else env.car
        track = env.unwrapped.track if hasattr(env.unwrapped, 'track') else env.track
        if car is None: return np.array([0,0,0], dtype=np.float32), 0, (0,0), (0,0)

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
                dist = np.linalg.norm(track_coords[idx] - car_pos)
                if dist < best_dist: best_dist, self.prev_idx = dist, idx

        L = 6.0 + (speed * 0.5) 
        target_idx = self.prev_idx
        for i in range(50): 
            idx = (self.prev_idx + i) % len(track_coords)
            if np.linalg.norm(track_coords[idx] - car_pos) > L: target_idx = idx; break
        
        target_pt = track_coords[target_idx]
        dx, dy = target_pt[0] - car_pos[0], target_pt[1] - car_pos[1]
        rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
        curvature = (2.0 * rx) / (L * L)
        steer = np.clip(curvature * 3.0, -1.0, 1.0) 

        future_curve = self._estimate_curvature(track_coords, self.prev_idx, int(speed * 0.8))
        target_speed = max(15.0, 40.0 - (future_curve * 30.0))
        speed_error = target_speed - speed
        
        if speed_error > 2: gas, brake = 0.6, 0.0
        elif speed_error < -5: gas, brake = 0.0, min(0.5, abs(speed_error)/15.0)
        else: gas, brake = 0.3, 0.0

        if abs(steer) > 0.6 and speed > 20: gas, brake = 0.0, 0.2
        return np.array([steer, gas, brake], dtype=np.float32), speed, track_coords[self.prev_idx], target_pt

    def _estimate_curvature(self, track_coords, start_idx, lookahead):
        if lookahead < 2: return 0.0
        angles = []
        L = len(track_coords)
        for i in range(min(lookahead, 40)):
            idx, next_idx = (start_idx + i) % L, (start_idx + i + 1) % L
            dx, dy = track_coords[next_idx][0] - track_coords[idx][0], track_coords[next_idx][1] - track_coords[idx][1]
            angles.append(math.atan2(dy, dx))
        changes = [abs(angles[i+1]-angles[i]) if abs(angles[i+1]-angles[i]) < math.pi else 2*math.pi - abs(angles[i+1]-angles[i]) for i in range(len(angles)-1)]
        return sum(changes) / len(changes) if changes else 0.0

    def reset(self):
        self.prev_idx = 0
        self.has_reset = False

def run_visual_verification():
    print("--- VISUAL VERIFICATION MODE ---")
    env = gym.make("CarRacing-v3", render_mode="human", max_episode_steps=MAX_STEPS)
    driver = PurePursuitDriver()
    obs, _ = env.reset(); driver.reset()
    for _ in range(50): env.step(np.array([0,0,0]))
    while True:
        try: action, _, _, _ = driver.get_action(env)
        except: action = np.array([0,0,0], dtype=np.float32)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(); driver.reset()
            for _ in range(50): env.step(np.array([0,0,0]))

def worker_func(worker_id):
    warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
    seed = int(time.time()) + worker_id * 10000
    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    driver = PurePursuitDriver()
    
    states, actions, next_states = [], [], []
    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed+episode)
        driver.reset()
        for _ in range(50): env.step(np.array([0,0,0]))
        obs = process_frame(obs)
        
        for step in range(MAX_STEPS):
            try: action, _, _, _ = driver.get_action(env)
            except: action = np.array([0,0,0], dtype=np.float32)
            
            next_obs_raw, reward, term, trunc, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            states.append(obs); actions.append(action); next_states.append(next_obs)
            obs = next_obs
            if reward < -5.0 or term or trunc: break
            
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(DATA_DIR, f"race_chunk_{worker_id}.npz"), states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
    env.close()
    return worker_id

if __name__ == "__main__":
    if VISUAL_VERIFY:
        run_visual_verification()
    else:
        print(f"Launching {NUM_WORKERS} Expert Drivers...")
        mp.set_start_method("spawn", force=True)
        with mp.Pool(NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(worker_func, range(NUM_WORKERS)), total=NUM_WORKERS, desc="Workers Completed"))