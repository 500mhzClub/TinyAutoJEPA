import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time

NUM_WORKERS = min(16, mp.cpu_count())
EPISODES_PER_WORKER = 30
MAX_STEPS = 800
DATA_DIR = "data_edge"
IMG_SIZE = 64

def process_frame(frame):
    if frame is None: return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def detect_black_proximity(obs):
    # Check bottom center for black pixels
    roi = obs[60:84, 30:66, :] 
    black_mask = (roi[:, :, 0] < 30) & (roi[:, :, 1] < 30) & (roi[:, :, 2] < 30)
    return np.sum(black_mask)

class EdgeExplorationDriver:
    def __init__(self):
        self.prev_idx = 0
        self.has_reset = False
        self.edge_bias = 0.0
        self.bias_timer = 0
        
    def get_action(self, env, obs_raw):
        car = env.unwrapped.car if hasattr(env.unwrapped, 'car') else env.car
        track = env.unwrapped.track if hasattr(env.unwrapped, 'track') else env.track
        if car is None: return np.array([0,0,0], dtype=np.float32)

        car_pos = np.array(car.hull.position)
        car_vel = np.array(car.hull.linearVelocity)
        speed = np.linalg.norm(car_vel)
        car_angle = car.hull.angle
        track_coords = np.array([t[2:4] for t in track])
        
        # --- Pure Pursuit to find baseline expert steer ---
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

        # --- EDGE BIAS ---
        black_prox = detect_black_proximity(obs_raw)
        self.bias_timer += 1
        
        if self.bias_timer > 100:
            if self.bias_timer == 101: self.edge_bias = np.random.choice([-0.4, 0.4])
            
            # If we are safe (no black pixels), push towards edge. If unsafe, let expert take over.
            if black_prox < 100: 
                final_steer = np.clip(expert_steer + self.edge_bias, -1.0, 1.0)
                gas = 0.5
            else:
                final_steer = expert_steer
                gas = 0.3
                
            if self.bias_timer > 125: self.bias_timer = 0
        else:
            final_steer = expert_steer
            gas = 0.5 if speed < 25 else 0.3

        return np.array([final_steer, gas, 0.0], dtype=np.float32)

    def reset(self):
        self.prev_idx = 0
        self.has_reset = False
        self.bias_timer = 0

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 10000
    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    driver = EdgeExplorationDriver()
    states, actions, next_states = [], [], []
    
    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed+episode)
        driver.reset()
        for _ in range(50): env.step(np.array([0,0,0]))
        obs_raw = obs
        obs = process_frame(obs_raw)
        
        for step in range(MAX_STEPS):
            try: action = driver.get_action(env, obs_raw)
            except: action = np.array([0,0,0], dtype=np.float32)
            
            next_obs_raw, reward, term, trunc, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            
            # Save if near edge or turning hard
            black_prox = detect_black_proximity(obs_raw)
            if black_prox > 20 or abs(action[0]) > 0.4:
                states.append(obs)
                actions.append(action)
                next_states.append(next_obs)
            
            obs_raw = next_obs_raw
            obs = next_obs
            if term or trunc: break
            
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(DATA_DIR, f"edge_chunk_{worker_id}.npz"), states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
    env.close()
    print(f"Worker {worker_id} Done.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(NUM_WORKERS)
    pool.map(worker_func, range(NUM_WORKERS))