import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time

# --- CONFIGURATION ---
VISUAL_VERIFY = False    # Set to TRUE to watch a full lap. Set to FALSE to generate data.
NUM_WORKERS = 32         
EPISODES_PER_WORKER = 20 # Reduced count because episodes are 3x longer now
MAX_STEPS = 3000         # Increased to ~60 seconds to allow FULL LAPS
DATA_DIR = "data_race"   
IMG_SIZE = 64
TARGET_SPEED = 25.0      # Increased slightly for better race data

def process_frame(frame):
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :] 
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return frame.astype(np.uint8)

class PurePursuitDriver:
    def __init__(self):
        self.prev_idx = 0
        self.has_reset = False
        
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
            
        if car is None:
            return np.array([0,0,0], dtype=np.float32), 0, (0,0), (0,0)

        car_pos = np.array(car.hull.position)
        car_vel = np.array(car.hull.linearVelocity)
        speed = np.linalg.norm(car_vel)
        car_angle = car.hull.angle
        
        track_coords = np.array([t[2:4] for t in track])
        track_len = len(track_coords)

        # 2. Strict Forward Search
        if not self.has_reset:
            dists = np.linalg.norm(track_coords - car_pos, axis=1)
            closest_idx = np.argmin(dists)
            self.has_reset = True
        else:
            # Forward Search Window (0 to +30 tiles)
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
        closest_pt = track_coords[closest_idx]

        # 3. Pure Pursuit Target
        # Look ahead distance scales with speed (adaptive)
        L = 6.0 + (speed * 0.5) 
        
        target_idx = closest_idx
        for i in range(50): 
            idx = (closest_idx + i) % track_len
            dist = np.linalg.norm(track_coords[idx] - car_pos)
            if dist > L:
                target_idx = idx
                break
        
        target_pt = track_coords[target_idx]

        # 4. Calculate Curvature
        dx = target_pt[0] - car_pos[0]
        dy = target_pt[1] - car_pos[1]
        
        rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
        # ry = dx * math.sin(-car_angle) + dy * math.cos(-car_angle)
        
        curvature = (2.0 * rx) / (L * L)
        steer = np.clip(curvature * 3.0, -1.0, 1.0) 

        # 5. Speed Control
        gas, brake = 0.0, 0.0
        if speed < TARGET_SPEED: gas = 0.4
        elif speed > TARGET_SPEED + 5: brake = 0.1
        
        # Corner safety
        if abs(steer) > 0.5:
            gas = 0.0
            if speed > 15: brake = 0.1

        return np.array([steer, gas, brake], dtype=np.float32), speed, closest_pt, target_pt

    def reset(self):
        self.prev_idx = 0
        self.has_reset = False

def run_visual_verification():
    print("--- VISUAL DEBUG MODE ---")
    print("Press CTRL+C to stop.")
    
    # max_episode_steps=3000 ensures the gym environment doesn't kill us early
    env = gym.make("CarRacing-v3", render_mode="rgb_array", max_episode_steps=3000)
    driver = PurePursuitDriver()
    
    obs, _ = env.reset()
    driver.reset()
    
    for _ in range(50): env.step(np.array([0,0,0]))
    
    step = 0
    while True:
        action, speed, closest_pt, target_pt = driver.get_action(env)
        obs, _, terminated, truncated, _ = env.step(action)
        
        debug_frame = env.render() 
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
        
        cv2.putText(debug_frame, f"Steps: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Debug View", debug_frame)
        cv2.waitKey(1) 
        step += 1

        if terminated or truncated:
            print("Lap Finished/Truncated. Resetting...")
            obs, _ = env.reset()
            driver.reset()
            step = 0
            for _ in range(50): env.step(np.array([0,0,0]))

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 10000
    # Increase max steps here too
    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    driver = PurePursuitDriver()
    
    states, actions, next_states = [], [], []
    
    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed + episode)
        driver.reset()
        obs = process_frame(obs)
        
        for _ in range(50): env.step(np.array([0, 0, 0]))
            
        for step in range(MAX_STEPS):
            try:
                action, _, _, _ = driver.get_action(env)
            except:
                action = np.array([0,0,0], dtype=np.float32)

            next_obs_raw, reward, terminated, truncated, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            obs = next_obs
            
            if reward < -5.0: break
            if done: break
            
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        
    filename = os.path.join(DATA_DIR, f"race_chunk_{worker_id}.npz")
    np.savez_compressed(filename, states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
    env.close()
    print(f"Worker {worker_id} Done.")

if __name__ == "__main__":
    if VISUAL_VERIFY:
        run_visual_verification()
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Launching {NUM_WORKERS} Expert Drivers (generating ~{NUM_WORKERS*EPISODES_PER_WORKER} full laps)...")
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(NUM_WORKERS)
        pool.map(worker_func, range(NUM_WORKERS))
        pool.close()
        pool.join()
        print("Race Data Collection Complete.")