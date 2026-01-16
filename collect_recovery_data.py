import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time
import warnings

# --- CONFIGURATION ---
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(min(32, mp.cpu_count()))))
EPISODES_PER_WORKER = int(os.getenv("EPISODES_PER_WORKER", "50"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "1000"))
DATA_DIR = os.getenv("DATA_DIR", "data_recovery")
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))

# Controller
BASE_LOOKAHEAD = 6.0
STEER_GAIN = 5.0
TARGET_SPEED = 35.0
MISTAKE_CHANCE = 0.02 

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None: return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :] 
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def robust_pure_pursuit(env, prev_idx, direction_sign):
    u = env.unwrapped
    car = getattr(u, "car", None)
    track = getattr(u, "track", None)
    if car is None or track is None or len(track) == 0:
        return np.zeros(3), prev_idx, direction_sign, False

    car_pos = np.array(car.hull.position, dtype=np.float32)
    car_angle = float(car.hull.angle)
    speed = np.linalg.norm(car.hull.linearVelocity)
    track_coords = np.array([t[2:4] for t in track], dtype=np.float32)
    track_len = len(track_coords)

    if prev_idx is None:
        dists = np.linalg.norm(track_coords - car_pos[None, :], axis=1)
        closest_idx = int(np.argmin(dists))
        next_idx = (closest_idx + 1) % track_len
        vec_forward = track_coords[next_idx] - track_coords[closest_idx]
        car_heading = np.array([math.cos(car_angle), math.sin(car_angle)])
        direction_sign = 1 if np.dot(vec_forward, car_heading) > 0 else -1
    else:
        search_r = 20
        indices = np.arange(prev_idx, prev_idx + search_r) % track_len if direction_sign == 1 else np.arange(prev_idx, prev_idx - search_r, -1) % track_len
        dists = np.linalg.norm(track_coords[indices.astype(int)] - car_pos[None, :], axis=1)
        closest_idx = int(indices[np.argmin(dists)])

    if np.linalg.norm(track_coords[closest_idx] - car_pos) > 15.0: 
        return np.zeros(3), closest_idx, direction_sign, True 

    L = BASE_LOOKAHEAD + speed * 0.4
    target_idx = closest_idx
    for i in range(1, 50):
        idx = (closest_idx + i * direction_sign) % track_len
        if np.linalg.norm(track_coords[idx] - car_pos) > L:
            target_idx = idx
            break

    target = track_coords[target_idx]
    dx, dy = target[0] - car_pos[0], target[1] - car_pos[1]
    rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
    ry = dx * math.sin(-car_angle) + dy * math.cos(-car_angle)
    curvature = (2.0 * ry) / (L * L)
    steer = np.clip(curvature * STEER_GAIN, -1.0, 1.0)
    
    gas, brake = (0.5, 0.0) if speed < TARGET_SPEED else (0.0, 0.1)
    if abs(steer) > 0.5 and speed < TARGET_SPEED: gas = 0.1

    return np.array([steer, gas, brake], dtype=np.float32), closest_idx, direction_sign, False

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 2000
    rng = np.random.RandomState(seed)
    try: env = gym.make("CarRacing-v3", render_mode=None)
    except: env = gym.make("CarRacing-v2", render_mode=None)

    states, actions = [], []
    success_count = 0
    fail_count = 0
    
    for ep in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=rng.randint(0, 100000))
        
        # FIXED: Warmup using numpy array
        for _ in range(40): 
            env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        prev_idx = None
        dir_sign = 1
        mistake_timer = 0
        mistake_action = np.zeros(3, dtype=np.float32)
        
        ep_states, ep_actions = [], []
        
        for _ in range(MAX_STEPS):
            s = process_frame(obs)
            
            expert_action, prev_idx, dir_sign, off_track = robust_pure_pursuit(env, prev_idx, dir_sign)
            
            if off_track:
                ep_states, ep_actions = [], []
                fail_count += 1
                break 

            if mistake_timer > 0:
                final_action = mistake_action
                mistake_timer -= 1
            else:
                if rng.rand() < MISTAKE_CHANCE:
                    mistake_timer = rng.randint(5, 20) 
                    mistake_type = rng.choice(['oversteer', 'understeer', 'noise'])
                    
                    # --- FIX START: Ensure numpy array creation ---
                    if mistake_type == 'oversteer':
                        raw_action = [np.clip(expert_action[0]*2.0, -1, 1), 0.3, 0]
                    elif mistake_type == 'understeer':
                        raw_action = [expert_action[0]*0.1, 0.3, 0]
                    else:
                        raw_action = [rng.uniform(-1, 1), 0.3, 0]
                    
                    mistake_action = np.array(raw_action, dtype=np.float32)
                    final_action = mistake_action
                    # --- FIX END ---
                else:
                    final_action = expert_action

            obs, _, term, trunc, _ = env.step(final_action)
            ep_states.append(s)
            ep_actions.append(final_action)
            
            if term or trunc: break
        
        if len(ep_states) > 100:
            states.extend(ep_states)
            actions.extend(ep_actions)
            success_count += 1
            
        if (ep+1) % 10 == 0:
             print(f"Worker {worker_id}: Ep {ep+1}/{EPISODES_PER_WORKER} | Success: {success_count} | Fail (OffTrack): {fail_count}")

    env.close()
    if len(states) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        fname = os.path.join(DATA_DIR, f"recovery_{worker_id}.npz")
        np.savez(fname, states=np.array(states), actions=np.array(actions))
        print(f"Worker {worker_id}: DONE. Saved {len(states)} RECOVERY frames -> {fname}")
    else:
        print(f"Worker {worker_id}: FAILED. No valid data collected.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    with mp.Pool(NUM_WORKERS) as p: p.map(worker_func, range(NUM_WORKERS))