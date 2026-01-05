import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time

# --- CONFIGURATION ---
VISUAL_VERIFY = False  # Set to True to watch one car drive (to verify steering fixes)
NUM_WORKERS = min(32, mp.cpu_count())
EPISODES_PER_WORKER = 20
MAX_STEPS = 3000
DATA_DIR = "data_race"
IMG_SIZE = 64

def process_frame(frame):
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

class PurePursuitDriver:
    """
    FIXED: Tuned for CarRacing-v3 tight corners.
    """
    def __init__(self):
        self.prev_idx = 0
        self.has_reset = False

    def get_action(self, env):
        car = None
        track = None
        # Handle different gym versions/wrappers
        if hasattr(env.unwrapped, 'car'):
            car = env.unwrapped.car
            track = env.unwrapped.track
        elif hasattr(env, 'car'):
            car = env.car
            track = env.track
        
        if car is None:
            return np.array([0, 0, 0], dtype=np.float32), 0
        
        car_pos = np.array(car.hull.position)
        car_vel = np.array(car.hull.linearVelocity)
        speed = np.linalg.norm(car_vel)
        car_angle = car.hull.angle
        
        track_coords = np.array([t[2:4] for t in track])
        track_len = len(track_coords)
        
        # Track following logic
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
        
        # LOOKAHEAD FIX: Shorter lookahead = more responsive steering
        # Old: 6.0 + speed * 0.5
        L = 4.0 + (speed * 0.3)
        
        target_idx = closest_idx
        for i in range(50):
            idx = (closest_idx + i) % track_len
            dist = np.linalg.norm(track_coords[idx] - car_pos)
            if dist > L:
                target_idx = idx
                break
        
        target_pt = track_coords[target_idx]
        
        # Calculate steering
        dx = target_pt[0] - car_pos[0]
        dy = target_pt[1] - car_pos[1]
        
        rx = dx * math.cos(-car_angle) - dy * math.sin(-car_angle)
        curvature = (2.0 * rx) / (L * L)
        
        # GAIN FIX: Increased from 3.0 to 8.0 to handle sharp turns
        steer = np.clip(curvature * 8.0, -1.0, 1.0)
        
        # Speed control based on upcoming curvature
        future_curvature = self._estimate_upcoming_curvature(
            track_coords, closest_idx, int(speed * 0.8)
        )
        
        base_target = 40.0
        curve_penalty = future_curvature * 30.0
        target_speed = max(15.0, base_target - curve_penalty)
        speed_error = target_speed - speed
        
        if speed_error > 2:
            gas = 0.6; brake = 0.0
        elif speed_error < -5:
            gas = 0.0; brake = min(0.5, abs(speed_error) / 15.0)
        else:
            gas = 0.3; brake = 0.0
        
        # Emergency slow down on hard turns
        if abs(steer) > 0.6 and speed > 20:
            gas = 0.0; brake = 0.2
        
        return np.array([steer, gas, brake], dtype=np.float32), speed

    def _estimate_upcoming_curvature(self, track_coords, current_idx, lookahead_tiles):
        if lookahead_tiles < 2: return 0.0
        angles = []
        track_len = len(track_coords)
        for i in range(min(lookahead_tiles, 40)):
            idx = (current_idx + i) % track_len
            next_idx = (current_idx + i + 1) % track_len
            dx = track_coords[next_idx][0] - track_coords[idx][0]
            dy = track_coords[next_idx][1] - track_coords[idx][1]
            angles.append(math.atan2(dy, dx))
        
        if len(angles) < 2: return 0.0
        angle_changes = []
        for i in range(len(angles) - 1):
            delta = abs(angles[i + 1] - angles[i])
            if delta > math.pi: delta = 2 * math.pi - delta
            angle_changes.append(delta)
        return sum(angle_changes) / len(angle_changes) if angle_changes else 0.0

    def reset(self):
        self.prev_idx = 0
        self.has_reset = False

def run_visual_verification():
    print("=== VISUAL VERIFICATION MODE ===")
    env = gym.make("CarRacing-v3", render_mode="human", max_episode_steps=3000)
    driver = PurePursuitDriver()
    obs, _ = env.reset()
    driver.reset()
    # Skip zoom
    for _ in range(50): env.step(np.array([0, 0, 0]))

    step = 0
    while True:
        action, speed = driver.get_action(env)
        obs, reward, terminated, truncated, _ = env.step(action)
        if step % 10 == 0:
            print(f"Step {step} | Speed: {speed:.1f} | Steer: {action[0]:.2f}")
        step += 1
        if terminated or truncated:
            obs, _ = env.reset()
            driver.reset()
            for _ in range(50): env.step(np.array([0, 0, 0]))

def worker_func(worker_id):
    seed = int(time.time()) + worker_id * 10000
    env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    driver = PurePursuitDriver()
    
    states, actions, next_states = [], [], []

    for episode in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=seed + episode)
        driver.reset()
        for _ in range(50): env.step(np.array([0, 0, 0]))
        
        obs = process_frame(obs)
        for step in range(MAX_STEPS):
            try:
                action, _ = driver.get_action(env)
            except:
                action = np.array([0, 0, 0], dtype=np.float32)
            
            next_obs_raw, reward, terminated, truncated, _ = env.step(action)
            next_obs = process_frame(next_obs_raw)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            obs = next_obs
            
            if reward < -5.0 or done:
                break

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        
    filename = os.path.join(DATA_DIR, f"race_chunk_{worker_id}.npz")
    np.savez_compressed(
        filename,
        states=np.array(states),
        actions=np.array(actions),
        next_states=np.array(next_states)
    )
    env.close()
    print(f"[Worker {worker_id}] Complete: {len(states)} frames")

if __name__ == "__main__":
    if VISUAL_VERIFY:
        run_visual_verification()
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"=== Expert Racing Data Collection (FIXED) ===")
        print(f"Fix: Steering gain 3.0 -> 8.0")
        print(f"Launching Workers: {NUM_WORKERS}")
        
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(NUM_WORKERS)
        pool.map(worker_func, range(NUM_WORKERS))
        pool.close()
        pool.join()
        
        import glob
        files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
        total_frames = 0
        for f in files:
            with np.load(f) as data:
                total_frames += len(data['states'])
        print(f"\n=== Complete ===")
        print(f"Total frames: {total_frames:,}")