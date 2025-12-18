import gymnasium as gym
import numpy as np
import os
import math
import time

# --- Configuration ---
NUM_EPISODES = 20       # 20 Episodes = ~20,000 frames (plenty for fine-tuning)
chunk_size = 1000       # Save every 1000 frames
data_dir = "./data_race" # Separate folder for "Pro" data

os.makedirs(data_dir, exist_ok=True)

def get_cheat_action(env):
    """
    Accesses the environment's internal state to drive perfectly.
    This creates 'Expert Data'.
    """
    car = env.unwrapped.car
    track = env.unwrapped.track
    
    # 1. Get Car Position & Angle
    car_x, car_y = car.hull.position
    car_angle = car.hull.angle
    
    # 2. Find the closest track tile
    # (The track is a list of (x, y) coordinates)
    closest_idx = 0
    min_dist = float('inf')
    
    for i, point in enumerate(track):
        x, y = point[2], point[3] # Track center coordinates
        dist = (x - car_x)**2 + (y - car_y)**2
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            
    # 3. Look Ahead (Target the track 5 tiles forward)
    # This creates smooth racing lines
    target_idx = (closest_idx + 5) % len(track)
    target_x, target_y = track[target_idx][2], track[target_idx][3]
    
    # 4. Calculate Steering Angle
    # Vector to target
    vec_x = target_x - car_x
    vec_y = target_y - car_y
    
    # Angle to target
    target_angle = math.atan2(vec_y, vec_x)
    
    # Angle difference (Error)
    diff = target_angle - car_angle
    
    # Normalize to -pi to pi
    while diff > math.pi: diff -= 2*math.pi
    while diff < -math.pi: diff += 2*math.pi
    
    # 5. PID Control (P-only is fine for this)
    steer = diff * 10.0
    steer = np.clip(steer, -1.0, 1.0)
    
    # Simple Gas Logic: Slow down for turns, floor it on straights
    if abs(steer) > 0.5:
        gas = 0.1
        brake = 0.0
    else:
        gas = 0.8
        brake = 0.0
        
    return np.array([steer, gas, brake])

def collect():
    print(f"Collecting RACE data to {data_dir}...")
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    
    total_frames = 0
    file_counter = 0
    
    obs_buffer = []
    action_buffer = []
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        step = 0
        
        # Initial zoom-in frames often have no car, skip them
        for _ in range(50):
            obs, _, _, _, _ = env.step([0,0,0])

        while not done:
            # Get the Expert Action
            action = get_cheat_action(env)
            
            # Step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Save (Resize is handled in training, but we can save raw 96x96)
            # To match your existing pipeline, we save whatever Gym gives (96,96,3)
            obs_buffer.append(obs)
            action_buffer.append(action)
            
            obs = next_obs
            step += 1
            total_frames += 1
            
            # Save Chunk
            if len(obs_buffer) >= chunk_size:
                filename = os.path.join(data_dir, f"race_{int(time.time())}_{file_counter}.npz")
                
                # Convert list to numpy
                np_obs = np.array(obs_buffer, dtype=np.uint8)
                np_actions = np.array(action_buffer, dtype=np.float32)
                
                # Save compressed
                np.savez_compressed(filename, states=np_obs, actions=np_actions)
                print(f"Saved {filename} | Total Frames: {total_frames}")
                
                file_counter += 1
                obs_buffer = []
                action_buffer = []
                
        print(f"Episode {episode+1}/{NUM_EPISODES} complete.")
        
    env.close()
    print("Race Data Collection Complete.")

if __name__ == "__main__":
    collect()