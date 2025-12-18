import gymnasium as gym
import numpy as np
import os
import math
import time

# --- Configuration ---
NUM_EPISODES = 20       # Approx 20k frames
DATA_DIR = "./data_race" 
os.makedirs(DATA_DIR, exist_ok=True)

def get_cheat_action(env):
    """
    Accesses environment internals to drive perfectly on the center line.
    """
    car = env.unwrapped.car
    track = env.unwrapped.track
    car_pos = np.array(car.hull.position)
    
    # Find closest track point
    dists = [np.linalg.norm(car_pos - np.array(t[2:4])) for t in track]
    closest_idx = np.argmin(dists)
    
    # Look Ahead (Target the track 6 tiles forward)
    target_idx = (closest_idx + 6) % len(track)
    target = np.array(track[target_idx][2:4])
    
    # Calculate Steering Angle
    angle_to_target = math.atan2(target[1] - car_pos[1], target[0] - car_pos[0])
    diff = angle_to_target - car.hull.angle
    while diff > math.pi: diff -= 2*math.pi
    while diff < -math.pi: diff += 2*math.pi
    
    steer = np.clip(diff * 10.0, -1.0, 1.0)
    
    # Gas Logic: Slow down for turns, gas on straights
    if abs(steer) > 0.5: gas, brake = 0.0, 0.0
    else: gas, brake = 0.6, 0.0
        
    return np.array([steer, gas, brake])

def collect():
    print(f"Collecting RACE data to {DATA_DIR}...")
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    
    obs_buffer, act_buffer = [], []
    file_count = 0
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        # Skip zoom-in
        for _ in range(50): obs, _, _, _, _ = env.step([0,0,0])

        done = False
        while not done:
            action = get_cheat_action(env)
            next_obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            
            obs_buffer.append(obs)
            act_buffer.append(action)
            obs = next_obs
            
            # Save every 1000 frames
            if len(obs_buffer) >= 1000:
                fname = f"{DATA_DIR}/race_{int(time.time())}_{file_count}.npz"
                np.savez_compressed(fname, states=np.array(obs_buffer), actions=np.array(act_buffer))
                print(f"Saved chunk {file_count}")
                file_count += 1
                obs_buffer, act_buffer = [], []
                
    env.close()
    print("Race Data Collection Complete.")

if __name__ == "__main__":
    collect()