import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import time
import warnings
import argparse
from stable_baselines3 import PPO

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Generate CarRacing-v3 Data")
parser.add_argument("--mode", type=str, required=True, choices=["random", "expert", "recover"], help="Generation mode")
parser.add_argument("--watch", action="store_true", help="Watch visually (forces 1 worker)")
parser.add_argument("--workers", type=int, default=min(16, mp.cpu_count()), help="Number of parallel workers")
parser.add_argument("--episodes", type=int, default=80, help="Episodes per worker")
parser.add_argument("--steps", type=int, default=600, help="Max steps per episode")
parser.add_argument("--model", type=str, default="ppo_carracing_v3_expert.zip", help="Path to local expert model")
args = parser.parse_args()

# --- CONFIGURATION ---
IMG_SIZE = 64
DATA_DIR = f"data_{args.mode}"

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :] 
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def get_random_action(prev_steer):
    noise = np.random.uniform(-1.0, 1.0)
    steer = np.clip(0.2 * prev_steer + 0.8 * noise, -1.0, 1.0)
    mode = np.random.choice(["accelerate", "brake", "coast"], p=[0.60, 0.10, 0.30])
    if mode == "accelerate": gas, brake = np.random.uniform(0.5, 1.0), 0.0
    elif mode == "brake":    gas, brake = 0.0, np.random.uniform(0.2, 0.9)
    else:                    gas, brake = np.random.uniform(0.0, 0.2), 0.0
    return np.array([steer, gas, brake], dtype=np.float32), steer

def run_session(worker_id, mode, model_path, visualize):
    render_mode = "human" if visualize else None
    
    # Force v3 environment
    try:
        env = gym.make("CarRacing-v3", render_mode=render_mode, max_episode_steps=args.steps)
    except Exception as e:
        print(f"Error making env: {e}")
        return

    model = None
    if mode in ["expert", "recover"]:
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found! Train it first.")
            return
        model = PPO.load(model_path, device="cpu")

    seed = int(time.time()) + worker_id * 10000
    rng = np.random.RandomState(seed)
    states, actions, next_states = [], [], []

    ep_count = 0
    target_eps = args.episodes if not visualize else 999999

    try:
        while ep_count < target_eps:
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            prev_steer = 0.0
            sabotage_timer = 0
            sabotage_action = np.zeros(3)
            curr_frames = 0
            
            while True:
                s = process_frame(obs)
                action = np.zeros(3, dtype=np.float32)

                if mode == "random":
                    action, prev_steer = get_random_action(prev_steer)
                
                elif mode == "expert":
                    action, _ = model.predict(obs, deterministic=True)
                
                elif mode == "recover":
                    if sabotage_timer > 0:
                        action = sabotage_action
                        sabotage_timer -= 1
                        if visualize: print(f"\r!!! SABOTAGE {sabotage_timer:02d} !!!", end="")
                    else:
                        if rng.rand() < 0.01: # 1% chance to sabotage
                            sabotage_timer = rng.randint(5, 20)
                            steer = -1.0 if rng.rand() < 0.5 else 1.0
                            action = np.array([steer, 0.6, 0.0], dtype=np.float32)
                            sabotage_action = action
                        else:
                            action, _ = model.predict(obs, deterministic=True)
                            if visualize: print(f"\r   Expert Drive     ", end="")

                obs, _, terminated, truncated, _ = env.step(action)
                ns = process_frame(obs)

                if not visualize:
                    states.append(s)
                    actions.append(action)
                    next_states.append(ns)
                
                curr_frames += 1
                if terminated or truncated or curr_frames >= args.steps:
                    break
            
            ep_count += 1
            if visualize: print(f"\nEpisode {ep_count} finished.")

    finally:
        env.close()

    if not visualize and len(states) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        fname = os.path.join(DATA_DIR, f"{mode}_chunk_{worker_id}.npz")
        np.savez(fname, states=np.array(states), actions=np.array(actions), next_states=np.array(next_states))
        print(f"[Worker {worker_id}] Saved {len(states)} frames.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if args.watch:
        run_session(0, args.mode, args.model, True)
    else:
        task_args = [(i, args.mode, args.model, False) for i in range(args.workers)]
        with mp.Pool(args.workers) as pool:
            pool.starmap(run_session, task_args)