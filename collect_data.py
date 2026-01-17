import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import time
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Generate CarRacing-v3 JEPA Data")
parser.add_argument("--mode", type=str, required=True, choices=["random", "expert", "recover"], help="Generation mode")
parser.add_argument("--visualize", action="store_true", help="Watch the generation (forces 1 worker)")
parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument("--episodes", type=int, default=50, help="Episodes per worker")
parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
parser.add_argument("--model", type=str, default="ppo_carracing_v3_perfected.zip", help="Path to Expert Model")
args = parser.parse_args()

# --- CONFIGURATION ---
IMG_SIZE = 64
DATA_DIR = f"data_{args.mode}"

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Resize and crop raw frame for JEPA (64x64)"""
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def get_random_action(prev_steer):
    """Correlated noise for smoother random driving"""
    noise = np.random.uniform(-1.0, 1.0)
    steer = np.clip(0.4 * prev_steer + 0.6 * noise, -1.0, 1.0)
    
    mode = np.random.choice(["accel", "brake", "coast"], p=[0.7, 0.1, 0.2])
    
    if mode == "accel": 
        gas, brake = np.random.uniform(0.3, 1.0), 0.0
    elif mode == "brake":    
        gas, brake = 0.0, np.random.uniform(0.1, 0.8)
    else:                    
        gas, brake = 0.0, 0.0
        
    return np.array([steer, gas, brake], dtype=np.float32), steer

def run_session(worker_id, mode, model_path, visualize):
    """Worker process to generate data"""
    
    def make_env():
        return gym.make("CarRacing-v3", render_mode="rgb_array", max_episode_steps=args.steps)

    vec_env = DummyVecEnv([make_env])
    env = VecFrameStack(vec_env, n_stack=4)
    
    model = None
    if mode in ["expert", "recover"]:
        try:
            model = PPO.load(model_path, device="cpu")
            if worker_id == 0: print(f"Worker {worker_id}: Loaded {model_path}")
        except:
            print(f"Worker {worker_id}: Error loading model! Check path.")
            return

    states_buffer = []      
    actions_buffer = []     
    next_states_buffer = [] 
    
    seed = int(time.time()) + worker_id * 999
    rng = np.random.RandomState(seed)
    
    ep_count = 0
    target_eps = args.episodes if not visualize else 100000

    print(f"Worker {worker_id}: Starting...")

    try:
        while ep_count < target_eps:
            obs = env.reset()
            
            # Capture raw frame for dataset
            raw_frame = env.envs[0].render()
            curr_state_img = process_frame(raw_frame)
            
            prev_steer = 0.0
            
            sabotage_steps_left = 0
            recovery_cooldown = 0
            # FIX: Initialize as (1, 3) to match VecEnv expectations
            sabotage_action = np.zeros((1, 3), dtype=np.float32)

            for step in range(args.steps):
                
                # --- ACTION SELECTION ---
                if mode == "random":
                    raw_action, prev_steer = get_random_action(prev_steer)
                    # FIX: Wrap random action to shape (1, 3)
                    env_action = np.array([raw_action], dtype=np.float32)

                elif mode == "expert":
                    # PPO predict returns (1, 3) naturally. No wrapping needed.
                    env_action, _ = model.predict(obs, deterministic=True)

                elif mode == "recover":
                    if sabotage_steps_left > 0:
                        env_action = sabotage_action
                        sabotage_steps_left -= 1
                        if visualize: print(f"\r[Worker {worker_id}] SABOTAGE! {sabotage_steps_left}   ", end="")
                    else:
                        env_action, _ = model.predict(obs, deterministic=True)
                        
                        if recovery_cooldown > 0:
                            recovery_cooldown -= 1
                        # TUNED: Less frequent (0.5%), shorter burst, less throttle
                        elif rng.rand() < 0.005: 
                            sabotage_steps_left = rng.randint(2, 6) # Only 2-6 frames (quick jerk)
                            recovery_cooldown = 200 # Wait longer before next attack
                            
                            force_steer = 1.0 if rng.rand() > 0.5 else -1.0
                            force_gas = np.random.uniform(0.2, 0.5) # Less gas so it doesn't fly off
                            
                            # FIX: Define sabotage as (1, 3)
                            sabotage_action = np.array([[force_steer, force_gas, 0.0]], dtype=np.float32)

                # --- STEP ---
                # Pass the correctly shaped (1, 3) action directly
                obs, _, dones, infos = env.step(env_action)
                
                next_raw_frame = env.envs[0].render()
                next_state_img = process_frame(next_raw_frame)

                # --- VISUALIZATION ---
                if visualize:
                    disp = cv2.resize(next_raw_frame, (400, 300))
                    status = "EXPERT"
                    if mode == "recover":
                        status = "SABOTAGE" if sabotage_steps_left > 0 else "RECOVERING"
                    
                    cv2.putText(disp, f"Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(f"Worker {worker_id}", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        env.close()
                        return

                # --- SAVE TO BUFFER ---
                # We save the flat action (3,) to disk, not the nested (1,3)
                flat_action = env_action[0]

                states_buffer.append(curr_state_img)
                actions_buffer.append(flat_action)
                next_states_buffer.append(next_state_img)
                
                curr_state_img = next_state_img
                
                if dones[0]:
                    break
            
            ep_count += 1
            if worker_id == 0: print(f"Worker {worker_id}: Episode {ep_count}/{target_eps} done.")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        env.close()
        if visualize: cv2.destroyAllWindows()

    if len(states_buffer) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = os.path.join(DATA_DIR, f"{mode}_w{worker_id}_{int(time.time())}.npz")
        
        print(f"Worker {worker_id}: Saving {len(states_buffer)} frames to {filename}...")
        np.savez_compressed(
            filename,
            states=np.array(states_buffer, dtype=np.uint8),
            actions=np.array(actions_buffer, dtype=np.float32),
            next_states=np.array(next_states_buffer, dtype=np.uint8)
        )
        print(f"Worker {worker_id}: Save Complete.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    print(f"--- Generating {args.mode.upper()} Data ---")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Total Target: {args.workers * args.episodes * args.steps} frames (approx)")
    
    if args.visualize:
        run_session(0, args.mode, args.model, True)
    else:
        with mp.Pool(args.workers) as pool:
            pool.starmap(run_session, [(i, args.mode, args.model, False) for i in range(args.workers)])