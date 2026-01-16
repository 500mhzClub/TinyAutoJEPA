import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
import numpy as np
import cv2
import os

# --- FIX 1: Silence the Qt/Wayland warning ---
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- CONFIG ---
MODELS_TO_TEST = [
    ("Perfected", "ppo_carracing_v3_perfected.zip") 
]
N_EPISODES = 3
VIDEO_FOLDER = "./videos/"
RENDER_LIVE = True 

def evaluate(model_name, model_path):
    print(f"\n--- Testing {model_name} ---")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Could not load {model_path}: {e}")
        return

    rewards = []
    
    # --- FIX 2: Recreate env inside the loop ---
    # This ensures every run starts at 'step 0' so the video recorder 
    # definitely triggers every single time.
    for i in range(N_EPISODES):
        
        # 1. Setup Env
        vec_env = make_vec_env("CarRacing-v3", n_envs=1, env_kwargs={"render_mode": "rgb_array"})
        env = VecFrameStack(vec_env, n_stack=4)
        
        # 2. Setup Recorder (Unique name for each run)
        video_name = f"{model_name.lower()}_run_{i+1}"
        env = VecVideoRecorder(
            env,
            VIDEO_FOLDER,
            record_video_trigger=lambda x: x == 0, # Triggers immediately because we just reset the env
            video_length=2000,
            name_prefix=video_name
        )
        
        obs = env.reset()
        done = False
        total_reward = 0
        
        print(f"Episode {i+1} started... (Check the popup window)")
        
        while not done:
            # Deterministic=False allows the "texture"/jitter the agent likes
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            
            # --- LIVE PREVIEW ---
            if RENDER_LIVE:
                img = env.render() 
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img_bgr, f"Score: {total_reward:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f"Live Preview - {model_name}", img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    done = True

        rewards.append(total_reward)
        print(f"Run {i+1}: {total_reward:.1f}")
        
        # --- FIX 3: Just close the env ---
        # This automatically saves the video. No need for close_video_recorder().
        env.close()

    avg = np.mean(rewards)
    std = np.std(rewards)
    print(f">>> Average: {avg:.1f} +/- {std:.1f}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    for name, path in MODELS_TO_TEST:
        evaluate(name, path)