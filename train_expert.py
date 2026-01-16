import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# --- CONFIGURATION ---
ENV_ID = "CarRacing-v3"
TOTAL_TIMESTEPS = 600_000  # 600k usually gets to ~850-900 reward
N_ENVS = 8                 # Parallel workers (speed up training)
MODEL_NAME = "ppo_carracing_v3_expert"

def main():
    print(f"--- Training {ENV_ID} Expert ---")
    print(f"Workers: {N_ENVS} | Steps: {TOTAL_TIMESTEPS}")
    
    # 1. Create Vectorized Environment
    # We use a stack of 4 frames so the agent detects speed/direction
    env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    
    # 2. Define PPO Model (Optimized Hyperparams for CarRacing)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,    # Lower entropy to force precise driving
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/"
    )

    # 3. Train
    print("Starting training... (This may take 30-60 mins)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    
    # 4. Save
    model.save(MODEL_NAME)
    print(f"Success! Model saved to '{MODEL_NAME}.zip'")
    env.close()

if __name__ == "__main__":
    main()