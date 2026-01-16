import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# --- CONFIGURATION ---
ENV_ID = "CarRacing-v3"
N_ENVS = 8
# We train for another 400k steps (Total will be ~1M)
EXTRA_TIMESTEPS = 400_000 
MODEL_PATH = "ppo_carracing_v3_fixed.zip" # The model you just saved
NEW_MODEL_NAME = "ppo_carracing_v3_expert_final"

def main():
    # 1. Recreate the environment EXACTLY as before
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    env = VecFrameStack(vec_env, n_stack=4)

    # 2. Load the saved model
    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env)

    # 3. LOWER the learning rate for fine-tuning
    # This helps reduce the 'std' and smooths out the driving
    model.learning_rate = 5e-5  # Reduced from 1e-4

    # 4. Continue Training
    print(f"Resuming training for {EXTRA_TIMESTEPS} steps...")
    model.learn(total_timesteps=EXTRA_TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    
    model.save(NEW_MODEL_NAME)
    print("Done! Saved final expert model.")
    env.close()

if __name__ == "__main__":
    main()