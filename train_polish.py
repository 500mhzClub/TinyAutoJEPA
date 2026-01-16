import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# --- CONFIG ---
ENV_ID = "CarRacing-v3"
N_ENVS = 8
EXTRA_TIMESTEPS = 500_000
MODEL_PATH = "ppo_carracing_v3_grandmaster.zip"
FINAL_MODEL_NAME = "ppo_carracing_v3_perfected"

def main():
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    env = VecFrameStack(vec_env, n_stack=4)

    print(f"Loading {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, env=env)

    # CRITICAL CHANGES FOR STABILITY:
    # 1. Kill exploration (Entropy = 0)
    model.ent_coef = 0.0
    # 2. Ultra-low learning rate (Fine-tuning only)
    model.learning_rate = 1e-5 

    print("Polishing the diamond: 0 Entropy, Ultra-low LR...")
    model.learn(total_timesteps=EXTRA_TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    
    model.save(FINAL_MODEL_NAME)
    print("Done. The agent should now be rock solid.")
    env.close()

if __name__ == "__main__":
    main()