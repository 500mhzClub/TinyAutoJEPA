import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# --- CONFIG ---
ENV_ID = "CarRacing-v3"
N_ENVS = 8
EXTRA_TIMESTEPS = 1_000_000  # The final push to 2M
MODEL_PATH = "ppo_carracing_v3_expert_final.zip"
FINAL_MODEL_NAME = "ppo_carracing_v3_grandmaster"

def main():
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    env = VecFrameStack(vec_env, n_stack=4)

    print(f"Loading {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, env=env)

    # LOWER Learning Rate slightly to refine cornering
    # 5e-5 is safe, 3e-5 is very precise (but slower)
    model.learning_rate = 5e-5  

    print(f"Training for {EXTRA_TIMESTEPS} steps to learn braking...")
    model.learn(total_timesteps=EXTRA_TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    
    model.save(FINAL_MODEL_NAME)
    print("Training Complete. Your agent should now be drifting properly.")
    env.close()

if __name__ == "__main__":
    main()