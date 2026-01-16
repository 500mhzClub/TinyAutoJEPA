import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack # Import needed

# --- CONFIGURATION ---
ENV_ID = "CarRacing-v3"
TOTAL_TIMESTEPS = 600_000 
N_ENVS = 8                 

def main():
    # 1. Create Env
    # Use make_vec_env to create the parallel envs
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    
    # CRITICAL FIX: Wrap the environment in VecFrameStack
    # This stacks 4 frames so the agent sees motion/velocity
    env = VecFrameStack(vec_env, n_stack=4)
    
    # 2. Define PPO Model (Tuned for Stability)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,  # REDUCED: 3e-4 -> 1e-4 to fix high KL/Clipping
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # INCREASED: 0.0 -> 0.01 to prevent getting stuck in local optima
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    print("Starting training with Frame Stacking...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save("ppo_carracing_v3_fixed")
    env.close()

if __name__ == "__main__":
    main()