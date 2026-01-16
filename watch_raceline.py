import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

# --- CONFIG ---
MODEL_PATH = "ppo_carracing_v3_expert_final.zip"
VIDEO_FOLDER = "./videos/"
VIDEO_LENGTH = 2000  # Steps to record (approx 2 full laps)

def main():
    # 1. Setup the environment exactly like training
    vec_env = make_vec_env("CarRacing-v3", n_envs=1)
    env = VecFrameStack(vec_env, n_stack=4)

    # 2. Wrap it to record video
    # This will save a .mp4 file to the ./videos/ folder
    env = VecVideoRecorder(
        env, 
        VIDEO_FOLDER, 
        record_video_trigger=lambda x: x == 0, 
        video_length=VIDEO_LENGTH,
        name_prefix="expert_agent"
    )

    # 3. Load Model
    print(f"Loading {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    # 4. Enjoy the show
    obs = env.reset()
    print("Recording video...")
    for _ in range(VIDEO_LENGTH + 1):
        action, _ = model.predict(obs, deterministic=True) # deterministic=True turns off the 'std' noise!
        obs, _, _, _ = env.step(action)
    
    env.close()
    print(f"Video saved to {VIDEO_FOLDER}")

if __name__ == "__main__":
    main()