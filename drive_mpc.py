import gymnasium as gym
import torch
import numpy as np
import cv2
import os
import glob
from networks import TinyEncoder, Predictor

# --- CONFIGURATION ---
MODEL_PATH_ENC  = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED = "./models/predictor_multistep_final.pth"

HORIZON = 5           # How many steps to look ahead (must match training)
NUM_SAMPLES = 1000    # How many parallel futures to imagine
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    return enc, pred

def get_expert_target(encoder):
    """
    Calculates the 'Average Expert State' by looking at the race data.
    This acts as a magnet for the MPC.
    """
    print("🧠 Calibrating 'Expert Behavior' from race data...")
    files = glob.glob("./data_race/*.npz")[:10] # Look at first 10 files
    if not files:
        raise FileNotFoundError("No race data found in ./data_race/")
    
    latents = []
    with torch.no_grad():
        for f in files:
            data = np.load(f)
            if 'states' in data: imgs = data['states']
            elif 'obs' in data: imgs = data['obs']
            else: continue
            
            # Take a random sample to save time
            indices = np.random.choice(len(imgs), size=min(100, len(imgs)), replace=False)
            batch = imgs[indices]
            
            # Preprocess
            batch = np.array([cv2.resize(img, (64,64)) for img in batch])
            tensor = torch.tensor(batch).float().to(DEVICE) / 255.0
            tensor = tensor.permute(0, 3, 1, 2) # NCHW
            
            z = encoder(tensor)
            latents.append(z)
            
    # Calculate the Mean Vector (The "Ideal State")
    all_z = torch.cat(latents, dim=0)
    target = torch.mean(all_z, dim=0)
    print(f"✅ Target Locked. Expert Cluster Size: {len(all_z)}")
    return target

def mpc_policy(encoder, predictor, target_z, current_frame):
    # 1. Encode Current State
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img) # [1, 512]
        
    # 2. Generate Random Action Sequences
    # Action space: [Steering (-1,1), Gas (0,1), Brake (0,1)]
    # We bias towards moving forward (Gas=0.5) to encourage speed
    action_seqs = torch.rand(NUM_SAMPLES, HORIZON, 3, device=DEVICE) 
    action_seqs[:, :, 0] = (action_seqs[:, :, 0] * 2) - 1 # Steering: -1 to 1
    action_seqs[:, :, 1] = action_seqs[:, :, 1] * 1.0     # Gas: 0 to 1
    action_seqs[:, :, 2] = 0                              # Disable Brake for now (Aggressive)
    
    # 3. Simulate Futures
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) # Start all samples at current state
    
    cumulative_cost = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            actions_t = action_seqs[:, t, :]
            
            # Predict Next State
            z_futures = predictor(z_futures, actions_t)
            
            # Cost: Euclidean Distance to "Expert Target"
            # We want to be CLOSE to the average expert state
            dist = torch.norm(z_futures - target_z, dim=1)
            cumulative_cost += dist

    # 4. Pick Best Action
    best_idx = torch.argmin(cumulative_cost)
    best_action = action_seqs[best_idx, 0, :].cpu().numpy()
    
    return best_action

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor = load_models()
    
    # Calibration
    target_z = get_expert_target(encoder)
    
    obs, _ = env.reset()
    
    print("\n AUTOPILOT ENGAGED. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Get MPC Action
            action = mpc_policy(encoder, predictor, target_z, obs)
            
            # Execute
            obs, reward, done, trunc, _ = env.step(action)
            
            if done or trunc:
                print(" Crash/Reset")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Stopping...")
        env.close()

if __name__ == "__main__":
    main()