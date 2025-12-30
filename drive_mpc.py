import gymnasium as gym
import torch
import numpy as np
import cv2
import os
from networks import TinyEncoder, Predictor

# --- CONFIGURATION ---
MODEL_PATH_ENC    = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED   = "./models/predictor_multistep_final.pth"
MODEL_PATH_MEMORY = "./models/memory_bank.pt"

# MPC SETTINGS
HORIZON = 5            
NUM_SAMPLES = 1000     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    if not os.path.exists(MODEL_PATH_MEMORY):
        raise FileNotFoundError("Run make_memory_bank.py first!")
        
    # Load bank to GPU memory for fast query
    memory_bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    print(f"Memory Bank Loaded: {memory_bank.shape}")
    
    return enc, pred, memory_bank

def mpc_policy(encoder, predictor, memory_bank, current_frame):
    # 1. Encode
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    # 2. Random Actions (Simple Monte Carlo)
    # [NUM_SAMPLES, HORIZON, 3]
    action_seqs = torch.rand(NUM_SAMPLES, HORIZON, 3, device=DEVICE)
    action_seqs[:, :, 0] = (action_seqs[:, :, 0] * 2) - 1 # Steer -1..1
    action_seqs[:, :, 1] = (action_seqs[:, :, 1] * 0.5) + 0.5 # Gas 0.5..1.0
    action_seqs[:, :, 2] = 0.0 
    
    # 3. Simulate Futures
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) 
    total_cost = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            # 4. KNN Search
            # Calculate distance from every future state to every memory state
            # Then find the single closest match for each future
            dists = torch.cdist(z_futures, memory_bank) 
            min_dist, _ = torch.min(dists, dim=1)
            
            # Add to cost
            total_cost += min_dist

    # 5. Pick Best
    best_idx = torch.argmin(total_cost)
    best_action = action_seqs[best_idx, 0, :].cpu().numpy()
    min_cost = total_cost[best_idx].item()
    
    return best_action, min_cost

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, memory_bank = load_models()
    
    obs, _ = env.reset()
    
    print("\nMEMORY-BANK AUTOPILOT ENGAGED.")
    
    try:
        step = 0
        while True:
            action, cost = mpc_policy(encoder, predictor, memory_bank, obs)
            obs, _, done, trunc, _ = env.step(action)
            
            if step % 10 == 0:
                print(f"Step {step} | Steer: {action[0]:.2f} | Dist: {cost:.2f}")
            step += 1
            
            if done or trunc:
                print("Crash/Reset")
                obs, _ = env.reset()
                step = 0
                
    except KeyboardInterrupt:
        print("Stopping...")
        env.close()

if __name__ == "__main__":
    main()