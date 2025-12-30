import gymnasium as gym
import torch
import numpy as np
import cv2
import os
from networks import TinyEncoder, Predictor, CostModel

# --- CONFIGURATION ---
MODEL_PATH_ENC    = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED   = "./models/predictor_multistep_final.pth"
MODEL_PATH_COST   = "./models/cost_model_final.pth"
MODEL_PATH_MAGNET = "./models/expert_magnet.pth"

# HYBRID CONTROL SETTINGS
HORIZON = 10           
NUM_SAMPLES = 1000     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TUNING WEIGHTS
ALPHA_ENERGY = 1.0     
ALPHA_MAGNET = 0.5     
MOMENTUM     = 0.9     

def load_models():
    print(f"🔌 Loading models on {DEVICE}...")
    
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    cost = CostModel().to(DEVICE).eval()
    cost.load_state_dict(torch.load(MODEL_PATH_COST, map_location=DEVICE))
    
    # LOAD THE MAGNET INSTANTLY
    if not os.path.exists(MODEL_PATH_MAGNET):
        raise FileNotFoundError("Magnet file missing! Run make_magnet.py first.")
        
    magnet = torch.load(MODEL_PATH_MAGNET, map_location=DEVICE)
    print("Magnet Loaded from disk.")
    
    return enc, pred, cost, magnet

def generate_action_sequences(current_action, num_samples, horizon):
    actions = torch.zeros(num_samples, horizon, 3, device=DEVICE)
    curr_t = torch.tensor(current_action, device=DEVICE).repeat(num_samples, 1)
    
    for t in range(horizon):
        noise_steer = torch.randn(num_samples, device=DEVICE) * 0.2 
        noise_gas   = torch.rand(num_samples, device=DEVICE) * 0.3 + 0.4 
        
        if t == 0:
            steer = (MOMENTUM * curr_t[:, 0]) + ((1-MOMENTUM) * noise_steer)
        else:
            steer = (MOMENTUM * actions[:, t-1, 0]) + ((1-MOMENTUM) * noise_steer)
            
        actions[:, t, 0] = torch.clamp(steer, -1.0, 1.0)
        actions[:, t, 1] = torch.clamp(noise_gas, 0.0, 1.0)
        actions[:, t, 2] = 0.0 
        
    return actions

def mpc_policy(encoder, predictor, cost_model, target_z, current_frame, last_action):
    # 1. Encode
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    # 2. Generate Actions
    action_seqs = generate_action_sequences(last_action, NUM_SAMPLES, HORIZON)
    
    # 3. Predict & Score
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) 
    total_cost = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            energy_score = cost_model(z_futures).squeeze()
            magnet_dist = torch.norm(z_futures - target_z, dim=1)
            
            step_cost = (energy_score * ALPHA_ENERGY) + (magnet_dist * ALPHA_MAGNET)
            
            discount = 0.95 ** t
            total_cost += step_cost * discount

    best_idx = torch.argmin(total_cost)
    return action_seqs[best_idx, 0, :].cpu().numpy(), total_cost[best_idx].item()

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    
    # Load everything instantly
    encoder, predictor, cost_model, target_z = load_models()
    
    obs, _ = env.reset()
    last_action = np.array([0.0, 0.0, 0.0])
    
    print("\n⚡ HYBRID JEPA AUTOPILOT ENGAGED. Press Ctrl+C to stop.")
    
    try:
        step = 0
        while True:
            action, cost = mpc_policy(encoder, predictor, cost_model, target_z, obs, last_action)
            obs, _, done, trunc, _ = env.step(action)
            last_action = action
            
            if step % 10 == 0:
                print(f"Step {step} | Steer: {action[0]:.2f} | Cost: {cost:.4f}")
            step += 1
            
            if done or trunc:
                print("Crash/Reset")
                obs, _ = env.reset()
                last_action = np.array([0.0, 0.0, 0.0])
                step = 0
                
    except KeyboardInterrupt:
        print("Stopping...")
        env.close()

if __name__ == "__main__":
    main()