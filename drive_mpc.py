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

# MPC SETTINGS
HORIZON = 5            # Reduced from 10 to keep predictions sharp
NUM_SAMPLES = 2000     # Increased samples to find "needle in haystack"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TUNING
ALPHA_ENERGY = 1.0     # 100% Trust in Texture Safety
ALPHA_MAGNET = 0.0     # Disabled (removes straight-line bias)
MOMENTUM     = 0.8     # Balance between smoothness and reactivity

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    cost = CostModel().to(DEVICE).eval()
    cost.load_state_dict(torch.load(MODEL_PATH_COST, map_location=DEVICE))
    
    return enc, pred, cost

def generate_action_sequences(current_action, num_samples, horizon):
    actions = torch.zeros(num_samples, horizon, 3, device=DEVICE)
    curr_t = torch.tensor(current_action, device=DEVICE).repeat(num_samples, 1)
    
    for t in range(horizon):
        # Increased variance (0.5) so it can imagine hard turns
        noise_steer = torch.randn(num_samples, device=DEVICE) * 0.5 
        noise_gas   = torch.rand(num_samples, device=DEVICE) * 0.3 + 0.4 
        
        if t == 0:
            steer = (MOMENTUM * curr_t[:, 0]) + ((1-MOMENTUM) * noise_steer)
        else:
            steer = (MOMENTUM * actions[:, t-1, 0]) + ((1-MOMENTUM) * noise_steer)
            
        actions[:, t, 0] = torch.clamp(steer, -1.0, 1.0)
        actions[:, t, 1] = torch.clamp(noise_gas, 0.0, 1.0)
        actions[:, t, 2] = 0.0 
        
    return actions

def mpc_policy(encoder, predictor, cost_model, current_frame, last_action):
    # Encode
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    action_seqs = generate_action_sequences(last_action, NUM_SAMPLES, HORIZON)
    
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) 
    total_cost = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            # Cost is purely based on "Does this look like expert driving?"
            # We ignore the Magnet to prevent corner-cutting
            step_cost = cost_model(z_futures).squeeze()
            
            discount = 0.90 ** t
            total_cost += step_cost * discount

    best_idx = torch.argmin(total_cost)
    best_cost = total_cost[best_idx].item()
    return action_seqs[best_idx, 0, :].cpu().numpy(), best_cost

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, cost_model = load_models()
    
    obs, _ = env.reset()
    last_action = np.array([0.0, 0.0, 0.0])
    
    print("MPC ENGAGED (Horizon=5, Texture Only)")
    
    try:
        step = 0
        while True:
            action, cost = mpc_policy(encoder, predictor, cost_model, obs, last_action)
            obs, _, done, trunc, _ = env.step(action)
            last_action = action
            
            if step % 10 == 0:
                # If Cost is < 1.0, the car feels safe.
                # If Cost is > 3.0, the car is panicking.
                print(f"Step {step} | Steer: {action[0]:.2f} | Risk: {cost:.2f}")
            step += 1
            
            if done or trunc:
                print("Resetting...")
                obs, _ = env.reset()
                last_action = np.array([0.0, 0.0, 0.0])
                step = 0
                
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    main()