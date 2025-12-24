import gymnasium as gym
import torch
import numpy as np
import cv2
from networks import TinyEncoder, Predictor, CostModel

# --- CONFIGURATION ---
MODEL_PATH_ENC  = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED = "./models/predictor_multistep_final.pth"
MODEL_PATH_COST = "./models/cost_model_final.pth"

HORIZON = 10           # Increased Horizon for better planning
NUM_SAMPLES = 1000     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """
    Generates 'Colored Noise' actions. 
    Instead of pure random, we blend the previous action with new random noise.
    This creates smooth, drivable trajectories (momentum).
    """
    # [NUM_SAMPLES, HORIZON, 3]
    actions = torch.zeros(num_samples, horizon, 3, device=DEVICE)
    
    # 1. Bias first step with current action (Momentum)
    # We treat the current real action as the starting point
    curr_t = torch.tensor(current_action, device=DEVICE).repeat(num_samples, 1)
    
    for t in range(horizon):
        # Sample random noise
        # Steering: Normal distribution centered on 0, Gas: Uniform
        noise_steer = torch.randn(num_samples, device=DEVICE) * 0.5 
        noise_gas   = torch.rand(num_samples, device=DEVICE) * 0.5 + 0.2 # Bias towards gas
        
        if t == 0:
            # First step is heavily influenced by current state
            steer = (0.7 * curr_t[:, 0]) + (0.3 * noise_steer)
        else:
            # Subsequent steps smooth from the previous plan
            steer = (0.8 * actions[:, t-1, 0]) + (0.2 * noise_steer)
            
        # Clip Physics
        steer = torch.clamp(steer, -1.0, 1.0)
        gas   = torch.clamp(noise_gas, 0.0, 1.0)
        brake = torch.zeros(num_samples, device=DEVICE) # Disable brake for racing
        
        actions[:, t, 0] = steer
        actions[:, t, 1] = gas
        actions[:, t, 2] = brake
        
    return actions

def mpc_policy(encoder, predictor, cost_model, current_frame, last_action):
    # 1. Encode Current State
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img) # [1, 512]
        
    # 2. Generate Smooth Action Sequences
    action_seqs = generate_action_sequences(last_action, NUM_SAMPLES, HORIZON)
    
    # 3. Simulate Futures
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) 
    cumulative_energy = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            actions_t = action_seqs[:, t, :]
            
            # Predict Next State
            z_futures = predictor(z_futures, actions_t)
            
            # --- ENERGY EVALUATION (The JEPA Fix) ---
            # Instead of distance to mean, we ask the Cost Model:
            # "How much does this state look like expert driving?"
            energy = cost_model(z_futures) # Returns prob (0=Good, 1=Bad)
            
            # Add to cost (weighted slightly more for near-term safety)
            cumulative_energy += energy.squeeze()

    # 4. Pick Best Action
    best_idx = torch.argmin(cumulative_energy)
    best_action = action_seqs[best_idx, 0, :].cpu().numpy()
    
    return best_action

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, cost_model = load_models()
    
    obs, _ = env.reset()
    last_action = np.array([0.0, 0.0, 0.0]) # Steering, Gas, Brake
    
    print("\n⚡ JEPA AUTOPILOT ENGAGED (Energy-Based). Press Ctrl+C to stop.")
    
    try:
        while True:
            # Get MPC Action
            action = mpc_policy(encoder, predictor, cost_model, obs, last_action)
            
            # Execute
            obs, reward, done, trunc, _ = env.step(action)
            last_action = action
            
            if done or trunc:
                print(" Crash/Reset")
                obs, _ = env.reset()
                last_action = np.array([0.0, 0.0, 0.0])
                
    except KeyboardInterrupt:
        print("Stopping...")
        env.close()

if __name__ == "__main__":
    main()