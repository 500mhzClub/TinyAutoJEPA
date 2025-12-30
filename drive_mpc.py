import gymnasium as gym
import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder, Predictor, CostModel

# --- CONFIGURATION ---
MODEL_PATH_ENC  = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED = "./models/predictor_multistep_final.pth"
MODEL_PATH_COST = "./models/cost_model_final.pth"

# HYBRID CONTROL SETTINGS
HORIZON = 10           # Look 10 steps ahead (~1.0s)
NUM_SAMPLES = 1000     # Parallel futures
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TUNING WEIGHTS
ALPHA_ENERGY = 1.0     # Discriminator Weight (Safety / Texture)
ALPHA_MAGNET = 0.5     # Magnet Weight (Direction / Centering)
MOMENTUM     = 0.9     # Action Smoothing (0.0=Jerky, 1.0=Frozen)

def load_models():
    print(f"🔌 Loading models on {DEVICE}...")
    
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    cost = CostModel().to(DEVICE).eval()
    cost.load_state_dict(torch.load(MODEL_PATH_COST, map_location=DEVICE))
    
    return enc, pred, cost

def get_expert_target(encoder):
    """
    Calibrates the 'Magnet' (Average Expert State).
    This acts as a compass to pull the car towards the center of the road.
    """
    print("🧲 Calibrating Magnet (Expert Centroid)...")
    files = glob.glob("./data_race/*.npz")
    if not files: files = glob.glob("./data/*.npz")
    
    # Use first 20 files for calibration
    files = files[:20] 
    latents = []
    
    with torch.no_grad():
        for f in files:
            try:
                data = np.load(f)
                if 'states' in data: imgs = data['states']
                elif 'obs' in data: imgs = data['obs']
                else: continue
                
                # Subsample: Take every 10th frame to get a diverse spread
                batch = imgs[::10] 
                
                # Resize & Encode
                if batch.shape[1] != 64:
                    batch = np.array([cv2.resize(img, (64,64)) for img in batch])
                
                tensor = torch.tensor(batch).float().to(DEVICE) / 255.0
                tensor = tensor.permute(0, 3, 1, 2)
                
                z = encoder(tensor)
                latents.append(z)
            except: pass
    
    # Calculate the Mean Vector (The Center of Gravity for 'Good Driving')
    all_z = torch.cat(latents, dim=0)
    target = torch.mean(all_z, dim=0)
    print(f"✅ Magnet Locked. (Based on {len(all_z)} frames)")
    return target

def generate_action_sequences(current_action, num_samples, horizon):
    """
    Generates 'Colored Noise' actions with high momentum.
    This prevents the car from twitching or snapping between left/right turns.
    """
    actions = torch.zeros(num_samples, horizon, 3, device=DEVICE)
    curr_t = torch.tensor(current_action, device=DEVICE).repeat(num_samples, 1)
    
    for t in range(horizon):
        # Reduced noise variance (0.2) for stability
        noise_steer = torch.randn(num_samples, device=DEVICE) * 0.2 
        noise_gas   = torch.rand(num_samples, device=DEVICE) * 0.3 + 0.4 # Bias gas to 0.55 avg
        
        if t == 0:
            # Heavy bias towards previous action
            steer = (MOMENTUM * curr_t[:, 0]) + ((1-MOMENTUM) * noise_steer)
        else:
            # Smooth transition from previous planned step
            steer = (MOMENTUM * actions[:, t-1, 0]) + ((1-MOMENTUM) * noise_steer)
            
        actions[:, t, 0] = torch.clamp(steer, -1.0, 1.0)
        actions[:, t, 1] = torch.clamp(noise_gas, 0.0, 1.0)
        actions[:, t, 2] = 0.0 # No Brakes
        
    return actions

def mpc_policy(encoder, predictor, cost_model, target_z, current_frame, last_action):
    # 1. Encode Current Reality
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    # 2. Generate Smooth Action Candidates
    action_seqs = generate_action_sequences(last_action, NUM_SAMPLES, HORIZON)
    
    # 3. Predict Futures & Calculate Hybrid Cost
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) 
    total_cost = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            # Rollout one step
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            # --- COST FUNCTION ---
            
            # A. The Guardrail (Discriminator)
            # "Is this state dangerous/random?" (0=Safe, 1=Bad)
            energy_score = cost_model(z_futures).squeeze()
            
            # B. The Compass (Magnet)
            # "How far is this from the ideal center of the track?"
            magnet_dist = torch.norm(z_futures - target_z, dim=1)
            
            # Combined Cost
            # We combine safety (energy) with direction (magnet)
            step_cost = (energy_score * ALPHA_ENERGY) + (magnet_dist * ALPHA_MAGNET)
            
            # Temporal Discounting
            # We care more about surviving the next 0.1s than 1.0s away
            discount = 0.95 ** t
            total_cost += step_cost * discount

    # 4. Pick Best Trajectory
    best_idx = torch.argmin(total_cost)
    best_action = action_seqs[best_idx, 0, :].cpu().numpy()
    
    # Optional: Return cost for debugging
    min_cost = total_cost[best_idx].item()
    
    return best_action, min_cost

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, cost_model = load_models()
    
    # Calibrate Magnet
    target_z = get_expert_target(encoder)
    
    obs, _ = env.reset()
    last_action = np.array([0.0, 0.0, 0.0])
    
    print("\n⚡ HYBRID JEPA AUTOPILOT ENGAGED. Press Ctrl+C to stop.")
    
    try:
        step = 0
        while True:
            # Run MPC
            action, cost = mpc_policy(encoder, predictor, cost_model, target_z, obs, last_action)
            
            # Execute
            obs, _, done, trunc, _ = env.step(action)
            last_action = action
            
            if step % 10 == 0:
                print(f"Step {step} | Steer: {action[0]:.2f} | Cost: {cost:.4f}")
            step += 1
            
            if done or trunc:
                print("💥 Crash/Reset")
                obs, _ = env.reset()
                last_action = np.array([0.0, 0.0, 0.0])
                step = 0
                
    except KeyboardInterrupt:
        print("Stopping...")
        env.close()

if __name__ == "__main__":
    main()