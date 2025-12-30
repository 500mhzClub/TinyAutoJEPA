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

# MPPI SETTINGS (The "Smooth" Controller)
HORIZON = 5            
NUM_SAMPLES = 1000     
TEMPERATURE = 0.05     # Controls how "picky" we are (Lower = sharper, Higher = smoother)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    if not os.path.exists(MODEL_PATH_MEMORY):
        raise FileNotFoundError("Run make_memory_bank.py first!")
        
    # Load bank and NORMALIZE it immediately for Cosine Similarity
    bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    bank = torch.nn.functional.normalize(bank, p=2, dim=1)
    print(f"Memory Bank Loaded & Normalized: {bank.shape}")
    
    return enc, pred, bank

def mpc_policy(encoder, predictor, memory_bank, current_frame, last_action):
    # 1. Encode
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    # 2. Generate Action Candidates (Gaussian Noise around Mean)
    # We bias the search around the *previous* action for smoothness
    mean_action = torch.tensor(last_action, device=DEVICE).float()
    
    # [NUM_SAMPLES, HORIZON, 3]
    # Noise scale: Steer=0.5, Gas=0.2
    noise = torch.randn(NUM_SAMPLES, HORIZON, 3, device=DEVICE)
    noise[:, :, 0] *= 0.5 
    noise[:, :, 1] *= 0.2
    noise[:, :, 2] = 0.0 # No brake
    
    # Base actions: repeat the last action across the horizon
    action_seqs = mean_action.view(1, 1, 3) + noise
    
    # Clip
    action_seqs[:, :, 0] = torch.clamp(action_seqs[:, :, 0], -1.0, 1.0) # Steer
    action_seqs[:, :, 1] = torch.clamp(action_seqs[:, :, 1], 0.0, 1.0)   # Gas
    action_seqs[:, :, 2] = 0.0
    
    # 3. Simulate Futures
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) 
    total_scores = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            # --- COSINE SIMILARITY SCORE ---
            # 1. Normalize the predicted vectors (Fixes the "Scale" issue)
            z_norm = torch.nn.functional.normalize(z_futures, p=2, dim=1)
            
            # 2. Matrix Multiply with Memory Bank (Efficient Cosine Sim)
            # [1000, 512] @ [512, 1600] -> [1000, 1600]
            similarity_matrix = torch.mm(z_norm, memory_bank.T)
            
            # 3. Find the BEST match for each future
            # Max Similarity = Nearest Neighbor
            max_sim, _ = torch.max(similarity_matrix, dim=1)
            
            # Cost is NEGATIVE similarity (we want to maximize sim)
            # We want sim to be 1.0, so cost is (1.0 - sim)
            step_cost = 1.0 - max_sim
            
            total_scores += step_cost

    # --- MPPI WEIGHTING (The Magic Sauce) ---
    # instead of argmin (Winner Takes All), we Softmax the negative costs
    
    # 1. Weights = exp(-Cost / Temperature)
    # Lower cost = Higher weight
    weights = torch.softmax(-total_scores / TEMPERATURE, dim=0)
    
    # 2. Weighted Average of the FIRST action in the sequence
    # We blend the Steering/Gas of all 1000 samples based on how good they are
    # [1000, 1] * [1000, 3] -> [3]
    weighted_action = torch.sum(weights.view(-1, 1) * action_seqs[:, 0, :], dim=0)
    
    # 3. Return the smooth action
    best_action = weighted_action.cpu().numpy()
    
    # Debug: Expected "goodness"
    expected_sim = torch.sum(weights * total_scores).item()
    
    return best_action, expected_sim

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, memory_bank = load_models()
    
    obs, _ = env.reset()
    last_action = np.array([0.0, 0.0, 0.0])
    
    print("\nMPPI CONTROLLER ENGAGED (Cosine Sim + Weighted Avg).")
    
    try:
        step = 0
        while True:
            # We pass last_action to smooth the noise generation
            action, score = mpc_policy(encoder, predictor, memory_bank, obs, last_action)
            
            # Action Smoothing (Exponential Moving Average)
            # This prevents twitching between MPPI steps
            action = (0.7 * last_action) + (0.3 * action)
            
            obs, _, done, trunc, _ = env.step(action)
            last_action = action
            
            if step % 10 == 0:
                # Score is (1.0 - Similarity). 
                # 0.00 = Perfect Match, 1.00 = Opposite
                print(f"Step {step} | Steer: {action[0]:.2f} | Gap: {score:.4f}")
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