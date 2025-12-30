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

# TENTACLE SETTINGS
HORIZON = 5            
NUM_TENTACLES = 60     # Number of smooth paths to test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    if not os.path.exists(MODEL_PATH_MEMORY):
        raise FileNotFoundError("Run make_memory_bank.py first!")
        
    bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    # Normalize bank for Cosine Similarity
    bank = torch.nn.functional.normalize(bank, p=2, dim=1)
    print(f"Memory Bank Loaded: {bank.shape}")
    
    return enc, pred, bank

def generate_tentacles(horizon, num_tentacles):
    """
    Creates a library of smooth, constant-curvature paths.
    From Hard Left (-1.0) to Hard Right (+1.0).
    """
    # [NUM_TENTACLES, HORIZON, 3]
    actions = torch.zeros(num_tentacles, horizon, 3, device=DEVICE)
    
    # Generate steering angles linearly from -1 to 1
    steer_angles = torch.linspace(-1.0, 1.0, num_tentacles, device=DEVICE)
    
    for i, steer in enumerate(steer_angles):
        # Action 0: Steering (Constant for the whole horizon)
        actions[i, :, 0] = steer
        
        # Action 1: Gas
        # Go fast on straights, slow on turns
        if abs(steer) > 0.5:
            actions[i, :, 1] = 0.2  # Slow down for sharp turns
        else:
            actions[i, :, 1] = 0.6  # Gas it on straights
            
        # Action 2: Brake (None)
        actions[i, :, 2] = 0.0
        
    return actions

def mpc_policy(encoder, predictor, memory_bank, current_frame):
    # 1. Encode
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    # 2. Generate Tentacles (Fixed library of smooth moves)
    action_seqs = generate_tentacles(HORIZON, NUM_TENTACLES)
    
    # 3. Simulate Futures
    z_futures = z_curr.repeat(NUM_TENTACLES, 1) 
    total_sim = torch.zeros(NUM_TENTACLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            # --- COSINE SIMILARITY ---
            z_norm = torch.nn.functional.normalize(z_futures, p=2, dim=1)
            
            # Match against memory bank
            similarity_matrix = torch.mm(z_norm, memory_bank.T)
            
            # Find best match for each tentacle
            max_sim, _ = torch.max(similarity_matrix, dim=1)
            
            # Add similarity to score (We want to MAXIMIZE this)
            total_sim += max_sim

    # 4. Pick Winner
    best_idx = torch.argmax(total_sim)
    best_action = action_seqs[best_idx, 0, :].cpu().numpy()
    
    # Score logic: Higher is better. 
    # Max possible score = HORIZON * 1.0 = 5.0
    best_score = total_sim[best_idx].item()
    
    return best_action, best_score

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, memory_bank = load_models()
    
    obs, _ = env.reset()
    
    print("\nTENTACLE CONTROLLER ENGAGED.")
    print(f"Testing {NUM_TENTACLES} smooth paths per step.")
    
    try:
        step = 0
        while True:
            action, score = mpc_policy(encoder, predictor, memory_bank, obs)
            
            obs, _, done, trunc, _ = env.step(action)
            
            if step % 10 == 0:
                # Score > 4.5 is great. Score < 3.0 is panic.
                print(f"Step {step} | Steer: {action[0]:.2f} | Match Score: {score:.2f}/5.0")
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