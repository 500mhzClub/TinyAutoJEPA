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

# CONTROLLER SETTINGS
HORIZON = 5            
NUM_TENTACLES = 60     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ROBUSTNESS TUNING
TOP_K = 10             # Require consensus from 10 memories
STEER_PENALTY = 0.5    # Cost for jerking the wheel (Smoothness)

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    if not os.path.exists(MODEL_PATH_MEMORY):
        raise FileNotFoundError("Run make_memory_bank.py first!")
        
    bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    bank = torch.nn.functional.normalize(bank, p=2, dim=1)
    print(f"Memory Bank Loaded: {bank.shape}")
    
    return enc, pred, bank

def generate_tentacles(horizon, num_tentacles):
    actions = torch.zeros(num_tentacles, horizon, 3, device=DEVICE)
    steer_angles = torch.linspace(-1.0, 1.0, num_tentacles, device=DEVICE)
    
    for i, steer in enumerate(steer_angles):
        actions[i, :, 0] = steer
        # Slower on turns to prevent spinning
        if abs(steer) > 0.4:
            actions[i, :, 1] = 0.3
        else:
            actions[i, :, 1] = 0.6
        actions[i, :, 2] = 0.0
        
    return actions

def mpc_policy(encoder, predictor, memory_bank, current_frame, last_steer):
    # 1. Encode
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    # 2. Generate Tentacles
    action_seqs = generate_tentacles(HORIZON, NUM_TENTACLES)
    
    # 3. Simulate Futures
    z_futures = z_curr.repeat(NUM_TENTACLES, 1) 
    total_sim = torch.zeros(NUM_TENTACLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            # --- ROBUST COSINE SIMILARITY ---
            z_norm = torch.nn.functional.normalize(z_futures, p=2, dim=1)
            similarity_matrix = torch.mm(z_norm, memory_bank.T)
            
            # Top-K Average (Consensus)
            # We take the top 10 matches and average them.
            # This filters out single "lucky" bad matches.
            top_sims, _ = torch.topk(similarity_matrix, k=TOP_K, dim=1)
            avg_sim = torch.mean(top_sims, dim=1)
            
            total_sim += avg_sim

    # 4. Add Physics Constraints (Smoothness)
    # Penalize tentacles that require jerking the wheel from the current position
    # action_seqs[:, 0, 0] is the steering angle of the tentacle
    # last_steer is where the wheel is NOW
    steer_diff = torch.abs(action_seqs[:, 0, 0] - last_steer)
    
    # Subtract penalty from the similarity score
    # We weight this so it breaks ties but doesn't override good driving
    final_scores = total_sim - (steer_diff * STEER_PENALTY)

    # 5. Pick Winner
    best_idx = torch.argmax(final_scores)
    best_action = action_seqs[best_idx, 0, :].cpu().numpy()
    
    # Debug info
    raw_score = total_sim[best_idx].item()
    
    return best_action, raw_score

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, memory_bank = load_models()
    
    obs, _ = env.reset()
    last_steer = 0.0
    
    print("\nROBUST CONTROLLER ENGAGED (Top-10 Consensus).")
    
    try:
        step = 0
        while True:
            action, score = mpc_policy(encoder, predictor, memory_bank, obs, last_steer)
            
            obs, _, done, trunc, _ = env.step(action)
            last_steer = action[0]
            
            if step % 10 == 0:
                print(f"Step {step} | Steer: {action[0]:.2f} | Match: {score:.2f}/5.0")
            step += 1
            
            if done or trunc:
                print("Crash/Reset")
                obs, _ = env.reset()
                last_steer = 0.0
                step = 0
                
    except KeyboardInterrupt:
        print("Stopping...")
        env.close()

if __name__ == "__main__":
    main()