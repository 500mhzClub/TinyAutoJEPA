import gymnasium as gym
import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder, Predictor

# --- CONFIGURATION ---
MODEL_PATH_ENC  = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED = "./models/predictor_multistep_final.pth"

# MPC SETTINGS
HORIZON = 5            
NUM_SAMPLES = 1000     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_SIZE = 2000     # How many expert frames to keep in RAM

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    return enc, pred

def create_memory_bank(encoder):
    """
    Instead of one 'Average' magnet, we keep 2,000 diverse examples.
    This allows the car to match 'Left Turn' memories when turning left.
    """
    print(f"Building Memory Bank ({MEMORY_SIZE} examples)...")
    files = glob.glob("./data_race/*.npz")[:50]
    
    bank = []
    with torch.no_grad():
        for f in files:
            if len(bank) >= MEMORY_SIZE: break
            try:
                data = np.load(f)
                if 'states' in data: imgs = data['states']
                elif 'obs' in data: imgs = data['obs']
                else: continue
                
                # Take random samples from this file
                indices = np.random.choice(len(imgs), size=min(100, len(imgs)), replace=False)
                batch = imgs[indices]
                
                # Resize & Encode
                if batch.shape[1] != 64:
                    batch = np.array([cv2.resize(img, (64,64)) for img in batch])
                    
                tensor = torch.tensor(batch).float().to(DEVICE) / 255.0
                tensor = tensor.permute(0, 3, 1, 2)
                
                z = encoder(tensor)
                bank.append(z)
            except: pass
            
    # [MEMORY_SIZE, 512]
    memory_bank = torch.cat(bank, dim=0)[:MEMORY_SIZE]
    print(f"Memory Bank Assembled. Shape: {memory_bank.shape}")
    return memory_bank

def mpc_policy(encoder, predictor, memory_bank, current_frame):
    # 1. Encode Current Reality
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    # 2. Generate Action Candidates
    # Simple random shooting often works best for short horizons
    action_seqs = torch.rand(NUM_SAMPLES, HORIZON, 3, device=DEVICE)
    action_seqs[:, :, 0] = (action_seqs[:, :, 0] * 2) - 1 # Steer -1 to 1
    action_seqs[:, :, 1] = (action_seqs[:, :, 1] * 0.5) + 0.5 # Gas 0.5 to 1.0
    action_seqs[:, :, 2] = 0.0 # No Brake
    
    # 3. Simulate Futures
    z_futures = z_curr.repeat(NUM_SAMPLES, 1) 
    total_cost = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            
            # --- NEAREST NEIGHBOR COST ---
            # Instead of distance to Mean, find distance to CLOSEST memory
            # We use CDIST (pairwise distance)
            # z_futures: [1000, 512]
            # memory_bank: [2000, 512]
            
            # This computes distance from every future to every memory
            dists = torch.cdist(z_futures, memory_bank) # [1000, 2000]
            
            # Find the single closest expert memory for each future
            min_dist, _ = torch.min(dists, dim=1) # [1000]
            
            total_cost += min_dist

    # 4. Pick Best
    best_idx = torch.argmin(total_cost)
    best_action = action_seqs[best_idx, 0, :].cpu().numpy()
    min_cost = total_cost[best_idx].item()
    
    return best_action, min_cost

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor = load_models()
    
    # Create the Brain Bank
    memory_bank = create_memory_bank(encoder)
    
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