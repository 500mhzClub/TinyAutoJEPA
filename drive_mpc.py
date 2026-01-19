import gymnasium as gym
import torch
import numpy as np
import cv2
import os
from networks import TinyEncoder, Predictor, TinyDecoder

# --- CONFIGURATION ---
MODEL_PATH_ENC    = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED   = "./models/predictor_final.pth" # Note: Check if this name matches your saved file
MODEL_PATH_DEC    = "./models/decoder_final.pth"
MODEL_PATH_MEMORY = "./models/memory_bank.pt"

# CONTROLLER PHYSICS
HORIZON       = 10     # Look 10 steps (1.0s) into the future
NUM_TENTACLES = 128    # More samples = smoother driving
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TUNING
TOP_K         = 20     # Average the top 20 matches (removes outliers)
STEER_PENALTY = 0.05   # Don't zigzag
TEMPERATURE   = 0.04   # Lower = More aggressive greedy choice
MOMENTUM      = 0.60   # High momentum = smoother steering (prevents twitching)

def load_models():
    print(f"Loading World Model on {DEVICE}...")
    
    # Encoder
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    # Predictor
    pred = Predictor().to(DEVICE).eval()
    # Try loading the latest predictor checkpoint
    try:
        pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    except:
        print(f"⚠️ Could not load {MODEL_PATH_PRED}, trying 'predictor_final.pth'...")
        pred.load_state_dict(torch.load("models/predictor_final.pth", map_location=DEVICE))

    # Decoder (Optional - For Visualization)
    dec = TinyDecoder().to(DEVICE).eval()
    if os.path.exists(MODEL_PATH_DEC):
        dec.load_state_dict(torch.load(MODEL_PATH_DEC, map_location=DEVICE))
        print("✅ Decoder Loaded (Dream View Enabled)")
    else:
        print("⚠️ Decoder NOT found. You won't see the imagination bubble.")
        dec = None
        
    # Memory Bank
    if not os.path.exists(MODEL_PATH_MEMORY):
        raise FileNotFoundError("Run make_memory_bank.py first!")
    bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    # Ensure bank is normalized
    bank = torch.nn.functional.normalize(bank, p=2, dim=1)
    
    return enc, pred, dec, bank

def generate_tentacles(horizon, num_tentacles):
    """
    Generates random action sequences.
    """
    actions = torch.zeros(num_tentacles, horizon, 3, device=DEVICE)
    
    # 1. Straight & Gentle Turns (Gaussian)
    steer_dist = torch.randn(num_tentacles, device=DEVICE) * 0.5
    steer_dist = torch.clamp(steer_dist, -1.0, 1.0)
    
    # 2. Hard Left/Right (Uniform) - To cover extremes
    num_uniform = num_tentacles // 4
    steer_dist[:num_uniform] = torch.rand(num_uniform, device=DEVICE) * 2 - 1
    
    for i in range(num_tentacles):
        s = steer_dist[i]
        actions[i, :, 0] = s
        
        # Physics Logic: Slow down for turns
        if abs(s) > 0.4:   
            actions[i, :, 1] = 0.0  # Gas
            actions[i, :, 2] = 0.2  # Brake
        else:                  
            actions[i, :, 1] = 0.4  # Gas
            actions[i, :, 2] = 0.0  # Brake
            
    return actions

def mpc_policy(encoder, predictor, decoder, memory_bank, current_frame, last_steer):
    # Prepare Input
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.from_numpy(img).float().to(DEVICE).div(255.0).permute(2, 0, 1).unsqueeze(0)
    
    # 1. Encode Reality
    with torch.no_grad():
        z_curr = encoder(t_img)
        z_curr = torch.nn.functional.normalize(z_curr, p=2, dim=1)
        
    action_seqs = generate_tentacles(HORIZON, NUM_TENTACLES)
    
    # Expand z_curr to match batch size
    z_futures = z_curr.repeat(NUM_TENTACLES, 1) 
    total_scores = torch.zeros(NUM_TENTACLES, device=DEVICE)
    
    # 2. Simulate All Tentacles (The "Dream")
    with torch.no_grad():
        for t in range(HORIZON):
            # Predict next state
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            z_norm = torch.nn.functional.normalize(z_futures, p=2, dim=1)
            
            # Compare to Expert Memory (Cosine Similarity)
            # Matrix Multiply: [Batch, 512] x [512, BankSize] = [Batch, BankSize]
            similarity_matrix = torch.mm(z_norm, memory_bank.T)
            
            # Take average of top K matches (Robustness)
            top_sims, _ = torch.topk(similarity_matrix, k=TOP_K, dim=1)
            step_score = torch.mean(top_sims, dim=1)
            
            # Discount future steps (Immediate future matters more)
            total_scores += step_score * (0.9 ** t)

    # 3. Pick Winner
    # Penalize changing steering too abruptly
    steer_diff = torch.abs(action_seqs[:, 0, 0] - last_steer)
    final_scores = total_scores - (steer_diff * STEER_PENALTY)
    
    best_idx = torch.argmax(final_scores)
    best_action = action_seqs[best_idx, 0].cpu().numpy()
    
    # 4. DREAM VISUALIZATION (Optional)
    dream_frame = None
    if decoder is not None:
        with torch.no_grad():
            # Rollout the WINNING path again to generate the image
            z_vis = z_curr.clone()
            for t in range(HORIZON):
                z_vis = predictor(z_vis, action_seqs[best_idx, t].unsqueeze(0))
            
            # Decode the final thought
            dream_frame = decoder(z_vis)

    # Calculate confidence for HUD
    confidence = final_scores[best_idx].item()
    return best_action, confidence, dream_frame

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, decoder, memory_bank = load_models()
    
    obs, _ = env.reset()
    last_steer = 0.0
    
    print("\n🚗 MPC AUTOPILOT ENGAGED")
    print("-------------------------")
    print("Red Line   = Reality Steering")
    print("Blue Image = What the car is imagining (T+10)")
    
    try:
        while True:
            # Run Policy
            action, score, dream_img = mpc_policy(
                encoder, predictor, decoder, memory_bank, obs, last_steer
            )
            
            # Smooth controls
            steer = (MOMENTUM * last_steer) + ((1 - MOMENTUM) * action[0])
            final_action = [steer, action[1], action[2]]
            
            # Step Environment
            obs, _, done, trunc, _ = env.step(final_action)
            last_steer = steer
            
            # --- VISUALIZATION HUD ---
            # 1. Real Camera
            vis_real = cv2.resize(obs, (400, 400))
            cv2.putText(vis_real, f"Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            
            # Draw Steering Bar
            center_x = 200
            end_x = int(center_x + (steer * 100))
            cv2.line(vis_real, (center_x, 350), (end_x, 350), (0, 0, 255), 8)
            
            # 2. Dream Camera
            if dream_img is not None:
                d_img = dream_img.squeeze().cpu().permute(1, 2, 0).numpy()
                d_img = cv2.resize(d_img, (400, 400))
                d_img = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                cv2.putText(d_img, "DREAM (T+10)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                
                # Stack Side-by-Side
                hud = np.hstack((vis_real, d_img))
            else:
                hud = vis_real

            cv2.imshow("World Model Pilot", hud)
            if cv2.waitKey(1) == 27: break # ESC to quit
            
            if done or trunc:
                obs, _ = env.reset()
                last_steer = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()