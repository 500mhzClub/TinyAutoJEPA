import gymnasium as gym
import torch
import numpy as np
import cv2
import os
from networks import TinyEncoder, Predictor, TinyDecoder

# --- CONFIGURATION ---
MODEL_PATH_ENC    = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED   = "./models/predictor_multistep_final.pth"
MODEL_PATH_DEC    = "./models/decoder_parallel_ep40.pth" # Use your best decoder checkpoint
MODEL_PATH_MEMORY = "./models/memory_bank.pt"

# CONTROLLER
HORIZON = 6            
NUM_TENTACLES = 60     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TUNING (Agile Settings)
TOP_K = 10             
STEER_PENALTY = 0.08   
TEMPERATURE = 0.06     
MOMENTUM = 0.25        

def load_models():
    print(f"Loading models on {DEVICE}...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    dec = TinyDecoder().to(DEVICE).eval()
    if os.path.exists(MODEL_PATH_DEC):
        dec.load_state_dict(torch.load(MODEL_PATH_DEC, map_location=DEVICE))
        print("Decoder Loaded (Dream Mode Active)")
    else:
        print("Warning: Decoder NOT found. Visualization will be limited.")
        dec = None
        
    if not os.path.exists(MODEL_PATH_MEMORY):
        raise FileNotFoundError("Run make_dense_memory.py first!")
        
    bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    bank = torch.nn.functional.normalize(bank, p=2, dim=1)
    
    return enc, pred, dec, bank

def generate_tentacles(horizon, num_tentacles):
    actions = torch.zeros(num_tentacles, horizon, 3, device=DEVICE)
    steer_angles = torch.linspace(-1.0, 1.0, num_tentacles, device=DEVICE)
    for i, steer in enumerate(steer_angles):
        actions[i, :, 0] = steer
        # Brake for corners
        if abs(steer) > 0.5:   actions[i, :, 1] = 0.25 
        elif abs(steer) > 0.2: actions[i, :, 1] = 0.45
        else:                  actions[i, :, 1] = 0.60
        actions[i, :, 2] = 0.0
    return actions

def draw_hud(frame, dream_frame, steer, gas, score):
    # 1. Prepare Reality View
    vis_real = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
    
    # Steering Line on Reality
    center = (200, 350)
    angle = steer * 45 * (np.pi / 180)
    length = 120
    end_x = int(center[0] + length * np.sin(angle))
    end_y = int(center[1] - length * np.cos(angle))
    
    if score > 0.7: color = (0, 255, 0)
    elif score > 0.5: color = (0, 255, 255)
    else: color = (0, 0, 255)
    
    cv2.line(vis_real, center, (end_x, end_y), color, 5)
    
    # Stats
    cv2.rectangle(vis_real, (0, 0), (400, 60), (0, 0, 0), -1)
    cv2.putText(vis_real, f"Sim: {score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(vis_real, "REALITY", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 2. Prepare Dream View
    if dream_frame is not None:
        # Convert torch tensor to numpy image
        d_img = dream_frame.squeeze(0).cpu().permute(1,2,0).numpy() * 255
        d_img = d_img.clip(0, 255).astype(np.uint8)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
        vis_dream = cv2.resize(d_img, (400, 400), interpolation=cv2.INTER_NEAREST)
        
        cv2.putText(vis_dream, "PREDICTED (T=6)", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        vis_dream = np.zeros((400, 400, 3), dtype=np.uint8)

    # 3. Combine Side-by-Side
    combined = np.hstack([vis_real, vis_dream])
    
    cv2.imshow("TinyAutoJEPA: Reality vs Imagination", combined)
    cv2.waitKey(1)

def mpc_policy(encoder, predictor, decoder, memory_bank, current_frame, last_steer):
    img = cv2.resize(current_frame, (64, 64))
    t_img = torch.tensor(img).float().to(DEVICE).unsqueeze(0) / 255.0
    t_img = t_img.permute(0, 3, 1, 2)
    
    # 1. Encode Reality
    with torch.no_grad():
        z_curr = encoder(t_img)
        
    action_seqs = generate_tentacles(HORIZON, NUM_TENTACLES)
    z_futures = z_curr.repeat(NUM_TENTACLES, 1) 
    total_sim = torch.zeros(NUM_TENTACLES, device=DEVICE)
    
    # 2. Simulate All Tentacles
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])
            z_norm = torch.nn.functional.normalize(z_futures, p=2, dim=1)
            similarity_matrix = torch.mm(z_norm, memory_bank.T)
            top_sims, _ = torch.topk(similarity_matrix, k=TOP_K, dim=1)
            total_sim += torch.mean(top_sims, dim=1) * (0.95 ** t)

    # 3. Pick Winner
    steer_diff = torch.abs(action_seqs[:, 0, 0] - last_steer)
    final_scores = total_sim - (steer_diff * STEER_PENALTY)
    weights = torch.softmax(final_scores / TEMPERATURE, dim=0) 
    best_steer = torch.sum(weights * action_seqs[:, 0, 0]).item()
    
    best_idx = torch.argmax(final_scores)
    confidence = total_sim[best_idx].item() / (HORIZON * 0.8)
    target_gas = np.clip(confidence * 0.9, 0.25, 0.6)
    
    # 4. DREAM GENERATION (Re-run prediction for the Winner ONLY)
    dream_frame = None
    if decoder is not None:
        winning_actions = action_seqs[best_idx].unsqueeze(0) # [1, H, 3]
        z_dream = z_curr.clone() # [1, 512]
        
        with torch.no_grad():
            # Fast forward to the end of the horizon
            for t in range(HORIZON):
                z_dream = predictor(z_dream, winning_actions[:, t, :])
            
            # Decode the FINAL state (What does the car think T=6 looks like?)
            dream_frame = decoder(z_dream)

    return best_steer, target_gas, confidence, dream_frame

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    encoder, predictor, decoder, memory_bank = load_models()
    
    obs, _ = env.reset()
    last_action = np.array([0.0, 0.0, 0.0])
    
    print("\nDREAM VISUALIZER ACTIVE.")
    
    try:
        step = 0
        while True:
            steer, gas, score, dream_img = mpc_policy(encoder, predictor, decoder, memory_bank, obs, last_action[0])
            
            # Agile Momentum
            smooth_steer = (MOMENTUM * last_action[0]) + ((1 - MOMENTUM) * steer)
            
            # Draw HUD
            draw_hud(obs, dream_img, smooth_steer, gas, score)
            
            final_action = np.array([smooth_steer, gas, 0.0])
            obs, _, done, trunc, _ = env.step(final_action)
            last_action = final_action
            
            if step % 10 == 0:
                 print(f"Step {step} | Steer: {smooth_steer:.2f} | Score: {score:.2f}")
            step += 1
            
            if done or trunc:
                print("Crash/Reset")
                obs, _ = env.reset()
                last_action = np.array([0.0, 0.0, 0.0])
                step = 0
                
    except KeyboardInterrupt:
        print("Stopping...")
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()