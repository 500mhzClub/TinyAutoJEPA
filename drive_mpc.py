import gymnasium as gym
import torch
import numpy as np
import cv2
import os
import pygame 
from networks import TinyEncoder, Predictor, TinyDecoder

# --- CONFIGURATION ---
MODEL_PATH_ENC    = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED   = "./models/predictor_final.pth" 
MODEL_PATH_DEC    = "./models/decoder_final.pth" 
MODEL_PATH_MEMORY = "./models/memory_bank.pt"

# MPC PHYSICS
HORIZON       = 10     
NUM_TENTACLES = 120    
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AGENT TUNING
TOP_K         = 20     
STEER_PENALTY = 0.05   
MOMENTUM      = 0.60   

def load_models():
    print(f"Loading World Model on {DEVICE}...")
    
    # 1. Encoder (Strict .eval to match debug script)
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    # 2. Predictor
    pred = Predictor().to(DEVICE).eval()
    if os.path.exists(MODEL_PATH_PRED):
        pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    elif os.path.exists("./models/predictor_multistep_final.pth"):
        pred.load_state_dict(torch.load("./models/predictor_multistep_final.pth", map_location=DEVICE))
    else:
        raise FileNotFoundError("Could not find a predictor_final.pth file!")

    # 3. Decoder (Strict .eval)
    dec = TinyDecoder().to(DEVICE).eval()
    if os.path.exists(MODEL_PATH_DEC):
        dec.load_state_dict(torch.load(MODEL_PATH_DEC, map_location=DEVICE))
    else:
        dec = None
        
    if not os.path.exists(MODEL_PATH_MEMORY):
        raise FileNotFoundError("Run make_memory_bank.py first!")
    bank = torch.load(MODEL_PATH_MEMORY, map_location=DEVICE)
    bank = torch.nn.functional.normalize(bank, p=2, dim=1)
    
    return enc, pred, dec, bank

def generate_tentacles(horizon, num_tentacles):
    actions = torch.zeros(num_tentacles, horizon, 3, device=DEVICE)
    steer_dist = torch.randn(num_tentacles, device=DEVICE) * 0.5
    steer_dist = torch.clamp(steer_dist, -1.0, 1.0)
    
    num_uniform = num_tentacles // 4
    steer_dist[:num_uniform] = torch.rand(num_uniform, device=DEVICE) * 2 - 1
    
    for i in range(num_tentacles):
        s = steer_dist[i]
        actions[i, :, 0] = s
        if abs(s) > 0.4:   
            actions[i, :, 1] = 0.0 
            actions[i, :, 2] = 0.2 
        else:                  
            actions[i, :, 1] = 0.6 
            actions[i, :, 2] = 0.0
    return actions

def capture_window(env):
    """
    Robust dynamic resolution capture.
    """
    raw_surface = env.unwrapped.screen
    if raw_surface is None: return None
    
    # 1. Ask PyGame for the REAL resolution
    w, h = raw_surface.get_size()
    
    # 2. Use array3d (Safe copy, no lock)
    frame_t = pygame.surfarray.array3d(raw_surface)
    
    # 3. Transpose to (Height, Width, 3) for OpenCV/Torch
    frame = frame_t.transpose(1, 0, 2)
    
    # 4. Force contiguous memory
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    
    return frame

def mpc_policy(encoder, predictor, decoder, memory_bank, full_frame, last_steer):
    # 1. Resize whatever we got -> 64x64
    img = cv2.resize(full_frame, (64, 64), interpolation=cv2.INTER_AREA)

    # 2. Normalize pixels
    t_img = torch.from_numpy(img).float().to(DEVICE).div(255.0).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        # IMPORTANT:
        # - Decoder training + debug_live_vision decode *raw* encoder latents.
        # - Memory bank similarity expects *normalized* latents (cosine).
        # We rollout in raw space; normalize only when scoring.
        z_raw = encoder(t_img)
        
    action_seqs = generate_tentacles(HORIZON, NUM_TENTACLES)

    # Rollout in raw latent space
    z_futures = z_raw.repeat(NUM_TENTACLES, 1)
    total_scores = torch.zeros(NUM_TENTACLES, device=DEVICE)
    
    with torch.no_grad():
        for t in range(HORIZON):
            z_futures = predictor(z_futures, action_seqs[:, t, :])

            # Normalize ONLY for similarity scoring
            z_score = torch.nn.functional.normalize(z_futures, p=2, dim=1)
            similarity = torch.mm(z_score, memory_bank.T)

            top_sims, _ = torch.topk(similarity, k=TOP_K, dim=1)
            total_scores += torch.mean(top_sims, dim=1) * (0.9 ** t)

    steer_diff = torch.abs(action_seqs[:, 0, 0] - last_steer)
    final_scores = total_scores - (steer_diff * STEER_PENALTY)
    
    best_idx = torch.argmax(final_scores)
    best_action = action_seqs[best_idx, 0].cpu().numpy()
    
    recon_now, dream_future = None, None
    if decoder is not None:
        with torch.no_grad():
            # Decode raw latents
            recon_now = decoder(z_raw)

            # Visualize predicted future by rolling out in raw space then decoding
            z_vis = z_raw.clone()
            for t in range(HORIZON):
                z_vis = predictor(z_vis, action_seqs[best_idx, t].unsqueeze(0))
            dream_future = decoder(z_vis)

    return best_action, final_scores[best_idx].item(), recon_now, dream_future, img

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    enc, pred, dec, bank = load_models()
    
    env.reset() 
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32)) 
    
    last_steer = 0.0
    
    print("\n🏎️ AUTOPILOT ENGAGED")
    print("-------------------------")
    
    print("1. Skipping 'Zoom In' (50 frames)...")
    for _ in range(50):
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        capture_window(env)
        
    print("2. 🚀 KICKSTART: Forcing Gas for 20 frames...")
    for _ in range(20):
        env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
        capture_window(env)

    print("3. CONTROL LOOP START")
    try:
        while True:
            full_frame = capture_window(env)
            if full_frame is None: continue
            
            action, score, recon, dream, tiny_input = mpc_policy(enc, pred, dec, bank, full_frame, last_steer)
            
            steer = (MOMENTUM * last_steer) + ((1 - MOMENTUM) * action[0])
            final_action = np.array([steer, action[1], action[2]], dtype=np.float32)
            
            _, _, done, trunc, _ = env.step(final_action)
            last_steer = steer
            
            # --- HUD ---
            vis_real = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
            vis_real = cv2.resize(vis_real, (400, 300))
            cv2.putText(vis_real, "LIVE GAME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            vis_input = cv2.cvtColor(tiny_input, cv2.COLOR_RGB2BGR)
            vis_input = cv2.resize(vis_input, (100, 100), interpolation=cv2.INTER_NEAREST)
            vis_real[200:300, 300:400] = vis_input
            cv2.putText(vis_real, "INPUT", (310, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            if recon is not None and dream is not None:
                r_img = torch.clamp(recon, 0, 1).squeeze().cpu().permute(1, 2, 0).numpy()
                r_img = cv2.resize(r_img, (400, 300))
                r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
                cv2.putText(r_img, "What Net Sees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                
                d_img = torch.clamp(dream, 0, 1).squeeze().cpu().permute(1, 2, 0).numpy()
                d_img = cv2.resize(d_img, (400, 300))
                d_img = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                cv2.putText(d_img, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                vis = np.hstack((vis_real, r_img, d_img))
            else:
                vis = vis_real

            cv2.imshow("World Model Pilot", vis)
            if cv2.waitKey(1) == 27 or done or trunc: 
                print("Reset.")
                env.reset()
                last_steer = 0.0
                for _ in range(50): 
                    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    capture_window(env)
                for _ in range(20): 
                    env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
                    capture_window(env)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
