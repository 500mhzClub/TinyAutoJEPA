import gymnasium as gym
import torch
import numpy as np
import cv2
import os
import pygame 
from networks import TinyEncoder, TinyDecoder

# --- CONFIG ---
ENC_PATH = "./models/encoder_mixed_final.pth"
DEC_PATH = "./models/decoder_final.pth"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print(f"Loading Models on {DEVICE}...")
    
    # Force .eval() mode
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(ENC_PATH, map_location=DEVICE))
    
    dec = TinyDecoder().to(DEVICE).eval()
    dec.load_state_dict(torch.load(DEC_PATH, map_location=DEVICE))
    
    return enc, dec

def capture_window(env):
    """
    Robust dynamic resolution capture.
    """
    raw_surface = env.unwrapped.screen
    if raw_surface is None: return None
    
    # 1. Ask PyGame for the REAL resolution
    # (On your system, this is likely 1000x800)
    w, h = raw_surface.get_size()
    
    # 2. Use array3d (Safe copy, no lock)
    # Returns (Width, Height, 3)
    frame_t = pygame.surfarray.array3d(raw_surface)
    
    # 3. Transpose to (Height, Width, 3) for OpenCV/Torch
    frame = frame_t.transpose(1, 0, 2)
    
    # 4. Force contiguous memory (Fixes artifacts)
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    
    return frame

def get_reconstruction(encoder, decoder, frame_b1):
    """
    Runs inference using the Phantom Batch Hack
    """
    # 1. Resize & Normalize
    # We resize whatever resolution we get (1000x800) down to 64x64
    img = cv2.resize(frame_b1, (64, 64), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().to(DEVICE).div(255.0)
    t = t.permute(2, 0, 1).unsqueeze(0) # [1, 3, 64, 64]

    with torch.no_grad():
        # --- PATH A: Naive Batch 1 ---
        z_b1 = encoder(t)
        recon_b1 = decoder(z_b1)
        
        # --- PATH B: Phantom Batch 8 ---
        # Duplicate the frame 8 times
        t_b8 = t.repeat(8, 1, 1, 1) 
        z_b8 = encoder(t_b8)
        out_b8 = decoder(z_b8)
        recon_b8 = out_b8[0:1] # Take just the first one
        
    return recon_b1, recon_b8

def main():
    env = gym.make("CarRacing-v3", render_mode="human")
    enc, dec = load_models()
    
    env.reset()
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    print("\n🎥 STARTING VISION TEST...")
    print("Compare 'Batch 1' (Left) vs 'Batch 8' (Right)")
    
    gas_timer = 0
    
    while True:
        # 1. Auto-Drive
        gas_timer += 1
        action = [0.0, 0.0, 0.0]
        if gas_timer < 50: action = [0.0, 0.6, 0.0] 
        elif gas_timer < 70: action = [-0.5, 0.0, 0.0] 
        else: gas_timer = 0
        
        env.step(np.array(action, dtype=np.float32))
        
        # 2. Capture
        full_frame = capture_window(env)
        if full_frame is None: continue
        
        # 3. Run Inference
        r_b1, r_b8 = get_reconstruction(enc, dec, full_frame)
        
        # 4. Visualize
        img_b1 = torch.clamp(r_b1, 0, 1).squeeze().cpu().permute(1, 2, 0).numpy()
        img_b1 = cv2.resize(img_b1, (300, 300))
        img_b1 = cv2.cvtColor(img_b1, cv2.COLOR_RGB2BGR)
        cv2.putText(img_b1, "BATCH 1 (Naive)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        img_b8 = torch.clamp(r_b8, 0, 1).squeeze().cpu().permute(1, 2, 0).numpy()
        img_b8 = cv2.resize(img_b8, (300, 300))
        img_b8 = cv2.cvtColor(img_b8, cv2.COLOR_RGB2BGR)
        cv2.putText(img_b8, "BATCH 8 (Phantom)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        vis = np.hstack((img_b1, img_b8))
        cv2.imshow("Vision Diagnostic", vis)
        
        if cv2.waitKey(1) == 27: break

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()