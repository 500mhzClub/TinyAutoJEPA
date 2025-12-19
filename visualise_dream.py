import torch
import numpy as np
import cv2
import glob
import os
import random
from networks import TinyEncoder, Predictor, TinyDecoder

# --- Configuration ---
NUM_DREAMS = 5         # How many videos to generate
SEQ_LEN = 150          # How long to dream (150 frames = ~3 seconds)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths
ENCODER_PATH   = "./models/encoder_final_mixed.pth"
PREDICTOR_PATH = "./models/predictor_race_final.pth" 
DECODER_PATH   = "./models/decoder_race_final.pth"     

# Fallbacks
if not os.path.exists(ENCODER_PATH): ENCODER_PATH = "./models/encoder_ep20.pth"

def load_models():
    print(f"--- Loading Models on {DEVICE} ---")
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    
    predictor = Predictor().to(DEVICE)
    # Try loading race predictor, fall back to final, then ep10
    if os.path.exists(PREDICTOR_PATH): 
        predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    elif os.path.exists("./models/predictor_final.pth"):
        predictor.load_state_dict(torch.load("./models/predictor_final.pth", map_location=DEVICE))
    
    decoder = TinyDecoder().to(DEVICE)
    if os.path.exists(DECODER_PATH): 
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    
    encoder.eval(); predictor.eval(); decoder.eval()
    return encoder, predictor, decoder

def generate_dream_batch():
    encoder, predictor, decoder = load_models()

    # 1. Find Data (Prioritize Race Data)
    files = glob.glob("./data_race/*.npz")
    if len(files) > 0:
        print(f"Found {len(files)} RACE recordings. Using these.")
    else:
        print("No race data found in ./data_race/. Falling back to random data.")
        files = glob.glob("./data/*.npz")
        
    if not files:
        print("CRITICAL: No data found anywhere.")
        return

    # 2. Loop to create batch
    for i in range(1, NUM_DREAMS + 1):
        # Pick random file
        filename = random.choice(files)
        
        try:
            data = np.load(filename)
            # Robust key fetch
            if 'states' in data: obs, actions = data['states'], data['actions']
            elif 'obs' in data: obs, actions = data['obs'], data['action']
            else: continue
            
            # Pick random start time (ensure enough frames left)
            if len(obs) < SEQ_LEN + 50: continue
            start_idx = random.randint(50, len(obs) - SEQ_LEN)
            
            print(f"Generating Dream {i}/{NUM_DREAMS} | Source: {os.path.basename(filename)} | Frame: {start_idx}")
            
            # --- PREPARE SEED ---
            seed_img = obs[start_idx] # HWC
            
            # Auto-Resize 96->64 if needed
            if seed_img.shape[0] == 96: seed_img = cv2.resize(seed_img, (64, 64))
            if seed_img.shape[0] != 64: seed_img = cv2.resize(seed_img, (64, 64))

            # To Tensor
            frame = torch.tensor(seed_img).float().to(DEVICE) / 255.0
            if frame.shape[0] != 3: frame = frame.permute(2, 0, 1) # HWC -> CHW
            
            # Encode Initial State
            with torch.no_grad():
                z = encoder(frame.unsqueeze(0))
            
            # --- DREAM LOOP ---
            video_frames = []
            
            for t in range(SEQ_LEN):
                with torch.no_grad():
                    # 1. Decode what we "see"
                    recon = decoder(z)
                    
                    # 2. Get the Action the PRO actually took
                    # (We feed this to the model to see if it predicts the result correctly)
                    real_action = actions[start_idx + t]
                    act_tensor = torch.tensor(real_action).float().to(DEVICE).unsqueeze(0)
                    
                    # 3. Predict Next State
                    z = predictor(z, act_tensor)

                # --- VISUALIZATION ---
                # Get Dream Image
                d_img = recon.squeeze(0).cpu().permute(1,2,0).numpy() * 255
                d_img = d_img.clip(0,255).astype(np.uint8)

                # Get Real Image (for comparison)
                r_img = obs[start_idx + t]
                if r_img.shape[0] == 96: r_img = cv2.resize(r_img, (64, 64))
                
                # Side-by-Side
                r_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
                d_bgr = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                
                # Upscale for viewing (x4)
                scale = 4
                h, w = r_bgr.shape[:2]
                r_big = cv2.resize(r_bgr, (w*scale, h*scale), interpolation=0)
                d_big = cv2.resize(d_bgr, (w*scale, h*scale), interpolation=0)
                
                combined = np.hstack((r_big, d_big))
                
                # Overlays
                cv2.putText(combined, f"Dream {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(combined, "REALITY", (10, h*scale - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(combined, "AI PREDICTION", (w*scale + 10, h*scale - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                video_frames.append(combined)

            # --- SAVE VIDEO ---
            out_name = f"dream_batch_{i}.avi"
            h_v, w_v, _ = video_frames[0].shape
            out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (w_v, h_v))
            for f in video_frames: out.write(f)
            out.release()
            print(f"--> Saved {out_name}")
            
        except Exception as e:
            print(f"Skipping file due to error: {e}")

if __name__ == "__main__":
    generate_dream_batch()