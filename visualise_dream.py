import torch
import numpy as np
import cv2
import glob
import os
import random
from networks import TinyEncoder, Predictor, TinyDecoder

# --- Configuration ---
NUM_DREAMS = 5         
SEQ_LEN = 500          # Longer dreams now possible!
GROUNDING_INTERVAL = 15 # Reset to reality every 30 frames (Disable with 9999)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths (Updated to your verified files)
ENCODER_PATH   = "./models/encoder_mixed_final.pth"
PREDICTOR_PATH = "./models/predictor_multistep_final.pth" 
DECODER_PATH   = "./models/decoder_parallel_ep40.pth"

def load_models():
    print(f"--- Loading Models on {DEVICE} ---")
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    
    predictor = Predictor().to(DEVICE).eval()
    predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    
    decoder = TinyDecoder().to(DEVICE).eval()
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    
    return encoder, predictor, decoder

def preprocess_frame(img_array):
    """Helper to turn numpy image into torch tensor"""
    if img_array.shape[0] == 96: img_array = cv2.resize(img_array, (64, 64))
    tensor = torch.tensor(img_array).float().to(DEVICE) / 255.0
    if tensor.shape[0] != 3: tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0)

def generate_dream_batch():
    encoder, predictor, decoder = load_models()

    files = glob.glob("./data_race/*.npz")
    if not files: files = glob.glob("./data/*.npz")
    if not files: return

    for i in range(1, NUM_DREAMS + 1):
        filename = random.choice(files)
        try:
            data = np.load(filename)
            if 'states' in data: obs, actions = data['states'], data['actions']
            elif 'obs' in data: obs, actions = data['obs'], data['action']
            else: continue
            
            if len(obs) < SEQ_LEN + 50: continue
            start_idx = random.randint(50, len(obs) - SEQ_LEN)
            
            print(f"Dream {i}: {os.path.basename(filename)} | Start: {start_idx}")
            
            # 1. Initial Encode
            current_frame = obs[start_idx]
            z = encoder(preprocess_frame(current_frame))
            
            video_frames = []
            
            for t in range(SEQ_LEN):
                # --- LOGIC: GROUNDING VS DREAMING ---
                is_sync_frame = (t > 0) and (t % GROUNDING_INTERVAL == 0)
                
                if is_sync_frame:
                    # CHEAT: Look at reality to reset error
                    real_t_frame = obs[start_idx + t]
                    with torch.no_grad():
                        z = encoder(preprocess_frame(real_t_frame))
                else:
                    # NORMAL: Predict next step based on last thought
                    real_action = actions[start_idx + t]
                    act_tensor = torch.tensor(real_action).float().to(DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        z = predictor(z, act_tensor)

                # --- VISUALIZE ---
                with torch.no_grad():
                    recon = decoder(z)

                d_img = recon.squeeze(0).cpu().permute(1,2,0).numpy() * 255
                d_img = d_img.clip(0,255).astype(np.uint8)
                
                r_img = obs[start_idx + t]
                if r_img.shape[0] == 96: r_img = cv2.resize(r_img, (64, 64))
                
                r_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
                d_bgr = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                
                # Scale up
                scale = 4
                h, w = r_bgr.shape[:2]
                r_big = cv2.resize(r_bgr, (w*scale, h*scale), interpolation=0)
                d_big = cv2.resize(d_bgr, (w*scale, h*scale), interpolation=0)
                
                # Visual Indicator for SYNC
                if is_sync_frame:
                    # Blue Border on Dream to show update
                    cv2.rectangle(d_big, (0,0), (w*scale, h*scale), (255, 0, 0), 10)
                    cv2.putText(d_big, "SYNC UPDATE", (20, h*scale//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                combined = np.hstack((r_big, d_big))
                video_frames.append(combined)

            # Save
            out_name = f"dream_grounded_{i}.avi"
            h_v, w_v, _ = video_frames[0].shape
            out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (w_v, h_v))
            for f in video_frames: out.write(f)
            out.release()
            print(f"--> Saved {out_name}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    generate_dream_batch()