import torch
import numpy as np
import cv2
import glob
import os
import random
from networks import TinyEncoder, Predictor, TinyDecoder

NUM_DREAMS = 5         
SEQ_LEN = 60           # Predict 1 context + 59 future frames
GROUNDING_INTERVAL = 1000 # Set high to DISABLE cheating (force pure prediction)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_PATH   = "./models/encoder_mixed_final.pth"
PREDICTOR_PATH = "./models/predictor_final.pth" 
# Checks for ep5 first, falls back to final if 5 isn't there yet
DECODER_PATH   = "./models/decoder_final.pth" 

def load_models():
    print(f"--- Loading Models on {DEVICE} ---")
    
    # 1. Encoder
    encoder = TinyEncoder().to(DEVICE).eval()
    if os.path.exists(ENCODER_PATH):
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
        print(f"Loaded Encoder: {ENCODER_PATH}")
    else:
        raise FileNotFoundError(f"Missing {ENCODER_PATH}")
    
    # 2. Predictor
    predictor = Predictor().to(DEVICE).eval()
    if os.path.exists(PREDICTOR_PATH):
        predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
        print(f"Loaded Predictor: {PREDICTOR_PATH}")
    else:
        raise FileNotFoundError(f"Missing {PREDICTOR_PATH}")

    # 3. Decoder (Check for ep5, else try final)
    decoder = TinyDecoder().to(DEVICE).eval()
    
    path_to_load = DECODER_PATH
    if not os.path.exists(path_to_load):
        # Fallback logic if ep5 isn't done but you have another file
        fallback = "./models/decoder_vicreg_final.pth"
        if os.path.exists(fallback):
            path_to_load = fallback
        else:
            print(f"⚠️  WARNING: Could not find {DECODER_PATH}. Waiting for training to finish...")
            raise FileNotFoundError(f"Missing Decoder checkpoint")

    decoder.load_state_dict(torch.load(path_to_load, map_location=DEVICE))
    print(f"✅ Loaded Decoder: {path_to_load}")
    
    return encoder, predictor, decoder

def preprocess_frame(img_array):
    """Helper to turn numpy image into torch tensor"""
    if img_array.shape[0] == 96: img_array = cv2.resize(img_array, (64, 64))
    tensor = torch.tensor(img_array).float().to(DEVICE) / 255.0
    if tensor.shape[0] != 3: tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0)

def generate_dream_batch():
    encoder, predictor, decoder = load_models()

    # Prefer race data (cleaner), fallback to regular
    files = glob.glob("./data_expert/*.npz")
    if not files: files = glob.glob("./data_recover/*.npz")
    if not files: 
        print("❌ No data files found.")
        return

    print(f"Found {len(files)} sequence files. Generating {NUM_DREAMS} dreams...")

    for i in range(1, NUM_DREAMS + 1):
        filename = random.choice(files)
        try:
            data = np.load(filename)
            if 'states' in data: obs, actions = data['states'], data['actions']
            elif 'obs' in data: obs, actions = data['obs'], data['action']
            else: continue
            
            # Ensure sequence is long enough
            if len(obs) < SEQ_LEN + 20: continue
            
            # Pick a random start point
            start_idx = random.randint(10, len(obs) - SEQ_LEN - 10)
            
            print(f"Dream {i}: {os.path.basename(filename)} | Frame Start: {start_idx}")
            
            current_frame = obs[start_idx]
            z = encoder(preprocess_frame(current_frame)) # Latent State (t=0)
            
            video_frames = []
            
            for t in range(SEQ_LEN):

                # We disable grounding to test if the model drifts
                is_sync_frame = (t > 0) and (t % GROUNDING_INTERVAL == 0)
                
                if is_sync_frame:
                    # Cheat: Reset to reality
                    real_t_frame = obs[start_idx + t]
                    with torch.no_grad():
                        z = encoder(preprocess_frame(real_t_frame))
                elif t > 0:
                    # Normal: Predict next step based on last thought + action
                    real_action = actions[start_idx + t - 1] # Action taken at previous step
                    act_tensor = torch.tensor(real_action).float().to(DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        z = predictor(z, act_tensor)

                with torch.no_grad():
                    recon = decoder(z)

                # Convert Dream to Image
                d_img = recon.squeeze(0).cpu().permute(1,2,0).numpy() * 255
                d_img = d_img.clip(0,255).astype(np.uint8)
                
                # Get Real Image for comparison
                r_img = obs[start_idx + t]
                if r_img.shape[0] == 96: r_img = cv2.resize(r_img, (64, 64))
                
                r_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
                d_bgr = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
                
                # Scale up for visibility
                scale = 4
                h, w = r_bgr.shape[:2]
                r_big = cv2.resize(r_bgr, (w*scale, h*scale), interpolation=0)
                d_big = cv2.resize(d_bgr, (w*scale, h*scale), interpolation=0)
                
                # Annotations
                if t == 0:
                    cv2.putText(d_big, "CONTEXT (REAL)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(d_big, f"PRED +{t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Stack: Real on Top, Dream on Bottom
                combined = np.vstack([r_big, d_big])
                video_frames.append(combined)

            # Save Video
            out_name = f"dream_pure_{i}.avi"
            h_v, w_v, _ = video_frames[0].shape
            out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'DIVX'), 20, (w_v, h_v))
            for f in video_frames: out.write(f)
            out.release()
            print(f"--> Saved {out_name}")
            
        except Exception as e:
            print(f"Skipping file due to error: {e}")

if __name__ == "__main__":
    generate_dream_batch()