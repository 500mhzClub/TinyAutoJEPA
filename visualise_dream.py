import torch
import numpy as np
import cv2
import glob
import os
import sys
from networks import TinyEncoder, Predictor, TinyDecoder

# --- Configuration ---
# We use the "Preview Build" models you just trained
ENCODER_PATH   = "./models/encoder_ep20.pth"
PREDICTOR_PATH = "./models/predictor_ep10.pth" 
DECODER_PATH   = "./models/decoder_ep5.pth"      # <--- We will wait for this one!
DATA_PATH      = "./data/*.npz"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dream_video():
    print(f"--- Generating Dream on {DEVICE} ---")
    
    # 1. Load the Trinity
    print("Loading models...")
    
    # Encoder
    if not os.path.exists(ENCODER_PATH):
        print(f"CRITICAL: {ENCODER_PATH} missing.")
        return
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()

    # Predictor
    if not os.path.exists(PREDICTOR_PATH):
        print(f"CRITICAL: {PREDICTOR_PATH} missing.")
        return
    predictor = Predictor().to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    predictor.eval()

    # Decoder
    if not os.path.exists(DECODER_PATH):
        print(f"CRITICAL: {DECODER_PATH} missing. (Did you wait for Epoch 5?)")
        # Fallback to checking for final if ep5 is missing
        if os.path.exists("./models/decoder_final.pth"):
            print("Found decoder_final.pth, using that instead.")
            decoder_path = "./models/decoder_final.pth"
        else:
            return
    else:
        decoder_path = DECODER_PATH

    decoder = TinyDecoder().to(DEVICE)
    decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))
    decoder.eval()

    # 2. Load Data
    files = glob.glob(DATA_PATH)
    if not files:
        print("No data files found.")
        return
    
    # Load a random file
    print(f"Loading data from {files[0]}...")
    try:
        data = np.load(files[0])
        # Robust key fetch
        if 'states' in data: 
            obs = data['states']
            actions = data['actions']
        elif 'obs' in data:
            obs = data['obs']
            actions = data['action']
        elif 'arr_0' in data:
            obs = data['arr_0']
            actions = data['arr_1']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. The Setup
    # Start deeper in the episode to ensure movement
    start_idx = 100 
    seq_len = 120 # 2 seconds of dreaming
    
    if len(obs) < start_idx + seq_len:
        print("File too short for sequence length.")
        return

    # Prepare initial frame (The "Seed")
    seed_frame = torch.tensor(obs[start_idx]).float().to(DEVICE) / 255.0
    if seed_frame.ndim == 3: seed_frame = seed_frame.unsqueeze(0) # (1, 3, 64, 64)
    if seed_frame.shape[1] != 3: seed_frame = seed_frame.permute(0, 3, 1, 2) # Ensure NCHW

    # Get the action sequence we actually took
    action_seq = torch.tensor(actions[start_idx : start_idx+seq_len]).float().to(DEVICE)
    if action_seq.ndim == 1: action_seq = action_seq.unsqueeze(1) # Ensure (T, 3)

    print("Dreaming sequence...")
    frames = []

    with torch.no_grad():
        # A. Encode the starting point (The "Open Eye" Phase)
        z = encoder(seed_frame)
        
        for t in range(seq_len):
            # 1. Decode current thought (Visualization)
            recon = decoder(z)
            
            # 2. Predict next state (The "Brain" Step)
            # Unsqueeze action to match batch size (1, 3)
            current_action = action_seq[t].unsqueeze(0) 
            z_next = predictor(z, current_action)
            
            # 3. Update state (Closed Loop: Input is previous output)
            z = z_next

            # --- Visualization Processing ---
            # Get Real Frame
            real_img = obs[start_idx + t] # HWC, 0-255
            
            # Get Dream Frame
            dream_img = recon.squeeze(0).permute(1,2,0).cpu().numpy() # HWC, 0-1
            dream_img = (dream_img * 255).clip(0, 255).astype(np.uint8)
            
            # Convert to BGR for OpenCV
            real_bgr = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
            dream_bgr = cv2.cvtColor(dream_img, cv2.COLOR_RGB2BGR)
            
            # Resize for visibility (Zoom x4)
            scale = 4
            h, w = real_bgr.shape[:2]
            real_big = cv2.resize(real_bgr, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
            dream_big = cv2.resize(dream_bgr, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
            
            # Labels
            cv2.putText(real_big, "REALITY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(dream_big, "DREAM (Closed Eyes)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Stitch
            combined = np.hstack((real_big, dream_big))
            frames.append(combined)

    # 4. Save Video
    out_path = 'dream_result.avi'
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    
    for f in frames:
        out.write(f)
    out.release()
    print(f"Saved {out_path} - Open this file to see the dream!")

if __name__ == "__main__":
    create_dream_video()