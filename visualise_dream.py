import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder, Predictor, TinyDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "./models/encoder_final_mixed.pth"
PREDICTOR_PATH = "./models/predictor_race_final.pth" 
DECODER_PATH = "./models/decoder_race_final.pth"     

if not os.path.exists(ENCODER_PATH): ENCODER_PATH = "./models/encoder_ep20.pth"

def create_dream_video():
    print("Loading models...")
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    
    predictor = Predictor().to(DEVICE)
    if os.path.exists(PREDICTOR_PATH): predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    
    decoder = TinyDecoder().to(DEVICE)
    if os.path.exists(DECODER_PATH): decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    
    encoder.eval(); predictor.eval(); decoder.eval()

    # Prefer Race Data
    race_files = glob.glob("./data_race/*.npz")
    if not race_files:
        print("No race files found, using random data.")
        race_files = glob.glob("./data/*.npz")
    
    # Load Data
    data = np.load(race_files[0])
    if 'states' in data: obs, actions = data['states'], data['actions']
    else: obs, actions = data['obs'], data['action']

    print(f"Dreaming from {race_files[0]}...")
    
    start_idx = 200
    seq_len = 100
    
    # --- RESIZE SEED FRAME IF NEEDED ---
    seed_img = obs[start_idx]
    if seed_img.shape[0] == 96: # HWC
        seed_img = cv2.resize(seed_img, (64, 64))
    # -----------------------------------

    # Prepare Tensor
    frame = torch.tensor(seed_img).float().to(DEVICE) / 255.0
    if frame.shape[0] != 3: frame = frame.permute(2, 0, 1) # HWC->CHW
    z = encoder(frame.unsqueeze(0))
    
    frames = []
    
    for t in range(seq_len):
        with torch.no_grad():
            recon = decoder(z)
            act = torch.tensor(actions[start_idx+t]).float().to(DEVICE).unsqueeze(0)
            z = predictor(z, act)
            
        # Vis Logic
        r_img = obs[start_idx+t] # HWC (might be 96 or 64)
        
        # If Real is 96, resize to 64 for visualization consistency
        if r_img.shape[0] == 96: r_img = cv2.resize(r_img, (64,64))

        d_img = recon.squeeze(0).cpu().permute(1,2,0).numpy() * 255
        d_img = d_img.clip(0,255).astype(np.uint8)
        
        r_bgr = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
        d_bgr = cv2.cvtColor(d_img, cv2.COLOR_RGB2BGR)
        
        s = 4
        h, w = r_bgr.shape[:2]
        r_big = cv2.resize(r_bgr, (w*s, h*s), interpolation=0)
        d_big = cv2.resize(d_bgr, (w*s, h*s), interpolation=0)
        
        combined = np.hstack((r_big, d_big))
        cv2.putText(combined, "Reality", (10,30), 0, 1, (0,255,0), 2)
        cv2.putText(combined, "Dream", (w*s+10,30), 0, 1, (0,0,255), 2)
        frames.append(combined)

    out = cv2.VideoWriter('dream_race.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames: out.write(f)
    out.release()
    print("Saved dream_race.avi")

if __name__ == "__main__":
    create_dream_video()