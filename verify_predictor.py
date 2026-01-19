import torch
import numpy as np
import cv2
import glob
import os
import random
from torchvision.utils import save_image
from networks import TinyEncoder, TinyDecoder, Predictor

# --- CONFIGURATION ---
ENCODER_PATH = "./models/encoder_mixed_final.pth"
DECODER_PATH = "./models/decoder_final.pth"
PRED_PATH    = "./models/predictor_final.pth"
# Fallback if you named it differently
PRED_PATH_ALT= "./models/predictor_multistep_final.pth"

DATA_PATTERN = "./data_expert/*.npz" 
HORIZON      = 10   # How many steps to dream into the future
NUM_SEQS     = 5    # How many different examples to generate
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print(f"Loading Models on {DEVICE}...")
    
    # Encoder
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    
    # Decoder
    decoder = TinyDecoder().to(DEVICE).eval()
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    
    # Predictor
    predictor = Predictor().to(DEVICE).eval()
    if os.path.exists(PRED_PATH):
        predictor.load_state_dict(torch.load(PRED_PATH, map_location=DEVICE))
        print(f"Loaded Predictor: {PRED_PATH}")
    elif os.path.exists(PRED_PATH_ALT):
        predictor.load_state_dict(torch.load(PRED_PATH_ALT, map_location=DEVICE))
        print(f"Loaded Predictor: {PRED_PATH_ALT}")
    else:
        raise FileNotFoundError("Predictor model not found!")
        
    return encoder, decoder, predictor

def process_frame(img):
    # Standardize to 64x64 uint8 -> float 0-1
    if img.shape[0] != 64 or img.shape[1] != 64:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    
    if img.dtype != np.uint8:
        if img.max() <= 1.05: img = (img * 255).astype(np.uint8)
        else: img = img.astype(np.uint8)
        
    t = torch.from_numpy(img).float().to(DEVICE).div(255.0)
    return t.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]

def main():
    encoder, decoder, predictor = load_models()
    
    files = glob.glob(DATA_PATTERN)
    if not files: raise FileNotFoundError("No expert data found!")
    
    print(f"Found {len(files)} sequence files. Generating {NUM_SEQS} dreams...")
    
    for seq_idx in range(NUM_SEQS):
        # 1. Load a random file
        f = random.choice(files)
        with np.load(f) as data:
            if 'states' in data: obs = data['states']
            elif 'obs' in data: obs = data['obs']
            else: continue
            
            if 'actions' in data: actions = data['actions']
            else: continue
        
        # Need enough length
        if len(obs) < HORIZON + 20: continue
        
        # Pick a random start point (avoiding the very beginning zoom-in)
        start_t = random.randint(10, len(obs) - HORIZON - 1)
        
        # 2. Get Initial State
        frame_0 = obs[start_t]
        z_current = encoder(process_frame(frame_0))
        
        # Lists to store images for the grid
        real_sequence = []
        dream_sequence = []
        
        # Add T=0
        real_sequence.append(process_frame(frame_0))
        dream_sequence.append(decoder(z_current)) # Reconstruct T=0
        
        # 3. Dream Loop
        with torch.no_grad():
            for t in range(HORIZON):
                # Get Real Action at this step
                # Action shape in file is usually (3,) -> Need (1, 3)
                act_np = actions[start_t + t]
                act_t = torch.from_numpy(act_np).float().to(DEVICE).unsqueeze(0)
                
                # PREDICT NEXT STATE
                z_next = predictor(z_current, act_t)
                
                # DECODE DREAM
                dream_img = decoder(z_next)
                dream_sequence.append(dream_img)
                
                # GET REALITY FOR COMPARISON
                real_next_frame = obs[start_t + t + 1]
                real_sequence.append(process_frame(real_next_frame))
                
                # Update loop
                z_current = z_next

        # 4. Save Grid
        # Stack Top (Real) and Bottom (Dream)
        row_real = torch.cat(real_sequence, dim=3) # Cat width-wise
        row_dream = torch.cat(dream_sequence, dim=3)
        grid = torch.cat([row_real, row_dream], dim=2) # Cat height-wise
        
        save_path = f"visuals/dream_test_{seq_idx+1}.png"
        os.makedirs("visuals", exist_ok=True)
        save_image(grid, save_path)
        print(f"--> Saved {save_path} | Frame start: {start_t}")

if __name__ == "__main__":
    main()