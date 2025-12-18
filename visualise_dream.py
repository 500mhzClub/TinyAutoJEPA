import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder, Predictor, TinyDecoder

# --- Config ---
# Adjust these paths to point to your best models later
ENCODER_PATH = "./models/encoder_ep20.pth" 
PREDICTOR_PATH = "./models/predictor_final.pth" # Or predictor_ep10.pth etc
DECODER_PATH = "./models/decoder_final.pth"     # You need to train this briefly!
DATA_PATH = "./data/*.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_dream_video():
    print(f"Loading models on {DEVICE}...")
    
    # 1. Load the Trinity
    encoder = TinyEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()

    predictor = Predictor().to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    predictor.eval()

    decoder = TinyDecoder().to(DEVICE)
    # Check if decoder exists, otherwise we can't visualize
    if not os.path.exists(DECODER_PATH):
        print("CRITICAL: No decoder found. Run train_decoder.py first!")
        return
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    decoder.eval()

    # 2. Load one real episode
    files = glob.glob(DATA_PATH)
    data = np.load(files[0]) # Just pick the first file
    
    # Handle keys again (just like in training)
    if 'states' in data: 
        obs = data['states']
        actions = data['actions']
    else:
        obs = data['obs']
        actions = data['action']

    # 3. The Setup
    # We will start at frame 50 to ensure the car is moving
    start_idx = 50
    seq_len = 60 # Dream for 60 frames (1 second at 60fps)
    
    # Prepare inputs
    # Get the "Seed" frame (Real Reality at t=0)
    current_frame = torch.tensor(obs[start_idx]).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
    
    # Get the sequence of actions we DID take in reality
    action_seq = torch.tensor(actions[start_idx : start_idx+seq_len]).float().to(DEVICE)
    
    print("Dreaming...")
    frames = []

    with torch.no_grad():
        # A. Encode the starting point
        z = encoder(current_frame)
        
        for t in range(seq_len):
            # 1. Decode the current thought (Visualize what the AI is thinking)
            recon = decoder(z)
            
            # 2. Predict the next thought (The Physics Step)
            # Take the current thought + the action we took at this time
            current_action = action_seq[t].unsqueeze(0) # (1, 3)
            z_next = predictor(z, current_action)
            
            # 3. Update state (Closed Loop: Input is previous output)
            z = z_next

            # --- Visualization Processing ---
            # Get the Real Frame for comparison
            real_img = obs[start_idx + t] # (64, 64, 3)
            
            # Process Dream Image
            dream_img = recon.squeeze(0).permute(1,2,0).cpu().numpy() # (64, 64, 3)
            dream_img = (dream_img * 255).astype(np.uint8)
            dream_img = cv2.cvtColor(dream_img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
            
            # Process Real Image
            real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
            
            # Stitch them side by side
            # Resize for easier viewing (200x200)
            real_big = cv2.resize(real_img, (200, 200), interpolation=cv2.INTER_NEAREST)
            dream_big = cv2.resize(dream_img, (200, 200), interpolation=cv2.INTER_NEAREST)
            
            # Add Text
            cv2.putText(real_big, "Reality", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(dream_big, "Dream", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Combine
            combined = np.hstack((real_big, dream_big))
            frames.append(combined)

    # 4. Save Video
    height, width, layers = frames[0].shape
    out = cv2.VideoWriter('dream_result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
    
    for f in frames:
        out.write(f)
    out.release()
    print("Saved dream_result.avi")

if __name__ == "__main__":
    create_dream_video()