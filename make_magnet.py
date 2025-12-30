import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder
import gc
from tqdm import tqdm

# --- CONFIG ---
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/expert_magnet.pth"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SAMPLES = 500 

def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print(f"Calibrating Magnet (Target: {TARGET_SAMPLES} frames)...")
    files = glob.glob("./data_race/*.npz")
    
    latents = []
    count = 0
    
    # Use tqdm on the file loop so you see overall progress
    pbar = tqdm(total=TARGET_SAMPLES, unit="frames")
    
    for f in files:
        if count >= TARGET_SAMPLES: break
        
        try:
            with np.load(f) as data:
                if 'states' in data: imgs = data['states']
                elif 'obs' in data: imgs = data['obs']
                else: continue
                
                # Take a random 100 frames from this file
                indices = np.random.choice(len(imgs), size=min(100, len(imgs)), replace=False)
                batch_cpu = imgs[indices]

            # Resize
            if batch_cpu.shape[1] != 64:
                batch_cpu = np.array([cv2.resize(img, (64,64)) for img in batch_cpu])

            # To GPU
            tensor = torch.tensor(batch_cpu).float() / 255.0
            tensor = tensor.permute(0, 3, 1, 2).to(DEVICE)
            
            with torch.no_grad():
                z = encoder(tensor)
                latents.append(z.cpu())
            
            # Update progress bar
            added = len(batch_cpu)
            count += added
            pbar.update(added)
            
        except Exception as e:
            # Print error above the progress bar so it doesn't break layout
            pbar.write(f"Skip: {e}")

    pbar.close()

    if not latents:
        print("Error: No data found.")
        return

    # Average them
    all_z = torch.cat(latents, dim=0)
    all_z = all_z[:TARGET_SAMPLES]
    
    target_mean = torch.mean(all_z, dim=0)
    
    print(f"Magnet Calculated from {len(all_z)} samples.")
    print(f"Saving to {SAVE_PATH}...")
    torch.save(target_mean, SAVE_PATH)

if __name__ == "__main__":
    main()