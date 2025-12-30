import torch
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from networks import TinyEncoder

# --- CONFIG ---
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/expert_magnet.pth"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print("Calibrating Magnet from Race Data...")
    files = glob.glob("./data_race/*.npz")
    
    # We can use MORE data now since we only run this once
    # Let's use 50 files for a super-stable average
    files = files[:50] 
    
    latents = []
    
    with torch.no_grad():
        for f in tqdm(files, desc="Processing Files"):
            try:
                data = np.load(f)
                if 'states' in data: imgs = data['states']
                elif 'obs' in data: imgs = data['obs']
                else: continue
                
                # Take every 5th frame (Dense sampling)
                batch = imgs[::5]
                
                # Resize if needed
                if batch.shape[1] != 64:
                    batch = np.array([cv2.resize(img, (64,64)) for img in batch])
                
                # Batch processing
                # Process in chunks of 256 to avoid VRAM overflow
                batch_tensor = torch.tensor(batch).float().to(DEVICE) / 255.0
                batch_tensor = batch_tensor.permute(0, 3, 1, 2)
                
                chunks = torch.split(batch_tensor, 256)
                for chunk in chunks:
                    z = encoder(chunk)
                    latents.append(z.cpu()) # Store on CPU to save GPU RAM
            except Exception as e:
                print(f"Skipping file: {e}")

    print("⚗️  Distilling Essence...")
    # Stack all vectors (on CPU)
    all_z = torch.cat(latents, dim=0)
    
    # Calculate Mean (The Magnet)
    target_mean = torch.mean(all_z, dim=0)
    
    print(f"Magnet Calculated from {len(all_z)} frames.")
    print(f"Saving to {SAVE_PATH}...")
    torch.save(target_mean, SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()