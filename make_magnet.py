import torch
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from networks import TinyEncoder
import time
import gc

# --- CONFIG ---
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/expert_magnet.pth"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 128  # Safe batch size

def main():
    # 1. Force Clean Start
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print("Calibrating Magnet from Race Data...")
    files = glob.glob("./data_race/*.npz")[:50] # Process up to 50 files
    
    latents = []
    total_frames = 0
    
    print(f"Found {len(files)} files.")

    for file_idx, f in enumerate(files):
        filename = os.path.basename(f)
        try:
            # Load Data (CPU)
            with np.load(f) as data:
                if 'states' in data: imgs = data['states']
                elif 'obs' in data: imgs = data['obs']
                else: continue
                
                # Subsample (Every 5th frame)
                batch_cpu = imgs[::5]

            # Resize (CPU)
            if batch_cpu.shape[1] != 64:
                batch_cpu = np.array([cv2.resize(img, (64,64)) for img in batch_cpu])

            # Encode (GPU Batching)
            num_samples = len(batch_cpu)
            
            # Print status every file
            print(f"[{file_idx+1}/{len(files)}] {filename}: Encoding {num_samples} frames...", end="", flush=True)
            
            with torch.no_grad():
                for i in range(0, num_samples, BATCH_SIZE):
                    chunk_cpu = batch_cpu[i : i + BATCH_SIZE]
                    
                    # To GPU
                    chunk_tensor = torch.tensor(chunk_cpu).float() / 255.0
                    chunk_tensor = chunk_tensor.permute(0, 3, 1, 2).to(DEVICE)
                    
                    # Encode
                    z = encoder(chunk_tensor)
                    latents.append(z.cpu()) # Move to CPU immediately
            
            print(" Done.")
            total_frames += num_samples
            
            # Aggressive Cleanup
            del batch_cpu
            del chunk_tensor
            del z
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError on {filename}: {e}")

    if not latents:
        print("Error: No data processed.")
        return

    print(f"\nDistilling Essence from {total_frames} frames...")
    all_z = torch.cat(latents, dim=0)
    target_mean = torch.mean(all_z, dim=0)
    
    print(f"Saving to {SAVE_PATH}...")
    torch.save(target_mean, SAVE_PATH)
    print("Magnet Created Successfully.")

if __name__ == "__main__":
    main()