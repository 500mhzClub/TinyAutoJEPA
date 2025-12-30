import torch
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from networks import TinyEncoder
import time

# --- CONFIG ---
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/expert_magnet.pth"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 256  # Larger batch since your GPU is bored

def main():
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print("Calibrating Magnet from Race Data...")
    files = glob.glob("./data_race/*.npz")
    files = files[:32] # Process 32 files
    
    latents = []
    total_frames = 0
    
    print(f"Found {len(files)} files. Starting processing...")

    # Outer Loop: Files
    for file_idx, f in enumerate(files):
        filename = os.path.basename(f)
        try:
            # Explicit load start message
            print(f"\n[File {file_idx+1}/{len(files)}] Loading {filename}...", end=" ", flush=True)
            t0 = time.time()
            
            with np.load(f) as data:
                if 'states' in data: imgs = data['states']
                elif 'obs' in data: imgs = data['obs']
                else: 
                    print("Skipped (No key found)")
                    continue
                
                # Subsample (Take every 5th frame)
                batch_cpu = imgs[::5]
                load_time = time.time() - t0
                print(f"✅ Loaded {len(batch_cpu)} frames in {load_time:.2f}s.")

            # Resize loop (CPU intensive)
            if batch_cpu.shape[1] != 64:
                # Use a quick list comprehension
                batch_cpu = np.array([cv2.resize(img, (64,64)) for img in batch_cpu])

            # Inner Loop: Batches (Shows progress bar PER FILE)
            num_samples = len(batch_cpu)
            
            # Create a mini-progress bar for just this file
            with tqdm(total=num_samples, desc="  Encoding", unit="fr", leave=False) as pbar:
                for i in range(0, num_samples, BATCH_SIZE):
                    chunk_cpu = batch_cpu[i : i + BATCH_SIZE]
                    
                    # To GPU
                    chunk_tensor = torch.tensor(chunk_cpu).float() / 255.0
                    chunk_tensor = chunk_tensor.permute(0, 3, 1, 2).to(DEVICE)
                    
                    # Encode
                    z = encoder(chunk_tensor)
                    latents.append(z.cpu())
                    
                    pbar.update(len(chunk_cpu))
            
            total_frames += num_samples

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    if not latents:
        print("Error: No data found.")
        return

    print(f"\nDistilling from {total_frames} frames...")
    all_z = torch.cat(latents, dim=0)
    target_mean = torch.mean(all_z, dim=0)
    
    print(f"Saving to {SAVE_PATH}...")
    torch.save(target_mean, SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()