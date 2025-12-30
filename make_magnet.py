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
BATCH_SIZE     = 128  # Safe batch size

def main():
    print(f"🔌 Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print("Calibrating Magnet from Race Data...")
    files = glob.glob("./data_race/*.npz")
    
    # Use 32 files for a solid average
    files = files[:32] 
    
    latents = []
    total_frames = 0
    
    with torch.no_grad():
        for f in tqdm(files, desc="Processing Files"):
            try:
                # 1. Load to RAM (CPU)
                with np.load(f) as data:
                    if 'states' in data: imgs = data['states']
                    elif 'obs' in data: imgs = data['obs']
                    else: continue
                
                # Subsample: Take every 5th frame
                batch_cpu = imgs[::5]
                
                # Resize on CPU if needed
                if batch_cpu.shape[1] != 64:
                    batch_cpu = np.array([cv2.resize(img, (64,64)) for img in batch_cpu])
                
                # 2. Process in mini-batches to save VRAM
                # We loop through the CPU array and only send small chunks to GPU
                num_samples = len(batch_cpu)
                for i in range(0, num_samples, BATCH_SIZE):
                    # Slice the CPU array
                    chunk_cpu = batch_cpu[i : i + BATCH_SIZE]
                    
                    # Prepare Tensor
                    chunk_tensor = torch.tensor(chunk_cpu).float() / 255.0
                    chunk_tensor = chunk_tensor.permute(0, 3, 1, 2) # NHWC -> NCHW
                    
                    # SEND TO GPU NOW (Small chunk only)
                    chunk_tensor = chunk_tensor.to(DEVICE)
                    
                    # Encode
                    z = encoder(chunk_tensor)
                    
                    # Move result back to CPU immediately
                    latents.append(z.cpu())
                    
                total_frames += num_samples
                
            except Exception as e:
                print(f"Skipping file {f}: {e}")

    if not latents:
        print("❌ Error: No data found or processed.")
        return

    print(f"⚗️  Distilling Essence from {total_frames} frames...")
    
    # Stack all vectors (Safe on CPU)
    all_z = torch.cat(latents, dim=0)
    
    # Calculate Mean (The Magnet)
    target_mean = torch.mean(all_z, dim=0)
    
    print(f"Magnet Calculated.")
    print(f"Saving to {SAVE_PATH}...")
    torch.save(target_mean, SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()