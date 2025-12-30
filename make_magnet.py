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
BATCH_SIZE     = 128 

def main():
    # Force clean start
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print("Calibrating Magnet from Race Data...")
    # Limit to 32 files to keep it quick
    files = glob.glob("./data_race/*.npz")[:32]
    
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

            num_samples = len(batch_cpu)
            
            # INNER LOOP PROGRESS BAR
            # This will show you exactly how fast the GPU is crunching
            print(f"[{file_idx+1}/{len(files)}] {filename}:")
            
            with torch.no_grad():
                with tqdm(total=num_samples, unit="frames", leave=False) as pbar:
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
            
            # Cleanup
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