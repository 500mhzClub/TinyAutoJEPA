import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder
import gc

# --- CONFIG ---
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/memory_bank.pt"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_SIZE    = 2000     # Keep 2000 diverse examples

def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print(f"Mining {MEMORY_SIZE} expert states...")
    files = glob.glob("./data_race/*.npz")
    
    bank = []
    total_scanned = 0
    
    # We shuffle files to get diversity from different races
    np.random.shuffle(files)
    
    with torch.no_grad():
        for f in files:
            if len(bank) >= MEMORY_SIZE: break
            
            try:
                with np.load(f) as data:
                    if 'states' in data: imgs = data['states']
                    elif 'obs' in data: imgs = data['obs']
                    else: continue
                
                # Take random diverse samples from this file
                # We don't want 100 frames from the same straightaway
                indices = np.random.choice(len(imgs), size=min(50, len(imgs)), replace=False)
                batch_cpu = imgs[indices]
                
                # Resize
                if batch_cpu.shape[1] != 64:
                    batch_cpu = np.array([cv2.resize(img, (64,64)) for img in batch_cpu])

                # Encode
                tensor = torch.tensor(batch_cpu).float().to(DEVICE) / 255.0
                tensor = tensor.permute(0, 3, 1, 2)
                z = encoder(tensor)
                
                # Add to bank
                bank.append(z.cpu())
                total_scanned += len(batch_cpu)
                print(f"Collected {len(bank)}/{MEMORY_SIZE} vectors...", end="\r")
                
            except Exception as e:
                print(f"Skip {f}: {e}")

    if not bank:
        print("Error: No data found.")
        return

    # Cat and Slice
    memory_bank = torch.cat(bank, dim=0)
    
    # Ensure we didn't go over (though loop handles it, cat might exceed slightly)
    memory_bank = memory_bank[:MEMORY_SIZE]
    
    print(f"\nBank Assembled. Shape: {memory_bank.shape}")
    print(f"Saving to {SAVE_PATH}...")
    torch.save(memory_bank, SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()