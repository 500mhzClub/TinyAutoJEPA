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
    
    # We shuffle files to get diversity from different races
    np.random.shuffle(files)
    
    with torch.no_grad():
        with tqdm(total=MEMORY_SIZE, unit="vecs") as pbar:
            for f in files:
                if len(bank) >= MEMORY_SIZE: break
                
                try:
                    with np.load(f) as data:
                        if 'states' in data: imgs = data['states']
                        elif 'obs' in data: imgs = data['obs']
                        else: continue
                    
                    # Take random diverse samples from this file
                    # We don't want 100 frames from the same straightaway
                    num_to_take = min(50, len(imgs))
                    indices = np.random.choice(len(imgs), size=num_to_take, replace=False)
                    batch_cpu = imgs[indices]
                    
                    # Resize
                    if batch_cpu.shape[1] != 64:
                        batch_cpu = np.array([cv2.resize(img, (64,64)) for img in batch_cpu])

                    # Encode
                    tensor = torch.tensor(batch_cpu).float().to(DEVICE) / 255.0
                    tensor = tensor.permute(0, 3, 1, 2)
                    z = encoder(tensor)
                    
                    # Add to bank
                    # We iterate to make sure we don't overshoot exactly 2000 in the list
                    # (though slicing at the end handles it, this keeps the bar accurate)
                    for vec in z:
                        if len(bank) >= MEMORY_SIZE: break
                        bank.append(vec.unsqueeze(0).cpu())
                        pbar.update(1)
                    
                except Exception as e:
                    # Write to a new line so we don't break the bar
                    pbar.write(f"Skip {f}: {e}")

    if not bank:
        print("Error: No data found.")
        return

    # Cat and Slice
    memory_bank = torch.cat(bank, dim=0)
    
    print(f"Bank Assembled. Shape: {memory_bank.shape}")
    print(f"Saving to {SAVE_PATH}...")
    torch.save(memory_bank, SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()