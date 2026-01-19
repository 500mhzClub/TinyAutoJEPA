import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder
from tqdm import tqdm

# --- CONFIG ---
# Only use the BEST data for the "Target" memory bank
DATA_PATTERN   = "./data_expert/*.npz" 
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/memory_bank.pt"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_SIZE    = 10000  # Increased for better density

def main():
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print(f"Mining {MEMORY_SIZE} EXPERT states...")
    files = glob.glob(DATA_PATTERN)
    
    if not files:
        # Fallback if you named it 'data_race'
        files = glob.glob("./data_race/*.npz")
        print(f"Warning: data_expert not found, using data_race ({len(files)} files)")

    np.random.shuffle(files)
    bank = []
    
    with torch.no_grad():
        with tqdm(total=MEMORY_SIZE, unit="vecs") as pbar:
            file_idx = 0
            while len(bank) < MEMORY_SIZE:
                if file_idx >= len(files): 
                    file_idx = 0
                    np.random.shuffle(files) # Reshuffle if we loop
                
                f = files[file_idx]
                file_idx += 1
                
                try:
                    with np.load(f) as data:
                        # Handle varied key names
                        if 'states' in data: imgs = data['states']
                        elif 'obs' in data: imgs = data['obs']
                        else: continue
                    
                    # Skip short files
                    if len(imgs) < 10: continue

                    # Sample sparsely (every 5th frame) to get variety, not clumps
                    indices = np.linspace(0, len(imgs)-1, num=50, dtype=int)
                    batch_cpu = imgs[indices]
                    
                    # Resize Loop (Vectorized for speed)
                    if batch_cpu.shape[1] != 64:
                        batch_cpu = np.array([cv2.resize(i, (64,64)) for i in batch_cpu])

                    tensor = torch.from_numpy(batch_cpu).float().to(DEVICE) / 255.0
                    tensor = tensor.permute(0, 3, 1, 2)
                    
                    # Encode
                    z = encoder(tensor)
                    
                    # Normalize! Critical for Cosine Similarity later
                    z = torch.nn.functional.normalize(z, p=2, dim=1)
                    
                    for vec in z:
                        if len(bank) >= MEMORY_SIZE: break
                        bank.append(vec.unsqueeze(0).cpu())
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"Skipping {f}: {e}")

    memory_bank = torch.cat(bank, dim=0)
    print(f"Dense Bank Assembled. Shape: {memory_bank.shape}")
    torch.save(memory_bank, SAVE_PATH)
    print("✅ Memory Bank Saved (Clean Expert Data Only)")

if __name__ == "__main__":
    main()