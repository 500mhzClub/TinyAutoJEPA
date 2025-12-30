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
MEMORY_SIZE    = 5000     

def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print(f"Mining {MEMORY_SIZE} expert/recovery states...")
    # Load BOTH datasets
    files = glob.glob("./data_race/*.npz") + glob.glob("./data_recovery/*.npz")
    np.random.shuffle(files)
    
    bank = []
    
    with torch.no_grad():
        with tqdm(total=MEMORY_SIZE, unit="vecs") as pbar:
            while len(bank) < MEMORY_SIZE:
                for f in files:
                    if len(bank) >= MEMORY_SIZE: break
                    try:
                        with np.load(f) as data:
                            if 'states' in data: imgs = data['states']
                            elif 'obs' in data: imgs = data['obs']
                            else: continue
                        
                        # Take 50 random samples per file
                        num_to_take = min(50, len(imgs))
                        indices = np.random.choice(len(imgs), size=num_to_take, replace=False)
                        batch_cpu = imgs[indices]
                        
                        if batch_cpu.shape[1] != 64:
                            batch_cpu = np.array([cv2.resize(img, (64,64)) for img in batch_cpu])

                        tensor = torch.tensor(batch_cpu).float().to(DEVICE) / 255.0
                        tensor = tensor.permute(0, 3, 1, 2)
                        
                        z = encoder(tensor)
                        
                        for vec in z:
                            if len(bank) >= MEMORY_SIZE: break
                            bank.append(vec.unsqueeze(0).cpu())
                            pbar.update(1)
                            
                    except Exception as e:
                        pass
                np.random.shuffle(files)

    memory_bank = torch.cat(bank, dim=0)
    print(f"Dense Bank Assembled. Shape: {memory_bank.shape}")
    torch.save(memory_bank, SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()