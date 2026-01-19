import torch
import numpy as np
import cv2
import glob
import os
from networks import TinyEncoder
from tqdm import tqdm

# --- CONFIG ---
DATA_PATTERN   = "./data_expert/*.npz" 
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/memory_bank.pt"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEMORY_SIZE    = 10000 
SKIP_START     = 60  # [NEW] Skip the 'Zoom In' and 'Wait' phase

def main():
    print(f"Loading Encoder on {DEVICE}...")
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")
        
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    print(f"Mining {MEMORY_SIZE} EXPERT states from {DATA_PATTERN}...")
    files = glob.glob(DATA_PATTERN)
    
    if not files:
        print("⚠️ data_expert not found. Checking data_race...")
        files = glob.glob("./data_race/*.npz")
    
    if not files:
        raise FileNotFoundError("No .npz files found!")

    np.random.shuffle(files)
    bank = []
    
    with torch.no_grad():
        with tqdm(total=MEMORY_SIZE, unit="vecs") as pbar:
            file_idx = 0
            while len(bank) < MEMORY_SIZE:
                if file_idx >= len(files): 
                    file_idx = 0
                    np.random.shuffle(files)
                
                f = files[file_idx]
                file_idx += 1
                
                try:
                    with np.load(f) as data:
                        if 'states' in data: imgs = data['states']
                        elif 'obs' in data: imgs = data['obs']
                        else: continue
                    
                    # [NEW] PURGE THE START LINE
                    # Skip the first SKIP_START frames where the car is static
                    if len(imgs) < (SKIP_START + 10): continue
                    imgs = imgs[SKIP_START:] 

                    # Sample sparsely
                    indices = np.linspace(0, len(imgs)-1, num=50, dtype=int)
                    batch_cpu = imgs[indices]
                    
                    # Resize if needed (matches training pipeline)
                    if batch_cpu.shape[1] != 64:
                        batch_cpu = np.array([cv2.resize(i, (64,64)) for i in batch_cpu])

                    tensor = torch.from_numpy(batch_cpu).float().to(DEVICE) / 255.0
                    tensor = tensor.permute(0, 3, 1, 2)
                    
                    z = encoder(tensor)
                    z = torch.nn.functional.normalize(z, p=2, dim=1)
                    
                    for vec in z:
                        if len(bank) >= MEMORY_SIZE: break
                        bank.append(vec.unsqueeze(0).cpu())
                        pbar.update(1)
                        
                except Exception:
                    continue

    memory_bank = torch.cat(bank, dim=0)
    torch.save(memory_bank, SAVE_PATH)
    print(f"✅ Memory Bank Saved: {SAVE_PATH} (Purged Start Line)")

if __name__ == "__main__":
    main()