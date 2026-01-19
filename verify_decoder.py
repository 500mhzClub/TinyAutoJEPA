import torch
import numpy as np
import cv2
import glob
import os
import random
from torchvision.utils import save_image
from networks import TinyEncoder, TinyDecoder

# --- CONFIGURATION ---
ENCODER_PATH = "./models/encoder_mixed_final.pth"
DECODER_PATH = "./models/decoder_final.pth"
DATA_PATTERN = "./data_expert/*.npz" # Only check expert data for now
BATCH_SIZE   = 8                     # How many examples to check
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    print(f"Loading Models on {DEVICE}...")
    
    # Encoder
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Missing {ENCODER_PATH}")
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    
    # Decoder
    if not os.path.exists(DECODER_PATH):
        raise FileNotFoundError(f"Missing {DECODER_PATH}")
    decoder = TinyDecoder().to(DEVICE).eval()
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    
    print("✅ Models Loaded.")
    return encoder, decoder

def get_real_batch(batch_size):
    """
    Loads random frames from disk and prepares them EXACTLY like training.
    """
    files = glob.glob(DATA_PATTERN)
    if not files:
        # Fallback to other folders if expert is missing
        print("⚠️ data_expert not found, trying data_race...")
        files = glob.glob("./data_race/*.npz")
        
    if not files:
        raise FileNotFoundError("No data found to verify against!")
        
    print(f"Sampling from {len(files)} files...")
    
    batch = []
    while len(batch) < batch_size:
        f = random.choice(files)
        try:
            with np.load(f) as data:
                if 'states' in data: obs = data['states']
                elif 'obs' in data: obs = data['obs']
                else: continue
            
            if len(obs) < 1: continue
            
            # Pick a random frame
            idx = random.randint(0, len(obs) - 1)
            img = obs[idx]
            
            # --- CRITICAL PREPROCESSING STEP ---
            # 1. Resize to 64x64 (Matches Training)
            if img.shape[0] != 64 or img.shape[1] != 64:
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            
            # 2. Ensure uint8 (0-255)
            if img.dtype != np.uint8:
                if img.max() <= 1.05: img = (img * 255).astype(np.uint8)
                else: img = img.astype(np.uint8)
            
            # 3. Add to batch
            batch.append(img)
            
        except Exception as e:
            print(f"Skipping bad file {f}: {e}")
            continue
            
    # Convert to Tensor [B, C, H, W] and Normalize 0.0-1.0
    batch_np = np.array(batch)
    batch_t = torch.from_numpy(batch_np).float().to(DEVICE)
    batch_t = batch_t.permute(0, 3, 1, 2).div(255.0)
    
    return batch_t

def main():
    encoder, decoder = load_models()
    
    print("Fetching Real Data Batch...")
    real_imgs = get_real_batch(BATCH_SIZE)
    
    print("Running Inference...")
    with torch.no_grad():
        # 1. Encode
        z = encoder(real_imgs)
        # 2. Decode
        recon_imgs = decoder(z)
        
    # Stack them: Top Row = Real, Bottom Row = Reconstructed
    print("Constructing Comparison Image...")
    comparison = torch.cat([real_imgs, recon_imgs], dim=0)
    
    os.makedirs("visuals", exist_ok=True)
    save_path = "visuals/verification_result.png"
    save_image(comparison, save_path, nrow=BATCH_SIZE)
    
    print(f"\n✅ Verification Complete!")
    print(f"Check the image at: {save_path}")
    print("Top Row    = Real Data from Disk")
    print("Bottom Row = Encoder -> Decoder Reconstruction")

if __name__ == "__main__":
    main()