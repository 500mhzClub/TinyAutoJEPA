import torch
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from networks import TinyEncoder

DATA_PATTERN   = "./data_expert/*.npz"
MODEL_PATH_ENC = "./models/encoder_mixed_final.pth"
SAVE_PATH      = "./models/memory_bank.pt"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEMORY_SIZE    = 15000
SKIP_START     = 60

# ROI ahead of car in 64x64 (y1:y2, x1:x2)
ROI = (34, 62, 16, 48)

# Filtering thresholds
MAX_GRASS_FRAC = 0.35   # allow some green
MIN_ROAD_FRAC  = 0.10   # require some grey road

def _roi_masks_rgb01(img_rgb01: torch.Tensor):
    """
    img_rgb01: [B,3,64,64] float 0..1
    returns grass_frac[B], road_frac[B]
    grass: green-dominant
    road: roughly grey (R~G~B) and mid intensity
    """
    y1, y2, x1, x2 = ROI
    roi = img_rgb01[:, :, y1:y2, x1:x2]  # [B,3,h,w]
    r = roi[:, 0]
    g = roi[:, 1]
    b = roi[:, 2]

    # green-dominant grass-like
    grass = (g > (r + 0.08)) & (g > (b + 0.08)) & (g > 0.20)

    # grey-ish road: channels close, mid intensity (avoid white HUD)
    m = (r + g + b) / 3.0
    road = (torch.abs(r - g) < 0.07) & (torch.abs(g - b) < 0.07) & (m > 0.15) & (m < 0.75)

    grass_frac = grass.float().mean(dim=(1, 2))
    road_frac = road.float().mean(dim=(1, 2))
    return grass_frac, road_frac

def main():
    if not os.path.exists(MODEL_PATH_ENC):
        raise FileNotFoundError(f"Missing {MODEL_PATH_ENC}")

    print(f"Loading Encoder on {DEVICE}...")
    encoder = TinyEncoder().to(DEVICE).eval()
    encoder.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))

    files = glob.glob(DATA_PATTERN)
    if not files:
        print("⚠️ No data_expert found; trying data_race...")
        files = glob.glob("./data_race/*.npz")
    if not files:
        raise FileNotFoundError("No .npz files found!")

    np.random.shuffle(files)

    bank = []
    accepted = 0
    seen = 0

    with torch.no_grad(), tqdm(total=MEMORY_SIZE, unit="vecs") as pbar:
        file_idx = 0
        while len(bank) < MEMORY_SIZE:
            if file_idx >= len(files):
                file_idx = 0
                np.random.shuffle(files)

            f = files[file_idx]
            file_idx += 1

            try:
                with np.load(f) as data:
                    if "states" in data:
                        imgs = data["states"]
                    elif "obs" in data:
                        imgs = data["obs"]
                    else:
                        continue

                if len(imgs) < (SKIP_START + 100):
                    continue
                imgs = imgs[SKIP_START:]

                # sample candidate indices from this file
                idxs = np.random.choice(len(imgs), size=min(256, len(imgs)), replace=False)
                batch = imgs[idxs]
                if batch.shape[1] != 64:
                    batch = np.array([cv2.resize(i, (64, 64), interpolation=cv2.INTER_AREA) for i in batch])

                # to torch [B,3,64,64] 0..1
                t = torch.from_numpy(batch).float().to(DEVICE) / 255.0
                t = t.permute(0, 3, 1, 2)

                grass_frac, road_frac = _roi_masks_rgb01(t)
                keep = (grass_frac <= MAX_GRASS_FRAC) & (road_frac >= MIN_ROAD_FRAC)

                keep_idx = torch.where(keep)[0]
                if keep_idx.numel() == 0:
                    continue

                t_keep = t[keep_idx]
                z = encoder(t_keep)
                z = torch.nn.functional.normalize(z, p=2, dim=1)

                for vec in z:
                    if len(bank) >= MEMORY_SIZE:
                        break
                    bank.append(vec.unsqueeze(0).cpu())
                    accepted += 1
                    pbar.update(1)

                seen += int(t.shape[0])

            except Exception:
                continue

    memory_bank = torch.cat(bank, dim=0)
    torch.save(memory_bank, SAVE_PATH)
    print(f"✅ Saved filtered memory bank: {SAVE_PATH}")
    print(f"Accepted {accepted} / Seen ~{seen} frames")

if __name__ == "__main__":
    main()
