import os
import glob
import numpy as np
from tqdm import tqdm

# The folders created by your collection script
DIRS = ["data_expert", "data_recover", "data_random"]

def count_frames():
    grand_total = 0
    
    print(f"{'FOLDER':<20} | {'FILES':<10} | {'FRAMES':<15} | {'SIZE (Est.)':<15}")
    print("-" * 70)
    
    for folder in DIRS:
        if not os.path.exists(folder):
            print(f"{folder:<20} | {'NOT FOUND':<45}")
            continue
            
        # 1. Count unpacked .npy files (Fastest)
        npy_files = glob.glob(os.path.join(folder, "*.npy"))
        
        # 2. Count compressed .npz files (Just in case)
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
        
        all_files = npy_files + npz_files
        folder_total = 0
        
        # Iterate and count
        for f in tqdm(all_files, desc=f"Scanning {folder}", leave=False):
            try:
                if f.endswith('.npy'):
                    # Mmap mode reads only the header info (instant)
                    shape = np.load(f, mmap_mode='r').shape
                    folder_total += shape[0]
                elif f.endswith('.npz'):
                    # NPZ requires opening the zip archive
                    with np.load(f) as data:
                        folder_total += data['states'].shape[0]
            except Exception as e:
                pass

        # Estimate size in memory (64x64x3 bytes per frame)
        size_gb = (folder_total * 64 * 64 * 3) / (1024**3)
        
        print(f"{folder:<20} | {len(all_files):<10} | {folder_total:<15,} | {size_gb:.2f} GB")
        grand_total += folder_total

    print("-" * 70)
    print(f"{'TOTAL':<20} | {'-':<10} | {grand_total:<15,} | {(grand_total * 64 * 64 * 3) / (1024**3):.2f} GB")

    # Recommendation for JEPA training
    print("\n--- ANALYSIS ---")
    if grand_total < 100_000:
        print("⚠️  Dataset is SMALL. Models might overfit. Aim for >300k frames.")
    elif grand_total < 500_000:
        print("✅ Dataset is GOOD. Sufficient for TinyAutoJEPA training.")
    else:
        print("🚀 Dataset is EXCELLENT. Large scale training enabled.")

if __name__ == "__main__":
    count_frames()