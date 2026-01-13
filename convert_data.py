import os
import glob
import numpy as np
from tqdm import tqdm

DATA_DIRS = [
    "./data", 
    "./data_race", 
    "./data_recovery", 
    "./data_edge"
]

def convert_npz_to_npy(folder):
    files = glob.glob(os.path.join(folder, "*.npz"))
    print(f"Found {len(files)} .npz files in {folder}")
    
    for f in tqdm(files):
        base_name = os.path.splitext(f)[0]
        target_path = base_name + ".npy"
        
        # Skip if already exists
        if os.path.exists(target_path):
            continue
            
        try:
            # Load the compressed data
            with np.load(f) as data:
                # Extract the main array
                if "states" in data:
                    arr = data["states"]
                elif "obs" in data:
                    arr = data["obs"]
                else:
                    print(f"Skipping {f}: No 'states' or 'obs' found.")
                    continue
                
                # Save as uncompressed .npy
                # This uses more disk space but allows instant seeking (mmap)
                np.save(target_path, arr)
                
        except Exception as e:
            print(f"Error converting {f}: {e}")

if __name__ == "__main__":
    for d in DATA_DIRS:
        if os.path.exists(d):
            convert_npz_to_npy(d)
        else:
            print(f"Directory not found: {d}")