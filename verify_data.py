import numpy as np
import cv2
import glob
import os
import random
from tqdm import tqdm

# --- Config ---
DIRS_TO_CHECK = ["./data_race", "./data_recovery", "./data"]
SAMPLES_PER_FOLDER = 5  # Check 5 random files per folder
FRAMES_TO_SHOW = 8      # Show 8 frames from each file
OUTPUT_DIR = "data_inspection"

def inspect_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.npz"))
    
    if not files:
        print(f" No files found in {folder_path}")
        return

    print(f"--- Inspecting {folder_path} ({len(files)} files) ---")
    
    # Create a giant canvas for this folder
    # Rows = Files, Cols = Frames
    folder_canvas = []

    # Pick random files
    files_to_check = random.sample(files, min(len(files), SAMPLES_PER_FOLDER))

    for f_idx, f in enumerate(files_to_check):
        try:
            data = np.load(f)
            
            # 1. Check Keys
            print(f"File: {os.path.basename(f)}")
            print(f"  Keys: {list(data.keys())}")
            
            # 2. Extract Images
            if 'states' in data: obs = data['states']
            elif 'obs' in data: obs = data['obs']
            else:
                print(" No image data found (missing 'states' or 'obs')")
                continue

            # 3. Check Statistics (CRITICAL for Green Blur diagnosis)
            print(f"  Shape: {obs.shape}")
            print(f"  Type:  {obs.dtype}")
            print(f"  Range: {obs.min()} - {obs.max()} (Should be 0-255)")
            
            # Check for "All Green/Black" corruption
            mean_color = obs.mean(axis=(0,1,2))
            print(f"  Avg Color (RGB): {mean_color.astype(int)}")

            # 4. Prepare Visuals
            row_frames = []
            
            # Resize if necessary
            total_frames = len(obs)
            step = max(1, total_frames // FRAMES_TO_SHOW)
            
            for i in range(0, total_frames, step):
                if len(row_frames) >= FRAMES_TO_SHOW: break
                
                img = obs[i]
                
                # Resize to standard thumbnail 64x64
                if img.shape[0] != 64:
                    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
                
                # Convert RGB (Numpy) to BGR (OpenCV)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Add text to first frame of row
                if len(row_frames) == 0:
                    cv2.putText(img_bgr, f"{os.path.basename(f)}", (2, 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                
                row_frames.append(img_bgr)
            
            # Pad row if short
            while len(row_frames) < FRAMES_TO_SHOW:
                row_frames.append(np.zeros((64, 64, 3), dtype=np.uint8))

            # Stack frames horizontally
            folder_canvas.append(np.hstack(row_frames))
            print("")

        except Exception as e:
            print(f"Error reading file: {e}")

    # Stack rows vertically
    if folder_canvas:
        final_grid = np.vstack(folder_canvas)
        save_name = os.path.join(OUTPUT_DIR, f"inspect_{os.path.basename(folder_path)}.png")
        cv2.imwrite(save_name, final_grid)
        print(f" saved inspection grid: {save_name}\n")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for d in DIRS_TO_CHECK:
        if os.path.exists(d):
            inspect_folder(d)
        else:
            print(f"Skipping {d} (Not found)")