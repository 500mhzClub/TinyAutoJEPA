import numpy as np
import glob
import os

# 1. Find a file
files = sorted(glob.glob("./data_expert/*.npy") + glob.glob("./data_recover/*.npy"))
if not files:
    print("No files found! Check paths.")
    exit()

f = files[0]
print(f"Inspecting: {f}")

# 2. Load it
try:
    data = np.load(f, allow_pickle=True)
    
    # Handle 0-d array (saved dictionary)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        print("Structure: 0-d Array containing Object")
        data = data.item()
    
    print(f"Base Type: {type(data)}")

    # 3. Check for Keys/Fields
    if isinstance(data, dict):
        print(f"Keys found: {list(data.keys())}")
        if 'action' in data: print(f"Action Shape: {data['action'].shape}")
        if 'actions' in data: print(f"Action Shape: {data['actions'].shape}")
        
    elif isinstance(data, np.ndarray):
        print(f"Array Shape: {data.shape}")
        if data.dtype.names:
            print(f"Structured Fields: {data.dtype.names}")
        else:
            print("⚠️ WARNING: This is a raw array (likely images only). No keys found.")

except Exception as e:
    print(f"Error reading file: {e}")