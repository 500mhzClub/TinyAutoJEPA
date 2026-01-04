import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def analyze(name, folder):
    files = glob.glob(os.path.join(folder, "*.npz"))
    if not files: 
        print(f"No files found for {name} in {folder}")
        return
    
    steer, gas, brake, frames = [], [], [], 0
    for f in tqdm(files, desc=f"Analyzing {name}"):
        try:
            with np.load(f) as d:
                if 'actions' in d:
                    steer.extend(d['actions'][:,0])
                    gas.extend(d['actions'][:,1])
                    brake.extend(d['actions'][:,2])
                    frames += len(d['actions'])
        except Exception as e:
            print(f"Error loading {f}: {e}")
            pass
        
    steer = np.array(steer)
    gas = np.array(gas)
    brake = np.array(brake)
    
    print(f"\nStats for {name}:")
    print(f"Frames: {frames:,}")
    print(f"Steer Mean: {steer.mean():.3f}, Std: {steer.std():.3f}")
    print(f"Straight (<0.1): {np.mean(np.abs(steer)<0.1):.1%}")
    print(f"Turning (>0.5): {np.mean(np.abs(steer)>0.5):.1%}")
    print(f"Gas Mean: {gas.mean():.3f}")
    print(f"Brake Mean: {brake.mean():.3f}")

if __name__ == "__main__":
    analyze("Random", "./data")
    analyze("Race", "./data_race")
    analyze("Recovery", "./data_recovery")
    analyze("Edge", "./data_edge")