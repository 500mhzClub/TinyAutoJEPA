import numpy as np
import os
import glob
from tqdm import tqdm

DATA_DIRS = {
    "Random": "data",
    "Race": "data_race",
    "Recovery": "data_recovery",
    # "Edge": "data_edge"  <-- Disabled as discussed
}

def check_dataset(name, path):
    if not os.path.exists(path):
        print(f"Skipping {name} (Not found)")
        return

    files = glob.glob(os.path.join(path, "*.npz"))
    if not files:
        print(f"Skipping {name} (No files)")
        return

    total_frames = 0
    all_steer = []
    all_gas = []
    all_brake = []

    print(f"Analyzing {name}...")
    for f in tqdm(files):
        try:
            with np.load(f) as data:
                actions = data['actions']
                total_frames += len(actions)
                
                # Sample 10% of data for speed
                indices = np.random.choice(len(actions), size=int(len(actions)*0.1), replace=False)
                sampled = actions[indices]
                
                all_steer.append(sampled[:, 0])
                all_gas.append(sampled[:, 1])
                all_brake.append(sampled[:, 2])
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if total_frames == 0:
        return

    steer = np.concatenate(all_steer)
    gas = np.concatenate(all_gas)
    brake = np.concatenate(all_brake)

    steer_mean = np.mean(steer)
    steer_std = np.std(steer)
    
    # Logic for distribution
    straight = np.mean(np.abs(steer) < 0.1) * 100
    turning = np.mean(np.abs(steer) > 0.5) * 100

    print(f"\nStats for {name}:")
    print(f"Frames: {total_frames:,}")
    print(f"Steer Mean: {steer_mean:.3f}, Std: {steer_std:.3f}")
    print(f"Straight (<0.1): {straight:.1f}%")
    print(f"Turning (>0.5): {turning:.1f}%")
    print(f"Gas Mean: {np.mean(gas):.3f}")
    print(f"Brake Mean: {np.mean(brake):.3f}")

    # --- AUTO GRADING ---
    print(f"Result: ", end="")
    if name == "Race":
        if steer_std < 0.2:
            print("❌ FAIL (Too straight - Gain too low?)")
        elif turning < 10.0:
            print("❌ FAIL (Not enough corners)")
        else:
            print("✅ PASS")
            
    elif name == "Random":
        if steer_std < 0.3:
            print("❌ FAIL (Not enough entropy)")
        elif turning < 5.0:
             print("❌ FAIL (Too safe/straight)")
        else:
            print("✅ PASS")
    else:
        print("ℹ️  Info only")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    for name, path in DATA_DIRS.items():
        check_dataset(name, path)