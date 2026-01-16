import gymnasium as gym
import numpy as np
import math

def cheat_drive_v3_perfect():
    # 1. Initialize CarRacing-v3
    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    env.reset()

    # --- CONSTANTS ---
    LOOKAHEAD = 8         # Look 8 tiles ahead (smoother lines)
    STEER_GAIN = -5.0     # NEGATIVE because Left is -1.0 in Gym, but +Angle in math
    SPEED_LIMIT = 50      # Safe speed limit

    print("🚗 Corrected Cheat Activated: Fixing Steering Sign...")

    while True:
        # --- WAIT FOR LANDING ---
        # The car falls from the sky. We wait until it's stationary-ish on the ground.
        # This prevents the "ghost car" issue where physics haven't started.
        for _ in range(60):
            env.step(np.array([0, 0, 0], dtype=np.float32))
            
        # Re-acquire internal objects (They are recreated after reset)
        track = env.unwrapped.track
        car = env.unwrapped.car
        
        # Safety check
        if car is None or track is None:
            env.reset()
            continue

        print("✅ Car grounded. Driving.")
        
        total_reward = 0
        terminated = False
        truncated = False

        # Extract track coordinates once to save performance
        # Track tiles are: [angle, beta, x, y] -> We want x,y (index 2 and 3)
        track_coords = np.array([tile[2:4] for tile in track])
        
        # Keep track of where we are in the array to prevent searching backwards
        current_track_index = 0

        while not (terminated or truncated):
            # A. Get Car State
            if car.hull is None: break # Safety
            car_pos = np.array(car.hull.position)
            car_angle = car.hull.angle # Radians, continuous (can be > 2pi)
            car_vel = np.linalg.norm(car.hull.linearVelocity)

            # B. efficient Track Localization
            # Only search indices near our last known position to prevent jumping to the wrong part of the loop
            # Search window: +/- 20 tiles from last known index
            start_search = current_track_index - 10
            end_search = current_track_index + 20
            
            # Handle wrapping around the array (the track is a loop)
            indices = np.arange(start_search, end_search) % len(track)
            
            # Distances only for local tiles
            local_distances = np.linalg.norm(track_coords[indices] - car_pos, axis=1)
            best_local_idx = np.argmin(local_distances)
            
            # Update our global index
            current_track_index = indices[best_local_idx]

            # C. Target Selection
            target_idx = (current_track_index + LOOKAHEAD) % len(track)
            target_point = track_coords[target_idx]

            # D. Calculate Error
            # Vector to target
            v_x = target_point[0] - car_pos[0]
            v_y = target_point[1] - car_pos[1]
            
            # Desired heading (where we want to point)
            desired_angle = math.atan2(v_y, v_x)
            
            # Calculate difference (Error)
            # We want to minimize (desired - current)
            angle_error = desired_angle - car_angle
            
            # Normalize error to [-pi, pi]
            # This handles the case where the car is at 359 degrees and target is at 1 degree
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

            # --- CONTROL LOGIC ---
            
            # 1. STEERING
            # KEY FIX: The gain is NEGATIVE. 
            # If target is to the Left (+angle error), we need Negative Steer (-1.0).
            steer = np.clip(angle_error * STEER_GAIN, -1.0, 1.0)
            
            # 2. GAS / BRAKE
            # Dynamic speed control based on steering angle
            if car_vel < SPEED_LIMIT:
                if abs(steer) > 0.3:
                    # Cornering: reduce gas, maybe tap brake
                    gas = 0.1
                    brake = 0.0
                else:
                    # Straight: Full throttle
                    gas = 1.0
                    brake = 0.0
            else:
                # Over speed limit
                gas = 0.0
                # Brake if we are going too fast AND entering a turn
                if abs(steer) > 0.2:
                    brake = 0.2
                else:
                    brake = 0.0

            # Execute
            action = np.array([steer, gas, brake], dtype=np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        print(f"🏁 Finished. Reward: {total_reward:.2f}")
        env.reset()

if __name__ == "__main__":
    cheat_drive_v3_perfect()