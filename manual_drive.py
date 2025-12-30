import gymnasium as gym
import torch
import numpy as np
import cv2
import pygame
from networks import TinyEncoder, Predictor, TinyDecoder

# --- CONFIG ---
MODEL_PATH_ENC    = "./models/encoder_mixed_final.pth"
MODEL_PATH_PRED   = "./models/predictor_multistep_final.pth"
MODEL_PATH_DEC    = "./models/decoder_parallel_ep40.pth"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_HORIZON      = 5

def load_models():
    print("Loading Brain...")
    enc = TinyEncoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load(MODEL_PATH_ENC, map_location=DEVICE))
    
    pred = Predictor().to(DEVICE).eval()
    pred.load_state_dict(torch.load(MODEL_PATH_PRED, map_location=DEVICE))
    
    dec = TinyDecoder().to(DEVICE).eval()
    dec.load_state_dict(torch.load(MODEL_PATH_DEC, map_location=DEVICE))
    return enc, pred, dec

def main():
    # 1. Init Gym in HEADLESS mode (No internal window)
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset()

    # 2. Init Pygame manually for our dashboard
    pygame.init()
    # Window size: 600 wide, 800 tall (Reality Top, Dream Bottom)
    screen = pygame.display.set_mode((600, 800)) 
    pygame.display.set_caption("TinyAutoJEPA: Manual vs Brain")
    clock = pygame.time.Clock()

    encoder, predictor, decoder = load_models()
    
    print("\n🎮 CONTROLS: Arrow Keys to Drive.")
    
    steer = 0.0
    gas = 0.0
    brake = 0.0

    running = True
    while running:
        # --- INPUT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        keys = pygame.key.get_pressed()
        
        # Smooth-ish control logic
        if keys[pygame.K_LEFT]:  steer -= 0.1
        elif keys[pygame.K_RIGHT]: steer += 0.1
        else: steer *= 0.8 # Return to center
        steer = np.clip(steer, -1.0, 1.0)
        
        gas = 0.6 if keys[pygame.K_UP] else 0.0
        brake = 0.8 if keys[pygame.K_DOWN] else 0.0
        
        action = np.array([steer, gas, brake])

        # --- PHYSICS STEP ---
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc: obs, _ = env.reset()

        # --- BRAIN STEP ---
        # 1. Resize & Tensorify Reality
        img_small = cv2.resize(obs, (64, 64))
        t_img = torch.tensor(img_small).float().to(DEVICE).unsqueeze(0) / 255.0
        t_img = t_img.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            # 2. Encode
            z = encoder(t_img)
            
            # 3. Predict Future (Repeat current action 5 times)
            t_action = torch.tensor(action).float().to(DEVICE).unsqueeze(0)
            for _ in range(PRED_HORIZON):
                z = predictor(z, t_action)
            
            # 4. Decode Dream
            recon = decoder(z)

        # --- RENDER DASHBOARD ---
        
        # Prepare Reality (Top)
        # Pygame uses Transposed Coordinate system compared to NumPy
        # Obs: [96, 96, 3] -> Pygame Surface
        surf_real = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        surf_real = pygame.transform.scale(surf_real, (600, 400))
        
        # Prepare Dream (Bottom)
        # Recon: [1, 3, 64, 64] -> Numpy [64, 64, 3]
        dream_np = recon.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255
        dream_np = np.clip(dream_np, 0, 255).astype(np.uint8)
        surf_dream = pygame.surfarray.make_surface(np.transpose(dream_np, (1, 0, 2)))
        surf_dream = pygame.transform.scale(surf_dream, (600, 400))

        # Draw to Screen
        screen.blit(surf_real, (0, 0))
        screen.blit(surf_dream, (0, 400))
        
        # UI Text
        font = pygame.font.SysFont("monospace", 24)
        label_real = font.render(f"REALITY (Input)", True, (255, 255, 0))
        label_dream = font.render(f"DREAM (T+{PRED_HORIZON} Prediction)", True, (0, 255, 255))
        label_act = font.render(f"Action: [{steer:.2f}, {gas:.2f}]", True, (255, 255, 255))
        
        screen.blit(label_real, (20, 20))
        screen.blit(label_dream, (20, 420))
        screen.blit(label_act, (20, 750))

        pygame.display.flip()
        clock.tick(30) # Lock to 30 FPS

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()