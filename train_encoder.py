import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
from tqdm import tqdm
from networks import TinyEncoder, Projector
from vicreg import vicreg_loss

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 20  # Fine-tuning doesn't need 50 epochs
LR = 1e-4    # Moderate LR for fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Combined Dataset ---
class CombinedDataset(Dataset):
    def __init__(self):
        # Load BOTH folders
        self.files = glob.glob("./data/*.npz") + glob.glob("./data_race/*.npz")
        if not self.files: raise FileNotFoundError("No files found in ./data or ./data_race")
            
        print(f"Loading Mixed Diet: {len(self.files)} files found.")
        self.data_list = []
        
        for f in self.files:
            try:
                with np.load(f) as arr:
                    if 'states' in arr: obs = arr['states']
                    elif 'obs' in arr: obs = arr['obs']
                    else: continue
                    self.data_list.append(obs)
            except: pass
            
        self.data = np.concatenate(self.data_list, axis=0)
        self.data = np.transpose(self.data, (0, 3, 1, 2)) # NHWC -> NCHW
        print(f"Total Combined Frames: {len(self.data)}")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)), # Less aggressive crop for racing
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.GaussianBlur(3),
        ])

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float() / 255.0
        return self.transform(img), self.transform(img)

# --- Training Loop ---
def train():
    print(f"Training Encoder on {DEVICE}")
    dataset = CombinedDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    encoder = TinyEncoder().to(DEVICE)
    projector = Projector().to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # --- SMART RESUME ---
    # We want to KEEP your current knowledge
    start_epoch = 0
    if os.path.exists("./models/encoder_final.pth"):
        path = "./models/encoder_final.pth"
    elif os.path.exists("./models/encoder_ep20.pth"):
        path = "./models/encoder_ep20.pth"
    else:
        path = None

    if path:
        print(f"--- RESUMING from {path} (Fine-Tuning Mode) ---")
        encoder.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        print("--- STARTING FROM SCRATCH ---")

    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        encoder.train()
        projector.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for x1, x2 in pbar:
            x1, x2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                z1 = projector(encoder(x1))
                z2 = projector(encoder(x2))
                loss = vicreg_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch+1) % 5 == 0:
            torch.save(encoder.state_dict(), f"models/encoder_fine_ep{epoch+1}.pth")

    torch.save(encoder.state_dict(), "models/encoder_final_mixed.pth")
    print("Encoder Fine-Tuning Complete.")

if __name__ == "__main__":
    train()