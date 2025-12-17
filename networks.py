import torch
import torch.nn as nn
from torchvision.models import resnet18

class TinyEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Use a lightweight ResNet18
        # We modify the first layer because CarRacing images are 64x64, not ImageNet 224x224
        self.backbone = resnet18(weights=None) # weights=None replaces pretrained=False in newer torchvision
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity() # Remove maxpool to preserve spatial info for small images
        self.backbone.fc = nn.Identity()      # Remove the classification head
        
        self.representation_dim = 512 # ResNet18 output size before FC

    def forward(self, x):
        return self.backbone(x)

class Projector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        return self.net(x)

class TinyDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # We need to project the 512 vector into a spatial map
        # ResNet18 bottleneck is usually 1/32 size (2x2 for 64px input), but we modified it.
        # Let's assume we project to 2x2x256 and upsample from there.
        
        self.fc_input = nn.Linear(latent_dim, 256 * 4 * 4) # Project to 4x4 spatial feature map
        
        self.net = nn.Sequential(
            # Input: (256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # -> (3, 64, 64)
            nn.Sigmoid() # Output pixels 0-1
        )

    def forward(self, z):
        x = self.fc_input(z)
        x = x.view(-1, 256, 4, 4) # Reshape to spatial
        return self.net(x)