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