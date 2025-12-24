import torch
import torch.nn as nn
from torchvision.models import resnet18

class TinyEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # ResNet18 modified for 64x64 input
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.representation_dim = 512

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

class Predictor(nn.Module):
    def __init__(self, input_dim=512, action_dim=3, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) 
        )

    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        return self.net(x)

class TinyDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc_input(z)
        x = x.view(-1, 256, 4, 4)
        return self.net(x)
    
class CostModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Output probability (0 to 1)
        )

    def forward(self, x):
        return self.net(x)
    
    