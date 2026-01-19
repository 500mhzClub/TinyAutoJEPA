import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RESNET BLOCK (UNCHANGED) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

# --- TEMPORAL SPATIAL ENCODER ---
# Input: (B, 12, 64, 64) -> 4 stacked RGB frames
# Output: (B, 512, 8, 8)
class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 12 channels = 4 frames x 3 channels
        self.in_channels = 64
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64,  blocks=1, stride=1)
        self.layer2 = self._make_layer(128, blocks=1, stride=2)
        self.layer3 = self._make_layer(256, blocks=1, stride=2)
        self.layer4 = self._make_layer(512, blocks=1, stride=2)
        
        self.representation_channels = 512 

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x 

# --- PROJECTOR (UNCHANGED logic) ---
class Projector(nn.Module):
    def __init__(self, input_channels=512, hidden_dim=512, output_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.net(x)

# --- PREDICTOR (Spatial + Speed Awareness) ---
# Input: Latent (Spatial), Action (3), Speed (Scalar)
class Predictor(nn.Module):
    def __init__(self, action_dim=3, features=512):
        super().__init__()
        
        # We append speed to the action vector, so action_input is 3+1=4
        total_action_dim = action_dim + 1 
        
        self.conv_in = nn.Conv2d(features + total_action_dim, features, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(features)
        
        self.res1 = ResidualBlock(features, features)
        self.res2 = ResidualBlock(features, features)
        
        self.conv_out = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        
        # Embed (Action + Speed)
        self.act_embed = nn.Sequential(
            nn.Linear(total_action_dim, total_action_dim * 2),
            nn.ReLU(),
            nn.Linear(total_action_dim * 2, total_action_dim)
        )

    def forward(self, z, action, speed):
        # z: (B, 512, 8, 8)
        # action: (B, 3)
        # speed: (B, 1)
        B, C, H, W = z.shape
        
        # 1. Combine Action + Speed
        # Ensure speed is (B, 1)
        if speed.dim() == 1: speed = speed.unsqueeze(1)
        a_vec = torch.cat([action, speed], dim=1) # (B, 4)
        
        # 2. Expand to Spatial Map
        a_emb = self.act_embed(a_vec)
        a_map = a_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # 3. Concat & Convolve
        x = torch.cat([z, a_map], dim=1)
        
        h = F.relu(self.bn_in(self.conv_in(x)))
        h = self.res1(h)
        h = self.res2(h)
        delta = self.conv_out(h)
        
        return z + delta

# --- DECODER (UNCHANGED) ---
# Reconstructs only the *current* frame (t=0) from the stack features
class TinyDecoder(nn.Module):
    def __init__(self, latent_channels=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)