import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class TinyEncoder(nn.Module):
    """
    Small ResNet-ish encoder.

    Expected input for your setup:
      - frame_stack=4 => in_ch=12 (4 stacked RGB frames), input (B, 12, 64, 64)
    Output:
      - feature map (B, emb_dim, 8, 8) by default
    """
    def __init__(self, in_ch: int = 12, emb_dim: int = 512, base_width: int = 64, blocks=(1, 1, 1, 1)):
        super().__init__()
        self.in_channels = base_width

        self.conv1 = nn.Conv2d(in_ch, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        # Stage widths. Keep canonical 64/128/256 then final emb_dim.
        w1, w2, w3, w4 = base_width, base_width * 2, base_width * 4, emb_dim

        self.layer1 = self._make_layer(w1, blocks=int(blocks[0]), stride=1)
        self.layer2 = self._make_layer(w2, blocks=int(blocks[1]), stride=2)
        self.layer3 = self._make_layer(w3, blocks=int(blocks[2]), stride=2)
        self.layer4 = self._make_layer(w4, blocks=int(blocks[3]), stride=2)

        self.representation_channels = emb_dim

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Projector(nn.Module):
    """
    MLP projector used for VICReg-style representation learning.

    Backward-compatible signature for train_encoder.py:
      Projector(in_dim=512, hid_dim=2048, out_dim=512)
    """
    def __init__(self, in_dim: int = 512, hid_dim: int = 2048, out_dim: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, action_dim: int = 3, features: int = 512):
        super().__init__()
        total_action_dim = action_dim + 1  # + speed

        self.conv_in = nn.Conv2d(features + total_action_dim, features, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(features)

        self.res1 = ResidualBlock(features, features)
        self.res2 = ResidualBlock(features, features)

        self.conv_out = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)

        self.act_embed = nn.Sequential(
            nn.Linear(total_action_dim, total_action_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_action_dim * 2, total_action_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape

        if speed.dim() == 1:
            speed = speed.unsqueeze(1)
        a_vec = torch.cat([action, speed], dim=1)

        a_emb = self.act_embed(a_vec)
        a_map = a_emb.view(b, -1, 1, 1).expand(-1, -1, h, w)

        x = torch.cat([z, a_map], dim=1)

        h1 = F.relu(self.bn_in(self.conv_in(x)))
        h1 = self.res1(h1)
        h1 = self.res2(h1)
        delta = self.conv_out(h1)

        return z + delta


class TinyDecoder(nn.Module):
    def __init__(self, latent_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
