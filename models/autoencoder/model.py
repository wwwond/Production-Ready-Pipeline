import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            # (B, 3, 256, 256) → (B, 32, 128, 128)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 32, 128, 128) → (B, 64, 64, 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 64, 64) → (B, 128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 32, 32) → (B, latent_dim, 16, 16)
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.decoder = nn.Sequential(
            # (B, latent_dim, 16, 16) → (B, 128, 32, 32)
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (B, 128, 32, 32) → (B, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (B, 64, 64, 64) → (B, 32, 128, 128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # (B, 32, 128, 128) → (B, 3, 256, 256)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # 픽셀값 0~1 정규화
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reconstructed = self.forward(x)
            score = torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3])
        return score