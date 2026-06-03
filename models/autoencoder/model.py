"""
models/autoencoder/model.py
===========================
역할
----
AutoEncoder 모델 구조를 정의하는 파일입니다.

AutoEncoder란?
--------------
정상 이미지만으로 학습하여 이상을 탐지하는 비지도 학습 모델입니다.

  입력 이미지 → [Encoder] → 잠재 벡터(압축) → [Decoder] → 복원 이미지

- 정상 이미지: 학습한 패턴이라 잘 복원됨 → 복원 오차(MSE) 작음 → 정상
- 이상 이미지: 학습한 적 없는 패턴이라 복원 못함 → 복원 오차 큼 → 이상

구조
----
Encoder
  Conv2d(3, 32)  → 이미지 특징 추출 (채널 3 → 32)
  Conv2d(32, 64) → 더 깊은 특징 추출 (채널 32 → 64)
  Conv2d(64, 128)→ 고수준 특징 추출 (채널 64 → 128)
  → 최종적으로 이미지를 작은 잠재 벡터로 압축

Decoder
  ConvTranspose2d(128, 64) → 압축된 벡터를 다시 이미지로 복원
  ConvTranspose2d(64, 32)
  ConvTranspose2d(32, 3)   → 원본 이미지 크기로 복원 (채널 3)

왜 이 구조인가?
--------------
- Conv2d: 이미지의 공간적 패턴(텍스처, 엣지 등)을 학습하는 데 적합
- BatchNorm2d: 학습 안정화, 빠른 수렴
- LeakyReLU: 음수 기울기를 허용해 gradient vanishing 방지
- Sigmoid (출력): 픽셀값을 0~1로 정규화
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    이미지를 점점 작은 잠재 벡터로 압축하는 인코더.
    Conv2d + BatchNorm + LeakyReLU 블록을 3번 반복합니다.
    """

    def __init__(self, latent_dim: int = 512):
        """
        Args:
            latent_dim: 압축된 잠재 벡터의 채널 수 (config.yaml의 autoencoder.latent_dim)
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: (B, 3, 256, 256) → (B, 32, 128, 128)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (B, 32, 128, 128) → (B, 64, 64, 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (B, 64, 64, 64) → (B, 128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: (B, 128, 32, 32) → (B, latent_dim, 16, 16)
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """
    잠재 벡터를 원본 이미지 크기로 복원하는 디코더.
    ConvTranspose2d로 Encoder와 반대 순서로 업샘플링합니다.
    """

    def __init__(self, latent_dim: int = 512):
        """
        Args:
            latent_dim: Encoder의 출력 채널 수와 동일해야 합니다.
        """
        super().__init__()

        self.decoder = nn.Sequential(
            # Block 1: (B, latent_dim, 16, 16) → (B, 128, 32, 32)
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2: (B, 128, 32, 32) → (B, 64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (B, 64, 64, 64) → (B, 32, 128, 128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 4: (B, 32, 128, 128) → (B, 3, 256, 256)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # 픽셀값 0~1 정규화
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class AutoEncoder(nn.Module):
    """
    Encoder + Decoder를 묶은 전체 AutoEncoder 모델.

    사용 예시:
        model = AutoEncoder(latent_dim=512)
        output = model(input_image)           # 복원 이미지 반환
        score  = model.anomaly_score(input_image)  # Anomaly Score 반환
    """

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 이미지 텐서 (B, 3, 256, 256)
        Returns:
            복원된 이미지 텐서 (B, 3, 256, 256)
        """
        z = self.encoder(x)
        return self.decoder(z)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        픽셀별 복원 오차(MSE)를 Anomaly Score로 반환합니다.
        오차가 클수록 이상일 가능성이 높습니다.

        Args:
            x: 입력 이미지 텐서 (B, 3, 256, 256)
        Returns:
            이미지별 Anomaly Score (B,) - 스칼라 값
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            # 픽셀별 MSE → 이미지 전체 평균
            score = torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3])
        return score