"""
models/autoencoder/train.py
===========================
역할
----
AutoEncoder를 학습시키는 파일입니다.

학습 전략
---------
- 정상 이미지만 사용 (MVTec AD의 train/good 폴더)
- 이상 이미지는 학습에 사용하지 않음
- 손실 함수: MSELoss (복원 오차)
- 옵티마이저: Adam

학습 흐름
---------
1. config.yaml에서 설정값 로드
2. MVTec 정상 이미지 데이터셋 로드
3. AutoEncoder 모델 초기화
4. 에폭마다 이미지 복원 오차(MSE) 최소화
5. 최적 모델 가중치를 results/checkpoints/에 저장

실행 방법
---------
  python models/autoencoder/train.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("C:\\Users\\User\\pro\\prp")
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from models.autoencoder.model import AutoEncoder


# ── 데이터셋 ──────────────────────────────────────────────────────────────────

class MVTecDataset(Dataset):
    """
    MVTec AD 데이터셋 로더.
    train/good 폴더의 정상 이미지만 로드합니다.

    폴더 구조:
        data/mvtec/{category}/train/good/*.png
    """

    def __init__(self, root: str, category: str, image_size: int):
        """
        Args:
            root      : 데이터셋 루트 경로 (config.yaml의 data.root)
            category  : 실험할 카테고리 (예: 'bottle', 'cable')
            image_size: 리사이즈 크기 (config.yaml의 data.image_size)
        """
        self.image_paths = list(
            Path(root, category, "train", "good").glob("*.png")
        )
        # jpg도 포함
        self.image_paths += list(
            Path(root, category, "train", "good").glob("*.jpg")
        )

        if not self.image_paths:
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다: {root}/{category}/train/good/\n"
                "MVTec AD 데이터셋을 다운로드했는지 확인하세요."
            )

        # 전처리: 리사이즈 → 텐서 변환 → 정규화 (0~1)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # (H, W, C) → (C, H, W), 0~255 → 0~1
        ])

        logger.info(f"데이터셋 로드 완료: {len(self.image_paths)}장 ({category}/train/good)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


# ── 학습 함수 ─────────────────────────────────────────────────────────────────

def train(config: dict):
    """
    AutoEncoder 학습 메인 함수.

    Args:
        config: config.yaml에서 로드한 설정 딕셔너리
    """
    # 설정값 파싱
    data_cfg  = config["data"]
    ae_cfg    = config["autoencoder"]
    res_cfg   = config["results"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"학습 장치: {device}")

    # 체크포인트 저장 폴더 생성
    checkpoint_dir = Path(res_cfg["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로더
    dataset = MVTecDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        image_size=data_cfg["image_size"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=ae_cfg["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows 호환성을 위해 0으로 설정
    )

    # 모델 초기화
    model = AutoEncoder(latent_dim=ae_cfg["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    criterion = nn.MSELoss()  # 복원 오차

    logger.info(f"학습 시작 | epochs={ae_cfg['epochs']} batch={ae_cfg['batch_size']} lr={ae_cfg['learning_rate']}")

    best_loss = float("inf")

    for epoch in range(1, ae_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for images in tqdm(dataloader, desc=f"Epoch {epoch}/{ae_cfg['epochs']}"):
            images = images.to(device)

            # Forward
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        # 최적 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoint_dir / f"autoencoder_best.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"모델 저장: {save_path} (loss={best_loss:.6f})")

    logger.info("학습 완료")


# ── 실행 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # config.yaml 로드
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train(config)