"""
models/patchcore/train.py
=========================
역할
----
PatchCore의 Memory Bank를 구축하는 파일입니다.

AutoEncoder train.py와 뭐가 다른가?
-------------------------------------
AutoEncoder : 수백 에폭 동안 가중치를 업데이트 (진짜 학습)
PatchCore   : 에폭 없음. 정상 이미지 한 번 통과시켜서 특징 저장 (Memory Bank 구축)
              → 훨씬 빠르고 GPU 없어도 괜찮음

실행 방법
---------
  python models/patchcore/train.py

결과물
------
  results/checkpoints/patchcore_memory_bank.npy  ← Memory Bank 파일
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("C:\\Users\\User\\pro\\prp")

import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from loguru import logger

from models.patchcore.model import PatchCore


def get_dataloader(config: dict) -> DataLoader:
    """
    정상 이미지만 로드하는 DataLoader를 반환합니다.
    MVTec의 train/good 폴더 구조를 활용합니다.

    Args:
        config: config.yaml 설정 딕셔너리

    Returns:
        정상 이미지 DataLoader
    """
    data_cfg = config["data"]

    transform = transforms.Compose([
        transforms.Resize((data_cfg["image_size"], data_cfg["image_size"])),
        transforms.ToTensor(),
        # ImageNet 정규화 (ResNet이 ImageNet으로 학습됐으므로 동일하게 정규화)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # train/good 폴더를 ImageFolder로 로드
    train_dir = Path(data_cfg["root"]) / data_cfg["category"] / "train"
    dataset = ImageFolder(root=str(train_dir), transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,   # Memory Bank 구축은 순서 상관없음
        num_workers=0,
    )

    logger.info(f"데이터 로드 완료: {len(dataset)}장 ({data_cfg['category']}/train/good)")
    return dataloader


def train(config: dict) -> None:
    """
    PatchCore Memory Bank 구축 메인 함수.

    Args:
        config: config.yaml 설정 딕셔너리
    """
    pc_cfg  = config["patchcore"]
    res_cfg = config["results"]

    # 체크포인트 폴더 생성
    checkpoint_dir = Path(res_cfg["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # PatchCore 모델 초기화
    model = PatchCore(
        backbone      = pc_cfg["backbone"],
        layers        = pc_cfg["layers"],
        coreset_ratio = pc_cfg["coreset_ratio"],
    )

    # 데이터 로더
    dataloader = get_dataloader(config)

    # Memory Bank 구축 (학습)
    # DataLoader가 (images, labels) 튜플을 반환하므로 images만 추출
    class ImageOnlyLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for images, _ in self.loader:
                yield images
        def __len__(self):
            return len(self.loader)

    model.fit(ImageOnlyLoader(dataloader))

    # Memory Bank 저장
    save_path = checkpoint_dir / "patchcore_memory_bank.npy"
    model.save(str(save_path))
    logger.info("PatchCore Memory Bank 구축 완료")


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # config에서 patchcore로 모델 변경
    config["model"]["current"] = "patchcore"

    train(config)