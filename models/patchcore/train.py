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
    data_cfg  = config["data"]
    transform = transforms.Compose([
        transforms.Resize((data_cfg["image_size"], data_cfg["image_size"])),
        transforms.ToTensor(),
        # ResNet이 ImageNet으로 학습됐으므로 동일한 정규화 적용
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = Path(data_cfg["root"]) / data_cfg["category"] / "train"
    dataset   = ImageFolder(root=str(train_dir), transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    logger.info(f"데이터 로드 완료: {len(dataset)}장 ({data_cfg['category']}/train/good)")
    return dataloader


def train(config: dict) -> None:
    pc_cfg  = config["patchcore"]
    res_cfg = config["results"]

    checkpoint_dir = Path(res_cfg["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = PatchCore(
        backbone      = pc_cfg["backbone"],
        layers        = pc_cfg["layers"],
        coreset_ratio = pc_cfg["coreset_ratio"],
    )

    dataloader = get_dataloader(config)

    # ImageFolder는 (images, labels) 튜플 반환 → images만 추출
    class ImageOnlyLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for images, _ in self.loader:
                yield images
        def __len__(self):
            return len(self.loader)

    model.fit(ImageOnlyLoader(dataloader))

    save_path = checkpoint_dir / "patchcore_memory_bank.npy"
    model.save(str(save_path))
    logger.info("PatchCore Memory Bank 구축 완료")


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["current"] = "patchcore"
    train(config)