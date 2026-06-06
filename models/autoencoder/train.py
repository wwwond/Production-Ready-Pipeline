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


class MVTecDataset(Dataset):
    def __init__(self, root: str, category: str, image_size: int):
        self.image_paths = list(Path(root, category, "train", "good").glob("*.png"))
        self.image_paths += list(Path(root, category, "train", "good").glob("*.jpg"))

        if not self.image_paths:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {root}/{category}/train/good/")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        logger.info(f"데이터셋 로드 완료: {len(self.image_paths)}장 ({category}/train/good)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def train(config: dict):
    data_cfg = config["data"]
    ae_cfg   = config["autoencoder"]
    res_cfg  = config["results"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"학습 장치: {device}")

    checkpoint_dir = Path(res_cfg["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = MVTecDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        image_size=data_cfg["image_size"],
    )
    dataloader = DataLoader(dataset, batch_size=ae_cfg["batch_size"], shuffle=True, num_workers=0)

    model     = AutoEncoder(latent_dim=ae_cfg["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    criterion = nn.MSELoss()

    logger.info(f"학습 시작 | epochs={ae_cfg['epochs']} batch={ae_cfg['batch_size']} lr={ae_cfg['learning_rate']}")

    best_loss = float("inf")

    for epoch in range(1, ae_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for images in tqdm(dataloader, desc=f"Epoch {epoch}/{ae_cfg['epochs']}"):
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoint_dir / "autoencoder_best.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"모델 저장: {save_path} (loss={best_loss:.6f})")

    logger.info("학습 완료")


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)