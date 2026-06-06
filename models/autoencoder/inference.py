import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("C:\\Users\\User\\pro\\prp")

import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from loguru import logger

from models.patchcore.model import PatchCore


class AnomalyDetector:
    # AutoEncoder의 AnomalyDetector와 동일한 인터페이스
    # consumer.py에서 import 경로만 바꾸면 바로 교체 가능
    # 이거 --> from models.autoencoder.inference import AnomalyDetector

    def __init__(self, config: dict):
        self.config     = config
        self.threshold  = config["model"]["threshold"]
        self.image_size = config["data"]["image_size"]

        # train.py와 동일한 정규화 (ResNet ImageNet 기준)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = self._load_model()
        logger.info(f"PatchCore AnomalyDetector 초기화 완료 | threshold={self.threshold}")

    def _load_model(self) -> PatchCore:
        pc_cfg           = self.config["patchcore"]
        memory_bank_path = Path(self.config["results"]["checkpoints"]) / "patchcore_memory_bank.npy"

        if not memory_bank_path.exists():
            raise FileNotFoundError(f"Memory Bank가 없습니다: {memory_bank_path}")

        model = PatchCore(
            backbone      = pc_cfg["backbone"],
            layers        = pc_cfg["layers"],
            coreset_ratio = pc_cfg["coreset_ratio"],
        )
        model.load(str(memory_bank_path))
        return model

    def predict(self, image_path: str) -> dict:
        image  = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)

        scores, dist_map = self.model.predict(tensor)
        score            = float(scores[0])
        heatmap_path     = self._save_heatmap(image_path, dist_map[0])

        result = {
            "image_path"  : str(image_path),
            "score"       : round(score, 6),
            "is_anomaly"  : score >= self.threshold,
            "heatmap_path": heatmap_path,
        }

        status = "🔴 이상" if result["is_anomaly"] else "🟢 정상"
        logger.info(f"{status} | score={score:.6f} | {Path(image_path).name}")
        return result

    def _save_heatmap(self, image_path: str, dist_map: np.ndarray) -> str:
        normalized = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)

        heatmap        = cv2.resize(normalized, (self.image_size, self.image_size))
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        heatmap_dir = Path(self.config["results"]["heatmaps"])
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        save_path = heatmap_dir / f"{Path(image_path).stem}_patchcore_heatmap.png"

        cv2.imwrite(str(save_path), heatmap_colored)
        return str(save_path)


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    detector  = AnomalyDetector(config)
    category  = config["data"]["category"]
    test_dir  = Path(config["data"]["root"]) / category / "test"

    good_scores = []
    bad_scores  = []

    for image_path in sorted(test_dir.rglob("*.png")):
        result = detector.predict(str(image_path))
        if image_path.parent.name == "good":
            good_scores.append(result["score"])
        else:
            bad_scores.append(result["score"])

    print(f"\n{'='*50}")
    print(f"카테고리: {category}")
    print(f"정상 - 평균: {sum(good_scores)/len(good_scores):.6f} | 최대: {max(good_scores):.6f} | 최소: {min(good_scores):.6f}")
    print(f"이상 - 평균: {sum(bad_scores)/len(bad_scores):.6f} | 최대: {max(bad_scores):.6f} | 최소: {min(bad_scores):.6f}")
    print(f"{'='*50}")

    threshold = config["model"]["threshold"]
    detected  = sum(1 for s in bad_scores if s >= threshold)
    print(f"\nthreshold={threshold} 기준")
    print(f"이상 탐지: {detected}/{len(bad_scores)}장 ({detected/len(bad_scores)*100:.1f}%)")
    print(f"오탐(정상→이상): {sum(1 for s in good_scores if s >= threshold)}/{len(good_scores)}장")