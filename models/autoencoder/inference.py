"""
models/autoencoder/inference.py
================================
역할
----
학습된 AutoEncoder로 이미지를 추론하는 파일입니다.
Anomaly Score 계산 + Anomaly Map(히트맵) 생성까지 담당합니다.

추론 흐름
---------
1. 학습된 가중치(autoencoder_best.pth) 로드
2. 테스트 이미지 입력
3. AutoEncoder로 이미지 복원
4. 원본 vs 복원 이미지의 픽셀별 오차 → Anomaly Score
5. 픽셀별 오차를 히트맵으로 시각화 → Anomaly Map
6. 결과를 results/heatmaps/ 에 저장

Anomaly Score 기준
------------------
- config.yaml의 model.threshold 값 기준
- score >= threshold → 이상 (anomaly)
- score <  threshold → 정상 (normal)

실행 방법
---------
  python models/autoencoder/inference.py
"""

import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from loguru import logger

from models.autoencoder.model import AutoEncoder


class AnomalyDetector:
    """
    학습된 AutoEncoder를 사용해 이미지의 이상 여부를 판단합니다.

    사용 예시:
        detector = AnomalyDetector(config)
        result = detector.predict("data/mvtec/bottle/test/broken_large/000.png")
        print(result["is_anomaly"], result["score"])
    """

    def __init__(self, config: dict):
        """
        Args:
            config: config.yaml에서 로드한 설정 딕셔너리
        """
        self.config    = config
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = config["model"]["threshold"]
        self.image_size = config["data"]["image_size"]

        # 이미지 전처리 (학습과 동일하게)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        # 모델 로드
        self.model = self._load_model()
        logger.info(f"AnomalyDetector 초기화 완료 | threshold={self.threshold}")

    def _load_model(self) -> AutoEncoder:
        """
        저장된 가중치를 로드해 추론 모드로 반환합니다.
        """
        checkpoint_path = Path(self.config["results"]["checkpoints"]) / "autoencoder_best.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"가중치 파일이 없습니다: {checkpoint_path}\n"
                "먼저 train.py를 실행해 모델을 학습시키세요."
            )

        model = AutoEncoder(latent_dim=self.config["autoencoder"]["latent_dim"])
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # 추론 모드 (BatchNorm, Dropout 비활성화)

        logger.info(f"모델 로드 완료: {checkpoint_path}")
        return model

    def predict(self, image_path: str) -> dict:
        """
        이미지 하나를 추론해 결과를 반환합니다.

        Args:
            image_path: 추론할 이미지 파일 경로

        Returns:
            {
                "image_path"  : 입력 이미지 경로,
                "score"       : Anomaly Score (float),
                "is_anomaly"  : 이상 여부 (bool),
                "heatmap_path": 저장된 히트맵 경로 (str),
            }
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # 추론
        with torch.no_grad():
            reconstructed = self.model(tensor)

        # Anomaly Score: 픽셀별 MSE 평균
        diff = (tensor - reconstructed) ** 2          # (1, 3, H, W)
        score = float(diff.mean().cpu().numpy())

        # Anomaly Map: 채널 평균 → 히트맵 이미지
        heatmap_path = self._save_heatmap(image_path, diff)

        result = {
            "image_path"  : str(image_path),
            "score"       : round(score, 6),
            "is_anomaly"  : score >= self.threshold,
            "heatmap_path": heatmap_path,
        }

        status = "🔴 이상" if result["is_anomaly"] else "🟢 정상"
        logger.info(f"{status} | score={score:.6f} | {Path(image_path).name}")

        return result

    def _save_heatmap(self, image_path: str, diff: torch.Tensor) -> str:
        """
        픽셀별 오차를 히트맵 이미지로 저장합니다.

        히트맵이 빨간색일수록 오차가 크고 이상일 가능성이 높습니다.

        Args:
            image_path: 원본 이미지 경로 (파일명 참조용)
            diff      : 픽셀별 오차 텐서 (1, 3, H, W)

        Returns:
            저장된 히트맵 이미지 경로
        """
        # 채널 평균 → (H, W) numpy 배열
        heatmap = diff.squeeze(0).mean(0).cpu().numpy()  # (H, W)

        # 0~255 스케일로 정규화
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

        # OpenCV COLORMAP_JET: 파랑(낮음) → 빨강(높음)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 저장 경로
        heatmap_dir = Path(self.config["results"]["heatmaps"])
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        save_path = heatmap_dir / f"{Path(image_path).stem}_heatmap.png"

        cv2.imwrite(str(save_path), heatmap_colored)
        return str(save_path)


# ── 실행 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # config.yaml 로드
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    detector = AnomalyDetector(config)

    # 테스트 이미지 폴더 전체 추론
    category = config["data"]["category"]
    test_dir = Path(config["data"]["root"]) / category / "test"

    results = []
    for image_path in sorted(test_dir.rglob("*.png")):
        result = detector.predict(str(image_path))
        results.append(result)

    # 결과 요약
    total     = len(results)
    anomalies = sum(1 for r in results if r["is_anomaly"])
    logger.info(f"\n추론 완료 | 전체: {total}장 | 이상 탐지: {anomalies}장 | 정상: {total - anomalies}장")
