import json
import os
from pathlib import Path
 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
 
import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image
from torchvision import transforms
 
from models.autoencoder.model import AutoEncoder
 
 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
 
 
def build_ae_transform(image_size: int, normalize: bool) -> transforms.Compose:
    """train.py 와 반드시 동일해야 한다. Sigmoid 출력과의 정합성 문제 때문."""
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)
 
 
class AnomalyDetector:
 
    def __init__(self, config: dict):
        self.config     = config
        self.category   = config["data"]["category"]
        self.image_size = config["data"]["image_size"]
 
        ae_cfg          = config.get("autoencoder", {})
        self.normalize  = bool(ae_cfg.get("normalize", False))
        self.reduction  = ae_cfg.get("score_reduction", "mean")
        self.latent_dim = ae_cfg.get("latent_dim", 512)
 
        if self.reduction not in ("mean", "max"):
            raise ValueError(f"score_reduction 은 mean 또는 max: {self.reduction}")
 
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_ae_transform(self.image_size, self.normalize)
 
        self.model = self._load_model()
 
        self.threshold   = self._load_threshold()
        self.score_range = ((self.threshold * 0.5, self.threshold * 1.5)
                            if self.threshold is not None else None)
 
        if self.normalize:
            logger.warning(
                "AE 입력에 ImageNet 정규화가 켜져 있습니다. Decoder 출력은 Sigmoid([0,1])라 "
                "입력 범위(-2.12~2.64)를 커버할 수 없습니다. train.py 전처리와 일치하는지 확인하세요."
            )
 
        logger.info(f"AutoEncoder AnomalyDetector 초기화 | category={self.category} "
                    f"normalize={self.normalize} reduction={self.reduction} "
                    f"threshold={self.threshold}")
 
    # 로딩
    def _ckpt_path(self) -> Path:
        ckpt_dir = Path(self.config["results"]["checkpoints"])
 
        # 카테고리별 파일 우선, 없으면 단일 파일 폴백
        candidates = [
            ckpt_dir / f"autoencoder_{self.category}_{self.image_size}.pth",
            ckpt_dir / f"autoencoder_{self.category}.pth",
            ckpt_dir / "autoencoder.pth",
            ckpt_dir / "autoencoder_best.pth",
        ]
        for p in candidates:
            if p.exists():
                return p
 
        found = sorted(ckpt_dir.glob("*autoencoder*.pt*")) if ckpt_dir.exists() else []
        if found:
            logger.warning(f"표준 파일명 없음 -> 추정 사용: {found[0]}")
            return found[0]
 
        raise FileNotFoundError(
            f"AutoEncoder 체크포인트를 찾지 못했습니다. 탐색 위치: {ckpt_dir}\n"
            f"시도한 이름: {[p.name for p in candidates]}\n"
            "먼저 실행: python models/autoencoder/train.py"
        )
 
    def _load_model(self) -> AutoEncoder:
        ckpt_path = self._ckpt_path()
        model     = AutoEncoder(latent_dim=self.latent_dim)
 
        state = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
 
        if isinstance(state, AutoEncoder):          # 모델 객체를 통째로 저장한 경우
            model = state
        else:
            model.load_state_dict(state)
 
        model.to(self.device).eval()
        logger.info(f"AutoEncoder 로드: {ckpt_path}")
        self.ckpt_path = ckpt_path
        return model
 
    def _metrics_path(self) -> Path:
        return Path(self.config["results"]["metrics"]) / f"{self.category}_autoencoder.json"
 
    def _load_threshold(self):
        p = self._metrics_path()
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("threshold") is not None:
                return float(m["threshold"])
        return None
 
    def holdout_paths(self) -> list:
        """AE는 train/good 전체로 학습되었으므로 holdout 도 학습 데이터에 포함된다.
        여기서 산출한 threshold 는 낙관적으로 편향되므로 evaluate.py 에서 사용하지 않는다.
        (인터페이스 호환을 위해 존재)"""
        ckpt_dir = Path(self.config["results"]["checkpoints"])
        p = ckpt_dir / f"holdout_{self.category}_{self.image_size}.json"
        if not p.exists():
            raise FileNotFoundError(f"Holdout 목록 없음: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
 
    # 추론
    def raw_score_map(self, image_path):
        """(score, score_map) 반환. score_map 은 (H, W) 픽셀별 복원 오차."""
        image  = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
 
        with torch.no_grad():
            recon = self.model(tensor)
            err   = (tensor - recon) ** 2          # (1, 3, H, W)
            pix   = err.mean(dim=1)[0]             # (H, W) 채널 평균
 
        score_map = pix.cpu().numpy()
        score = float(score_map.mean()) if self.reduction == "mean" else float(score_map.max())
        return score, score_map
 
    def raw_score(self, image_path) -> float:
        return self.raw_score_map(image_path)[0]
 
    def predict(self, image_path: str) -> dict:
        score, score_map = self.raw_score_map(image_path)
        heatmap_path     = self._save_heatmap(image_path, score_map)
 
        is_anomaly = (score >= self.threshold) if self.threshold is not None else None
 
        result = {
            "image_path"  : str(image_path),
            "model_type"  : "autoencoder",
            "score"       : round(score, 8),
            "is_anomaly"  : is_anomaly,
            "threshold"   : self.threshold,
            "heatmap_path": heatmap_path,
        }
 
        status = "미판정" if is_anomaly is None else ("이상" if is_anomaly else "정상")
        logger.info(f"[{status}] score={score:.8f} | {Path(image_path).name}")
        return result
 
    def _save_heatmap(self, image_path: str, score_map: np.ndarray) -> str:
        # 전역 고정 스케일. 이미지별 min-max 는 정상 이미지도 최대 강도로 표시한다.
        if self.score_range is not None:
            lo, hi = self.score_range
        else:
            lo, hi = 0.0, float(score_map.max() + 1e-8)
 
        normalized = np.clip((score_map - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        normalized = (normalized * 255).astype(np.uint8)
 
        if normalized.shape != (self.image_size, self.image_size):
            normalized = cv2.resize(normalized, (self.image_size, self.image_size),
                                    interpolation=cv2.INTER_LINEAR)
        normalized      = cv2.GaussianBlur(normalized, (0, 0), sigmaX=4)
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
 
        heatmap_dir = Path(self.config["results"]["heatmaps"]) / self.category
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        save_path = heatmap_dir / f"{Path(image_path).stem}_autoencoder_heatmap.png"
 
        cv2.imwrite(str(save_path), heatmap_colored)
        return str(save_path)
 
 
if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
 
    detector = AnomalyDetector(config)
    test_dir = Path(config["data"]["root"]) / config["data"]["category"] / "test"
 
    good_scores, bad_scores = [], []
    for image_path in sorted(test_dir.rglob("*.png")):
        score = detector.raw_score(str(image_path))
        (good_scores if image_path.parent.name == "good" else bad_scores).append(score)
 
    print(f"\n{'=' * 50}")
    print(f"카테고리: {config['data']['category']} (AutoEncoder, {detector.reduction})")
    print(f"정상 - 평균: {np.mean(good_scores):.8f} | 최대: {max(good_scores):.8f}")
    print(f"이상 - 평균: {np.mean(bad_scores):.8f} | 최대: {max(bad_scores):.8f}")
    print(f"{'=' * 50}")
    print("\n정량 지표는: python evaluate.py --model autoencoder")
 