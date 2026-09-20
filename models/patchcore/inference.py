import json
import os
from pathlib import Path
 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
 
import cv2
import numpy as np
import yaml
from loguru import logger
from PIL import Image
 
from models.patchcore.model import PatchCore, build_transform
 
 
class AnomalyDetector:
 
    def __init__(self, config: dict):
        self.config     = config
        self.category   = config["data"]["category"]
        self.image_size = config["data"]["image_size"]
        self.transform  = build_transform(self.image_size)
 
        self.model = self._load_model()
 
        self.threshold   = self._load_threshold()
        self.score_range = ((self.threshold * 0.5, self.threshold * 1.5)
                            if self.threshold is not None else (0.0, 6.0))
 
        logger.info(f"PatchCore AnomalyDetector 초기화 | category={self.category} "
                    f"threshold={self.threshold}")
 
    # 로딩
    def _bank_path(self) -> Path:
        return Path(self.config["results"]["checkpoints"]) / \
            f"patchcore_{self.category}_{self.image_size}.npy"
 
    def _load_model(self) -> PatchCore:
        pc_cfg   = self.config["patchcore"]
        mb_path  = self._bank_path()
 
        if not mb_path.exists():
            raise FileNotFoundError(
                f"Memory Bank가 없습니다: {mb_path}\n"
                f"먼저 실행: python models/patchcore/train.py --category {self.category}"
            )
 
        model = PatchCore(
            backbone          = pc_cfg["backbone"],
            layers            = pc_cfg["layers"],
            coreset_ratio     = pc_cfg["coreset_ratio"],
            neighborhood_size = pc_cfg.get("neighborhood_size", 3),
        )
        model.load(
            str(mb_path),
            expect={
                "category"  : self.category,
                "image_size": self.image_size,
                "backbone"  : pc_cfg["backbone"],
            },
        )
        self.mb_path = mb_path
        return model
 
    def _metrics_path(self) -> Path:
        return Path(self.config["results"]["metrics"]) / f"{self.category}.json"
 
    def _load_threshold(self):
        """evaluate.py가 holdout 분위수로 산출한 threshold. 없으면 None."""
        p = self._metrics_path()
        if p.exists():
            return float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
 
        # 구버전 호환: config에 값이 남아 있으면 사용
        legacy = self.config.get("model", {}).get("threshold")
        if legacy is not None:
            logger.warning(f"metrics 없음 -> config의 legacy threshold 사용: {legacy}")
            return float(legacy)
 
        logger.warning(f"threshold 미설정 ({p} 없음). evaluate.py를 먼저 실행하세요.")
        return None
 
    def holdout_paths(self) -> list:
        p = self.mb_path.with_name(
            self.mb_path.name.replace("patchcore_", "holdout_").replace(".npy", ".json")
        )
        if not p.exists():
            raise FileNotFoundError(f"Holdout 목록 없음: {p}. train.py를 다시 실행하세요.")
        return json.loads(p.read_text(encoding="utf-8"))
 
    # 추론
    def raw_score_map(self, image_path):
        """히트맵 저장 없이 (score, dist_map) 반환. 평가 루프용."""
        image  = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        scores, dist_map = self.model.predict(tensor)
        return float(scores[0]), dist_map[0]
 
    def raw_score(self, image_path) -> float:
        return self.raw_score_map(image_path)[0]
 
    def predict(self, image_path: str) -> dict:
        score, dist_map = self.raw_score_map(image_path)
        heatmap_path    = self._save_heatmap(image_path, dist_map)
 
        is_anomaly = (score >= self.threshold) if self.threshold is not None else None
 
        result = {
            "image_path"  : str(image_path),
            "score"       : round(score, 6),
            "is_anomaly"  : is_anomaly,
            "threshold"   : self.threshold,
            "heatmap_path": heatmap_path,
        }
 
        status = "미판정" if is_anomaly is None else ("이상" if is_anomaly else "정상")
        logger.info(f"[{status}] score={score:.6f} | {Path(image_path).name}")
        return result
 
    def _save_heatmap(self, image_path: str, dist_map: np.ndarray) -> str:
        # 전역 고정 스케일. 이미지별 min-max를 쓰면 정상 이미지도 최대 강도로 표시된다.
        lo, hi     = self.score_range
        normalized = np.clip((dist_map - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        normalized = (normalized * 255).astype(np.uint8)
 
        heatmap         = cv2.resize(normalized, (self.image_size, self.image_size),
                                     interpolation=cv2.INTER_LINEAR)
        heatmap         = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
 
        heatmap_dir = Path(self.config["results"]["heatmaps"]) / self.category
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        save_path = heatmap_dir / f"{Path(image_path).stem}_patchcore_heatmap.png"
 
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
    print(f"카테고리: {config['data']['category']}")
    print(f"정상 - 평균: {np.mean(good_scores):.6f} | 최대: {max(good_scores):.6f} | 최소: {min(good_scores):.6f}")
    print(f"이상 - 평균: {np.mean(bad_scores):.6f} | 최대: {max(bad_scores):.6f} | 최소: {min(bad_scores):.6f}")
    print(f"{'=' * 50}")
    print("\n정량 지표(AUROC 등)는 evaluate.py 를 실행하세요.")
 