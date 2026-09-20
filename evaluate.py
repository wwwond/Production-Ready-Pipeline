import argparse
import json
import os
from pathlib import Path
 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
 
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
 
 
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
 
HOLDOUT_PERCENTILE = 99.0
 
 
def get_detector_cls(model_type: str):
    if model_type == "patchcore":
        from models.patchcore.inference import AnomalyDetector
    elif model_type == "autoencoder":
        from models.autoencoder.inference import AnomalyDetector
    else:
        raise ValueError(f"알 수 없는 model: {model_type}")
    return AnomalyDetector
 
 
def upsample_map(score_map: np.ndarray, size: int, sigma: float = 4.0) -> np.ndarray:
    """(H, W) -> (size, size) anomaly map. 논문대로 가우시안 평활화.
    AE 처럼 이미 입력 해상도인 경우 보간은 항등에 가깝다."""
    t  = torch.from_numpy(score_map.astype(np.float32))[None, None]
    up = torch.nn.functional.interpolate(t, size=(size, size),
                                         mode="bilinear", align_corners=False)
    return gaussian_filter(up[0, 0].numpy(), sigma=sigma)
 
 
def load_gt_mask(root: Path, defect: str, stem: str, size: int) -> np.ndarray:
    mask_path = root / "ground_truth" / defect / f"{stem}_mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"GT 마스크 없음: {mask_path}")
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)
 
 
def evaluate(config: dict, model_type: str = "patchcore", with_pixel: bool = True) -> dict:
    category = config["data"]["category"]
    size     = config["data"]["image_size"]
    root     = Path(config["data"]["root"]) / category
 
    detector = get_detector_cls(model_type)(config)
 
    # 1) 운영 임계값: 학습용 정상 holdout 에서 산출 (테스트셋 미사용)
    #    AE 는 holdout 이 학습 데이터라 편향 -> 산출하지 않음
    threshold = None
    if model_type == "patchcore":
        hold_scores = [detector.raw_score(p) for p in detector.holdout_paths()]
        threshold   = float(np.percentile(hold_scores, HOLDOUT_PERCENTILE))
        logger.info(f"[{category}/{model_type}] threshold={threshold:.6f} "
                    f"(holdout {len(hold_scores)}장, p{HOLDOUT_PERCENTILE})")
    else:
        logger.info(f"[{category}/{model_type}] threshold 산출 생략 "
                    f"(AE는 train 전체로 학습 -> holdout이 학습 데이터)")
 
    # 2) 테스트셋 전수 평가
    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []
 
    test_paths = sorted((root / "test").rglob("*.png"))
    for i, p in enumerate(test_paths, 1):
        score, score_map = detector.raw_score_map(str(p))
        defect = p.parent.name
        is_bad = defect != "good"
 
        img_scores.append(score)
        img_labels.append(int(is_bad))
 
        if with_pixel:
            amap = upsample_map(score_map, size)
            gt   = (load_gt_mask(root, defect, p.stem, size) if is_bad
                    else np.zeros((size, size), np.uint8))
            pix_scores.append(amap.ravel().astype(np.float32))
            pix_labels.append(gt.ravel())
 
        if i % 25 == 0:
            logger.info(f"  {i}/{len(test_paths)}")
 
    img_scores = np.array(img_scores)
    img_labels = np.array(img_labels)
 
    normal, anomaly = img_scores[img_labels == 0], img_scores[img_labels == 1]
 
    metrics = {
        "category"      : category,
        "model"         : model_type,
        "image_size"    : size,
        "n_normal"      : int(len(normal)),
        "n_anomaly"     : int(len(anomaly)),
        "image_auroc"   : float(roc_auc_score(img_labels, img_scores)),
        "image_aupr"    : float(average_precision_score(img_labels, img_scores)),
        "score_normal_mean" : float(normal.mean()),
        "score_anomaly_mean": float(anomaly.mean()),
    }
 
    if model_type == "patchcore":
        metrics["coreset_ratio"] = config["patchcore"]["coreset_ratio"]
    else:
        metrics["score_reduction"] = detector.reduction
        metrics["input_normalize"] = detector.normalize
 
    # 운영 임계값 기준 성능
    metrics["threshold"] = threshold
    if threshold is not None:
        metrics["recall_at_thr"] = float((anomaly >= threshold).mean())
        metrics["fpr_at_thr"]    = float((normal  >= threshold).mean())
 
    # FPR=0 진단 지표 (테스트셋 기반이므로 운영 임계값이 아님)
    thr_zero_fp = float(normal.max())
    metrics["thr_zero_fp"]       = thr_zero_fp
    metrics["recall_at_zero_fp"] = float((anomaly > thr_zero_fp).mean())
 
    if with_pixel:
        metrics["pixel_auroc"] = float(
            roc_auc_score(np.concatenate(pix_labels), np.concatenate(pix_scores))
        )
 
    out_dir = Path(config["results"]["metrics"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{category}.json" if model_type == "patchcore" else f"{category}_{model_type}.json"
    (out_dir / fname).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
 
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics
 
 
def print_summary(all_metrics: list) -> None:
    nan = float("nan")
    print(f"\n{'category':<14}{'image AUROC':>13}{'pixel AUROC':>13}"
          f"{'R@thr':>9}{'FPR@thr':>10}{'R@FPR0':>9}")
    print("-" * 68)
    for m in all_metrics:
        print(f"{m['category']:<14}{m['image_auroc']:>13.4f}"
              f"{m.get('pixel_auroc', nan):>13.4f}"
              f"{m.get('recall_at_thr', nan):>9.3f}"
              f"{m.get('fpr_at_thr', nan):>10.3f}"
              f"{m['recall_at_zero_fp']:>9.3f}")
    print("-" * 68)
    print(f"{'MEAN':<14}{np.mean([m['image_auroc'] for m in all_metrics]):>13.4f}"
          f"{np.nanmean([m.get('pixel_auroc', np.nan) for m in all_metrics]):>13.4f}"
          f"{'':>19}{np.mean([m['recall_at_zero_fp'] for m in all_metrics]):>9.3f}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default="patchcore",
                        choices=["patchcore", "autoencoder"])
    parser.add_argument("--all", action="store_true", help="15개 카테고리 전체")
    parser.add_argument("--no-pixel", action="store_true", help="pixel AUROC 생략")
    args = parser.parse_args()
 
    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
 
    categories = MVTEC_CATEGORIES if args.all else \
        [args.category or base_config["data"]["category"]]
 
    results = []
    for cat in categories:
        cfg = json.loads(json.dumps(base_config))  # deep copy
        cfg["data"]["category"] = cat
        try:
            results.append(evaluate(cfg, model_type=args.model,
                                    with_pixel=not args.no_pixel))
        except FileNotFoundError as e:
            logger.error(f"[{cat}] 건너뜀: {e}")
 
    if len(results) > 1:
        print_summary(results)
        suffix = "" if args.model == "patchcore" else f"_{args.model}"
        Path(base_config["results"]["metrics"], f"_summary{suffix}.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
 