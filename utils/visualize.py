import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def generate_heatmap(
    diff_map: np.ndarray,
    save_path: str,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    # 0~255 정규화 후 컬러맵 적용
    normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    normalized = (normalized * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(normalized, colormap)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, heatmap)
    logger.debug(f"히트맵 저장: {save_path}")
    return heatmap


def generate_overlay(
    original_path: str,
    diff_map: np.ndarray,
    save_path: str,
    alpha: float = 0.5,
) -> np.ndarray:
    original = cv2.imread(original_path)
    h, w     = original.shape[:2]

    normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    normalized = (normalized * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap    = cv2.resize(heatmap, (w, h))

    # 원본 * (1 - alpha) + 히트맵 * alpha
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, overlay)
    logger.debug(f"오버레이 저장: {save_path}")
    return overlay


def save_comparison(
    original_path: str,
    reconstructed: np.ndarray,
    diff_map: np.ndarray,
    save_path: str,
) -> None:
    original = cv2.imread(original_path)
    h, w     = original.shape[:2]

    reconstructed_resized = cv2.resize(reconstructed, (w, h))

    normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    normalized = (normalized * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap    = cv2.resize(heatmap, (w, h))

    # [원본 | 복원 | 히트맵] 가로로 이어붙이기
    comparison = np.hstack([original, reconstructed_resized, heatmap])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original",      (10, 30),         font, 0.8, (255,255,255), 2)
    cv2.putText(comparison, "Reconstructed", (w + 10, 30),     font, 0.8, (255,255,255), 2)
    cv2.putText(comparison, "Anomaly Map",   (w * 2 + 10, 30), font, 0.8, (255,255,255), 2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, comparison)
    logger.info(f"비교 이미지 저장: {save_path}")