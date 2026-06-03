"""
utils/visualize.py
==================
역할
----
Anomaly Map(히트맵)을 생성하고 시각화하는 유틸리티 파일입니다.
inference.py에서 호출해서 사용합니다.

왜 별도 파일로 분리했나?
------------------------
시각화 로직을 inference.py 안에 넣으면 코드가 길어지고
나중에 PatchCore로 교체할 때도 똑같이 써야 해서 분리했습니다.
AutoEncoder든 PatchCore든 이 파일 하나로 히트맵을 만듭니다.

Anomaly Map이란?
----------------
픽셀별 복원 오차를 컬러맵으로 표현한 이미지입니다.
  파랑(낮은 오차) → 초록 → 노랑 → 빨강(높은 오차)

빨간 영역 = 이상이 발생한 위치

저장 결과물
-----------
results/heatmaps/{파일명}_heatmap.png  ← 히트맵만
results/heatmaps/{파일명}_overlay.png  ← 원본 + 히트맵 오버레이 (반투명)
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def generate_heatmap(
    diff_map: np.ndarray,
    save_path: str,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    픽셀별 오차 배열을 히트맵 이미지로 변환하고 저장합니다.

    Args:
        diff_map  : 픽셀별 오차 배열 (H, W) - float32
        save_path : 저장할 파일 경로
        colormap  : OpenCV 컬러맵 (기본: COLORMAP_JET)

    Returns:
        히트맵 이미지 배열 (H, W, 3) - uint8
    """
    # 0~255로 정규화
    normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    normalized = (normalized * 255).astype(np.uint8)

    # 컬러맵 적용 (파랑→빨강)
    heatmap = cv2.applyColorMap(normalized, colormap)

    # 저장
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
    """
    원본 이미지 위에 히트맵을 반투명하게 오버레이합니다.
    어디가 이상한지 원본 이미지와 함께 직관적으로 확인할 수 있습니다.

    Args:
        original_path: 원본 이미지 경로
        diff_map     : 픽셀별 오차 배열 (H, W) - float32
        save_path    : 저장할 파일 경로
        alpha        : 히트맵 투명도 (0=완전투명, 1=완전불투명, 기본 0.5)

    Returns:
        오버레이 이미지 배열 (H, W, 3) - uint8
    """
    # 원본 이미지 로드
    original = cv2.imread(original_path)
    h, w = original.shape[:2]

    # 히트맵 생성 (원본과 동일한 크기로 리사이즈)
    normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    normalized = (normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h))

    # 오버레이: 원본 * (1 - alpha) + 히트맵 * alpha
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

    # 저장
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
    """
    원본 / 복원 / 히트맵을 나란히 붙여서 비교 이미지로 저장합니다.
    포트폴리오 시각화, 실험 결과 정리에 유용합니다.

    결과 이미지 구성:
        [ 원본 이미지 | 복원 이미지 | Anomaly Map ]

    Args:
        original_path : 원본 이미지 경로
        reconstructed : 복원된 이미지 배열 (H, W, 3) - uint8
        diff_map      : 픽셀별 오차 배열 (H, W) - float32
        save_path     : 저장할 파일 경로
    """
    # 원본 이미지
    original = cv2.imread(original_path)
    h, w = original.shape[:2]

    # 복원 이미지 리사이즈
    reconstructed_resized = cv2.resize(reconstructed, (w, h))

    # 히트맵
    normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    normalized = (normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h))

    # 3개 이미지를 가로로 이어붙임
    comparison = np.hstack([original, reconstructed_resized, heatmap])

    # 라벨 텍스트 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original",      (10, 30),         font, 0.8, (255,255,255), 2)
    cv2.putText(comparison, "Reconstructed", (w + 10, 30),     font, 0.8, (255,255,255), 2)
    cv2.putText(comparison, "Anomaly Map",   (w * 2 + 10, 30), font, 0.8, (255,255,255), 2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, comparison)
    logger.info(f"비교 이미지 저장: {save_path}")