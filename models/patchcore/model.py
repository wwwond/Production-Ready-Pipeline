"""
models/patchcore/model.py
=========================
역할
----
PatchCore 모델 구조를 정의하는 파일입니다.

AutoEncoder와 뭐가 다른가?
--------------------------
AutoEncoder : 모델이 이미지를 직접 복원 → 복원 오차로 이상 탐지
PatchCore   : 사전학습된 ResNet으로 특징 추출 → Memory Bank와 거리 비교로 이상 탐지

PatchCore가 AutoEncoder보다 좋은 이유
-------------------------------------
1. ResNet은 ImageNet으로 이미 학습된 강력한 특징 추출기
   → AutoEncoder처럼 처음부터 학습할 필요 없음
2. 패치(이미지 부분) 단위로 특징을 추출해서 이상 위치도 정확하게 탐지
3. Memory Bank와의 거리로 판단하기 때문에 score 분리가 훨씬 잘 됨

동작 구조
---------
[학습]
정상 이미지
  → ResNet layer2, layer3에서 중간 특징 추출 (패치 단위)
  → 특징 벡터들을 Memory Bank에 저장
  → Coreset Sampling으로 Memory Bank 압축 (전체의 10%)

[추론]
테스트 이미지
  → ResNet에서 패치 특징 추출
  → Memory Bank에서 가장 가까운 벡터까지의 거리 계산
  → 거리가 멀수록 이상 (본 적 없는 패턴)
  → 패치별 거리를 히트맵으로 시각화

왜 layer2, layer3인가?
----------------------
- layer1: 너무 저수준 (엣지, 색상)
- layer2, layer3: 중간 수준 (텍스처, 패턴) ← 이상 탐지에 가장 적합
- layer4: 너무 고수준 (객체 전체 의미)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Wide_ResNet50_2_Weights
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from loguru import logger


class PatchCore(nn.Module):
    """
    PatchCore 이상 탐지 모델.

    사용 예시:
        model = PatchCore(backbone='wide_resnet50_2', layers=['layer2', 'layer3'])
        model.fit(dataloader)                    # Memory Bank 구축
        score, heatmap = model.predict(image)    # 추론
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list = ["layer2", "layer3"],
        coreset_ratio: float = 0.1,
    ):
        """
        Args:
            backbone     : 특징 추출에 사용할 ResNet 종류
            layers       : 특징을 추출할 레이어 이름 목록
            coreset_ratio: Memory Bank 압축 비율 (0.1 = 전체의 10%만 저장)
        """
        super().__init__()

        self.layers        = layers
        self.coreset_ratio = coreset_ratio
        self.memory_bank   = None   # 학습 후 정상 특징 벡터가 저장되는 곳
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 사전학습된 ResNet 로드 (ImageNet 가중치)
        self.backbone = self._load_backbone(backbone)
        self.backbone.to(self.device)
        self.backbone.eval()  # 추론 모드 고정 (학습하지 않음)

        # 중간 레이어 특징을 저장할 딕셔너리
        self._features = {}
        self._register_hooks()

        logger.info(f"PatchCore 초기화 완료 | backbone={backbone} layers={layers} coreset_ratio={coreset_ratio}")

    def _load_backbone(self, backbone: str) -> nn.Module:
        """
        사전학습된 백본 모델을 로드합니다.
        마지막 분류 레이어(fc)는 필요 없으므로 제거합니다.
        """
        if backbone == "wide_resnet50_2":
            model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone}")

        # fc 레이어 제거 (특징 추출만 할 것이므로)
        model.fc = nn.Identity()
        return model

    def _register_hooks(self) -> None:
        """
        지정한 레이어에서 특징을 자동으로 캡처하는 hook을 등록합니다.

        hook이란?
        ---------
        forward 실행 중 특정 레이어의 출력을 가로채는 함수입니다.
        layer2, layer3의 출력을 self._features에 저장합니다.
        """
        def make_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(make_hook(layer_name))

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        이미지에서 패치 특징 벡터를 추출합니다.

        추출 과정:
        1. ResNet forward 실행 (hook이 자동으로 layer2, layer3 출력 캡처)
        2. layer2, layer3 특징을 같은 크기로 interpolate
        3. 두 레이어 특징을 채널 방향으로 합치기 (concat)
        4. (B, C, H, W) → (B*H*W, C) 패치 단위로 reshape

        Args:
            images: 입력 이미지 텐서 (B, 3, 256, 256)

        Returns:
            패치 특징 벡터 (B*H*W, C)
        """
        with torch.no_grad():
            self.backbone(images)

        features = []
        target_size = None

        for layer_name in self.layers:
            feat = self._features[layer_name]  # (B, C, H, W)

            # 첫 번째 레이어 크기 기준으로 맞춤
            if target_size is None:
                target_size = feat.shape[2:]

            # 크기 맞추기 (layer3는 layer2보다 작으므로 interpolate)
            if feat.shape[2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            features.append(feat)

        # 채널 방향으로 합치기: (B, C2+C3, H, W)
        combined = torch.cat(features, dim=1)

        # 패치 단위로 reshape: (B, C, H, W) → (B*H*W, C)
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)

        return patches, (B, H, W)

    def fit(self, dataloader) -> None:
        """
        정상 이미지들로 Memory Bank를 구축합니다.
        (PatchCore에서 '학습'은 가중치 업데이트가 아니라 Memory Bank 구축입니다)

        과정:
        1. 모든 정상 이미지에서 패치 특징 추출
        2. 전체 특징을 모아서 Coreset Sampling으로 압축
        3. 압축된 특징을 Memory Bank에 저장

        Args:
            dataloader: 정상 이미지 DataLoader
        """
        logger.info("Memory Bank 구축 시작...")
        all_features = []

        for images in dataloader:
            images = images.to(self.device)
            patches, _ = self._extract_features(images)
            all_features.append(patches.cpu().numpy())

        # 전체 특징 합치기
        all_features = np.concatenate(all_features, axis=0)
        logger.info(f"전체 패치 특징 수: {len(all_features)}")

        # Coreset Sampling: 전체에서 대표적인 것만 선택
        self.memory_bank = self._coreset_sampling(all_features)
        logger.info(f"Memory Bank 구축 완료: {len(self.memory_bank)}개 (압축률={self.coreset_ratio})")

        # NearestNeighbors 인덱스 구축 (추론 속도 대폭 향상)
        logger.info("NearestNeighbors 인덱스 구축 중...")
        self.nn_index = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        logger.info("인덱스 구축 완료")

    def _coreset_sampling(self, features: np.ndarray) -> np.ndarray:
        """
        Coreset Sampling으로 대표 특징만 선택합니다.

        왜 필요한가?
        ------------
        정상 이미지 200장 * 패치 수 = 수십만 개의 특징 벡터
        전부 Memory Bank에 저장하면 추론 시 거리 계산이 너무 느려요.
        대표적인 특징만 골라서 속도와 성능을 모두 잡습니다.

        방법:
        1. SparseRandomProjection으로 차원 축소 (거리 계산 속도 향상)
        2. 목표 개수만큼 랜덤 샘플링

        Args:
            features: 전체 패치 특징 벡터 (N, C)

        Returns:
            선택된 대표 특징 벡터 (N*coreset_ratio, C)
        """
        n_samples = max(1, int(len(features) * self.coreset_ratio))

        # 차원 축소 (고차원 벡터의 거리 계산 속도 향상)
        reducer = SparseRandomProjection(n_components=128, random_state=42)
        reduced = reducer.fit_transform(features)

        # 랜덤 샘플링 (간단한 구현)
        # 논문의 greedy coreset은 더 정교하지만 속도가 느려서 랜덤으로 대체
        indices = np.random.choice(len(reduced), n_samples, replace=False)
        return features[indices]

    def predict(self, images: torch.Tensor):
        """
        이미지의 Anomaly Score와 히트맵용 거리 맵을 반환합니다.

        과정:
        1. 이미지에서 패치 특징 추출
        2. 각 패치 특징과 Memory Bank 간 최소 거리 계산
        3. 최대 거리 = Anomaly Score (이미지 전체의 이상 정도)
        4. 패치별 거리를 (H, W) 맵으로 reshape → 히트맵 생성에 사용

        Args:
            images: 입력 이미지 텐서 (B, 3, 256, 256)

        Returns:
            scores  : 이미지별 Anomaly Score (B,)
            dist_map: 패치별 거리 맵 (B, H, W) - 히트맵 생성용
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory Bank가 없습니다. fit()을 먼저 실행하세요.")

        images = images.to(self.device)
        patches, (B, H, W) = self._extract_features(images)
        patches_np = patches.cpu().numpy()

        # Memory Bank와의 거리 계산 (배치 처리)
        # 각 패치에서 Memory Bank의 가장 가까운 벡터까지의 거리
        distances = self._compute_distances(patches_np)

        # (B*H*W,) → (B, H, W) reshape
        dist_map = distances.reshape(B, H, W)

        # 이미지별 Anomaly Score = 패치 거리의 최대값
        scores = dist_map.max(axis=(1, 2))

        return scores, dist_map

    def _compute_distances(self, patches: np.ndarray) -> np.ndarray:
        """
        각 패치와 Memory Bank 사이의 최소 거리를 계산합니다.

        Args:
            patches: 테스트 패치 특징 벡터 (N, C)

        Returns:
            각 패치의 최소 거리 (N,)
        """
        """
        각 패치와 Memory Bank 사이의 최소 거리를 계산합니다.
        NearestNeighbors 인덱스가 있으면 사용하고, 없으면 직접 계산합니다.

        Args:
            patches: 테스트 패치 특징 벡터 (N, C)

        Returns:
            각 패치의 최소 거리 (N,)
        """
        if hasattr(self, "nn_index") and self.nn_index is not None:
            # NearestNeighbors 인덱스 사용 (빠름)
            distances, _ = self.nn_index.kneighbors(patches)
            return distances.flatten()

        # 인덱스 없을 때 직접 계산 (느림, 폴백용)
        patch_batch_size  = 50
        memory_batch_size = 1000
        all_min_distances = []

        for i in range(0, len(patches), patch_batch_size):
            patch_batch = patches[i:i + patch_batch_size]
            min_dists   = np.full(len(patch_batch), np.inf)

            for j in range(0, len(self.memory_bank), memory_batch_size):
                mem_batch = self.memory_bank[j:j + memory_batch_size]
                diff  = patch_batch[:, np.newaxis, :] - mem_batch[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_dists = np.minimum(min_dists, dists.min(axis=1))

            all_min_distances.append(min_dists)

        return np.concatenate(all_min_distances)

    def save(self, path: str) -> None:
        """Memory Bank를 파일로 저장합니다."""
        np.save(path, self.memory_bank)
        logger.info(f"Memory Bank 저장: {path}")

    def load(self, path: str) -> None:
        """저장된 Memory Bank를 로드하고 NearestNeighbors 인덱스를 구축합니다."""
        self.memory_bank = np.load(path)
        logger.info(f"Memory Bank 로드: {path} ({len(self.memory_bank)}개)")

        # 인덱스 구축 (추론 속도 향상)
        logger.info("NearestNeighbors 인덱스 구축 중...")
        self.nn_index = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        logger.info("인덱스 구축 완료")
