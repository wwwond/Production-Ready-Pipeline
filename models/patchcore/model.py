import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Wide_ResNet50_2_Weights
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from loguru import logger


class PatchCore(nn.Module):
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list = ["layer2", "layer3"],
        coreset_ratio: float = 0.1,
    ):
        super().__init__()

        self.layers        = layers
        self.coreset_ratio = coreset_ratio
        self.memory_bank   = None
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = self._load_backbone(backbone)
        self.backbone.to(self.device)
        self.backbone.eval()  # 가중치 고정 - 학습하지 않음

        self._features = {}
        self._register_hooks()

        logger.info(f"PatchCore 초기화 완료 | backbone={backbone} layers={layers} coreset_ratio={coreset_ratio}")

    def _load_backbone(self, backbone: str) -> nn.Module:
        if backbone == "wide_resnet50_2":
            model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone}")

        model.fc = nn.Identity()  # 분류 레이어 제거 (특징 추출만 사용)
        return model

    def _register_hooks(self) -> None:
        # forward 실행 시 layer2, layer3 출력을 자동으로 캡처
        def make_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(make_hook(layer_name))

    def _extract_features(self, images: torch.Tensor):
        with torch.no_grad():
            self.backbone(images)

        features    = []
        target_size = None

        for layer_name in self.layers:
            feat = self._features[layer_name]  # (B, C, H, W)

            if target_size is None:
                target_size = feat.shape[2:]

            # layer3는 layer2보다 작으므로 layer2 크기로 맞춤
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
        logger.info("Memory Bank 구축 시작...")
        all_features = []

        for images in dataloader:
            images = images.to(self.device)
            patches, _ = self._extract_features(images)
            all_features.append(patches.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        logger.info(f"전체 패치 특징 수: {len(all_features)}")

        self.memory_bank = self._coreset_sampling(all_features)
        logger.info(f"Memory Bank 구축 완료: {len(self.memory_bank)}개 (압축률={self.coreset_ratio})")

        logger.info("NearestNeighbors 인덱스 구축 중...")
        self.nn_index = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        logger.info("인덱스 구축 완료")

    def _coreset_sampling(self, features: np.ndarray) -> np.ndarray:
        n_samples = max(1, int(len(features) * self.coreset_ratio))

        # 차원 축소 후 랜덤 샘플링 (논문의 greedy coreset 대신 속도 최적화)
        reducer = SparseRandomProjection(n_components=128, random_state=42)
        reducer.fit_transform(features)

        indices = np.random.choice(len(features), n_samples, replace=False)
        return features[indices]

    def predict(self, images: torch.Tensor):
        if self.memory_bank is None:
            raise RuntimeError("Memory Bank가 없습니다. fit()을 먼저 실행하세요.")

        images = images.to(self.device)
        patches, (B, H, W) = self._extract_features(images)
        patches_np = patches.cpu().numpy()

        distances = self._compute_distances(patches_np)

        # (B*H*W,) → (B, H, W)
        dist_map = distances.reshape(B, H, W)

        # 이미지별 Anomaly Score = 패치 거리의 최대값
        scores = dist_map.max(axis=(1, 2))

        return scores, dist_map

    def _compute_distances(self, patches: np.ndarray) -> np.ndarray:
        if hasattr(self, "nn_index") and self.nn_index is not None:
            distances, _ = self.nn_index.kneighbors(patches)
            return distances.flatten()

        # NearestNeighbors 없을 때 폴백 (느림)
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
        np.save(path, self.memory_bank)
        logger.info(f"Memory Bank 저장: {path}")

    def load(self, path: str) -> None:
        self.memory_bank = np.load(path)
        logger.info(f"Memory Bank 로드: {path} ({len(self.memory_bank)}개)")

        logger.info("NearestNeighbors 인덱스 구축 중...")
        self.nn_index = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        logger.info("인덱스 구축 완료")