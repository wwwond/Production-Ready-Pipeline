import json
from datetime import datetime
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.random_projection import SparseRandomProjection
from torchvision import models, transforms
from torchvision.models import Wide_ResNet50_2_Weights
 
 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
 
 
def build_transform(image_size: int) -> transforms.Compose:
    """train / inference / evaluate 가 공유하는 단일 전처리 정의."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
 
 
class PatchCore(nn.Module):
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list = ["layer2", "layer3"],
        coreset_ratio: float = 0.01,
        neighborhood_size: int = 3,
        seed: int = 42,
    ):
        super().__init__()
 
        self.backbone_name     = backbone
        self.layers            = layers
        self.coreset_ratio     = coreset_ratio
        self.neighborhood_size = neighborhood_size
        self.seed              = seed
 
        self.memory_bank = None
        self.bank_t      = None
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.backbone = self._load_backbone(backbone)
        self.backbone.to(self.device)
        self.backbone.eval()  # 가중치 고정 - 학습하지 않음
 
        self._features = {}
        self._register_hooks()
 
        logger.info(
            f"PatchCore 초기화 | backbone={backbone} layers={layers} "
            f"coreset_ratio={coreset_ratio} neighborhood={neighborhood_size} device={self.device}"
        )
 
    # backbone / feature 추출
    def _load_backbone(self, backbone: str) -> nn.Module:
        if backbone == "wide_resnet50_2":
            model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone}")
 
        model.fc = nn.Identity()  # 분류 레이어 제거 (특징 추출만 사용)
        return model
 
    def _register_hooks(self) -> None:
        def make_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook
 
        for layer_name in self.layers:
            getattr(self.backbone, layer_name).register_forward_hook(make_hook(layer_name))
 
    def _extract_features(self, images: torch.Tensor):
        with torch.no_grad():
            self.backbone(images)
 
        features, target_size = [], None
 
        for layer_name in self.layers:
            feat = self._features[layer_name]  # (B, C, H, W)
 
            if target_size is None:
                target_size = feat.shape[2:]
 
            # layer3는 layer2보다 작으므로 layer2 크기로 맞춤
            if feat.shape[2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
 
            # local neighborhood aggregation: 각 위치를 3x3 이웃 평균으로 대체.
            # 수용영역을 넓혀 패치 표현의 노이즈 민감도를 낮춘다 (논문 3.1)
            if self.neighborhood_size > 1:
                k = self.neighborhood_size
                feat = nn.functional.avg_pool2d(feat, kernel_size=k, stride=1, padding=k // 2)
 
            features.append(feat)
 
        combined = torch.cat(features, dim=1)  # (B, C2+C3, H, W)
 
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)
 
        return patches, (B, H, W)
 
    # memory bank 구축
    def fit(self, dataloader) -> None:
        logger.info("Memory Bank 구축 시작...")
        all_features = []
 
        for images in dataloader:
            images = images.to(self.device)
            patches, _ = self._extract_features(images)
            all_features.append(patches.cpu().numpy().astype(np.float32))
 
        all_features = np.concatenate(all_features, axis=0)
        logger.info(f"전체 패치 특징: {all_features.shape} "
                    f"({all_features.nbytes / 1024**2:.1f} MB)")
 
        self.memory_bank = self._coreset_sampling(all_features)
        logger.info(f"Memory Bank 완료: {self.memory_bank.shape} "
                    f"({self.memory_bank.nbytes / 1024**2:.1f} MB)")
 
        self._build_index()
 
    def _coreset_sampling(self, features: np.ndarray) -> np.ndarray:
        """greedy k-center coreset.
 
        거리 계산은 SparseRandomProjection으로 128차원에 축소한 공간에서 수행하고,
        선택된 인덱스로 '원본' 차원의 특징을 반환한다.
        Johnson-Lindenstrauss 보조정리에 의해 축소 공간의 쌍거리는 원 공간의 거리를
        (1 +- eps) 범위로 보존하므로, 선택 결과는 원 공간 greedy와 거의 동일하다.
        """
        n_total   = len(features)
        n_samples = max(1, int(n_total * self.coreset_ratio))
 
        if n_samples >= n_total:
            logger.warning("coreset_ratio >= 1.0 -> 전체 특징 사용")
            return features
 
        logger.info(f"Greedy coreset: {n_total} -> {n_samples} ({self.coreset_ratio:.1%})")
 
        reducer = SparseRandomProjection(n_components=128, random_state=self.seed)
        reduced = np.asarray(reducer.fit_transform(features), dtype=np.float32)
        reduced_t = torch.from_numpy(reduced)
 
        rng   = np.random.default_rng(self.seed)
        start = int(rng.integers(n_total))
 
        selected = [start]
        min_dist = torch.cdist(reduced_t, reduced_t[start:start + 1]).squeeze(1)
 
        for i in range(1, n_samples):
            nxt = int(torch.argmax(min_dist))
            selected.append(nxt)
            d = torch.cdist(reduced_t, reduced_t[nxt:nxt + 1]).squeeze(1)
            min_dist = torch.minimum(min_dist, d)
 
            if i % 500 == 0:
                logger.info(f"  coreset {i}/{n_samples} | max_min_dist={float(min_dist.max()):.4f}")
 
        return features[np.array(selected)]
 
    def _build_index(self) -> None:
        """고차원(1536d)에서는 ball_tree/kd_tree의 가지치기가 동작하지 않으므로
        BLAS 행렬곱을 타는 brute force가 더 빠르다."""
        self.bank_t = torch.from_numpy(self.memory_bank.astype(np.float32)).to(self.device)
        logger.info(f"NN 인덱스 구축 완료 (brute/torch): {tuple(self.bank_t.shape)}")
 
    # 추론
    def predict(self, images: torch.Tensor):
        if self.memory_bank is None:
            raise RuntimeError("Memory Bank가 없습니다. fit() 또는 load()를 먼저 실행하세요.")
 
        images = images.to(self.device)
        patches, (B, H, W) = self._extract_features(images)
 
        distances = self._compute_distances(patches)  # (B*H*W,)
        dist_map  = distances.reshape(B, H, W)
 
        # 이미지별 Anomaly Score = 패치 거리의 최대값
        scores = dist_map.max(axis=(1, 2))
 
        return scores, dist_map
 
    def _compute_distances(self, patches: torch.Tensor) -> np.ndarray:
        q = patches.to(self.device).float()
        out = []
 
        for i in range(0, len(q), 4096):  # 메모리 상한용 청크
            d = torch.cdist(q[i:i + 4096], self.bank_t)
            out.append(d.min(dim=1).values)
 
        return torch.cat(out).cpu().numpy()
 
    # 저장 / 로드 (+ 메타데이터 검증)
    def _meta_path(self, path: str) -> Path:
        return Path(path).with_suffix(".json")
 
    def save(self, path: str, meta: dict = None) -> None:
        np.save(path, self.memory_bank)
 
        full_meta = {
            **(meta or {}),
            "backbone"         : self.backbone_name,
            "layers"           : self.layers,
            "coreset_ratio"    : self.coreset_ratio,
            "neighborhood_size": self.neighborhood_size,
            "seed"             : self.seed,
            "dim"              : int(self.memory_bank.shape[1]),
            "n_patches"        : int(len(self.memory_bank)),
            "created_at"       : datetime.now().isoformat(timespec="seconds"),
        }
        self._meta_path(path).write_text(
            json.dumps(full_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Memory Bank 저장: {path} (+ meta)")
 
    def load(self, path: str, expect: dict = None) -> None:
        self.memory_bank = np.load(path)
 
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"메타데이터 없음: {meta_path}\n"
                "구버전 memory bank입니다. train.py를 다시 실행하세요."
            )
 
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
 
        # config와 memory bank가 어긋난 채 조용히 잘못된 결과를 내는 것을 차단
        for key, want in (expect or {}).items():
            got = meta.get(key)
            if got != want:
                raise ValueError(
                    f"Memory Bank 불일치 [{key}] 저장={got} 요청={want}\n"
                    f"  -> {path} 는 다른 설정으로 만들어졌습니다. train.py 재실행 필요."
                )
 
        # hook은 __init__ 시점의 self.layers로 이미 등록되어 있으므로
        # 덮어쓰지 않고 '검증'만 한다 (덮어쓰면 hook과 조용히 어긋남)
        if meta["layers"] != self.layers:
            raise ValueError(
                f"Memory Bank 불일치 [layers] 저장={meta['layers']} 요청={self.layers}\n"
                f"  -> train.py 재실행 필요."
            )
        self.coreset_ratio     = meta["coreset_ratio"]
        self.neighborhood_size = meta.get("neighborhood_size", 3)
 
        logger.info(f"Memory Bank 로드: {path} ({meta['n_patches']}개 x {meta['dim']}d)")
        self._build_index()