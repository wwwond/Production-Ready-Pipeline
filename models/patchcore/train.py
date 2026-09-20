import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import json
import os
from pathlib import Path
 
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
 
import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
 
from models.patchcore.model import PatchCore, build_transform
 
 
SEED = 42
HOLDOUT_RATIO = 0.2
 
 
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
 
 
def bank_path(config: dict) -> Path:
    d = config["data"]
    return Path(config["results"]["checkpoints"]) / \
        f"patchcore_{d['category']}_{d['image_size']}.npy"
 
 
def holdout_path(config: dict) -> Path:
    d = config["data"]
    return Path(config["results"]["checkpoints"]) / \
        f"holdout_{d['category']}_{d['image_size']}.json"
 
 
def get_dataloaders(config: dict):
    """train/good 을 memory bank 용과 threshold 산출용으로 분리해 반환."""
    d = config["data"]
    transform = build_transform(d["image_size"])
 
    train_dir = Path(d["root"]) / d["category"] / "train"
    dataset   = ImageFolder(root=str(train_dir), transform=transform)
 
    n_hold = max(1, int(len(dataset) * HOLDOUT_RATIO))
    n_bank = len(dataset) - n_hold
 
    g = torch.Generator().manual_seed(SEED)
    bank_set, hold_set = random_split(dataset, [n_bank, n_hold], generator=g)
 
    n_workers = config.get("runtime", {}).get("num_workers", 4)
 
    def mk(ds):
        return DataLoader(ds, batch_size=16, shuffle=False, num_workers=n_workers)
 
    logger.info(f"{d['category']}/train/good {len(dataset)}장 -> bank {n_bank} / holdout {n_hold}")
    return mk(bank_set), hold_set
 
 
def images_only(loader):
    """ImageFolder는 (images, labels) 튜플 반환 -> images만 추출."""
    for images, _ in loader:
        yield images
 
 
def save_holdout_paths(config: dict, hold_set) -> None:
    """Subset.indices -> 원본 ImageFolder.samples 로 실제 파일 경로 복원."""
    base  = hold_set.dataset            # ImageFolder
    paths = [base.samples[i][0] for i in hold_set.indices]
 
    holdout_path(config).write_text(
        json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Holdout 경로 저장: {holdout_path(config)} ({len(paths)}장)")
 
 
def train(config: dict) -> None:
    set_seed()
 
    pc_cfg = config["patchcore"]
    d      = config["data"]
 
    Path(config["results"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
 
    model = PatchCore(
        backbone          = pc_cfg["backbone"],
        layers            = pc_cfg["layers"],
        coreset_ratio     = pc_cfg["coreset_ratio"],
        neighborhood_size = pc_cfg.get("neighborhood_size", 3),
        seed              = SEED,
    )
 
    bank_loader, hold_set = get_dataloaders(config)
 
    model.fit(images_only(bank_loader))
 
    save_path = bank_path(config)
    model.save(
        str(save_path),
        meta={
            "category"  : d["category"],
            "image_size": d["image_size"],
            "backbone"  : pc_cfg["backbone"],
        },
    )
    save_holdout_paths(config, hold_set)
    logger.info(f"[{d['category']}] PatchCore Memory Bank 구축 완료")
 
 
def load_config(path: str = "config/config.yaml", category: str = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if category:
        config["data"]["category"] = category
    return config
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="config의 category를 덮어씀")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
 
    train(load_config(args.config, args.category))