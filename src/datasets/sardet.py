"""SARDet-100K Veri Seti Yükleyicisi.

SARDet-100K (Li et al., NeurIPS 2024); ~117K SAR görüntü, 6 sınıf.
Kullanımı: SAR backbone'unun pre-training'i için.

Beklenen klasör yapısı:
    data_root/
    ├── images/{train, val, test}/img_*.png
    └── labels/{train, val, test}/img_*.txt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .m4_sar import _load_yolo_labels, _resize_pair  # paired olmasa da boyutlandırma yardımcısı
from .preprocess import (
    SARPreprocessConfig,
    paired_random_flip,
    preprocess_sar,
)


SARDET_CLASSES = ["aircraft", "ship", "car", "tank", "bridge", "harbor"]


@dataclass
class SARDetConfig:
    data_root: str
    split: str = "train"
    img_size: int = 640
    sar_preprocess: SARPreprocessConfig = None
    augment: bool = True
    p_lr: float = 0.5

    def __post_init__(self):
        if self.sar_preprocess is None:
            self.sar_preprocess = SARPreprocessConfig()


class SARDetDataset(Dataset):
    """Tek modal SAR detection veri seti."""

    def __init__(self, config: SARDetConfig):
        self.cfg = config
        root = Path(config.data_root)
        self.img_dir = root / "images" / config.split
        self.label_dir = root / "labels" / config.split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Klasör yok: {self.img_dir}")

        # PNG, JPG her ikisi de olabilir
        self.paths = sorted(
            list(self.img_dir.glob("*.png")) +
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.tif"))
        )
        if not self.paths:
            raise RuntimeError(f"Görüntü bulunamadı: {self.img_dir}")

        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.paths[idx]
        img_id = img_path.stem

        # PIL ile yükle (gri veya RGB kabul eder)
        from PIL import Image
        arr = np.array(Image.open(img_path))
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1, H, W)
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
            if arr.shape[0] == 3:
                arr = arr.mean(axis=0, keepdims=True)  # gri yap (SAR)
        sar = torch.from_numpy(arr.astype(np.float32))

        labels = _load_yolo_labels(self.label_dir / f"{img_id}.txt")

        # Boyut
        import torch.nn.functional as F
        sar = F.interpolate(sar.unsqueeze(0), size=(self.cfg.img_size, self.cfg.img_size),
                             mode='bilinear', align_corners=False).squeeze(0)

        # Tek modal: SAR'ı 2 kanala genişlet (model 2 kanal bekliyor)
        if sar.size(0) == 1:
            sar = sar.repeat(2, 1, 1)

        # Augment (sadece flip)
        if self.cfg.augment:
            # paired_random_flip optik için 3 kanal bekler — burada sar ile dummy optic kullanıyoruz
            dummy_opt = sar[:1].repeat(3, 1, 1)
            dummy_opt, sar, labels = paired_random_flip(
                dummy_opt, sar, labels, p_lr=self.cfg.p_lr, rng=self._rng
            )

        sar = preprocess_sar(sar, self.cfg.sar_preprocess)

        return {
            "sar": sar,
            "labels": labels,
            "image_id": img_id,
        }


def collate_fn_sar(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """SAR-only collate."""
    sar = torch.stack([b["sar"] for b in batch])
    label_chunks = []
    for i, b in enumerate(batch):
        if b["labels"].numel():
            n = b["labels"].size(0)
            batch_idx = torch.full((n, 1), i, dtype=torch.float32)
            label_chunks.append(torch.cat([batch_idx, b["labels"]], dim=1))
    labels = torch.cat(label_chunks, dim=0) if label_chunks \
        else torch.zeros((0, 6), dtype=torch.float32)
    return {
        "sar": sar,
        "labels": labels,
        "image_ids": [b["image_id"] for b in batch],
    }
