"""M4-SAR Veri Seti Yükleyicisi.

M4-SAR (Wang et al., 2024); Sentinel-1 SAR + Sentinel-2 optik eşleştirilmiş
çiftler. Her çift için YOLO formatında etiket dosyası beklenir.

Beklenen klasör yapısı:
    data_root/
    ├── optical/
    │   ├── train/img_001.png ...
    │   ├── val/
    │   └── test/
    ├── sar/
    │   ├── train/img_001.tif ...    # 2 kanal (VV, VH) ya da .npy
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/img_001.txt        # YOLO: class cx cy w h (normalize 0-1)
        ├── val/
        └── test/

Etiket sınıf indeksleri (M4-SAR varsayılan):
    0: Aircraft, 1: Ship, 2: Vehicle, 3: Bridge, 4: Storage tank, 5: Oil tank
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import (
    SARPreprocessConfig,
    paired_random_flip,
    preprocess_optical,
    preprocess_sar,
)


M4_SAR_CLASSES = ["aircraft", "ship", "vehicle", "bridge", "storage", "oiltank"]
NUM_CLASSES = len(M4_SAR_CLASSES)


def _load_optical(path: Path) -> torch.Tensor:
    """RGB optik görüntüyü (3, H, W) uint8 tensör olarak yükle."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Görüntü okunamadı: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()
    except ImportError:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_sar(path: Path) -> torch.Tensor:
    """SAR görüntüsünü (C, H, W) float tensör yükle.

    Dosya .npy ise numpy array bekler; .tif ise rasterio kullanır;
    .png/.jpg ise PIL ile yükler (M4-SAR uyumluluk).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        try:
            import rasterio  # type: ignore
            with rasterio.open(path) as src:
                arr = src.read()  # (C, H, W)
        except ImportError:
            from PIL import Image
            arr = np.array(Image.open(path))
            if arr.ndim == 2:
                arr = arr[None, ...]
    elif suffix in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        arr = np.array(Image.open(path))
        if arr.ndim == 2:
            arr = arr[None, ...]  # grayscale → (1, H, W)
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
    else:
        raise ValueError(f"Desteklenmeyen SAR formatı: {suffix}")

    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 2, 3):
        # HWC -> CHW
        if arr.shape[-1] < arr.shape[0]:
            arr = arr.transpose(2, 0, 1)

    # SAR model 2 kanal (VV+VH) bekler; tek kanallıyı çoğalt
    arr = arr.astype(np.float32)
    if arr.shape[0] == 1:
        arr = np.concatenate([arr, arr], axis=0)  # (1,H,W) → (2,H,W)
    elif arr.shape[0] == 3:
        # RGB SAR → ilk 2 kanal al (genelde R≈VV, G≈VH)
        arr = arr[:2]

    return torch.from_numpy(arr)


def _load_yolo_labels(path: Path) -> torch.Tensor:
    """YOLO formatlı etiket dosyası -> (N, 5) [class, cx, cy, w, h]."""
    if not path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(p) for p in parts[:5]])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _resize_pair(optical: torch.Tensor, sar: torch.Tensor,
                 target: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optik ve SAR'ı aynı kare boyuta yeniden boyutlandır.

    Boxes normalize olduğu için yeniden hesap gerekmez.
    """
    import torch.nn.functional as F
    optical = F.interpolate(optical.unsqueeze(0).float(), size=(target, target),
                             mode='bilinear', align_corners=False).squeeze(0)
    sar = F.interpolate(sar.unsqueeze(0).float(), size=(target, target),
                         mode='bilinear', align_corners=False).squeeze(0)
    return optical, sar


@dataclass
class M4SARConfig:
    data_root: str
    split: str = "train"
    img_size: int = 640
    sar_preprocess: SARPreprocessConfig = None
    augment: bool = True
    p_lr: float = 0.5
    p_ud: float = 0.0
    optical_ext: str = ".jpg"
    sar_ext: str = ".jpg"
    label_ext: str = ".txt"

    def __post_init__(self):
        if self.sar_preprocess is None:
            self.sar_preprocess = SARPreprocessConfig()


class M4SARDataset(Dataset):
    """M4-SAR multimodal eşleştirilmiş veri seti."""

    def __init__(self, config: M4SARConfig,
                 camo_synth_aug: Optional[Any] = None):
        self.cfg = config
        self.root = Path(config.data_root)
        self.split = config.split

        self.optical_dir = self.root / "optical" / self.split
        self.sar_dir = self.root / "sar" / self.split
        self.label_dir = self.root / "labels" / self.split

        import os
        # Kaggle raw dataset yapıları için SÜPER ESNEK ve GÜÇLÜ dinamik arama
        if not self.optical_dir.exists() or "data/m4_sar" in str(self.optical_dir):
            print(f"[SEARCH] [{self.split.upper()}] Veri seti hiyerarsisi taraniyor (Derinlik: 10)...", flush=True)
            queue = [(str(self.root), 0)]
            max_depth = 10
            
            # Başlangıçta hepsini None yapalım ki gerçekten bulup bulmadığımızı bilelim
            found_opt, found_sar, found_lbl = None, None, None
            
            while queue and not (found_opt and found_sar and found_lbl):
                curr_dir, depth = queue.pop(0)
                if depth > max_depth: continue
                
                try:
                    with os.scandir(curr_dir) as it:
                        for entry in it:
                            if entry.is_dir(follow_symlinks=True):
                                # Eğer klasör adı 'train' veya 'val' veya 'test' ise (split adımız)
                                if entry.name.lower() == self.split.lower():
                                    full_p = entry.path.lower()
                                    p = Path(entry.path)
                                    
                                    # Daha spesifik eşleştirme (Yolun son parçalarına bakıyoruz)
                                    rel_p = str(p.relative_to(self.root)).lower()
                                    
                                    if any(k in rel_p for k in ["label", "ann", "mask", "gt"]):
                                        found_lbl = p
                                        print(f"  [OK] Etiketler bulundu: {p}", flush=True)
                                    elif any(k in rel_p for k in ["sar", "radar", "s1", "sentinel1"]):
                                        found_sar = p
                                        print(f"  [OK] SAR/Radar bulundu: {p}", flush=True)
                                    elif any(k in rel_p for k in ["opt", "image", "visual", "rgb"]):
                                        found_opt = p
                                        print(f"  [OK] Optik bulundu: {p}", flush=True)
                                else:
                                    queue.append((entry.path, depth + 1))
                except Exception:
                    pass
            
            # Bulunanları asıl değişkenlere ata
            if found_opt: self.optical_dir = found_opt
            if found_sar: self.sar_dir = found_sar
            if found_lbl: self.label_dir = found_lbl

        # KRİTİK: Eğer biri bile eksikse dummy veriye GEÇME, hata ver ki düzeltelim!
        missing = []
        if not self.optical_dir.exists(): missing.append("Optik")
        if not self.sar_dir.exists(): missing.append("SAR")
        if not self.label_dir.exists(): missing.append("Label")
        
        if missing:
            raise FileNotFoundError(f"Kritik klasörler bulunamadı: {', '.join(missing)}. "
                                    f"Root: {self.root} taranmasına rağmen bulunamadı.")
        import os
        # Optik dosyalardan ID listesi çıkar
        if not self.optical_dir.exists():
            raise FileNotFoundError(f"Optik klasörü yok: {self.optical_dir}. Veriyi data_root altına yerleştirdiğinden emin ol.")
            
        print(f"[BUSY] [{self.split.upper()}] Dosyalar taraniyor (Kaggle ag surucusunde 100.000+ dosya okumak 2-3 DAKIKA surebilir, lutfen bekleyin)...", flush=True)
        
        # Optik dosyalardan ID listesi ve uzantı çıkar
        self.ids = []
        possible_exts = [".jpg", ".png", ".jpeg", ".tif", ".tiff"]
        all_files = os.listdir(self.optical_dir)
        
        for ext in possible_exts:
            matched_ids = [f[:-len(ext)] for f in all_files if f.lower().endswith(ext)]
            if matched_ids:
                self.ids = sorted(matched_ids)
                self.cfg.optical_ext = ext
                break
        
        # SAR uzantısını da otomatik tespit et
        if self.sar_dir.exists():
            sar_files = os.listdir(self.sar_dir)
            for ext in [".jpg", ".png", ".jpeg", ".tif", ".tiff", ".npy"]:
                if any(f.lower().endswith(ext) for f in sar_files):
                    self.cfg.sar_ext = ext
                    break
                
        print(f"[OK] [{self.split.upper()}] Taramasi bitti: {len(self.ids)} adet goruntu bulundu.", flush=True)
        
        if not self.ids:
            raise RuntimeError(f"Hiç optik görüntü bulunamadı: {self.optical_dir}")
            
        print(f"[LINK] Sensor verileri eslestiriliyor (Optical + SAR + Label)...", flush=True)
        
        # SAR ve Label klasorlerindeki ID'leri de listele
        sar_ids = set()
        for f in os.listdir(self.sar_dir):
            if "." in f: sar_ids.add(f.rsplit(".", 1)[0])
            
        label_ids = set()
        for f in os.listdir(self.label_dir):
            if f.endswith(".txt"): label_ids.add(f[:-4])
            
        # Kesisimi al: Sadece 3 klasorde de olanlari tut
        final_ids = []
        for img_id in self.ids:
            if img_id in sar_ids and img_id in label_ids:
                final_ids.append(img_id)
                
        self.ids = sorted(final_ids)
        print(f"[OK] Eslestirme tamamlandi: {len(self.ids)} adet TAM UYUMLU veri cifti bulundu.", flush=True)
        
        if not self.ids:
            raise RuntimeError(f"Eşleşen veri bulunamadı! Klasörleri kontrol edin: {self.optical_dir}, {self.sar_dir}, {self.label_dir}")
        self.camo_synth_aug = camo_synth_aug
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.ids[idx]

        # Yükle
        opt = _load_optical(self.optical_dir / f"{img_id}{self.cfg.optical_ext}")

        # SAR: konfigüredeki uzantı yoksa alternatif uzantıları dene
        sar_path = self.sar_dir / f"{img_id}{self.cfg.sar_ext}"
        if not sar_path.exists():
            for alt_ext in [".npy", ".png", ".tif", ".jpg"]:
                alt_path = self.sar_dir / f"{img_id}{alt_ext}"
                if alt_path.exists():
                    sar_path = alt_path
                    break
        sar = _load_sar(sar_path)

        labels = _load_yolo_labels(self.label_dir / f"{img_id}{self.cfg.label_ext}")

        # Aynı boyuta getir
        opt, sar = _resize_pair(opt, sar, self.cfg.img_size)

        # Sentetik kamuflaj (sadece eğitimde, optiğe uygulanır)
        if self.cfg.augment and self.camo_synth_aug is not None:
            opt = self.camo_synth_aug(opt, labels, rng=self._rng)

        # Geometrik augment (her iki modaliteye senkron)
        if self.cfg.augment:
            opt, sar, labels = paired_random_flip(
                opt, sar, labels,
                p_lr=self.cfg.p_lr, p_ud=self.cfg.p_ud, rng=self._rng
            )

        # Önişle
        opt = preprocess_optical(opt)
        sar = preprocess_sar(sar, self.cfg.sar_preprocess)

        return {
            "optical": opt,             # (3, H, W) float
            "sar": sar,                 # (C, H, W) float
            "labels": labels,           # (N, 5) [class, cx, cy, w, h]
            "image_id": img_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Variable-length labels için custom collate.

    Çıktı:
        optical:  (B, 3, H, W)
        sar:      (B, C, H, W)
        labels:   (M, 6) [batch_idx, class, cx, cy, w, h]   <- birleştirilmiş
        image_ids: List[str]
    """
    optical = torch.stack([b["optical"] for b in batch])
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
        "optical": optical,
        "sar": sar,
        "labels": labels,
        "image_ids": [b["image_id"] for b in batch],
    }


# ============================================================
# DUMMY VERİ SETİ — sandbox/test için (gerçek veri yokken)
# ============================================================

class DummyM4SARDataset(Dataset):
    """Sentetik dummy multimodal veri — pipeline doğrulama için.

    Gerçek M4-SAR yokken testler bunu kullanır.
    """

    def __init__(self, num_samples: int = 64, img_size: int = 256,
                 num_classes: int = NUM_CLASSES, sar_channels: int = 2):
        self.n = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.sar_channels = sar_channels
        self._rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H = W = self.img_size
        opt = torch.rand(3, H, W)
        sar = torch.rand(self.sar_channels, H, W) * 100  # ham SAR genliği

        # Rastgele 0-3 hedef
        n_targets = int(self._rng.integers(0, 4))
        if n_targets > 0:
            cls = torch.tensor(self._rng.integers(0, self.num_classes, n_targets),
                                dtype=torch.float32).unsqueeze(1)
            # Kutu merkezleri 0.1-0.9, genişlik 0.05-0.3
            cxcy = torch.tensor(self._rng.uniform(0.15, 0.85, (n_targets, 2)),
                                 dtype=torch.float32)
            wh = torch.tensor(self._rng.uniform(0.05, 0.25, (n_targets, 2)),
                               dtype=torch.float32)
            labels = torch.cat([cls, cxcy, wh], dim=1)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        # Önişle
        from .preprocess import SARPreprocessConfig as _C
        opt = preprocess_optical(opt)
        sar = preprocess_sar(sar, _C())

        return {
            "optical": opt,
            "sar": sar,
            "labels": labels,
            "image_id": f"dummy_{idx:04d}",
        }


if __name__ == "__main__":
    ds = DummyM4SARDataset(num_samples=4, img_size=128)
    print(f"Dummy dataset uzunluğu: {len(ds)}")
    sample = ds[0]
    print(f"  optical: {sample['optical'].shape}")
    print(f"  sar: {sample['sar'].shape}")
    print(f"  labels: {sample['labels'].shape}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  optical: {batch['optical'].shape}")
    print(f"  sar: {batch['sar'].shape}")
    print(f"  labels: {batch['labels'].shape}  (kolonlar: batch_idx, cls, cx, cy, w, h)")
