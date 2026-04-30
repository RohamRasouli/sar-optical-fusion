"""Sentetik Kamuflaj Augmentation Pipeline.

Eğitim sırasında optik görüntülerdeki hedeflerin üzerine programatik kamuflaj
efekti uygular. SAR görüntüsü değiştirilmez (kamuflaj boyaları radar dalgalarını
genelde geçirir; bu fiziksel asimetri projenin akademik özgünlük noktasıdır).

Adımlar:
    1) Bbox'ı YOLO formatından piksel koordinatlarına çevir
    2) Hedef bölge maskesi oluştur (basit yaklaşım: dikdörtgen veya elips)
    3) Bbox dışından arka plan dokusu örnekle
    4) Renk uydurma: hedef pikseli arka plan istatistiğine kaydır
    5) Doku alpha-blending (random alpha)
    6) Opsiyonel: hex desen kamuflaj ağı overlay
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class CamoSynthConfig:
    """Sentetik kamuflaj parametreleri."""
    probability: float = 0.3                  # Bu kadar görüntüde uygula
    per_box_probability: float = 0.7          # Uygulananlardan %70 kutuya
    texture_blend_alpha: Tuple[float, float] = (0.4, 0.8)
    color_match_strength: Tuple[float, float] = (0.5, 1.0)
    net_overlay_prob: float = 0.5
    net_cell_size: Tuple[int, int] = (8, 16)  # hex pattern boyut aralığı
    use_elliptical_mask: bool = True


def _yolo_to_xyxy(box: torch.Tensor, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
    """[cx, cy, w, h] normalize -> piksel (x1, y1, x2, y2)."""
    cx, cy, w, h = box.tolist()
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w, int((cx + w / 2) * img_w))
    y2 = min(img_h, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


def _make_elliptical_mask(h: int, w: int) -> np.ndarray:
    """h × w boyutunda elips şeklinde [0, 1] mask. Kenarda 0, merkezde 1, soft."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    a, b = max(w / 2.0, 1.0), max(h / 2.0, 1.0)
    r = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
    mask = np.clip(1.0 - r, 0.0, 1.0)
    # Yumuşak sınır için Gaussian benzeri yumuşat
    mask = mask ** 0.6
    return mask


def _sample_background_patch(img_np: np.ndarray, target_xyxy: Tuple[int, int, int, int],
                              all_boxes: list, rng: np.random.Generator,
                              max_tries: int = 12) -> Optional[np.ndarray]:
    """Hedef olmayan bir bölgeden hedef boyutunda doku patch'i örnekle.

    Args:
        img_np: (H, W, 3) [0,1] float
        target_xyxy: hedef bbox
        all_boxes: tüm bbox'lar [(x1,y1,x2,y2), ...]
        rng: random generator
    """
    H, W = img_np.shape[:2]
    tx1, ty1, tx2, ty2 = target_xyxy
    th, tw = ty2 - ty1, tx2 - tx1
    if th <= 1 or tw <= 1:
        return None

    for _ in range(max_tries):
        x1 = int(rng.integers(0, max(W - tw, 1)))
        y1 = int(rng.integers(0, max(H - th, 1)))
        x2, y2 = x1 + tw, y1 + th

        # Diğer bbox'larla overlap kontrolü
        overlap = False
        for bx1, by1, bx2, by2 in all_boxes:
            if not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2):
                overlap = True
                break
        if not overlap:
            return img_np[y1:y2, x1:x2].copy()

    return None  # bulunamadı


def _make_hex_pattern(h: int, w: int, cell_size: int = 12,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Kamuflaj ağı benzeri hex desen [0, 1] mask."""
    rng = rng or np.random.default_rng()
    mask = np.ones((h, w), dtype=np.float32) * 0.95
    # Hex grid yaklaşımı: çift satırlar shift edilmiş daire
    radius = cell_size / 2.0
    for row in range(0, h + cell_size, cell_size):
        shift = (cell_size // 2) if (row // cell_size) % 2 else 0
        for col in range(-cell_size, w + cell_size, cell_size):
            cy_, cx_ = row, col + shift
            if 0 <= cy_ < h and 0 <= cx_ < w:
                # Dairesel "delik"
                yy, xx = np.ogrid[0:h, 0:w]
                d = np.sqrt((yy - cy_) ** 2 + (xx - cx_) ** 2)
                mask = np.where(d < radius * 0.6, 0.6, mask)
    # Hafif gürültü
    noise = rng.normal(0, 0.05, (h, w)).astype(np.float32)
    return np.clip(mask + noise, 0.0, 1.0)


def synthetic_camouflage(optical: torch.Tensor, labels: torch.Tensor,
                          cfg: Optional[CamoSynthConfig] = None,
                          rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Sentetik kamuflaj uygula (sadece optiğe).

    Args:
        optical: (3, H, W) float [0, 1]
        labels: (N, 5) [class, cx, cy, w, h] normalize, ya da (N, 4)
        cfg: CamoSynthConfig
        rng: numpy random generator

    Returns:
        Aynı şekilde optik tensör (kamuflajlı).
    """
    cfg = cfg or CamoSynthConfig()
    rng = rng or np.random.default_rng()

    if labels.numel() == 0:
        return optical
    if rng.random() > cfg.probability:
        return optical

    # Tensor -> numpy (H, W, 3) [0, 1]
    img_np = optical.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if img_np.max() > 1.5:
        img_np = img_np / 255.0
    H, W = img_np.shape[:2]

    # YOLO -> piksel xyxy
    if labels.size(1) == 5:
        boxes_yolo = labels[:, 1:5]
    else:
        boxes_yolo = labels[:, :4]
    all_xyxy = [_yolo_to_xyxy(b, H, W) for b in boxes_yolo]

    for box_xyxy in all_xyxy:
        if rng.random() > cfg.per_box_probability:
            continue
        x1, y1, x2, y2 = box_xyxy
        bh, bw = y2 - y1, x2 - x1
        if bh < 4 or bw < 4:
            continue

        target_patch = img_np[y1:y2, x1:x2].copy()

        # 1) Arka plan örnekle
        bg_patch = _sample_background_patch(img_np, box_xyxy, all_xyxy, rng)
        if bg_patch is None or bg_patch.shape != target_patch.shape:
            continue

        # 2) Renk istatistiği uydurma
        match_strength = float(rng.uniform(*cfg.color_match_strength))
        bg_mean = bg_patch.mean(axis=(0, 1), keepdims=True)
        bg_std = bg_patch.std(axis=(0, 1), keepdims=True) + 1e-6
        tg_mean = target_patch.mean(axis=(0, 1), keepdims=True)
        tg_std = target_patch.std(axis=(0, 1), keepdims=True) + 1e-6
        # Pixel-by-pixel: (x - tg_mean) * (bg_std / tg_std) + bg_mean
        color_matched = (target_patch - tg_mean) * (bg_std / tg_std) + bg_mean
        target_patch = (1 - match_strength) * target_patch + match_strength * color_matched

        # 3) Doku alpha-blending
        alpha = float(rng.uniform(*cfg.texture_blend_alpha))
        blended = (1 - alpha) * target_patch + alpha * bg_patch

        # 4) Mask (elips ya da kare)
        if cfg.use_elliptical_mask:
            mask = _make_elliptical_mask(bh, bw)[..., None]  # (h, w, 1)
        else:
            mask = np.ones((bh, bw, 1), dtype=np.float32)

        # 5) Hex ağ overlay (opsiyonel)
        if rng.random() < cfg.net_overlay_prob:
            cell = int(rng.integers(cfg.net_cell_size[0], cfg.net_cell_size[1] + 1))
            net_mask = _make_hex_pattern(bh, bw, cell_size=cell, rng=rng)[..., None]
            blended = blended * net_mask

        # 6) Hedef bölgeye geri yaz (mask ile soft)
        img_np[y1:y2, x1:x2] = (
            mask * np.clip(blended, 0.0, 1.0) +
            (1 - mask) * img_np[y1:y2, x1:x2]
        )

    out = torch.from_numpy(img_np.transpose(2, 0, 1)).to(optical.dtype).to(optical.device)
    return out


class CamoSynthAugmenter:
    """Dataset içinde kullanılabilen callable wrapper."""

    def __init__(self, cfg: Optional[CamoSynthConfig] = None):
        self.cfg = cfg or CamoSynthConfig()

    def __call__(self, optical: torch.Tensor, labels: torch.Tensor,
                  rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        return synthetic_camouflage(optical, labels, self.cfg, rng)


if __name__ == "__main__":
    # Sanity test
    img = torch.rand(3, 128, 128)
    boxes = torch.tensor([
        [0, 0.3, 0.4, 0.15, 0.2],
        [1, 0.7, 0.6, 0.1, 0.1],
    ])
    cfg = CamoSynthConfig(probability=1.0, per_box_probability=1.0)
    aug = CamoSynthAugmenter(cfg)
    out = aug(img, boxes, rng=np.random.default_rng(42))
    print(f"Sentetik kamuflaj: input {img.shape} -> output {out.shape}")
    print(f"  fark normu (uygulandığını gösterir): {(out - img).abs().mean():.4f}")
