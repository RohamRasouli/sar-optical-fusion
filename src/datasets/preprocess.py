"""Veri ön işleme yardımcıları: SAR dB dönüşümü, normalize, paired transform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .augmentation.lee_filter import lee_filter

# ImageNet istatistikleri (RGB, [0,1] aralığında)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_db(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """SAR genliğini dB ölçeğine çevir: I_dB = 10 * log10(I + eps)."""
    return 10.0 * torch.log10(x.clamp(min=0.0) + eps)


def quantile_clip(x: torch.Tensor, lo: float = 0.01, hi: float = 0.99) -> torch.Tensor:
    """%lo - %hi quantile'ları arasında kırp (outlier temizliği)."""
    flat = x.reshape(x.size(0), -1) if x.dim() > 1 else x.reshape(-1)
    q_lo = torch.quantile(flat, lo, dim=-1, keepdim=True)
    q_hi = torch.quantile(flat, hi, dim=-1, keepdim=True)
    if x.dim() > 1:
        q_lo = q_lo.view(-1, *([1] * (x.dim() - 1)))
        q_hi = q_hi.view(-1, *([1] * (x.dim() - 1)))
    return x.clamp(min=q_lo, max=q_hi)


def min_max_norm(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] aralığına normalize."""
    if x.dim() <= 1:
        mn, mx = x.min(), x.max()
    else:
        # Her görüntü ayrı ayrı (B, C, H, W) veya (C, H, W)
        flat = x.reshape(*x.shape[:-2], -1)
        mn = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        mx = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
    return (x - mn) / (mx - mn + 1e-8)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet mean/std ile normalize. Girdi [0,1] aralığında olmalı, RGB."""
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    if x.dim() == 4:
        return (x - mean.unsqueeze(0)) / std.unsqueeze(0)
    return (x - mean) / std


@dataclass
class SARPreprocessConfig:
    use_lee: bool = True
    lee_window: int = 7
    lee_damping: float = 1.0
    convert_to_db: bool = True
    clip_quantile: Tuple[float, float] = (0.01, 0.99)
    min_max: bool = True


def preprocess_sar(sar: torch.Tensor, cfg: SARPreprocessConfig) -> torch.Tensor:
    """SAR görüntüsünü model girişine hazırlar.

    Args:
        sar: (C, H, W) ham SAR genliği (lineer ölçek). C tipik 1 veya 2.
        cfg: SARPreprocessConfig

    Returns:
        (C, H, W) önişlenmiş SAR.
    """
    if cfg.convert_to_db:
        sar = to_db(sar)

    if cfg.use_lee:
        sar = lee_filter(sar, window_size=cfg.lee_window,
                         damping_factor=cfg.lee_damping)

    if cfg.clip_quantile is not None:
        sar = quantile_clip(sar, lo=cfg.clip_quantile[0], hi=cfg.clip_quantile[1])

    if cfg.min_max:
        sar = min_max_norm(sar)

    return sar


def preprocess_optical(rgb: torch.Tensor) -> torch.Tensor:
    """Optik (RGB) görüntüyü model girişine hazırlar.

    Args:
        rgb: (3, H, W) [0, 255] uint8 ya da [0, 1] float

    Returns:
        (3, H, W) ImageNet normalize edilmiş.
    """
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    elif rgb.max() > 1.5:
        rgb = rgb.float() / 255.0
    return imagenet_normalize(rgb)


# ============================================================
# PAIRED GEOMETRIC TRANSFORMS — optik+SAR'ı senkronize çevirir
# ============================================================

def _flip_boxes_lr(boxes: torch.Tensor, img_w: int) -> torch.Tensor:
    """YOLO formatında (cx, cy, w, h, normalize) bbox'lar için yatay flip."""
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    out[:, 0] = 1.0 - out[:, 0]  # cx -> 1 - cx
    return out


def _flip_boxes_ud(boxes: torch.Tensor, img_h: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    out[:, 1] = 1.0 - out[:, 1]  # cy -> 1 - cy
    return out


def paired_random_flip(optical: torch.Tensor, sar: torch.Tensor,
                        boxes: torch.Tensor,
                        p_lr: float = 0.5, p_ud: float = 0.0,
                        rng: np.random.Generator | None = None
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optik + SAR + bbox'a senkronize rastgele flip uygular.

    Args:
        optical: (3, H, W)
        sar: (C, H, W)
        boxes: (N, 5) [class, cx, cy, w, h] normalized YOLO formatı, ya da (N, 4)
        p_lr, p_ud: yatay/dikey flip olasılığı

    Returns:
        Aynı şekilde flip edilmiş optical, sar, boxes.
    """
    rng = rng or np.random.default_rng()
    H, W = optical.shape[-2:]
    bbox_xywh = boxes[:, 1:5] if boxes.numel() and boxes.size(1) == 5 else boxes

    if rng.random() < p_lr:
        optical = torch.flip(optical, dims=[-1])
        sar = torch.flip(sar, dims=[-1])
        if bbox_xywh.numel():
            bbox_xywh = _flip_boxes_lr(bbox_xywh, W)

    if rng.random() < p_ud:
        optical = torch.flip(optical, dims=[-2])
        sar = torch.flip(sar, dims=[-2])
        if bbox_xywh.numel():
            bbox_xywh = _flip_boxes_ud(bbox_xywh, H)

    if boxes.numel() and boxes.size(1) == 5:
        boxes = torch.cat([boxes[:, :1], bbox_xywh], dim=1)
    else:
        boxes = bbox_xywh

    return optical, sar, boxes


def hsv_jitter_optical(rgb: torch.Tensor,
                        h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4,
                        rng: np.random.Generator | None = None) -> torch.Tensor:
    """Sadece optiğe HSV gürültüsü uygular (SAR'a uygulanmaz).

    Args:
        rgb: (3, H, W) float [0, 1] aralığında.
    """
    import colorsys

    rng = rng or np.random.default_rng()
    h_shift = rng.uniform(-h_gain, h_gain)
    s_shift = rng.uniform(1.0 - s_gain, 1.0 + s_gain)
    v_shift = rng.uniform(1.0 - v_gain, 1.0 + v_gain)

    # PyTorch'ta vektörel HSV; basit yaklaşım: çarpan formundaki shift
    # Hızlı yaklaşım: V (parlaklık) ve S (doygunluk) için multiplicative shift
    np_rgb = rgb.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    # Hue için RGB rotasyonu yerine basit ofset (tahmini)
    # Tam HSV dönüşümü için cv2.cvtColor kullanılabilir; bağımlılık azaltmak için
    # burada basitleştirilmiş yaklaşım uyguluyoruz
    np_rgb = np_rgb * v_shift                              # value
    np_rgb = (np_rgb - np_rgb.mean(axis=(0, 1), keepdims=True)) * s_shift + \
              np_rgb.mean(axis=(0, 1), keepdims=True)      # saturation benzeri
    np_rgb = np.clip(np_rgb, 0.0, 1.0)
    return torch.from_numpy(np_rgb).permute(2, 0, 1).to(rgb.dtype).to(rgb.device)


if __name__ == "__main__":
    # Sanity
    sar = torch.rand(2, 64, 64) * 100  # ham SAR genlik
    cfg = SARPreprocessConfig()
    out_sar = preprocess_sar(sar, cfg)
    print(f"SAR preprocess: {sar.shape} ({sar.min():.1f}, {sar.max():.1f}) -> "
          f"{out_sar.shape} ({out_sar.min():.3f}, {out_sar.max():.3f}) ✓")

    rgb = torch.rand(3, 64, 64)
    out_rgb = preprocess_optical(rgb)
    print(f"Optical preprocess: {rgb.shape} -> {out_rgb.shape} "
          f"(mean {out_rgb.mean():.3f}) ✓")

    # Paired flip
    boxes = torch.tensor([[0, 0.3, 0.5, 0.1, 0.1], [1, 0.7, 0.5, 0.2, 0.2]])
    o, s, b = paired_random_flip(rgb, sar, boxes, p_lr=1.0, rng=np.random.default_rng(0))
    print(f"Paired flip: boxes cx 0.3,0.7 -> {b[:, 1].tolist()} (beklenen: 0.7, 0.3) ✓")
