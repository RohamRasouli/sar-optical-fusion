"""Stres test augmentations — sadece TEST AŞAMASINDA kullanılır.

Üç senaryo:
  1) Bulut overlay (optik bozulur, SAR temiz kalır)
  2) Gece simülasyonu (optik kararır, SAR temiz)
  3) Kombine: bulut + gece + kamuflaj

Bu augmentation'lar tezde stres test sonuçları için uygulanır.
Eğitim sırasında uygulanmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


# ============================================================
# Perlin noise tabanlı bulut
# ============================================================

def _generate_perlin_noise(shape: Tuple[int, int], scale: float = 50.0,
                            octaves: int = 4, persistence: float = 0.5,
                            seed: Optional[int] = None) -> np.ndarray:
    """Basit Perlin-benzeri 2D gürültü (octave karışımı).

    Tam Perlin noise için noise kütüphanesi gerekir; burada hızlı
    noise piramidi yaklaşımı kullanıyoruz.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    out = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    total_amp = 0.0

    for o in range(octaves):
        s = max(1, int(scale / (2 ** o)))
        # Düşük çözünürlüklü gürültü oluştur, sonra interp ile büyüt
        nh = max(2, H // s)
        nw = max(2, W // s)
        low = rng.standard_normal((nh, nw)).astype(np.float32)

        # Bilinear upsampling
        try:
            from PIL import Image
            img = Image.fromarray((low * 127 + 127).clip(0, 255).astype(np.uint8))
            img = img.resize((W, H), Image.BILINEAR)
            up = np.array(img, dtype=np.float32) / 127.0 - 1.0
        except ImportError:
            # NumPy fallback — kaba zoom
            up = np.repeat(np.repeat(low, H // nh + 1, axis=0)[:H],
                            W // nw + 1, axis=1)[:, :W]

        out += amp * up
        total_amp += amp
        amp *= persistence

    out = out / max(total_amp, 1e-6)
    # Normalize [0, 1]
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out


def add_cloud_overlay(optical: torch.Tensor, sar: torch.Tensor,
                      coverage: float = 0.5, seed: Optional[int] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optiğe sentetik bulut ekle. SAR değişmez.

    Args:
        optical: (3, H, W) float [0, 1]
        sar: (C, H, W) — değişmeden döner
        coverage: 0 (bulutsuz) - 1 (tamamen bulutlu)

    Returns:
        (cloudy_optical, sar)
    """
    H, W = optical.shape[-2:]
    noise = _generate_perlin_noise((H, W), scale=80.0, octaves=4, seed=seed)
    # Coverage'a göre eşik: yüksek coverage -> düşük eşik -> daha çok bulut
    threshold = 1.0 - coverage
    cloud_mask = np.clip((noise - threshold) / (1.0 - threshold + 1e-6), 0, 1)
    cloud_mask = cloud_mask ** 0.5  # daha smooth

    cloud_t = torch.from_numpy(cloud_mask).to(optical.device, optical.dtype)
    cloud_t = cloud_t.unsqueeze(0)  # (1, H, W)

    # Beyaz/gri bulut rengi
    cloud_color = torch.tensor([0.92, 0.92, 0.95],
                                  device=optical.device, dtype=optical.dtype).view(3, 1, 1)
    out = optical * (1 - cloud_t) + cloud_color * cloud_t
    out = out.clamp(0, 1)
    return out, sar


# ============================================================
# Gece simülasyonu (low-light)
# ============================================================

def simulate_low_light(optical: torch.Tensor, sar: torch.Tensor,
                       brightness: float = 0.2, noise_std: float = 0.04,
                       blue_shift: float = 0.05, seed: Optional[int] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gece görüntüsü: parlaklık düşür, mavi kanal artır, gürültü ekle.

    Args:
        brightness: parlaklık çarpanı (0.1 - 0.3 tipik)
        noise_std: Gaussian gürültü std (0.02 - 0.06)
        blue_shift: mavi kanala +shift (gece mavi tonu)
    """
    rng = np.random.default_rng(seed)
    out = optical * brightness

    # Mavi shift (gece mavi tonu)
    blue_boost = torch.zeros_like(optical)
    blue_boost[2] = blue_shift  # B kanalı (RGB sırası)
    out = out + blue_boost

    # Gürültü
    noise = rng.standard_normal(out.shape).astype(np.float32) * noise_std
    out = out + torch.from_numpy(noise).to(optical.device, optical.dtype)
    out = out.clamp(0, 1)
    return out, sar


# ============================================================
# Kombine stres
# ============================================================

@dataclass
class StressConfig:
    name: str
    cloud_coverage: float = 0.0
    night: bool = False
    night_brightness: float = 0.2
    apply_camo: bool = False


PRESET_STRESS = {
    "clean": StressConfig(name="clean"),
    "cloud_light": StressConfig(name="cloud_light", cloud_coverage=0.2),
    "cloud_medium": StressConfig(name="cloud_medium", cloud_coverage=0.5),
    "cloud_heavy": StressConfig(name="cloud_heavy", cloud_coverage=0.8),
    "night_light": StressConfig(name="night_light", night=True, night_brightness=0.3),
    "night_dark": StressConfig(name="night_dark", night=True, night_brightness=0.1),
    "camo_only": StressConfig(name="camo_only", apply_camo=True),
    "cloud_camo": StressConfig(name="cloud_camo", cloud_coverage=0.5, apply_camo=True),
    "night_camo": StressConfig(name="night_camo", night=True, night_brightness=0.2, apply_camo=True),
    "all_combined": StressConfig(name="all_combined", cloud_coverage=0.4, night=True,
                                    night_brightness=0.3, apply_camo=True),
}


def apply_stress(optical: torch.Tensor, sar: torch.Tensor,
                  labels: torch.Tensor,
                  cfg: StressConfig,
                  seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Belirli stres senaryosunu uygula."""
    out_opt = optical
    out_sar = sar

    # Normalize input ([0,1] aralığına alın)
    if out_opt.max() > 1.5:
        out_opt = out_opt / 255.0
    out_opt = out_opt.clamp(0, 1).float()

    if cfg.apply_camo:
        from .camo_synth import CamoSynthAugmenter, CamoSynthConfig
        camo = CamoSynthAugmenter(CamoSynthConfig(probability=1.0,
                                                     per_box_probability=0.9))
        out_opt = camo(out_opt, labels, rng=np.random.default_rng(seed))

    if cfg.night:
        out_opt, out_sar = simulate_low_light(
            out_opt, out_sar,
            brightness=cfg.night_brightness, seed=seed,
        )

    if cfg.cloud_coverage > 0:
        out_opt, out_sar = add_cloud_overlay(
            out_opt, out_sar, coverage=cfg.cloud_coverage, seed=seed,
        )

    return out_opt, out_sar


class StressEvaluator:
    """Belirli bir stres senaryosu altında değerlendirme yapar.

    Kullanımı:
        evaluator = StressEvaluator(model, dataset, "cloud_medium")
        results = evaluator.run()
    """

    def __init__(self, model, dataset, stress_name: str = "clean", device="cuda"):
        self.model = model
        self.dataset = dataset
        self.cfg = PRESET_STRESS[stress_name]
        self.device = device

    def __call__(self, idx: int):
        sample = self.dataset[idx]
        opt, sar = apply_stress(
            sample["optical"], sample["sar"], sample["labels"],
            self.cfg, seed=idx,
        )
        return opt, sar, sample["labels"]


if __name__ == "__main__":
    import torch
    opt = torch.rand(3, 128, 128)
    sar = torch.rand(2, 128, 128)
    labels = torch.tensor([[0, 0.5, 0.5, 0.2, 0.2]])

    for preset_name, cfg in PRESET_STRESS.items():
        o, s = apply_stress(opt, sar, labels, cfg, seed=42)
        diff_o = (o - opt.clamp(0, 1)).abs().mean().item()
        print(f"  {preset_name:20s} optic_diff={diff_o:.3f}")
