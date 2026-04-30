"""SAR Speckle Gürültüsü için Lee Filtresi.

Klasik (parametrik) ve öğrenilebilir (learnable) iki varyant.

Lee filtresi formülü:
    W(x,y) = var_local / (var_local + sigma_n^2)
    output(x,y) = mean_local + W(x,y) * (input(x,y) - mean_local)

W -> 1: kenar bölgesi (varyans yüksek), filtre etkisiz
W -> 0: homojen bölge, çıkış yerel ortalamaya yakınsar
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def lee_filter(image: torch.Tensor, window_size: int = 7,
               damping_factor: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Klasik Lee filtresi (PyTorch, batch destekli).

    Args:
        image: (B, C, H, W) veya (C, H, W) veya (H, W) tensör. Float değerler.
        window_size: yerel pencere boyutu (tek sayı, ör. 5/7/9).
        damping_factor: filtre agresifliği; 1.0 = standart, < 1 = daha az filtreleme.
        eps: sayısal kararlılık.

    Returns:
        Aynı shape'te filtrelenmiş tensör.
    """
    assert window_size % 2 == 1, "window_size tek sayı olmalı"

    orig_shape = image.shape
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    pad = window_size // 2

    # Yerel ortalama
    kernel_mean = torch.ones(C, 1, window_size, window_size,
                              device=image.device, dtype=image.dtype) / (window_size ** 2)
    mean = F.conv2d(F.pad(image, [pad] * 4, mode='reflect'),
                    kernel_mean, groups=C)

    # Yerel varyans = E[X²] - (E[X])²
    mean_sq = F.conv2d(F.pad(image ** 2, [pad] * 4, mode='reflect'),
                       kernel_mean, groups=C)
    var = mean_sq - mean ** 2
    var = var.clamp(min=0.0)

    # Gürültü varyansı tahmini: tüm görüntü üzerinde ortalama varyans
    # (alternatif: koefisyent of variation tabanlı)
    sigma_n_sq = var.mean(dim=(-2, -1), keepdim=True) * damping_factor

    # Adaptif ağırlık
    w = var / (var + sigma_n_sq + eps)

    # Filtrelenmiş çıktı
    out = mean + w * (image - mean)

    return out.reshape(orig_shape)


class LearnableLeeFilter(nn.Module):
    """Öğrenilebilir Lee-benzeri filtre.

    Klasik Lee filtresinin sabit ağırlıklarını öğrenmeli hale getirir.
    Yerel istatistikleri toplayan bir kanaldır + 1x1 convolution ile birleştirme.
    Eğitim sırasında modelin SAR'a en uygun denoising'i bulması beklenir.
    """

    def __init__(self, in_channels: int, window_size: int = 7,
                 hidden_dim: int = 16):
        super().__init__()
        assert window_size % 2 == 1
        self.in_channels = in_channels
        self.window_size = window_size
        self.pad = window_size // 2

        # Yerel istatistikler için sabit kernel'ler
        self.register_buffer('mean_kernel',
            torch.ones(in_channels, 1, window_size, window_size) / (window_size ** 2))

        # Öğrenilebilir birleştirme: [input, mean, var, mean_sq] -> output
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels * 4, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Yerel istatistikler
        x_pad = F.pad(x, [self.pad] * 4, mode='reflect')
        mean = F.conv2d(x_pad, self.mean_kernel, groups=self.in_channels)
        mean_sq = F.conv2d(x_pad ** 2, self.mean_kernel, groups=self.in_channels)
        var = (mean_sq - mean ** 2).clamp(min=0.0)

        # Birleştir
        feat = torch.cat([x, mean, var, mean_sq], dim=1)
        residual = self.combine(feat)
        return x + residual  # residual connection — başlangıçta x ≈ output


if __name__ == "__main__":
    # Hızlı sanity test
    sar = torch.randn(2, 2, 64, 64)
    out = lee_filter(sar, window_size=7)
    assert out.shape == sar.shape
    print(f"Klasik Lee: input {sar.shape} -> output {out.shape} ✓")

    learnable = LearnableLeeFilter(in_channels=2)
    out2 = learnable(sar)
    assert out2.shape == sar.shape
    n_params = sum(p.numel() for p in learnable.parameters() if p.requires_grad)
    print(f"Learnable Lee: input {sar.shape} -> output {out2.shape}, params: {n_params} ✓")
