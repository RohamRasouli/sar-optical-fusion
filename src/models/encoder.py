"""Çift Akımlı Encoder.

Optik (RGB) ve SAR (VV+VH) girişlerini paralel işleyen iki adet CSPDarknet-temelli
backbone. Üç ölçekli özellik haritası üretir: 1/8, 1/16, 1/32.

Mimari özet (YOLOv8-s benzeri):
    Stem(3->32 ya da 2->32) → C2f(32->64) → C2f(64->128, 1/8 OUT)
    → C2f(128->256, 1/16 OUT) → SPPF + C2f(256->512, 1/32 OUT)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _autopad(k: int, p: int = None) -> int:
    """\"same\" padding hesaplama."""
    return k // 2 if p is None else p


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + SiLU."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1,
                 p: int = None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, _autopad(k, p),
                               groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck."""

    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_h = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, c_h, k=3)
        self.cv2 = ConvBNAct(c_h, c_out, k=3)
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions, fast (YOLOv8 stili)."""

    def __init__(self, c_in: int, c_out: int, n: int = 1, shortcut: bool = True,
                 e: float = 0.5):
        super().__init__()
        self.c = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, 2 * self.c, k=1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c_out, k=1)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0) for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        c_h = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_h, k=1)
        self.cv2 = ConvBNAct(c_h * 4, c_out, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class CSPDarknetBackbone(nn.Module):
    """Tek modaliteli CSPDarknet backbone.

    Üç ölçekli çıktı: 1/8, 1/16, 1/32 oranında özellik haritaları.
    """

    def __init__(self, in_channels: int = 3,
                 base_channels: int = 32,
                 depth_mult: float = 0.33,
                 width_mult: float = 0.5,
                 out_channels: Tuple[int, int, int] = (128, 256, 512)):
        super().__init__()

        def w(c: int) -> int:
            return max(8, int(c * width_mult))

        def d(n: int) -> int:
            return max(1, int(round(n * depth_mult)))

        # Stem: in -> base, /2
        self.stem = ConvBNAct(in_channels, w(base_channels), k=3, s=2)

        # Stage 1: /4
        self.stage1 = nn.Sequential(
            ConvBNAct(w(base_channels), w(base_channels * 2), k=3, s=2),
            C2f(w(base_channels * 2), w(base_channels * 2), n=d(3), shortcut=True),
        )

        # Stage 2: /8 (out 1)
        self.stage2 = nn.Sequential(
            ConvBNAct(w(base_channels * 2), w(base_channels * 4), k=3, s=2),
            C2f(w(base_channels * 4), w(base_channels * 4), n=d(6), shortcut=True),
        )

        # Stage 3: /16 (out 2)
        self.stage3 = nn.Sequential(
            ConvBNAct(w(base_channels * 4), w(base_channels * 8), k=3, s=2),
            C2f(w(base_channels * 8), w(base_channels * 8), n=d(6), shortcut=True),
        )

        # Stage 4: /32 (out 3) + SPPF
        self.stage4 = nn.Sequential(
            ConvBNAct(w(base_channels * 8), w(base_channels * 16), k=3, s=2),
            C2f(w(base_channels * 16), w(base_channels * 16), n=d(3), shortcut=True),
            SPPF(w(base_channels * 16), w(base_channels * 16)),
        )

        # Çıkış kanal sayılarını isteğe göre projection ile sabitle
        self.proj1 = nn.Conv2d(w(base_channels * 4), out_channels[0], kernel_size=1)
        self.proj2 = nn.Conv2d(w(base_channels * 8), out_channels[1], kernel_size=1)
        self.proj3 = nn.Conv2d(w(base_channels * 16), out_channels[2], kernel_size=1)

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)            # /2
        x = self.stage1(x)           # /4
        f1 = self.stage2(x)          # /8
        f2 = self.stage3(f1)         # /16
        f3 = self.stage4(f2)         # /32
        return [self.proj1(f1), self.proj2(f2), self.proj3(f3)]


class DualStreamEncoder(nn.Module):
    """İki paralel CSPDarknet (optik + SAR)."""

    def __init__(self,
                 optical_channels: int = 3,
                 sar_channels: int = 2,
                 base_channels: int = 32,
                 depth_mult: float = 0.33,
                 width_mult: float = 0.5,
                 out_channels: Tuple[int, int, int] = (128, 256, 512)):
        super().__init__()
        common = dict(base_channels=base_channels, depth_mult=depth_mult,
                       width_mult=width_mult, out_channels=out_channels)
        self.optical_encoder = CSPDarknetBackbone(in_channels=optical_channels, **common)
        self.sar_encoder = CSPDarknetBackbone(in_channels=sar_channels, **common)
        self.out_channels = out_channels

    def forward(self, optical: torch.Tensor, sar: torch.Tensor
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        opt_feats = self.optical_encoder(optical)
        sar_feats = self.sar_encoder(sar)
        return opt_feats, sar_feats


if __name__ == "__main__":
    enc = DualStreamEncoder(optical_channels=3, sar_channels=2)
    n_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"DualStreamEncoder param sayısı: {n_params/1e6:.2f}M")

    opt = torch.randn(2, 3, 256, 256)
    sar = torch.randn(2, 2, 256, 256)
    opt_feats, sar_feats = enc(opt, sar)
    print("Optik özellikler:")
    for i, f in enumerate(opt_feats):
        print(f"  scale 1/{8 * (2 ** i)}: {tuple(f.shape)}")
    print("SAR özellikler:")
    for i, f in enumerate(sar_feats):
        print(f"  scale 1/{8 * (2 ** i)}: {tuple(f.shape)}")
