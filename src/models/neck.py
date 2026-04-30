"""PAN-FPN Neck.

Üç ölçekli özellik haritalarını yukarı-aşağı yönlü kaynaştırır.
YOLOv8 standart Path Aggregation Network mimarisi.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import C2f, ConvBNAct


class PANFPN(nn.Module):
    """Bidirectional FPN (Top-down + Bottom-up).

    Girdi:  [P3 (1/8), P4 (1/16), P5 (1/32)] — kanal sayıları farklı olabilir
    Çıktı:  [N3, N4, N5] — hepsi out_channels'a normalize edilmiş
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256,
                 depth_mult: float = 0.33):
        super().__init__()
        assert len(in_channels) == 3
        c3, c4, c5 = in_channels

        n = max(1, int(round(3 * depth_mult)))

        # 1x1 girişleri unify et
        self.lateral3 = nn.Conv2d(c3, out_channels, 1)
        self.lateral4 = nn.Conv2d(c4, out_channels, 1)
        self.lateral5 = nn.Conv2d(c5, out_channels, 1)

        # Top-down (semantic up)
        self.td_block4 = C2f(out_channels * 2, out_channels, n=n, shortcut=False)
        self.td_block3 = C2f(out_channels * 2, out_channels, n=n, shortcut=False)

        # Bottom-up (localization)
        self.down3 = ConvBNAct(out_channels, out_channels, k=3, s=2)
        self.bu_block4 = C2f(out_channels * 2, out_channels, n=n, shortcut=False)
        self.down4 = ConvBNAct(out_channels, out_channels, k=3, s=2)
        self.bu_block5 = C2f(out_channels * 2, out_channels, n=n, shortcut=False)

        self.out_channels = out_channels

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = feats
        l3 = self.lateral3(p3)
        l4 = self.lateral4(p4)
        l5 = self.lateral5(p5)

        # Top-down: l5 -> up -> concat with l4 -> ... -> n3
        u5 = F.interpolate(l5, size=l4.shape[-2:], mode='nearest')
        m4 = self.td_block4(torch.cat([u5, l4], dim=1))
        u4 = F.interpolate(m4, size=l3.shape[-2:], mode='nearest')
        n3 = self.td_block3(torch.cat([u4, l3], dim=1))

        # Bottom-up: n3 -> down -> concat -> n4 -> n5
        d3 = self.down3(n3)
        n4 = self.bu_block4(torch.cat([d3, m4], dim=1))
        d4 = self.down4(n4)
        n5 = self.bu_block5(torch.cat([d4, l5], dim=1))

        return [n3, n4, n5]


if __name__ == "__main__":
    neck = PANFPN(in_channels=[128, 256, 512], out_channels=256)
    feats = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    outs = neck(feats)
    for i, o in enumerate(outs):
        print(f"N{i+3}: {tuple(o.shape)}")
    n_params = sum(p.numel() for p in neck.parameters() if p.requires_grad)
    print(f"PANFPN param: {n_params/1e6:.2f}M")
