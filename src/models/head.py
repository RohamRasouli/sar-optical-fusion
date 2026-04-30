"""YOLOv8 Decoupled Detection Head.

Anchor-free, decoupled (sınıflandırma + regresyon ayrı dallar).
DFL (Distribution Focal Loss) için reg_max+1 değerli regresyon dağılımı çıkarır.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .encoder import ConvBNAct


def make_anchors(feats: List[torch.Tensor], strides: List[int],
                 grid_offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Her özellik haritasının her hücresi için (x, y) merkez noktası ve stride.

    Returns:
        anchor_points: (N, 2) tüm seviyelerden birleşik
        stride_tensor: (N, 1) her noktanın stride'ı
    """
    anchor_points, stride_tensor = [], []
    for f, s in zip(feats, strides):
        _, _, h, w = f.shape
        sx = torch.arange(w, dtype=torch.float32, device=f.device) + grid_offset
        sy = torch.arange(h, dtype=torch.float32, device=f.device) + grid_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack([sx, sy], dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), s, dtype=torch.float32, device=f.device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor,
              xywh: bool = True) -> torch.Tensor:
    """Anchor merkezinden l, t, r, b mesafelerini bbox'a çevir."""
    lt, rb = distance.chunk(2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], -1)
    return torch.cat([x1y1, x2y2], -1)


class DFL(nn.Module):
    """Distribution Focal Loss bileşeni.

    Reg_max+1 boyutlu olasılık dağılımını ortalama mesafeye çevirir.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        # Sabit projeksiyon: [0, 1, ..., reg_max]
        self.proj = nn.Parameter(
            torch.arange(reg_max + 1, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4*(reg_max+1), N) -> (B, 4, reg_max+1, N) -> softmax -> ağırlıklı toplam
        b, _, n = x.shape
        x = x.view(b, 4, self.reg_max + 1, n).softmax(dim=2)
        return (x * self.proj.view(1, 1, -1, 1)).sum(dim=2)


class DetectionHead(nn.Module):
    """Decoupled detection head — her ölçekte ayrı cls/reg dalları."""

    def __init__(self, num_classes: int = 6, in_channels: List[int] = (256, 256, 256),
                 reg_max: int = 16):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.no = num_classes + 4 * (reg_max + 1)
        self.strides = [8, 16, 32]

        c2 = max(16, in_channels[0] // 4, 4 * (reg_max + 1))
        c3 = max(in_channels[0], num_classes)

        self.cv2 = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, c2, k=3),
                ConvBNAct(c2, c2, k=3),
                nn.Conv2d(c2, 4 * (reg_max + 1), kernel_size=1),
            ) for c in in_channels
        ])
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, c3, k=3),
                ConvBNAct(c3, c3, k=3),
                nn.Conv2d(c3, num_classes, kernel_size=1),
            ) for c in in_channels
        ])

        self.dfl = DFL(reg_max=reg_max)

    def forward(self, feats: List[torch.Tensor]):
        """Eğitimde: ham çıktıları döndür (loss tarafı işler).
        Inference'ta: bbox + sınıf skorları döndür.
        """
        outs = []
        for i, f in enumerate(feats):
            box = self.cv2[i](f)   # (B, 4*(reg_max+1), H, W)
            cls = self.cv3[i](f)   # (B, num_classes, H, W)
            outs.append(torch.cat([box, cls], dim=1))

        if self.training:
            return outs

        # Inference: anchor üret, dist'i çöz
        return self._inference(feats, outs)

    def _inference(self, feats: List[torch.Tensor],
                   outs: List[torch.Tensor]) -> torch.Tensor:
        anchor_points, stride_tensor = make_anchors(feats, self.strides)
        # Her ölçeği flatten et ve birleştir
        x_cat = torch.cat([o.view(o.size(0), self.no, -1) for o in outs], dim=2)
        box, cls = x_cat.split([4 * (self.reg_max + 1), self.nc], dim=1)
        box = self.dfl(box)                  # (B, 4, N)
        bbox_xywh = dist2bbox(box.transpose(1, 2), anchor_points, xywh=True) * stride_tensor
        scores = cls.sigmoid()
        return torch.cat([bbox_xywh, scores.transpose(1, 2)], dim=2)  # (B, N, 4+nc)


if __name__ == "__main__":
    head = DetectionHead(num_classes=6, in_channels=[256, 256, 256])
    feats = [
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]
    head.train()
    train_outs = head(feats)
    for i, o in enumerate(train_outs):
        print(f"  train scale {i}: {tuple(o.shape)}")
    head.eval()
    with torch.no_grad():
        infer = head(feats)
    print(f"  inference: {tuple(infer.shape)}  (B, N, 4+nc)")
    print(f"  param: {sum(p.numel() for p in head.parameters() if p.requires_grad)/1e6:.2f}M")
