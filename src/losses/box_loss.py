"""Bounding box kayıpları: CIoU + DFL.

CIoU = IoU - (ρ²/c²) - α·v
  ρ² = merkez mesafe karesi
  c² = en küçük çevreleyen kutu köşegen karesi
  v  = en-boy oranı tutarlılığı
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(b1: torch.Tensor, b2: torch.Tensor, ciou: bool = True,
             eps: float = 1e-7) -> torch.Tensor:
    """CIoU veya IoU hesapla.

    Args:
        b1, b2: (..., 4) [cx, cy, w, h] formatında
        ciou: True -> CIoU, False -> IoU

    Returns:
        (...,) skoru
    """
    # cxcywh -> xyxy
    b1_x1 = b1[..., 0] - b1[..., 2] / 2
    b1_y1 = b1[..., 1] - b1[..., 3] / 2
    b1_x2 = b1[..., 0] + b1[..., 2] / 2
    b1_y2 = b1[..., 1] + b1[..., 3] / 2
    b2_x1 = b2[..., 0] - b2[..., 2] / 2
    b2_y1 = b2[..., 1] - b2[..., 3] / 2
    b2_x2 = b2[..., 0] + b2[..., 2] / 2
    b2_y2 = b2[..., 1] + b2[..., 3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter + eps
    iou = inter / union

    if not ciou:
        return iou

    # En küçük çevreleyen kutu
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Merkez mesafe
    rho2 = ((b2[..., 0] - b1[..., 0]) ** 2 + (b2[..., 1] - b1[..., 1]) ** 2)

    # En-boy uyumu
    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(b2[..., 2] / (b2[..., 3] + eps)) -
        torch.atan(b1[..., 2] / (b1[..., 3] + eps)),
        2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - (rho2 / c2 + alpha * v)


class CIoULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 1.0 - bbox_iou(pred, target, ciou=True)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DFLLoss(nn.Module):
    """Distribution Focal Loss — regresyon dağılımının iki komşu binine cross-entropy."""

    def __init__(self, reg_max: int = 16, reduction: str = "mean"):
        super().__init__()
        self.reg_max = reg_max
        self.reduction = reduction

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            pred_dist: (N, reg_max+1) — softmax öncesi logit
            target:    (N,) — sürekli hedef değeri (0 ile reg_max arası)
        """
        target = target.clamp(0, self.reg_max - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr.float() - target
        wr = 1 - wl

        loss_l = F.cross_entropy(pred_dist, tl, reduction='none') * wl
        loss_r = F.cross_entropy(pred_dist, tr, reduction='none') * wr
        loss = loss_l + loss_r

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
