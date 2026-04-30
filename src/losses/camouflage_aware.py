"""CamouflageAware Loss — projenin özgün loss formülasyonu.

Üç bileşen:
  1) DynamicFocalLoss: zorluğa göre uyarlanan focal loss
     L = -α (1 - p_t)^γ_dynamic log(p_t)
     γ_dynamic = γ_base * (1 + β · D(x))
     D(x): örnek-bazlı kamuflaj zorluk skoru (modelin belirsizliği)

  2) BoundaryAwareLoss: kutu sınır piksellerine ek ağırlık
     L = mean over boundary pixels of L1(grad(GT) - grad(P))

  3) CrossModalConsistencyLoss: tek-modalite head'leri ile ana head arasında
     KL divergence:
     L = KL(P_fused || stop_grad(P_opt)) + KL(P_fused || stop_grad(P_sar))

Toplam loss = w_focal · L_focal + w_bound · L_bound + w_consist · L_consist
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CALConfig:
    use_focal: bool = True
    focal_gamma_base: float = 2.0
    focal_beta: float = 1.5
    focal_alpha: float = 0.25
    focal_lambda: float = 1.0

    use_boundary: bool = True
    boundary_lambda: float = 0.3
    boundary_band_pixels: int = 3

    use_consistency: bool = True
    consistency_lambda: float = 0.2
    consistency_warmup_epochs: int = 5


class DynamicFocalLoss(nn.Module):
    """Dynamic Focal Loss — zorluk-uyarlı γ ile."""

    def __init__(self, gamma_base: float = 2.0, beta: float = 1.5,
                 alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma_base = gamma_base
        self.beta = beta
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                difficulty: torch.Tensor = None) -> torch.Tensor:
        """Args:
            logits:     (..., C) ya da (...,) — sigmoid öncesi
            targets:    (..., C) ya da (...,) — 0/1
            difficulty: (..., 1) — örnek başına zorluk skoru [0, 1]; None ise 0
        """
        if logits.dim() == targets.dim() + 1 and logits.size(-1) > 1:
            # multi-class one-hot bekleniyor; targets ise sınıf indeksi olabilir
            targets = F.one_hot(targets.long(), logits.size(-1)).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # Dinamik gamma
        if difficulty is None:
            gamma = self.gamma_base
        else:
            # difficulty broadcast olacak şekilde shape uyarlaması
            while difficulty.dim() < logits.dim():
                difficulty = difficulty.unsqueeze(-1)
            gamma = self.gamma_base * (1.0 + self.beta * difficulty)

        focal_term = (1.0 - p_t) ** gamma
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BoundaryAwareLoss(nn.Module):
    """Kutu sınır piksellerinde gradyan tutarlılığı.

    GT mask'tan ve tahmin yoğunluk haritasından sınır şeridi alınır;
    sınır içinde Sobel gradient mismatch hesaplanır.
    """

    def __init__(self, band_pixels: int = 3, reduction: str = "mean"):
        super().__init__()
        self.band = band_pixels
        # Sobel kernel'leri (kayıt bufferi)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("sobel_x", kx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", ky.view(1, 1, 3, 3))
        self.reduction = reduction

    def _sobel_grad(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> (B, 2, H, W) gradient (gx, gy)."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.cat([gx, gy], dim=1)

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """Args:
            pred_mask, gt_mask: (B, 1, H, W) float [0, 1]
        """
        # Sınır şeridi: GT mask'in dilation - erosion farkı (kalın bant)
        kernel = torch.ones(1, 1, self.band * 2 + 1, self.band * 2 + 1,
                             device=gt_mask.device, dtype=gt_mask.dtype)
        dil = F.conv2d(gt_mask, kernel, padding=self.band).clamp(0, 1)
        ero = 1 - F.conv2d(1 - gt_mask, kernel, padding=self.band).clamp(0, 1)
        band = (dil - ero).clamp(0, 1)

        gp = self._sobel_grad(pred_mask)
        gg = self._sobel_grad(gt_mask)

        # Sadece sınır bandında L1 hatası
        diff = (gp - gg).abs().sum(dim=1, keepdim=True)
        loss = (diff * band).sum() / (band.sum() + 1e-6)
        return loss


class ConsistencyLoss(nn.Module):
    """Cross-modal consistency: ana çıktı, tek-modal yardımcılarla tutarlı olsun."""

    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean"):
        super().__init__()
        self.t = temperature
        self.reduction = reduction

    def forward(self, main_logits: torch.Tensor,
                aux_opt_logits: torch.Tensor,
                aux_sar_logits: torch.Tensor) -> torch.Tensor:
        """Hepsi (..., C) ya da flat tensör olabilir; KL hesaplanmadan önce
        son boyutta softmax alınır.
        """
        # Aux'lar gradient akmaz
        with torch.no_grad():
            p_opt = F.softmax(aux_opt_logits / self.t, dim=-1)
            p_sar = F.softmax(aux_sar_logits / self.t, dim=-1)

        log_p_main = F.log_softmax(main_logits / self.t, dim=-1)
        kl_o = F.kl_div(log_p_main, p_opt, reduction=self.reduction)
        kl_s = F.kl_div(log_p_main, p_sar, reduction=self.reduction)
        return 0.5 * (kl_o + kl_s) * (self.t ** 2)


class CamouflageAwareLoss(nn.Module):
    """Üç bileşen birleşik."""

    def __init__(self, cfg: CALConfig = None, num_classes: int = 6):
        super().__init__()
        self.cfg = cfg or CALConfig()
        self.num_classes = num_classes

        self.focal = DynamicFocalLoss(
            gamma_base=self.cfg.focal_gamma_base,
            beta=self.cfg.focal_beta,
            alpha=self.cfg.focal_alpha,
        ) if self.cfg.use_focal else None

        self.boundary = BoundaryAwareLoss(
            band_pixels=self.cfg.boundary_band_pixels,
        ) if self.cfg.use_boundary else None

        self.consistency = ConsistencyLoss() if self.cfg.use_consistency else None

    def forward(self, *,
                cls_logits: torch.Tensor,
                cls_targets: torch.Tensor,
                difficulty: torch.Tensor = None,
                pred_mask: torch.Tensor = None, gt_mask: torch.Tensor = None,
                aux_opt_logits: torch.Tensor = None,
                aux_sar_logits: torch.Tensor = None,
                epoch: int = 0) -> dict:
        """Tüm bileşenleri toplar.

        Yalnız geçerli (None olmayan) bileşenler hesaplanır.
        """
        out = {}

        if self.focal is not None:
            l_focal = self.focal(cls_logits, cls_targets, difficulty=difficulty)
            out["focal"] = l_focal * self.cfg.focal_lambda

        if self.boundary is not None and pred_mask is not None and gt_mask is not None:
            l_bound = self.boundary(pred_mask, gt_mask)
            out["boundary"] = l_bound * self.cfg.boundary_lambda

        if self.consistency is not None and \
                aux_opt_logits is not None and aux_sar_logits is not None:
            warmup = max(1, self.cfg.consistency_warmup_epochs)
            scale = min(1.0, epoch / warmup)
            l_consist = self.consistency(cls_logits, aux_opt_logits, aux_sar_logits)
            out["consistency"] = l_consist * self.cfg.consistency_lambda * scale

        out["total"] = sum(out.values()) if out else torch.tensor(0.0)
        return out


if __name__ == "__main__":
    cfg = CALConfig()
    cal = CamouflageAwareLoss(cfg, num_classes=6)

    cls_logits = torch.randn(4, 100, 6)  # batch, anchors, classes
    cls_targets = torch.zeros(4, 100, 6)
    cls_targets[0, 5, 2] = 1
    cls_targets[1, 10, 4] = 1
    difficulty = torch.rand(4, 100, 1)

    aux_opt = torch.randn(4, 100, 6)
    aux_sar = torch.randn(4, 100, 6)

    pred_mask = torch.rand(2, 1, 64, 64)
    gt_mask = torch.zeros(2, 1, 64, 64)
    gt_mask[:, :, 20:40, 20:40] = 1

    out = cal(
        cls_logits=cls_logits,
        cls_targets=cls_targets,
        difficulty=difficulty,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        aux_opt_logits=aux_opt,
        aux_sar_logits=aux_sar,
        epoch=10,
    )
    for k, v in out.items():
        print(f"  {k}: {v.item():.4f}")
