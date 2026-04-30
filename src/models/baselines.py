"""Multimodal Baseline Modeller — Ablation Karşılaştırması İçin.

Üç baseline:
  1. SingleModalModel — sadece optik veya sadece SAR (literatürdeki standart)
  2. ConcatFusionModel — early fusion (kanal boyutunda concat)
  3. LateFusionModel — iki bağımsız model, skor ortalaması
  4. SimpleAttentionFusionModel — tek ölçekte basit cross-attention (CMAFM yok)

Bu baseline'lar bizim CMAFM'in ne kadar değer kattığını ölçmek için kullanılır.
Tezde bu karşılaştırma kritik — sadece YOLOv8 ile değil.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CSPDarknetBackbone, DualStreamEncoder
from .head import DetectionHead
from .neck import PANFPN


@dataclass
class BaselineConfig:
    num_classes: int = 6
    optical_channels: int = 3
    sar_channels: int = 2
    base_channels: int = 32
    depth_mult: float = 0.33
    width_mult: float = 0.5
    feature_channels: tuple = (128, 256, 512)
    neck_out_channels: int = 256
    head_reg_max: int = 16


# ============================================================
# 1) Tek modaliteli model (sadece optik veya sadece SAR)
# ============================================================

class SingleModalModel(nn.Module):
    """Tek modaliteli YOLOv8 stili model — optik VEYA SAR."""

    def __init__(self, cfg: BaselineConfig = None, modality: str = "optical"):
        super().__init__()
        self.cfg = cfg or BaselineConfig()
        assert modality in {"optical", "sar"}
        self.modality = modality

        in_ch = self.cfg.optical_channels if modality == "optical" else self.cfg.sar_channels
        self.encoder = CSPDarknetBackbone(
            in_channels=in_ch,
            base_channels=self.cfg.base_channels,
            depth_mult=self.cfg.depth_mult,
            width_mult=self.cfg.width_mult,
            out_channels=self.cfg.feature_channels,
        )
        self.neck = PANFPN(
            in_channels=list(self.cfg.feature_channels),
            out_channels=self.cfg.neck_out_channels,
        )
        self.head = DetectionHead(
            num_classes=self.cfg.num_classes,
            in_channels=[self.cfg.neck_out_channels] * 3,
            reg_max=self.cfg.head_reg_max,
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor):
        x = optical if self.modality == "optical" else sar
        feats = self.encoder(x)
        feats = self.neck(feats)
        return {"main": self.head(feats), "gates": []}


# ============================================================
# 2) Concat (Early) Fusion
# ============================================================

class ConcatFusionModel(nn.Module):
    """Optik + SAR'ı kanal boyutunda concat eden basit baseline.

    Tek encoder; giriş kanalları (optical_channels + sar_channels) toplam.
    Multimodal'ın en basit versiyonu — bizim CMAFM'in ne kadar daha iyi
    olduğunu göstermek için kritik baseline.
    """

    def __init__(self, cfg: BaselineConfig = None):
        super().__init__()
        self.cfg = cfg or BaselineConfig()
        in_ch = self.cfg.optical_channels + self.cfg.sar_channels

        self.encoder = CSPDarknetBackbone(
            in_channels=in_ch,
            base_channels=self.cfg.base_channels,
            depth_mult=self.cfg.depth_mult,
            width_mult=self.cfg.width_mult,
            out_channels=self.cfg.feature_channels,
        )
        self.neck = PANFPN(
            in_channels=list(self.cfg.feature_channels),
            out_channels=self.cfg.neck_out_channels,
        )
        self.head = DetectionHead(
            num_classes=self.cfg.num_classes,
            in_channels=[self.cfg.neck_out_channels] * 3,
            reg_max=self.cfg.head_reg_max,
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor):
        x = torch.cat([optical, sar], dim=1)
        feats = self.encoder(x)
        feats = self.neck(feats)
        return {"main": self.head(feats), "gates": []}


# ============================================================
# 3) Late Fusion (skor ortalaması)
# ============================================================

class LateFusionModel(nn.Module):
    """İki bağımsız encoder + head. Çıktı seviyesinde skor ortalaması.

    Eğitim: iki head ayrı ayrı eğitilir.
    Inference: skor (sigmoid) ortalaması, box ortalaması.
    """

    def __init__(self, cfg: BaselineConfig = None):
        super().__init__()
        self.cfg = cfg or BaselineConfig()

        # Optic branch
        self.encoder_opt = CSPDarknetBackbone(
            in_channels=self.cfg.optical_channels,
            base_channels=self.cfg.base_channels,
            depth_mult=self.cfg.depth_mult,
            width_mult=self.cfg.width_mult,
            out_channels=self.cfg.feature_channels,
        )
        self.neck_opt = PANFPN(
            in_channels=list(self.cfg.feature_channels),
            out_channels=self.cfg.neck_out_channels,
        )
        self.head_opt = DetectionHead(
            num_classes=self.cfg.num_classes,
            in_channels=[self.cfg.neck_out_channels] * 3,
            reg_max=self.cfg.head_reg_max,
        )

        # SAR branch
        self.encoder_sar = CSPDarknetBackbone(
            in_channels=self.cfg.sar_channels,
            base_channels=self.cfg.base_channels,
            depth_mult=self.cfg.depth_mult,
            width_mult=self.cfg.width_mult,
            out_channels=self.cfg.feature_channels,
        )
        self.neck_sar = PANFPN(
            in_channels=list(self.cfg.feature_channels),
            out_channels=self.cfg.neck_out_channels,
        )
        self.head_sar = DetectionHead(
            num_classes=self.cfg.num_classes,
            in_channels=[self.cfg.neck_out_channels] * 3,
            reg_max=self.cfg.head_reg_max,
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor):
        # Optic
        f_opt = self.encoder_opt(optical)
        f_opt = self.neck_opt(f_opt)
        h_opt = self.head_opt(f_opt)

        # SAR
        f_sar = self.encoder_sar(sar)
        f_sar = self.neck_sar(f_sar)
        h_sar = self.head_sar(f_sar)

        if self.training:
            # Eğitimde iki head bağımsız — loss tarafı her ikisini de hesaplar
            return {"main": h_opt, "main_sar": h_sar, "gates": []}

        # Inference: skor ortalaması
        # h_opt ve h_sar (B, N, 4+nc) — box concat, score average
        avg_box = (h_opt[..., :4] + h_sar[..., :4]) / 2
        avg_score = (h_opt[..., 4:] + h_sar[..., 4:]) / 2
        return {"main": torch.cat([avg_box, avg_score], dim=-1), "gates": []}


# ============================================================
# 4) Tek-ölçekli Cross-Attention (CMAFM'siz, sadece 1/16'da)
# ============================================================

class SimpleCrossAttentionFusionModel(nn.Module):
    """Sadece tek ölçekte (1/16) basit cross-attention.

    CMAFM'in çok-ölçekli olmasının ve gating'in faydasını göstermek için
    bir ablation baseline.
    """

    def __init__(self, cfg: BaselineConfig = None,
                 attn_scale_idx: int = 1, num_heads: int = 4):
        super().__init__()
        self.cfg = cfg or BaselineConfig()
        self.attn_scale_idx = attn_scale_idx

        self.encoder = DualStreamEncoder(
            optical_channels=self.cfg.optical_channels,
            sar_channels=self.cfg.sar_channels,
            base_channels=self.cfg.base_channels,
            depth_mult=self.cfg.depth_mult,
            width_mult=self.cfg.width_mult,
            out_channels=self.cfg.feature_channels,
        )

        # Sadece bir ölçekte basit cross-attention
        c = self.cfg.feature_channels[attn_scale_idx]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=c, num_heads=num_heads, batch_first=True,
        )
        self.proj_out = nn.Conv2d(c, c, kernel_size=1)

        self.neck = PANFPN(
            in_channels=list(self.cfg.feature_channels),
            out_channels=self.cfg.neck_out_channels,
        )
        self.head = DetectionHead(
            num_classes=self.cfg.num_classes,
            in_channels=[self.cfg.neck_out_channels] * 3,
            reg_max=self.cfg.head_reg_max,
        )

    def _fuse(self, opt_feats: List[torch.Tensor], sar_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Sadece self.attn_scale_idx'inci ölçekte fusion uygula, diğerleri concat."""
        fused = []
        for i, (fo, fs) in enumerate(zip(opt_feats, sar_feats)):
            if i == self.attn_scale_idx:
                # Cross-attention: opt -> sar
                B, C, H, W = fo.shape
                q = fo.flatten(2).transpose(1, 2)
                k = fs.flatten(2).transpose(1, 2)
                v = k
                attn_out, _ = self.cross_attn(q, k, v)
                attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
                f = self.proj_out(attn_out) + fo + fs  # residual + sum
            else:
                f = (fo + fs) / 2  # diğer ölçeklerde basit ortalama
            fused.append(f)
        return fused

    def forward(self, optical: torch.Tensor, sar: torch.Tensor):
        opt_feats, sar_feats = self.encoder(optical, sar)
        fused = self._fuse(opt_feats, sar_feats)
        feats = self.neck(fused)
        return {"main": self.head(feats), "gates": []}


# ============================================================
# Fabrika
# ============================================================

def build_baseline(name: str, cfg: BaselineConfig = None) -> nn.Module:
    """Baseline modeli isimle oluştur.

    Args:
        name: 'optical_only', 'sar_only', 'concat', 'late', 'single_attn'
    """
    cfg = cfg or BaselineConfig()
    if name == "optical_only":
        return SingleModalModel(cfg, modality="optical")
    if name == "sar_only":
        return SingleModalModel(cfg, modality="sar")
    if name == "concat":
        return ConcatFusionModel(cfg)
    if name == "late":
        return LateFusionModel(cfg)
    if name == "single_attn":
        return SimpleCrossAttentionFusionModel(cfg)
    raise ValueError(f"Bilinmeyen baseline: {name}")


if __name__ == "__main__":
    import torch
    cfg = BaselineConfig()
    for name in ["optical_only", "sar_only", "concat", "late", "single_attn"]:
        m = build_baseline(name, cfg)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:20s} {n_params/1e6:6.2f} M params")
