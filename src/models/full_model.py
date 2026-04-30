"""Tam Model Assembly.

DualStreamEncoder + MultiScaleCMAFM + PANFPN + DetectionHead
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn

from .cmafm import MultiScaleCMAFM
from .encoder import DualStreamEncoder
from .head import DetectionHead
from .neck import PANFPN


@dataclass
class ModelConfig:
    num_classes: int = 6
    optical_channels: int = 3
    sar_channels: int = 2
    encoder_base_channels: int = 32
    encoder_depth_mult: float = 0.33
    encoder_width_mult: float = 0.5
    feature_channels: Tuple[int, int, int] = (128, 256, 512)
    cmafm_num_heads: Tuple[int, int, int] = (4, 4, 8)
    cmafm_window_size: int = 8
    cmafm_attn_drop: float = 0.1
    cmafm_drop_path: float = 0.1
    neck_out_channels: int = 256
    head_reg_max: int = 16
    aux_heads: bool = True   # tek modal yardımcı head'ler (consistency loss için)


class SAROpticalFusionModel(nn.Module):
    """Birleşik multimodal hedef tespit modeli."""

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        # 1) Çift akımlı encoder
        self.encoder = DualStreamEncoder(
            optical_channels=c.optical_channels,
            sar_channels=c.sar_channels,
            base_channels=c.encoder_base_channels,
            depth_mult=c.encoder_depth_mult,
            width_mult=c.encoder_width_mult,
            out_channels=c.feature_channels,
        )

        # 2) CMAFM (3 ölçek)
        self.cmafm = MultiScaleCMAFM(
            channels_list=list(c.feature_channels),
            num_heads_list=list(c.cmafm_num_heads),
            window_size=c.cmafm_window_size,
            attn_drop=c.cmafm_attn_drop,
            drop_path=c.cmafm_drop_path,
        )

        # 3) Neck (birleştirilmiş özelliklere)
        self.neck = PANFPN(
            in_channels=list(c.feature_channels),
            out_channels=c.neck_out_channels,
        )

        # 4) Ana detection head
        self.head = DetectionHead(
            num_classes=c.num_classes,
            in_channels=[c.neck_out_channels] * 3,
            reg_max=c.head_reg_max,
        )

        # 5) Auxiliary head'ler — sadece consistency loss için
        if c.aux_heads:
            self.aux_neck_opt = PANFPN(
                in_channels=list(c.feature_channels),
                out_channels=c.neck_out_channels,
            )
            self.aux_neck_sar = PANFPN(
                in_channels=list(c.feature_channels),
                out_channels=c.neck_out_channels,
            )
            self.aux_head_opt = DetectionHead(
                num_classes=c.num_classes,
                in_channels=[c.neck_out_channels] * 3,
                reg_max=c.head_reg_max,
            )
            self.aux_head_sar = DetectionHead(
                num_classes=c.num_classes,
                in_channels=[c.neck_out_channels] * 3,
                reg_max=c.head_reg_max,
            )
        else:
            self.aux_neck_opt = self.aux_neck_sar = None
            self.aux_head_opt = self.aux_head_sar = None

        self._gating_cache = None  # son gates (görselleştirme için)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor):
        """Forward.

        Returns dict with keys:
            'main':  ana head çıktısı (eğitimde liste, eval'de (B,N,4+nc) tensör)
            'aux_opt', 'aux_sar': yardımcı head çıktıları (varsa)
            'gates': CMAFM gating bilgisi (her ölçek için (sigma_opt, sigma_sar))
        """
        opt_feats, sar_feats = self.encoder(optical, sar)
        fused_feats, gates = self.cmafm(opt_feats, sar_feats)
        self._gating_cache = gates

        neck_feats = self.neck(fused_feats)
        main_out = self.head(neck_feats)

        result = {"main": main_out, "gates": gates}

        if self.aux_neck_opt is not None and self.training:
            with torch.set_grad_enabled(True):
                aux_opt_neck = self.aux_neck_opt(opt_feats)
                aux_sar_neck = self.aux_neck_sar(sar_feats)
                result["aux_opt"] = self.aux_head_opt(aux_opt_neck)
                result["aux_sar"] = self.aux_head_sar(aux_sar_neck)
        return result


def build_model(cfg: ModelConfig = None) -> SAROpticalFusionModel:
    """Yardımcı fabrika fonksiyonu."""
    return SAROpticalFusionModel(cfg)


def count_parameters(model: nn.Module) -> dict:
    out = {}
    total = 0
    for name, module in [
        ("encoder.optical", getattr(model.encoder, "optical_encoder", None)),
        ("encoder.sar", getattr(model.encoder, "sar_encoder", None)),
        ("cmafm", model.cmafm),
        ("neck", model.neck),
        ("head", model.head),
    ]:
        if module is None:
            continue
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        out[name] = n
        total += n
    if model.aux_head_opt is not None:
        n_aux = sum(p.numel() for p in model.aux_head_opt.parameters() if p.requires_grad)
        n_aux += sum(p.numel() for p in model.aux_head_sar.parameters() if p.requires_grad)
        n_aux += sum(p.numel() for p in model.aux_neck_opt.parameters() if p.requires_grad)
        n_aux += sum(p.numel() for p in model.aux_neck_sar.parameters() if p.requires_grad)
        out["aux"] = n_aux
        total += n_aux
    out["TOTAL"] = total
    return out


if __name__ == "__main__":
    cfg = ModelConfig()
    model = build_model(cfg)
    counts = count_parameters(model)
    print("Parametre sayıları:")
    for name, n in counts.items():
        print(f"  {name:20s} {n/1e6:6.2f} M")

    # Test forward
    opt = torch.randn(2, 3, 256, 256)
    sar = torch.randn(2, 2, 256, 256)
    model.train()
    out = model(opt, sar)
    print(f"\nTrain mod çıktısı: keys = {list(out.keys())}")
    print(f"  main: {len(out['main'])} ölçek")
    for i, m in enumerate(out["main"]):
        print(f"    scale {i}: {tuple(m.shape)}")
    if "aux_opt" in out:
        print(f"  aux_opt: {len(out['aux_opt'])} ölçek")
