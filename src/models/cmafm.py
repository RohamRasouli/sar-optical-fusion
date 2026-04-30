"""Cross-Modal Attention Fusion Module (CMAFM).

Projenin asıl özgün katkısı. Üç ölçekte (1/8, 1/16, 1/32) optik ve SAR
özellikleri arasında pencere-tabanlı iki yönlü çapraz dikkat ile bilgi alışverişi
sağlar; sigmoid kapı ile koşula bağlı modalite ağırlığı öğretilir.

Matematiksel formülasyon:
    F_o2s = MultiHeadCrossAttention(Q=F_opt, K=F_sar, V=F_sar)   # opt sorar
    F_s2o = MultiHeadCrossAttention(Q=F_sar, K=F_opt, V=F_opt)   # sar sorar

    sigma_opt = sigmoid(Conv1x1(Concat(F_opt, F_o2s)))
    sigma_sar = sigmoid(Conv1x1(Concat(F_sar, F_s2o)))

    F_opt' = F_opt + sigma_opt * F_o2s
    F_sar' = F_sar + sigma_sar * F_s2o

    F_fused = Conv1x1(Concat(F_opt', F_sar'))
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, C, H, W) -> (B * nW, C, ws, ws), nW = (H/ws) * (W/ws)."""
    B, C, H, W = x.shape
    assert H % ws == 0 and W % ws == 0, f"H,W ({H},{W}) ws ({ws})'a bölünebilmeli"
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nh, nw, C, ws, ws)
    x = x.view(-1, C, ws, ws)
    return x


def window_reverse(x: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """Pencere bölmenin tersi: (B*nW, C, ws, ws) -> (B, C, H, W)."""
    nW = (H // ws) * (W // ws)
    B = x.size(0) // nW
    C = x.size(1)
    x = x.view(B, H // ws, W // ws, C, ws, ws)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x


class WindowCrossAttention(nn.Module):
    """Pencere içi multi-head cross-attention.

    Girdi: q, k, v hepsi (B*nW, C, ws, ws) şeklinde.
    Çıktı: (B*nW, C, ws, ws).
    """

    def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.1,
                 proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
        # (B*nW, C, ws, ws) -> (B*nW, ws*ws, C)
        BnW, C, ws, _ = q.shape
        N = ws * ws
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        # Multi-head: (B*nW, N, C) -> (B*nW, num_heads, N, head_dim)
        q = q.view(BnW, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(BnW, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(BnW, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Skor: (B*nW, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Çıkış
        out = attn @ v  # (B*nW, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(BnW, N, C)
        out = self.proj_drop(self.proj(out))

        # (B*nW, ws*ws, C) -> (B*nW, C, ws, ws)
        return out.transpose(1, 2).view(BnW, C, ws, ws)


class CMAFMBlock(nn.Module):
    """Tek bir CMAFM bloğu — bir ölçekte uygulanır."""

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8,
                 attn_drop: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        self.channels = channels
        self.ws = window_size

        # Q, K, V projeksiyonları (modalite-paylaşımlı)
        self.q_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.q_sar = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_sar = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_sar = nn.Conv2d(channels, channels, kernel_size=1)

        # Cross-attention
        self.attn_o2s = WindowCrossAttention(channels, num_heads=num_heads,
                                               attn_drop=attn_drop)
        self.attn_s2o = WindowCrossAttention(channels, num_heads=num_heads,
                                               attn_drop=attn_drop)

        # Gating
        self.gate_opt = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_sar = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Final füzyon
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # Drop path için (basitleştirilmiş)
        self.drop_path = drop_path

    def _maybe_pad(self, x: torch.Tensor) -> tuple:
        """H, W pencereye bölünebilecek şekilde pad et."""
        _, _, H, W = x.shape
        pad_h = (self.ws - H % self.ws) % self.ws
        pad_w = (self.ws - W % self.ws) % self.ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, H, W, pad_h, pad_w

    def forward(self, F_opt: torch.Tensor, F_sar: torch.Tensor
                ) -> tuple:
        """Returns (F_fused, sigma_opt, sigma_sar) — gating'ler görselleştirme için."""
        assert F_opt.shape == F_sar.shape
        B, C, H, W = F_opt.shape

        # Q/K/V projeksiyonları
        Q_opt = self.q_opt(F_opt)
        K_opt = self.k_opt(F_opt)
        V_opt = self.v_opt(F_opt)
        Q_sar = self.q_sar(F_sar)
        K_sar = self.k_sar(F_sar)
        V_sar = self.v_sar(F_sar)

        # Pencereye böl (gerekirse pad)
        Q_opt_p, H_orig, W_orig, ph, pw = self._maybe_pad(Q_opt)
        K_sar_p, _, _, _, _ = self._maybe_pad(K_sar)
        V_sar_p, _, _, _, _ = self._maybe_pad(V_sar)
        Q_sar_p, _, _, _, _ = self._maybe_pad(Q_sar)
        K_opt_p, _, _, _, _ = self._maybe_pad(K_opt)
        V_opt_p, _, _, _, _ = self._maybe_pad(V_opt)
        Hp, Wp = Q_opt_p.shape[-2:]

        Q_opt_w = window_partition(Q_opt_p, self.ws)
        K_sar_w = window_partition(K_sar_p, self.ws)
        V_sar_w = window_partition(V_sar_p, self.ws)
        Q_sar_w = window_partition(Q_sar_p, self.ws)
        K_opt_w = window_partition(K_opt_p, self.ws)
        V_opt_w = window_partition(V_opt_p, self.ws)

        # İki yönlü çapraz dikkat
        F_o2s_w = self.attn_o2s(Q_opt_w, K_sar_w, V_sar_w)  # opt -> sar bilgisi
        F_s2o_w = self.attn_s2o(Q_sar_w, K_opt_w, V_opt_w)

        # Pencere ters çevirme
        F_o2s = window_reverse(F_o2s_w, self.ws, Hp, Wp)
        F_s2o = window_reverse(F_s2o_w, self.ws, Hp, Wp)

        # Pad'i kaldır
        if ph or pw:
            F_o2s = F_o2s[:, :, :H_orig, :W_orig]
            F_s2o = F_s2o[:, :, :H_orig, :W_orig]

        # Gating
        sigma_opt = self.gate_opt(torch.cat([F_opt, F_o2s], dim=1))
        sigma_sar = self.gate_sar(torch.cat([F_sar, F_s2o], dim=1))

        # Drop path (eğitim sırasında stochastic)
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask_opt = (torch.rand(B, 1, 1, 1, device=F_opt.device) < keep).float() / keep
            mask_sar = (torch.rand(B, 1, 1, 1, device=F_opt.device) < keep).float() / keep
            sigma_opt = sigma_opt * mask_opt
            sigma_sar = sigma_sar * mask_sar

        F_opt_new = F_opt + sigma_opt * F_o2s
        F_sar_new = F_sar + sigma_sar * F_s2o

        F_fused = self.fuse(torch.cat([F_opt_new, F_sar_new], dim=1))

        return F_fused, sigma_opt, sigma_sar


class MultiScaleCMAFM(nn.Module):
    """Üç ölçekli CMAFM (1/8, 1/16, 1/32)."""

    def __init__(self, channels_list: List[int],
                 num_heads_list: List[int],
                 window_size: int = 8,
                 attn_drop: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        assert len(channels_list) == len(num_heads_list)
        self.blocks = nn.ModuleList([
            CMAFMBlock(c, h, window_size=window_size,
                       attn_drop=attn_drop, drop_path=drop_path)
            for c, h in zip(channels_list, num_heads_list)
        ])
        self.channels_list = channels_list

    def forward(self, opt_feats: List[torch.Tensor],
                sar_feats: List[torch.Tensor]
                ) -> tuple:
        """Returns (fused_feats, gating_info_list)."""
        assert len(opt_feats) == len(sar_feats) == len(self.blocks)
        fused = []
        gates = []
        for block, fo, fs in zip(self.blocks, opt_feats, sar_feats):
            f, sg_o, sg_s = block(fo, fs)
            fused.append(f)
            gates.append((sg_o.detach(), sg_s.detach()))
        return fused, gates


if __name__ == "__main__":
    cmafm = MultiScaleCMAFM(channels_list=[128, 256, 512],
                              num_heads_list=[4, 4, 8],
                              window_size=8)
    n_params = sum(p.numel() for p in cmafm.parameters() if p.requires_grad)
    print(f"MultiScaleCMAFM param: {n_params/1e6:.2f}M")

    opt_feats = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    sar_feats = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    fused, gates = cmafm(opt_feats, sar_feats)
    for i, f in enumerate(fused):
        print(f"  fused scale {i}: {tuple(f.shape)}")
    for i, (sg_o, sg_s) in enumerate(gates):
        print(f"  gate scale {i}: opt mean={sg_o.mean():.3f}, sar mean={sg_s.mean():.3f}")
