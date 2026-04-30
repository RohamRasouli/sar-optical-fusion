"""Tam pipeline sanity test.

Dummy verilerle tüm boru hattının uçtan uca çalıştığını doğrular:
  - Encoder forward
  - CMAFM forward
  - Neck + Head forward
  - Loss hesabı
  - Backward + optimizer step

Bu test PyTorch'un yüklü olduğu herhangi bir ortamda çalışır.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Proje root'unu Python path'e ekle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model(device):
    import torch
    from src.models.full_model import ModelConfig, build_model

    cfg = ModelConfig(
        num_classes=6,
        optical_channels=3,
        sar_channels=2,
        encoder_base_channels=16,    # Test için küçük
        encoder_depth_mult=0.33,
        encoder_width_mult=0.25,
        feature_channels=(64, 128, 256),
        cmafm_num_heads=(2, 2, 4),
        cmafm_window_size=4,
        neck_out_channels=128,
        head_reg_max=8,
    )
    model = build_model(cfg).to(device)
    return model, cfg


# ============================================================
# UNIT TESTS
# ============================================================

def test_lee_filter_output_shape():
    import torch
    from src.datasets.augmentation.lee_filter import lee_filter

    sar = torch.randn(2, 2, 64, 64)
    out = lee_filter(sar, window_size=7)
    assert out.shape == sar.shape


def test_learnable_lee_filter_grad():
    import torch
    from src.datasets.augmentation.lee_filter import LearnableLeeFilter

    f = LearnableLeeFilter(in_channels=2)
    x = torch.randn(1, 2, 32, 32, requires_grad=True)
    y = f(x)
    y.sum().backward()
    # Filter parametreleri grad almalı
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in f.parameters())
    assert has_grad


def test_camo_synth_changes_image():
    import numpy as np
    import torch
    from src.datasets.augmentation.camo_synth import (
        CamoSynthAugmenter,
        CamoSynthConfig,
    )

    cfg = CamoSynthConfig(probability=1.0, per_box_probability=1.0)
    aug = CamoSynthAugmenter(cfg)
    img = torch.rand(3, 128, 128)
    boxes = torch.tensor([[0, 0.3, 0.4, 0.2, 0.2]])
    out = aug(img, boxes, rng=np.random.default_rng(42))
    diff = (out - img).abs().mean().item()
    assert diff > 1e-3, "Augmentation değişiklik yapmamış"


def test_dummy_dataset_collate():
    import torch
    from torch.utils.data import DataLoader

    from src.datasets.m4_sar import DummyM4SARDataset, collate_fn

    ds = DummyM4SARDataset(num_samples=8, img_size=64)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["optical"].shape == (4, 3, 64, 64)
    assert batch["sar"].shape == (4, 2, 64, 64)
    # labels: (M, 6) [batch_idx, cls, cx, cy, w, h]
    assert batch["labels"].dim() == 2
    assert batch["labels"].size(1) == 6


def test_encoder_forward(device):
    import torch
    from src.models.encoder import DualStreamEncoder

    enc = DualStreamEncoder(optical_channels=3, sar_channels=2,
                              base_channels=16, width_mult=0.25,
                              out_channels=(64, 128, 256)).to(device)
    opt = torch.randn(1, 3, 128, 128, device=device)
    sar = torch.randn(1, 2, 128, 128, device=device)
    opt_f, sar_f = enc(opt, sar)
    assert len(opt_f) == 3 and len(sar_f) == 3
    # 1/8, 1/16, 1/32 ölçek
    expected = [(64, 16, 16), (128, 8, 8), (256, 4, 4)]
    for f, exp in zip(opt_f, expected):
        assert tuple(f.shape[1:]) == exp


def test_cmafm_forward(device):
    import torch
    from src.models.cmafm import MultiScaleCMAFM

    cmafm = MultiScaleCMAFM(channels_list=[64, 128, 256],
                              num_heads_list=[2, 2, 4],
                              window_size=4).to(device)
    feats_o = [
        torch.randn(2, 64, 16, 16, device=device),
        torch.randn(2, 128, 8, 8, device=device),
        torch.randn(2, 256, 4, 4, device=device),
    ]
    feats_s = [f.clone() for f in feats_o]
    fused, gates = cmafm(feats_o, feats_s)
    assert len(fused) == 3
    for f, fo in zip(fused, feats_o):
        assert f.shape == fo.shape
    # Gate'ler 0-1 arasında
    for sg_o, sg_s in gates:
        assert sg_o.min() >= 0 and sg_o.max() <= 1
        assert sg_s.min() >= 0 and sg_s.max() <= 1


def test_full_model_forward_shape(small_model, device):
    import torch

    model, cfg = small_model
    model.eval()
    opt = torch.randn(1, 3, 64, 64, device=device)
    sar = torch.randn(1, 2, 64, 64, device=device)
    with torch.no_grad():
        out = model(opt, sar)
    # Inference modunda main: (B, N, 4+nc)
    assert "main" in out
    assert out["main"].dim() == 3
    assert out["main"].size(2) == 4 + cfg.num_classes


def test_full_model_train_forward(small_model, device):
    import torch

    model, cfg = small_model
    model.train()
    opt = torch.randn(2, 3, 64, 64, device=device)
    sar = torch.randn(2, 2, 64, 64, device=device)
    out = model(opt, sar)
    # Train modunda main: liste (3 ölçek)
    assert isinstance(out["main"], list)
    assert len(out["main"]) == 3
    if cfg.aux_heads:
        assert "aux_opt" in out and "aux_sar" in out


def test_loss_backward(small_model, device):
    import torch
    from src.losses.camouflage_aware import CALConfig
    from src.losses.detection_loss import DetectionLoss

    model, cfg = small_model
    model.train()
    cal_cfg = CALConfig()  # default tüm açık
    loss_fn = DetectionLoss(num_classes=cfg.num_classes,
                              reg_max=cfg.head_reg_max,
                              cal_cfg=cal_cfg, img_size=64)

    opt = torch.randn(2, 3, 64, 64, device=device)
    sar = torch.randn(2, 2, 64, 64, device=device)

    # Sentetik etiket: 1 hedef her batchde
    targets = torch.tensor([
        [0, 1, 0.5, 0.5, 0.2, 0.2],  # batch 0, class 1
        [1, 3, 0.3, 0.7, 0.1, 0.1],  # batch 1, class 3
    ], dtype=torch.float32, device=device)

    out = model(opt, sar)
    loss_dict = loss_fn(out, targets, epoch=0)
    assert "total" in loss_dict
    assert torch.is_tensor(loss_dict["total"])
    assert loss_dict["total"].requires_grad

    loss_dict["total"].backward()

    # Gradient'lerin akıp akmadığını kontrol et
    has_grad = sum(1 for p in model.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0)
    assert has_grad > 10, f"Çok az parametrede grad var: {has_grad}"


def test_training_step(small_model, device):
    """Tam bir eğitim adımı: forward + loss + backward + optimizer."""
    import torch
    from src.losses.camouflage_aware import CALConfig
    from src.losses.detection_loss import DetectionLoss

    model, cfg = small_model
    model.train()
    loss_fn = DetectionLoss(num_classes=cfg.num_classes,
                              reg_max=cfg.head_reg_max,
                              cal_cfg=CALConfig(), img_size=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    opt = torch.randn(2, 3, 64, 64, device=device)
    sar = torch.randn(2, 2, 64, 64, device=device)
    targets = torch.tensor([
        [0, 1, 0.5, 0.5, 0.2, 0.2],
        [1, 3, 0.3, 0.7, 0.1, 0.1],
    ], dtype=torch.float32, device=device)

    losses_before = []
    for step in range(3):
        out = model(opt, sar)
        loss_dict = loss_fn(out, targets, epoch=0)
        optimizer.zero_grad()
        loss_dict["total"].backward()
        optimizer.step()
        losses_before.append(loss_dict["total"].item())

    # 3 adımda loss sonsuz değilmeli
    for v in losses_before:
        assert v == v, "NaN!"
        assert abs(v) < 1e6, f"Loss patladı: {v}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
