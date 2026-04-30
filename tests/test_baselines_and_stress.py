"""Baseline modeller ve stres augmentations için ek testler."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_baseline_cfg():
    from src.models.baselines import BaselineConfig
    return BaselineConfig(
        num_classes=6, optical_channels=3, sar_channels=2,
        base_channels=16, depth_mult=0.33, width_mult=0.25,
        feature_channels=(64, 128, 256), neck_out_channels=128,
        head_reg_max=8,
    )


# ============================================================
# Baseline modelleri
# ============================================================

@pytest.mark.parametrize("name", ["optical_only", "sar_only", "concat", "single_attn"])
def test_baseline_forward(name, small_baseline_cfg, device):
    import torch
    from src.models.baselines import build_baseline

    model = build_baseline(name, small_baseline_cfg).to(device)
    model.eval()
    opt = torch.randn(1, 3, 64, 64, device=device)
    sar = torch.randn(1, 2, 64, 64, device=device)
    with torch.no_grad():
        out = model(opt, sar)
    assert "main" in out
    # Eval modunda main: (B, N, 4+nc) ya da liste
    main = out["main"]
    if isinstance(main, list):
        assert len(main) == 3
    else:
        assert main.dim() == 3


def test_late_fusion_train_outputs(small_baseline_cfg, device):
    import torch
    from src.models.baselines import build_baseline

    model = build_baseline("late", small_baseline_cfg).to(device)
    model.train()
    opt = torch.randn(2, 3, 64, 64, device=device)
    sar = torch.randn(2, 2, 64, 64, device=device)
    out = model(opt, sar)
    # Train modunda iki ayrı head çıktısı
    assert "main" in out
    assert "main_sar" in out


# ============================================================
# Stres augmentations
# ============================================================

def test_cloud_overlay_changes_optical_only():
    import torch
    from src.datasets.augmentation.stress import add_cloud_overlay

    opt = torch.rand(3, 64, 64) * 0.5 + 0.25
    sar = torch.rand(2, 64, 64)
    o2, s2 = add_cloud_overlay(opt, sar, coverage=0.7, seed=42)
    assert o2.shape == opt.shape
    assert s2.shape == sar.shape
    assert (o2 - opt).abs().mean().item() > 0.05, "Bulut etki çok zayıf"
    assert (s2 - sar).abs().mean().item() < 1e-6, "SAR değişmemeli"


def test_low_light_darkens():
    import torch
    from src.datasets.augmentation.stress import simulate_low_light

    opt = torch.ones(3, 64, 64) * 0.7  # parlak
    sar = torch.zeros(2, 64, 64)
    o2, _ = simulate_low_light(opt, sar, brightness=0.2, noise_std=0.0,
                                blue_shift=0.0, seed=42)
    assert o2.mean() < opt.mean(), "Görüntü kararmalı"


@pytest.mark.parametrize("preset", [
    "clean", "cloud_medium", "night_dark", "camo_only", "all_combined"
])
def test_stress_presets(preset):
    import torch
    from src.datasets.augmentation.stress import PRESET_STRESS, apply_stress

    opt = torch.rand(3, 128, 128)
    sar = torch.rand(2, 128, 128)
    labels = torch.tensor([[0, 0.5, 0.5, 0.2, 0.2]])
    cfg = PRESET_STRESS[preset]
    o, s = apply_stress(opt, sar, labels, cfg, seed=0)
    assert o.shape == opt.shape
    assert s.shape == sar.shape


# ============================================================
# Visualization (matplotlib opsiyonel)
# ============================================================

def test_draw_predictions_runs():
    import torch
    from src.utils.visualization import HAS_MPL, draw_predictions

    if not HAS_MPL:
        pytest.skip("matplotlib yüklü değil")

    img = torch.rand(3, 128, 128)
    detections = [
        {"class": 0, "score": 0.8, "bbox": [10, 20, 60, 80]},
        {"class": 1, "score": 0.6, "bbox": [70, 80, 110, 120]},
    ]
    fig = draw_predictions(img, detections, score_threshold=0.5)
    assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
