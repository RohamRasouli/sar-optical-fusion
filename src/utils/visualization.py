"""Görselleştirme yardımcıları — tez figürleri ve demo için.

Üç ana fonksiyon:
  1) draw_predictions: bbox + sınıf etiketi ile tahminleri görüntüye ekler
  2) plot_attention_maps: CMAFM gating haritalarını overlay olarak göster
  3) side_by_side: optik + SAR + füzyon yan yana karşılaştırma görseli
  4) plot_loss_curves: eğitim sırasında loss eğrileri
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib opsiyonel — yoksa basit numpy fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Renkler — sınıf başına renk paleti
DEFAULT_COLORS = [
    (1.0, 0.3, 0.3),   # kırmızı
    (0.3, 0.9, 0.3),   # yeşil
    (0.3, 0.5, 1.0),   # mavi
    (1.0, 0.8, 0.2),   # sarı
    (0.9, 0.4, 0.9),   # pembe
    (0.4, 0.9, 0.9),   # turkuaz
    (1.0, 0.6, 0.2),   # turuncu
    (0.7, 0.5, 0.9),   # mor
]


def _to_numpy_image(img) -> np.ndarray:
    """Tensor veya numpy'i (H, W, 3) [0, 1] float numpy'a çevirir."""
    import torch
    if torch.is_tensor(img):
        if img.dim() == 4:
            img = img[0]
        if img.size(0) <= 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] == 2:
        # SAR 2-kanal -> 3-kanal pseudo-RGB
        img = np.concatenate([img, img.mean(-1, keepdims=True)], axis=-1)
    return np.clip(img, 0, 1)


def draw_predictions(image, detections: List[dict],
                      class_names: Optional[List[str]] = None,
                      score_threshold: float = 0.25,
                      ax=None, save_path: Optional[str] = None):
    """Görüntüye bbox ve sınıf etiketi çiz.

    Args:
        image: tensor veya numpy (H, W, 3) [0, 1]
        detections: [{'class': int, 'score': float, 'bbox': [x1,y1,x2,y2]}, ...]
        class_names: opsiyonel sınıf isimleri
        score_threshold: bu skorun altındakileri çizme
    """
    if not HAS_MPL:
        print("matplotlib yok; görselleştirme atlanıyor")
        return None

    img = _to_numpy_image(image)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = ax.figure

    ax.imshow(img)
    ax.axis("off")

    for d in detections:
        if d["score"] < score_threshold:
            continue
        x1, y1, x2, y2 = d["bbox"]
        cls = d["class"]
        color = DEFAULT_COLORS[cls % len(DEFAULT_COLORS)]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.0, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)
        label = class_names[cls] if class_names else f"cls{cls}"
        text = f"{label} {d['score']:.2f}"
        ax.text(x1, max(y1 - 4, 0), text,
                fontsize=9, color='white',
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor='none'))

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_attention_maps(optical, sar, gates: List[Tuple],
                         save_path: Optional[str] = None):
    """CMAFM gating haritalarını her ölçek için overlay olarak göster.

    Args:
        optical: (3, H, W) tensör
        sar: (2 veya 3, H, W) tensör
        gates: [(sigma_opt, sigma_sar), ...] her ölçek için
    """
    if not HAS_MPL:
        return None

    n_scales = len(gates)
    fig, axes = plt.subplots(n_scales + 1, 4, figsize=(16, 4 * (n_scales + 1)))
    if n_scales == 0:
        return None

    # Üst satır: optik, SAR, başlıklar
    axes[0, 0].imshow(_to_numpy_image(optical))
    axes[0, 0].set_title("Optik")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(_to_numpy_image(sar), cmap='gray')
    axes[0, 1].set_title("SAR")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    for i, (sg_o, sg_s) in enumerate(gates):
        # sg_o: (B, C, H, W) — kanal ortalaması alınmış skor (B=1)
        if sg_o.dim() == 4:
            sg_o = sg_o[0]
            sg_s = sg_s[0]
        m_o = sg_o.mean(0).cpu().numpy()
        m_s = sg_s.mean(0).cpu().numpy()
        diff = m_o - m_s

        # Optik üzerine optik-gate overlay
        axes[i + 1, 0].imshow(_to_numpy_image(optical))
        axes[i + 1, 0].set_title(f"σ_opt scale {i+1}")
        axes[i + 1, 0].axis("off")

        # SAR üzerine sar-gate overlay
        axes[i + 1, 1].imshow(_to_numpy_image(sar), cmap='gray')
        axes[i + 1, 1].set_title(f"σ_sar scale {i+1}")
        axes[i + 1, 1].axis("off")

        # σ_opt heatmap
        im2 = axes[i + 1, 2].imshow(m_o, cmap='hot', vmin=0, vmax=1)
        axes[i + 1, 2].set_title(f"σ_opt heatmap")
        axes[i + 1, 2].axis("off")
        plt.colorbar(im2, ax=axes[i + 1, 2], fraction=0.046)

        # Modalite tercihi (opt - sar)
        im3 = axes[i + 1, 3].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i + 1, 3].set_title("σ_opt - σ_sar")
        axes[i + 1, 3].axis("off")
        plt.colorbar(im3, ax=axes[i + 1, 3], fraction=0.046)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def side_by_side(optical, sar, prediction_image=None,
                  titles: Tuple[str, str, str] = ("Optik", "SAR", "Füzyon"),
                  save_path: Optional[str] = None):
    """Üç panelli karşılaştırma görseli — tez için."""
    if not HAS_MPL:
        return None

    n = 2 if prediction_image is None else 3
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    axes[0].imshow(_to_numpy_image(optical))
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(_to_numpy_image(sar), cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    if prediction_image is not None:
        axes[2].imshow(_to_numpy_image(prediction_image))
        axes[2].set_title(titles[2])
        axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_loss_curves(history: Dict[str, List[float]],
                      save_path: Optional[str] = None,
                      title: str = "Eğitim Eğrileri"):
    """Loss eğrilerini çiz."""
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                            save_path: Optional[str] = None):
    """Sınıflandırma karışıklık matrisi heatmap."""
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    plt.colorbar(im, ax=ax)
    # Hücre değerleri
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def make_thesis_figure_grid(samples: List[Dict], save_path: str,
                             ncols: int = 4, class_names: Optional[List[str]] = None):
    """Tez için 'başarılı tespit örnekleri' grid figürü.

    samples: [{'image': tensor, 'detections': [...], 'caption': str}, ...]
    """
    if not HAS_MPL:
        return
    n = len(samples)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if n > 1 else [axes]
    for i, sample in enumerate(samples):
        ax = axes[i]
        draw_predictions(sample["image"], sample["detections"],
                          class_names=class_names, ax=ax)
        if "caption" in sample:
            ax.set_title(sample["caption"], fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
