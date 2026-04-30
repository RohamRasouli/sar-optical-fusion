#!/usr/bin/env python3
"""Gerçekçi sentetik demo veri seti oluşturucu.

Tam M4-SAR (19.4 GB) disk alanı gerektirdiğinde, bu script gerçekçi
görünümlü sentetik veri oluşturur. Amacı:
  - Pipeline'ın çalıştığını doğrulamak
  - Eğitim kodunun hata vermeden çalıştığını test etmek
  - Demo ve sunumlar için görselleştirme materyali üretmek

Oluşturulan veri seti, M4-SAR ile aynı klasör yapısına sahiptir.

Kullanım:
    python scripts/generate_demo_dataset.py --n-train 500 --n-val 100 --n-test 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# Sınıf bilgileri
# ============================================================

CLASSES = ["aircraft", "ship", "vehicle", "bridge", "storage", "oil_tank"]


def _random_color(rng: np.random.Generator) -> tuple:
    """Rastgele ama makul uydu görüntüsü pikseli rengi."""
    palette_type = rng.choice(["urban", "vegetation", "water", "desert"])
    if palette_type == "urban":
        v = rng.integers(80, 200)
        return (v + rng.integers(-20, 20), v + rng.integers(-20, 20), v + rng.integers(-30, 10))
    elif palette_type == "vegetation":
        return (rng.integers(30, 80), rng.integers(80, 160), rng.integers(30, 80))
    elif palette_type == "water":
        return (rng.integers(20, 60), rng.integers(40, 100), rng.integers(80, 180))
    else:
        return (rng.integers(150, 220), rng.integers(130, 200), rng.integers(90, 160))


def _draw_target_optical(draw: ImageDraw.Draw, cls: int, cx: int, cy: int,
                          w: int, h: int, rng: np.random.Generator):
    """Sınıfa göre basit hedef şekli çiz (optik)."""
    x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2

    if cls == 0:  # aircraft - çapraz
        color = (200 + rng.integers(0, 55), 200 + rng.integers(0, 55), 200 + rng.integers(0, 55))
        draw.rectangle([x1, y1, x2, y2], fill=color)
        draw.line([cx, y1, cx, y2], fill=(180, 180, 180), width=max(1, w // 6))
        draw.line([x1, cy, x2, cy], fill=(180, 180, 180), width=max(1, h // 8))
    elif cls == 1:  # ship - dikdörtgen
        color = (rng.integers(100, 160), rng.integers(100, 160), rng.integers(100, 180))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(80, 80, 80))
    elif cls == 2:  # vehicle - küçük kare
        color = (rng.integers(120, 200), rng.integers(80, 160), rng.integers(80, 140))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    elif cls == 3:  # bridge - yatay çizgi
        color = (rng.integers(140, 200), rng.integers(140, 200), rng.integers(120, 180))
        draw.rectangle([x1, cy - h // 6, x2, cy + h // 6], fill=color)
        draw.rectangle([x1, y1, x1 + w // 8, y2], fill=color)
        draw.rectangle([x2 - w // 8, y1, x2, y2], fill=color)
    elif cls == 4:  # storage - daire
        color = (rng.integers(160, 220), rng.integers(160, 220), rng.integers(160, 220))
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(120, 120, 120))
    elif cls == 5:  # oil_tank - daire + koyu
        color = (rng.integers(60, 120), rng.integers(60, 120), rng.integers(60, 120))
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(40, 40, 40))


def generate_pair(img_size: int, rng: np.random.Generator,
                   n_targets_range: tuple = (1, 8)):
    """Sentetik optik + SAR çifti ve etiketler oluştur.

    Returns:
        optical: PIL Image (RGB)
        sar_vv: numpy array (H, W) float32
        sar_vh: numpy array (H, W) float32
        labels: list of [cls, cx_norm, cy_norm, w_norm, h_norm]
    """
    # 1) Arka plan oluştur
    bg_color = _random_color(rng)
    optical = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(optical)

    # Doku ekle (basit gürültü)
    bg_np = np.array(optical).astype(np.float32)
    noise = rng.standard_normal(bg_np.shape).astype(np.float32) * 8
    bg_np = np.clip(bg_np + noise, 0, 255).astype(np.uint8)
    optical = Image.fromarray(bg_np)
    draw = ImageDraw.Draw(optical)

    # 2) Hedefler ekle
    n_targets = int(rng.integers(n_targets_range[0], n_targets_range[1] + 1))
    labels = []

    for _ in range(n_targets):
        cls = int(rng.integers(0, len(CLASSES)))

        # Boyut sınıfa göre
        if cls in [0, 3]:  # aircraft, bridge — uzun
            w = int(rng.integers(40, 100))
            h = int(rng.integers(15, 40))
        elif cls in [4, 5]:  # storage, oil_tank — kare
            s = int(rng.integers(20, 60))
            w, h = s, s
        elif cls == 2:  # vehicle — küçük
            w = int(rng.integers(10, 25))
            h = int(rng.integers(10, 25))
        else:  # ship
            w = int(rng.integers(20, 80))
            h = int(rng.integers(8, 30))

        # Konum (sınırda taşmayı önle)
        margin = max(w, h) // 2 + 5
        cx = int(rng.integers(margin, img_size - margin))
        cy = int(rng.integers(margin, img_size - margin))

        _draw_target_optical(draw, cls, cx, cy, w, h, rng)

        # YOLO normalize
        labels.append([
            cls,
            cx / img_size,
            cy / img_size,
            w / img_size,
            h / img_size,
        ])

    # 3) SAR oluştur (optikten türetilmiş + speckle)
    opt_gray = np.array(optical.convert("L")).astype(np.float32) / 255.0
    # VV: gri değer + speckle
    speckle_vv = rng.gamma(shape=5, scale=0.2, size=opt_gray.shape).astype(np.float32)
    sar_vv = opt_gray * speckle_vv
    sar_vv = np.clip(sar_vv, 0, 1)
    # VH: VV'den türetilmiş, daha düşük
    speckle_vh = rng.gamma(shape=3, scale=0.3, size=opt_gray.shape).astype(np.float32)
    sar_vh = opt_gray * 0.6 * speckle_vh
    sar_vh = np.clip(sar_vh, 0, 1)

    return optical, sar_vv, sar_vh, labels


def save_split(target_dir: Path, split: str, n_samples: int,
               img_size: int, seed: int):
    """Bir split için tüm dosyaları oluştur ve kaydet."""
    opt_dir = target_dir / "optical" / split
    sar_dir = target_dir / "sar" / split
    lbl_dir = target_dir / "labels" / split

    opt_dir.mkdir(parents=True, exist_ok=True)
    sar_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for i in range(n_samples):
        img_id = f"img_{i:05d}"

        optical, sar_vv, sar_vh, labels = generate_pair(img_size, rng)

        # Kaydet
        optical.save(opt_dir / f"{img_id}.png")
        sar_2ch = np.stack([sar_vv, sar_vh], axis=0)  # (2, H, W)
        np.save(sar_dir / f"{img_id}.npy", sar_2ch.astype(np.float32))

        # Etiket
        with open(lbl_dir / f"{img_id}.txt", "w") as f:
            for lbl in labels:
                f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        if (i + 1) % 100 == 0:
            print(f"   {split}: {i + 1}/{n_samples}")

    print(f"   ✅ {split}: {n_samples} çift oluşturuldu")
    return n_samples


def main():
    parser = argparse.ArgumentParser(
        description="Sentetik demo veri seti oluştur"
    )
    parser.add_argument("--target-dir", type=str, default="./data/m4_sar",
                        help="Çıktı dizini")
    parser.add_argument("--img-size", type=int, default=512,
                        help="Görüntü boyutu (kare)")
    parser.add_argument("--n-train", type=int, default=500,
                        help="Eğitim seti boyutu")
    parser.add_argument("--n-val", type=int, default=100,
                        help="Validasyon seti boyutu")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Test seti boyutu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    target_dir = Path(args.target_dir)

    print("=" * 60)
    print("  Sentetik Demo Veri Seti Oluşturucu")
    print("=" * 60)
    print(f"  Hedef: {target_dir}")
    print(f"  Boyut: {args.img_size}×{args.img_size}")
    print(f"  Train: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print()

    stats = {}
    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        stats[split] = save_split(target_dir, split, n, args.img_size,
                                   seed=args.seed + hash(split) % 1000)

    # Dataset info
    info = f"""# Demo Veri Seti (Sentetik)
- Train: {stats['train']} çift
- Val: {stats['val']} çift
- Test: {stats['test']} çift
- Boyut: {args.img_size}×{args.img_size}
- Sınıflar: {', '.join(CLASSES)}
- NOT: Bu sentetik veridir, pipeline doğrulama amaçlıdır.
- Gerçek sonuçlar için M4-SAR veri setini kullanın.
"""
    (target_dir / "dataset_info.md").write_text(info, encoding="utf-8")

    total = sum(stats.values())
    print(f"\n{'=' * 60}")
    print(f"  ✅ Toplam {total} çift oluşturuldu!")
    print(f"  Konum: {target_dir.resolve()}")
    print(f"{'=' * 60}")
    print(f"\nSonraki adım:")
    print(f"  python -m src.train --config configs/multimodal_full.yaml")


if __name__ == "__main__":
    main()
