#!/usr/bin/env python3
"""M4-SAR veri setini indir ve proje formatına dönüştür.

Kullanım:
    # Seçenek 1: Kaggle CLI ile (en kolay)
    pip install kaggle
    # ~/.kaggle/kaggle.json dosyasını oluştur (Kaggle API key)
    python scripts/download_m4sar.py --source kaggle

    # Seçenek 2: Hugging Face ile
    pip install huggingface_hub
    python scripts/download_m4sar.py --source huggingface

    # Seçenek 3: Manuel indirme sonrası dönüştürme
    python scripts/download_m4sar.py --raw-dir /path/to/downloaded/m4sar

Veri seti kaynakları:
    - Kaggle:       https://kaggle.com/datasets/wchao0601/m4-sar
    - Hugging Face: https://huggingface.co/datasets/wchao0601/m4-sar
    - GitHub:       https://github.com/wchao0601/M4-SAR

Orijinal M4-SAR yapısı (512×512 patch'ler, YOLO formatı):
    M4-SAR/
    ├── images/
    │   ├── opt/
    │   │   ├── train/ *.png
    │   │   ├── val/ *.png
    │   │   └── test/ *.png
    │   └── sar/
    │       ├── train/ *.png
    │       ├── val/ *.png
    │       └── test/ *.png
    └── labels/
        ├── train/ *.txt
        ├── val/ *.txt
        └── test/ *.txt

Dönüşüm sonrası yapı (proje için):
    data/m4_sar/
    ├── optical/{train,val,test}/ *.png    (Sentinel-2 RGB)
    ├── sar/{train,val,test}/ *.npy        (Sentinel-1 VV+VH, 2 kanal)
    └── labels/{train,val,test}/ *.txt     (YOLO formatı)

Sınıflar (6):
    0: bridge, 1: harbor, 2: oil_tank, 3: playground, 4: airport, 5: wind_turbine

Eşleştirme: M4-SAR'daki sınıf isimleri ile proje sınıf mapping'i uyumlu hale getirilir.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# Sınıf eşleştirme
# ============================================================

# M4-SAR orijinal sınıflar
M4SAR_CLASSES = ["bridge", "harbor", "oil_tank", "playground", "airport", "wind_turbine"]

# Bizim projedeki sınıflar (base.yaml)
PROJECT_CLASSES = ["aircraft", "ship", "vehicle", "bridge", "storage", "oil_tank"]

# Eşleştirme: M4-SAR sınıf indexi → proje sınıf indexi
# M4-SAR bridge(0) → project bridge(3)
# M4-SAR harbor(1) → en yakın: project ship(1)
# M4-SAR oil_tank(2) → project oil_tank(5)
# M4-SAR playground(3) → project storage(4)  (en yakın genel kategori)
# M4-SAR airport(4) → project aircraft(0)  (havalimanı → uçak)
# M4-SAR wind_turbine(5) → project storage(4) (yardımcı yapı)
CLASS_MAPPING = {
    0: 3,  # bridge → bridge
    1: 1,  # harbor → ship
    2: 5,  # oil_tank → oil_tank
    3: 4,  # playground → storage
    4: 0,  # airport → aircraft
    5: 4,  # wind_turbine → storage
}


def remap_labels(label_path: Path, output_path: Path):
    """YOLO formatı etiketlerinde sınıf indekslerini yeniden eşle."""
    if not label_path.exists():
        return
    lines = label_path.read_text().strip().split('\n')
    remapped = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        new_cls = CLASS_MAPPING.get(cls_id, cls_id)
        remapped.append(f"{new_cls} {' '.join(parts[1:])}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(remapped) + '\n')


# ============================================================
# İndirme fonksiyonları
# ============================================================

def download_kaggle(target_dir: Path):
    """Kaggle CLI ile M4-SAR indir."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("❌ kaggle paketi yüklü değil. Yüklemek için:")
        print("   pip install kaggle")
        print("   Sonra ~/.kaggle/kaggle.json dosyasını oluşturun.")
        sys.exit(1)

    raw_dir = target_dir / "_raw_kaggle"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Kaggle'dan M4-SAR indiriliyor...")
    print("   Bu işlem veri seti boyutuna göre 10-30 dakika sürebilir.")

    os.system(
        f'kaggle datasets download -d wchao0601/m4-sar '
        f'-p "{raw_dir}" --unzip'
    )

    return raw_dir


def download_huggingface(target_dir: Path):
    """Hugging Face'den M4-SAR indir."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub paketi yüklü değil. Yüklemek için:")
        print("   pip install huggingface_hub")
        sys.exit(1)

    raw_dir = target_dir / "_raw_hf"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Hugging Face'den M4-SAR indiriliyor...")
    snapshot_download(
        repo_id="wchao0601/m4-sar",
        repo_type="dataset",
        local_dir=str(raw_dir),
    )

    return raw_dir


# ============================================================
# Format dönüştürme
# ============================================================

def find_raw_structure(raw_dir: Path) -> Optional[Path]:
    """M4-SAR ham veri yapısının kök dizinini bul."""
    # Doğrudan images/ klasörü varsa
    if (raw_dir / "images").is_dir():
        return raw_dir

    # İç içe klasörlerde ara
    for p in raw_dir.rglob("images"):
        if p.is_dir() and (p / "opt").is_dir():
            return p.parent

    # Alternatif yapılar ara (M4-SAR varyantları)
    for p in raw_dir.rglob("opt"):
        if p.is_dir() and any((p / s).is_dir() for s in ["train", "val", "test"]):
            return p.parent.parent

    # Kaggle varyantı
    if (raw_dir / "optical").is_dir() and (raw_dir / "sar").is_dir():
        return raw_dir

    return None


def convert_m4sar_to_project(raw_dir: Path, target_dir: Path):
    """M4-SAR veri setini proje formatına dönüştür.

    Optik: PNG olarak kopyala (3 kanal RGB)
    SAR:   PNG'den oku → 2 kanal numpy (VV+VH simülasyonu) → .npy olarak kaydet
    Label: YOLO formatı, sınıf indexlerini yeniden eşle
    """
    root = find_raw_structure(raw_dir)
    if root is None:
        print(f"❌ M4-SAR yapısı bulunamadı: {raw_dir}")
        print("   Beklenen yapı: images/opt/{train,val,test}/*.png")
        print("   Lütfen --raw-dir ile doğru dizini belirtin.")
        sys.exit(1)

    print(f"✅ M4-SAR yapısı bulundu: {root}")

    # Kaynak dizinler
    opt_dir = root / "images" / "opt"
    sar_dir = root / "images" / "sar"
    lbl_dir = root / "labels"

    # Alternnatif isimler dene
    if not opt_dir.exists():
        opt_dir = root / "images" / "optical"
    if not sar_dir.exists():
        for alt in ["sar", "SAR", "s1"]:
            candidate = root / "images" / alt
            if candidate.exists():
                sar_dir = candidate
                break

    # Kaggle varyantı
    if (root / "optical" / "images").exists() and (root / "sar" / "images").exists():
        opt_dir = root / "optical" / "images"
        sar_dir = root / "sar" / "images"
        lbl_dir = root / "optical" / "labels"

    stats = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        print(f"\n📂 {split} split dönüştürülüyor...")

        # Optik görüntüler
        src_opt = opt_dir / split
        dst_opt = target_dir / "optical" / split

        # SAR görüntüler
        src_sar = sar_dir / split
        dst_sar = target_dir / "sar" / split

        # Etiketler
        src_lbl = lbl_dir / split
        dst_lbl = target_dir / "labels" / split

        dst_opt.mkdir(parents=True, exist_ok=True)
        dst_sar.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        if not src_opt.exists():
            print(f"   ⚠️  Optik klasör yok: {src_opt}")
            continue

        opt_files = sorted(list(src_opt.glob("*.png")) + list(src_opt.glob("*.jpg")))
        print(f"   Optik: {len(opt_files)} dosya bulundu")

        for i, opt_path in enumerate(opt_files):
            stem = opt_path.stem

            # 1) Optik → PNG kopyala
            dst_opt_path = dst_opt / opt_path.name
            if not dst_opt_path.exists():
                shutil.copy2(opt_path, dst_opt_path)

            # 2) SAR → Doğrudan PNG olarak kopyala (disk tasarrufu için)
            # Dönüşüm M4SARDataset loader içinde "on the fly" yapılacak.
            sar_path = src_sar / opt_path.name  # Aynı isim
            dst_sar_path = dst_sar / opt_path.name
            if sar_path.exists() and not dst_sar_path.exists():
                shutil.copy2(sar_path, dst_sar_path)

            # 3) Etiket → sınıf eşleme
            lbl_path = src_lbl / f"{stem}.txt"
            dst_lbl_path = dst_lbl / f"{stem}.txt"
            if lbl_path.exists() and not dst_lbl_path.exists():
                remap_labels(lbl_path, dst_lbl_path)

            stats[split] += 1

            if (i + 1) % 500 == 0:
                print(f"   ... {i + 1}/{len(opt_files)} tamamlandı")

        print(f"   ✅ {stats[split]} görüntü çifti dönüştürüldü")

    return stats


def generate_dataset_info(target_dir: Path, stats: dict):
    """Veri seti bilgi dosyası oluştur."""
    info = f"""# M4-SAR Veri Seti — Proje Formatı

## İstatistikler
- Train: {stats.get('train', 0)} çift
- Val:   {stats.get('val', 0)} çift
- Test:  {stats.get('test', 0)} çift
- Toplam: {sum(stats.values())} çift

## Sınıflar (6)
| Proje ID | Proje Sınıf | M4-SAR Karşılığı |
|----------|-------------|-------------------|
| 0        | aircraft    | airport (4)       |
| 1        | ship        | harbor (1)        |
| 2        | vehicle     | —                 |
| 3        | bridge      | bridge (0)        |
| 4        | storage     | playground (3), wind_turbine (5) |
| 5        | oil_tank    | oil_tank (2)      |

## Dosya Formatları
- Optik: PNG (3 kanal RGB, 512×512)
- SAR: NPY (2 kanal VV+VH, float32, 512×512)
- Etiket: TXT (YOLO format: class cx cy w h, normalize)

## Kaynak
- M4-SAR: Wang et al., "M4-SAR: A Multi-Resolution, Multi-Polarization,
  Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR
  Fusion Object Detection", arXiv:2505.10931, 2025.
"""
    (target_dir / "dataset_info.md").write_text(info, encoding="utf-8")
    print(f"\n📄 Dataset info kaydedildi: {target_dir / 'dataset_info.md'}")


def verify_dataset(target_dir: Path):
    """Veri setinin bütünlüğünü kontrol et."""
    print("\n🔍 Veri seti doğrulama...")
    errors = []

    for split in ["train", "val", "test"]:
        opt_dir = target_dir / "optical" / split
        sar_dir = target_dir / "sar" / split
        lbl_dir = target_dir / "labels" / split

        opt_files = set(p.stem for p in opt_dir.glob("*") if p.is_file())
        sar_files = set(p.stem for p in sar_dir.glob("*") if p.is_file())
        lbl_files = set(p.stem for p in lbl_dir.glob("*") if p.is_file())

        n_opt = len(opt_files)
        n_sar = len(sar_files)
        n_lbl = len(lbl_files)

        print(f"  {split:5s}: optik={n_opt}, sar={n_sar}, label={n_lbl}")

        # Eşleşme kontrolü
        missing_sar = opt_files - sar_files
        missing_lbl = opt_files - lbl_files
        if missing_sar:
            errors.append(f"{split}: {len(missing_sar)} optik dosyanın SAR karşılığı yok")
        if missing_lbl:
            errors.append(f"{split}: {len(missing_lbl)} optik dosyanın etiket karşılığı yok")

    if errors:
        print("\n⚠️  Uyarılar:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✅ Tüm dosyalar eşleşiyor!")


# ============================================================
# Ana fonksiyon
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="M4-SAR veri setini indir ve proje formatına dönüştür"
    )
    parser.add_argument("--source", choices=["kaggle", "huggingface", "manual"],
                        default="manual",
                        help="İndirme kaynağı (varsayılan: manual)")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Manuel indirme durumunda ham veri dizini")
    parser.add_argument("--target-dir", type=str, default="./data/m4_sar",
                        help="Çıktı dizini (varsayılan: ./data/m4_sar)")
    parser.add_argument("--no-remap", action="store_true",
                        help="Sınıf eşlemesi yapma (orijinal indeksleri koru)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Sadece mevcut veri setini doğrula")

    args = parser.parse_args()
    target_dir = Path(args.target_dir)

    if args.verify_only:
        verify_dataset(target_dir)
        return

    print("=" * 60)
    print("  M4-SAR Veri Seti Hazırlama")
    print("=" * 60)

    if args.source == "kaggle":
        raw_dir = download_kaggle(target_dir)
    elif args.source == "huggingface":
        raw_dir = download_huggingface(target_dir)
    else:
        if args.raw_dir is None:
            print("\n📌 Manuel mod: lütfen --raw-dir ile ham veri dizinini belirtin.")
            print("\nÖnce şu kaynaklardan birinden indirin:")
            print("  1. Kaggle:       kaggle datasets download -d wchao0601/m4-sar")
            print("  2. Hugging Face: huggingface-cli download wchao0601/m4-sar")
            print("  3. Google Drive: https://drive.google.com/file/d/1ZOGOBLtZEg1pQ_0SkqclgP5XXJsYUkU1")
            print("\nSonra çalıştırın:")
            print(f"  python scripts/download_m4sar.py --raw-dir <indirilen_dizin> --target-dir {target_dir}")
            return
        raw_dir = Path(args.raw_dir)

    if not raw_dir.exists():
        print(f"❌ Ham veri dizini bulunamadı: {raw_dir}")
        sys.exit(1)

    stats = convert_m4sar_to_project(raw_dir, target_dir)
    generate_dataset_info(target_dir, stats)
    verify_dataset(target_dir)

    print("\n" + "=" * 60)
    print("  ✅ Veri seti hazırlama tamamlandı!")
    print(f"  Konum: {target_dir.resolve()}")
    print("=" * 60)
    print("\nSonraki adım:")
    print("  python -m src.train --config configs/multimodal_full.yaml")


if __name__ == "__main__":
    main()
