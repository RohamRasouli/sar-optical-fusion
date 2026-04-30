#!/usr/bin/env bash
# M4-SAR veri seti hazırlama scripti
# Kullanım: bash scripts/prepare_m4sar.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data/m4_sar}"

echo "=== M4-SAR veri seti hazırlama ==="
echo "Hedef klasör: ${DATA_ROOT}"

mkdir -p "${DATA_ROOT}"/{optical/train,optical/val,optical/test}
mkdir -p "${DATA_ROOT}"/{sar/train,sar/val,sar/test}
mkdir -p "${DATA_ROOT}"/{labels/train,labels/val,labels/test}

cat <<'EOF'

Klasör yapısı oluşturuldu:
    ${DATA_ROOT}/
    ├── optical/{train,val,test}/   <- Sentinel-2 RGB png/jpg
    ├── sar/{train,val,test}/       <- Sentinel-1 npy ya da tif (VV+VH)
    └── labels/{train,val,test}/    <- YOLO formatında .txt

Sonraki adımlar:
  1. M4-SAR'ı resmi sayfasından indir:
     https://github.com/wangbingdi/M4-SAR
  2. Dosyaları yukarıdaki klasör yapısına yerleştir
  3. Etiketleri YOLO formatına çevir (her satır: class cx cy w h, normalize)
  4. Train/val/test bölümünü %70/15/15 olarak ayır

Eğer kendi etiketleme aracın varsa, normalize bbox formatına dönüştürmek için:
    python scripts/convert_labels.py --src <orijinal> --dst ${DATA_ROOT}/labels/

EOF
