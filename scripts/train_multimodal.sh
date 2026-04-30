#!/usr/bin/env bash
# Multimodal eğitim — M4-SAR üzerinde tam pipeline
# Kullanım: bash scripts/train_multimodal.sh [epoch_sayısı]

set -euo pipefail

EPOCHS="${1:-80}"
CONFIG="${CONFIG:-configs/multimodal_full.yaml}"
OUTPUT="${OUTPUT:-runs/multimodal_$(date +%Y%m%d_%H%M%S)}"

echo "=== Multimodal eğitim başlıyor ==="
echo "  Config: ${CONFIG}"
echo "  Epoch:  ${EPOCHS}"
echo "  Output: ${OUTPUT}"
echo ""

mkdir -p "${OUTPUT}"

python -m src.train \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --output "${OUTPUT}" \
    2>&1 | tee "${OUTPUT}/train.log"

echo ""
echo "=== Eğitim tamamlandı. Çıktı: ${OUTPUT} ==="
