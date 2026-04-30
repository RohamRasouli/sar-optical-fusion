# SAR + Optik Çapraz-Modal Füzyon ile Hedef Tespiti

> Sentinel-1 SAR ve Sentinel-2 optik uydu görüntülerinde çift akımlı çapraz-modal
> dikkat füzyon ağı ile düşük görünürlüklü ve kamuflajlı hedef tespiti.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)]()

## Tek Cümlede Ne Yapıyor

İki tip uydu görüntüsünü (optik kamera + radar) öğrenmeli olarak birleştirip, gizlenmeye çalışan veya zor koşullarda bulunan hedefleri tespit eden bir derin öğrenme modeli.

## Hızlı Başlangıç

```bash
# Klonla ve kur
git clone https://github.com/RohamRasouli/sar-optical-fusion.git
cd sar-optical-fusion
pip install -e .

# Sanity test (CPU'da, dummy data ile çalışır)
pytest tests/ -v

# Eğitim (Kaggle/Colab GPU önerilir)
python -m src.train --config configs/multimodal_full.yaml

# Değerlendirme
python -m src.eval --checkpoint runs/best.pt --config configs/multimodal_full.yaml

# Tek görüntü çıkarımı
python -m src.predict --optical sample.jpg --sar sample_sar.tif --checkpoint runs/best.pt

# Ablation çalışması (10 farklı yapılandırma otomatik)
python scripts/run_ablations.py --base configs/multimodal_full.yaml --epochs 30

# ONNX export (Jetson dağıtımı için)
python -m src.export --checkpoint runs/final.pt --format onnx --output model.onnx

# REST API servisi
SAR_FUSION_CKPT=runs/final.pt uvicorn src.api:app --port 8080

# Streamlit demo
streamlit run demo/app.py
```

## Mimari Özet

```
Optik (3, H, W) ──▶ Encoder_OPT ──▶ F_opt^{1/8, 1/16, 1/32}
                                          │
                                          ▼
                              ┌── 3× CMAFM ──┐
                                          ▲
                                          │
SAR    (2, H, W) ──▶ Encoder_SAR ──▶ F_sar^{1/8, 1/16, 1/32}

                                          │
                                          ▼
                                  PAN-FPN Neck
                                          │
                                          ▼
                          Decoupled Detection Head
                                          │
                                          ▼
                              Bounding boxes + sınıflar
```

**CMAFM** (Cross-Modal Attention Fusion Module):
- İki yönlü çapraz-dikkat (multi-head)
- Pencere-tabanlı verimli implementasyon
- Sigmoid kapı (gating) ile koşula bağlı modalite ağırlığı

**CamouflageAware Loss** = CIoU + DFL + BCE
                        + λ₁·DynamicFocal + λ₂·BoundaryLoss + λ₃·ConsistencyLoss

## Klasör Yapısı

```
.
├── configs/                  # YAML konfigürasyon dosyaları
├── src/
│   ├── datasets/             # Veri yükleyiciler + augmentation + stres testleri
│   ├── models/               # Encoder, CMAFM, neck, head, baseline'lar
│   ├── losses/               # CamouflageAware Loss + TAL Assigner
│   ├── utils/                # WandB logger, görselleştirme
│   ├── train.py · eval.py · predict.py · export.py · api.py
├── scripts/                  # Veri hazırlama + ablation runner
├── notebooks/                # Kaggle/Colab şablonları
├── demo/                     # Streamlit demo UI
├── tests/                    # PyTest birim testler
└── docs/                     # Mimari + dağıtım rehberi
```

## Eklenen Bileşenler

**Modeller:** Tam multimodal (CMAFM + CAL) + 5 baseline (optical only, SAR only,
concat early fusion, late fusion, single-scale attention) — ablation karşılaştırması için.

**Loss:** Task-Aligned Assigner (YOLOv8 stili) + CIoU + DFL + BCE + Dynamic Focal +
Boundary-Aware + Cross-Modal Consistency.

**Augmentation:** Lee filter + sentetik kamuflaj + 10 stres preset
(`clean`, `cloud_light/medium/heavy`, `night_light/dark`, `camo_only`, `cloud_camo`,
`night_camo`, `all_combined`).

**Görselleştirme:** Bbox overlay, attention heatmap, σ_opt/σ_sar gating maps,
loss eğrileri, confusion matrix, tez figür gridi.

**Dağıtım:** ONNX/TorchScript/TensorRT export, FastAPI REST servisi
(`/health`, `/info`, `/detect`, `/detect_geo`), Streamlit demo UI,
Jetson Orin Docker imajı.

**Operasyon:** Ablation runner (10+ deney otomatize), WandB logging
(loss eğrileri + gating histogramları)._

## Veri Setleri

| Veri Seti      | Görüntü      | Modalite          | Kullanım                |
|----------------|-------------:|-------------------|-------------------------|
| M4-SAR         | ~14,500 çift | Optik + SAR       | Birincil eğitim/val/test |
| SARDet-100K    | ~117,000     | SAR               | SAR backbone pre-training |
| Sentetik kamuflaj | ~%30 augment | Optik (SAR sabit) | Eğitim sırasında         |
| In-the-wild    | 200-500      | Optik + SAR       | Sadece final test        |

## Sonuçlar

> Sayılar tam eğitim tamamlandığında doldurulacak. Hedef:

| Model                              | mAP@50 | mAP@50-95 | FPS (Jetson Orin) |
|------------------------------------|-------:|----------:|------------------:|
| YOLOv8 (yalnız optik)              |  ~75   |    ~50    |       —           |
| YOLOv8 (yalnız SAR)                |  ~70   |    ~45    |       —           |
| Concat early fusion                |  ~78   |    ~53    |       —           |
| **Bu proje (CMAFM + CAL)**         | **~82**|  **~58**  |     **~25**       |

## Atıf

Eğer bu çalışmayı kullanırsanız:

```bibtex
@thesis{kerahroudi2026sarfusion,
  author = {Kerahroudi, Roham R.},
  title  = {SAR ve Optik Uydu Görüntülerinde Çift Akımlı Çapraz-Modal Füzyon ile Hedef Tespiti},
  school = {Sakarya Üniversitesi},
  year   = {2026},
  type   = {Bitirme Tezi}
}
```

## Lisans

MIT — bkz. [LICENSE](LICENSE).

## Teşekkürler

Danışman: **Prof. Dr. Cemil Öz** — Sakarya Üniversitesi Bilgisayar Mühendisliği Bölümü.

## İletişim

Roham Rasouli Kerahroudi · roham.kerahroudi@ogr.sakarya.edu.tr
