# CLAUDE.md — Proje Hafızası ve Devir Notu

> **Bu dosya Claude'un (ve gelecekteki herhangi bir aracın) bu projeyle ilgili
> tüm bağlamı tek bakışta anlaması için yazıldı.** Konuşma sıfırlanırsa, yeni
> Claude bu dosyayı okuyarak kaldığı yerden devam edebilir.
>
> **Asıl çalışma klasörü:** `C:\Users\roham\Documents\bitirme projem deneme`
> **Yedek/sync klasör:** `C:\Users\roham\Documents\bitirme projem` (robocopy /XO ile sync)

---

## 1. Proje Kimliği

| Alan | Değer |
|------|-------|
| **Başlık** | SAR ve Optik Uydu Görüntülerinde Çift Akımlı Çapraz-Modal Dikkat Füzyon Ağı ile Kamuflajlı Askeri Hedef Tespiti |
| **Öğrenci** | Roham Rasouli Kerahroudi (B211210561) |
| **Danışman** | Prof. Dr. Cemil Öz |
| **Üniversite** | Sakarya Üniversitesi |
| **Bölüm** | Bilgisayar Mühendisliği — Bitirme Projesi |
| **E-posta** | roham.kerahroudi@ogr.sakarya.edu.tr |
| **Kaggle username** | rohamrasouli |

### İkili kullanım çerçevesi (önerilen)

Saf askeri çerçeve yerine ikili kullanım: **Sahil Güvenlik (yasadışı tekne tespiti) +
AFAD (deprem hasar) + Orman GM (yasadışı kesim)** → aynı teknoloji, açık kaynak veri,
askeri uzantı opsiyonel. Tezde "Düşük-Görünürlüklü Hedef Tespiti" başlığı önerildi.

---

## 2. Proje Yapısı

```
bitirme projem deneme/
├── src/
│   ├── datasets/
│   │   ├── m4_sar.py              # M4-SAR loader (Optik+SAR çift)
│   │   ├── sardet.py              # SARDet-100K (pre-training için)
│   │   ├── preprocess.py          # dB, normalize, paired flip
│   │   └── augmentation/
│   │       ├── lee_filter.py      # SAR speckle filtresi (klasik + öğrenilebilir)
│   │       ├── camo_synth.py      # Sentetik kamuflaj
│   │       └── stress.py          # 10 stres preset (cloud/night/camo)
│   ├── models/
│   │   ├── encoder.py             # CSPDarknet × 2 (DualStreamEncoder)
│   │   ├── cmafm.py               # ★ Cross-Modal Attention Fusion (3 ölçek + gating)
│   │   ├── neck.py                # PAN-FPN
│   │   ├── head.py                # YOLOv8 decoupled head + DFL
│   │   ├── full_model.py          # Tüm parçaların birleşimi + aux head'ler
│   │   └── baselines.py           # 5 baseline (optical_only, sar_only, concat, late, single_attn)
│   ├── losses/
│   │   ├── box_loss.py            # CIoU + DFL
│   │   ├── camouflage_aware.py    # ★ Dynamic Focal + Boundary + Consistency
│   │   └── detection_loss.py      # TaskAlignedAssigner + birleşik loss
│   ├── utils/
│   │   ├── visualization.py       # Bbox overlay, attention heatmap, gridler
│   │   └── wandb_logger.py        # WandB hooks (gating histogramları)
│   ├── train.py                   # Eğitim — dict_merge yaml defaults destekli
│   ├── eval.py                    # mAP@50 / mAP@50-95
│   ├── predict.py                 # Tek görüntü çıkarımı
│   ├── export.py                  # ONNX / TorchScript / TensorRT
│   └── api.py                     # FastAPI servisi
├── configs/
│   ├── base.yaml                  # Tüm hiperparametreler
│   ├── multimodal_full.yaml       # 80 epoch, batch=2, grad_accum=16, img_size=416
│   ├── kaggle_p100.yaml           # batch=16, grad_accum=2, img_size=640
│   └── pretrain_sar.yaml          # SARDet-100K pre-training
├── scripts/
│   ├── pack_for_kaggle.py         # Kod zip paketleyici
│   ├── kaggle_auto_train.ps1      # End-to-end Kaggle PowerShell automation
│   ├── run_ablations.py           # 10 ablation otomatize (A_full → J_no_camo)
│   ├── prepare_m4sar.sh           # Veri klasör hazırlama
│   ├── download_m4sar.py          # M4-SAR indirme (kullanıcı eklemiş)
│   ├── generate_demo_dataset.py   # Demo veri seti üretici (kullanıcı eklemiş)
│   └── train_multimodal.sh
├── notebooks/
│   ├── kaggle_train.ipynb         # Kaggle eğitim şablonu
│   └── kaggle_train.py            # .py versiyonu (kullanıcı eklemiş)
├── demo/
│   └── app.py                     # Streamlit UI
├── tests/                         # 16+ pytest
├── docs/
│   ├── architecture.md            # Mimari detay
│   ├── deployment.md              # Jetson + ONNX rehberi
│   └── kaggle_setup.md            # Adım adım Kaggle rehberi
├── data/
│   └── m4_sar/                    # Lokalde gerçek veri (56,116 train + 22,112 val)
├── runs/                          # Eğitim checkpoint'leri (.pt dosyaları)
├── _kaggle_upload/                # Kaggle dataset upload klasörü (zip + metadata)
├── .kkernel/                      # Kaggle kernel push klasörü (script + metadata)
├── _KAGGLE_AUTO.bat               # Fire-and-forget launcher
├── _KAGGLE_BASLAT.bat             # Interactive launcher
├── _FIX_VE_BASLAT.bat             # Token fix + upload + push
├── _sync_from_deneme.bat          # robocopy /XO senkron
├── sar-optical-fusion-code.zip    # ~76 KB pack çıktısı (Kaggle upload için)
├── README.md
├── pyproject.toml
├── requirements.txt
├── train_log.txt                  # İlk başarısız run (CPU, generator bug)
├── train_log_gpu.txt              # GPU run #1 (E1 B26960'a kadar)
├── train_log_gpu_overnight.txt    # Resume run başlangıcı
└── train_log_gpu_final.txt        # Resume run sonu (E1 B25920)
```

---

## 3. Mimari Özet

### Çift Akımlı Encoder + CMAFM Pipeline

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
                                  PAN-FPN Neck (3 ölçek)
                                          │
                                          ▼
                          Decoupled Detection Head (cls + box-DFL)
                                          │
                                          ▼
                               Bounding boxes + sınıflar
```

### CMAFM (Cross-Modal Attention Fusion Module) — projenin özü

```
F_o2s = MultiHead(Q=F_opt, K=F_sar, V=F_sar)   # opt → sar
F_s2o = MultiHead(Q=F_sar, K=F_opt, V=F_opt)   # sar → opt (window-tabanlı)

σ_opt = sigmoid(Conv1x1(Concat(F_opt, F_o2s)))
σ_sar = sigmoid(Conv1x1(Concat(F_sar, F_s2o)))

F_opt' = F_opt + σ_opt · F_o2s
F_sar' = F_sar + σ_sar · F_s2o
F_fused = Conv1x1(Concat(F_opt', F_sar'))
```

3 ölçekte (1/8, 1/16, 1/32) ayrı CMAFM bloğu. Window-attention kullanılarak hesap maliyeti
makul tutuldu. Gating mekanizması koşula bağlı (bulutlu/gece/açık) modalite ağırlığı öğrenir.

### CamouflageAware Loss

```
L_total = w_box·CIoU + w_cls·BCE + w_dfl·DFL
        + λ_focal·DynamicFocal      # γ_dynamic = γ_base·(1 + β·D(x))
        + λ_bound·BoundaryAware     # Sobel gradient mismatch (GT mask gerek)
        + λ_consist·CrossModalConsistency  # KL(P_fused || stop_grad(P_opt/sar))
```

---

## 4. Bu Sohbette Tamamlananlar (Konuşma Tarihçesi)

### Aşama 1 — Strateji & Plan (PDF'ler üretildi)
- **Bitirme_Yol_Haritasi.pdf** (37 sayfa) — teknik mimari, eğitim, değerlendirme
- **Operasyonel_Yol_Haritasi.pdf** (25 sayfa) — paydaş, ikili kullanım, kariyer

### Aşama 2 — Kod Tabanı (5400+ satır Python, 33 dosya)
- Tüm modeller, loss'lar, dataset loader'ları yazıldı
- TaskAlignedAssigner (YOLOv8 stili) implementasyonu
- 5 baseline model
- 10 stres preset
- WandB entegrasyonu
- ONNX/TensorRT export
- FastAPI servisi
- Streamlit demo
- 16+ pytest

### Aşama 3 — Eğitim Denemeleri (Kullanıcının lokali)

| Run | Konum | Durum |
|-----|-------|-------|
| `train_log.txt` | CPU, multimodal_full | Crash: `_prepare_targets` line 229'da generator bug → kullanıcı düzeltti |
| `train_log_gpu.txt` | GPU run #1 | E1 B26960 — loss ~0.6 stabil ama düşmüyor; cal_focal=0.000 |
| `train_log_gpu_overnight.txt` | Resume başlangıç | "Epoch 1 noktasından devam" — 56116 train, 22112 val sample |
| `train_log_gpu_final.txt` | Resume sonu | E1 B25920 — loss 4-7 (config farklı); cal_focal hala 0 |

### Aşama 4 — Kaggle/Colab Aktarım Çabası

**Adımlar:**
1. ✅ pack_for_kaggle.py → `sar-optical-fusion-code.zip` (76 KB)
2. ✅ Kaggle CLI yüklendi
3. ✅ kaggle.json yenilendi (rohamrasouli)
4. ✅ Phone verification yapıldı
5. ❌ **`kaggle datasets create` → 401 Unauthorized** (sürekli)
6. ❌ Kernel push da 401
7. Colab denemesi: `FileNotFoundError: Kritik klasörler bulunamadı: Optik, SAR, Label`

**JSON BOM bug'ı:** `Set-Content -Encoding UTF8` UTF-8 with BOM yazıyordu, kaggle CLI Python json.loads parse edemiyordu. `[System.IO.File]::WriteAllText(..., UTF8Encoding(false))` ile düzeltildi.

---

## 5. KRİTİK BUG'LAR / AÇIK SORUNLAR

### 🔴 cal_focal = 0.000 sürekli (Dynamic Focal Loss EFFECTIVELY KAPALI)
- Bütün eğitim log'larında `cal_focal=0.000` yazıyor
- Dynamic Focal Loss class match yapamıyor ya da gamma yanlış
- **Etkisi: CamouflageAware Loss'un asıl özelliği etkisiz** → tezde ablation güçsüz
- **Düzeltme yeri:** `src/losses/camouflage_aware.py` `DynamicFocalLoss.forward()` — gamma_dynamic hesaplaması ve target_score broadcast

### 🔴 Loss düşüşü yok (Epoch 1 boyunca yatay)
- 26,000+ batch'te bile loss ~0.6 stabil → öğrenme YOK
- Olası sebepler:
  - TaskAlignedAssigner positive anchor yetersiz seçiyor
  - LR çok yüksek (6.5e-4 warmup'ta)
  - Veya box loss ağırlığı çok yüksek (config: w_box=7.5)
- **Doğrulamak için:** epoch sonunda `eval.py` çağırıp gerçek mAP ölçmek

### 🔴 Lokal eğitim ASIRLAR sürer
- 56,116 sample × batch_size=2 = ~28,000 batch/epoch
- 1 epoch ~12-15 saat → 80 epoch = **40+ gün lokalde**
- **Çözüm: Kaggle T4×2 veya Colab L4 zorunlu** (ama her ikisi de blocker'da)

### 🟡 Kaggle 401 Unauthorized
- Yeni token + phone verified → hala 401
- Olası: Kaggle hesabı yeni → trust score düşük → upload kilitli
- **Plan B: Browser drag-drop manuel upload** (CLI bypass) — denenmedi

### 🟡 Colab klasör adı uyumsuzluğu
- Kod `data/m4_sar/{Optik,SAR,Label}` Türkçe büyük harfli klasör arıyor
- Zip ile çıkarılan yapıda farklı (lowercase ya da farklı isimler)
- **Çözüm:** Colab'da `find` ile gerçek yapı tespit et + symlink ya da rename

### 🟡 Boundary Loss devre dışı
- GT mask gerekiyor, M4-SAR'da bbox var ama mask yok
- `configs/multimodal_full.yaml`: `boundary.enabled: false`
- **Düzeltme:** bbox'tan dikdörtgen mask üret → yaklaşık ama çalışır

### 🟡 train.py user'ın güncel versiyonu farklı
- User `dict_merge` ekledi (defaults config inheritance)
- `--resume`, `--time_budget_min`, `--save_every` flag'leri kullanıcının versiyonunda **yok**
- Kullanıcı muhtemelen Kaggle'da kullanmadı

---

## 6. Veri Setleri

| Set | Boyut | Modalite | Durum |
|-----|-------|----------|-------|
| **M4-SAR train** | 56,116 görüntü çifti | Optik (jpg) + SAR + Label | Lokalde `data/m4_sar/`, Kaggle'a yüklenmemiş |
| **M4-SAR val** | 22,112 | aynı | Lokalde |
| **SARDet-100K** | ~117,000 | SAR | Pre-training için (kullanılmadı) |
| **Sentetik kamuflaj** | augmentation | Optik | %30 prob, eğitim sırasında |
| **In-the-wild** | yok | — | Tezde planlandı, üretilmedi |

**Klasör yapısı (lokalde):** `Optik`, `SAR`, `Label` (Türkçe büyük harf)
**Dosya formatı:** Optik = jpg, SAR = ?, Label = txt (YOLO format)

---

## 7. Konfigürasyon Detayları

### `configs/multimodal_full.yaml`
```yaml
defaults: [base.yaml]
model:
  img_size: 416             # KÜÇÜK — RAM kısıtı
training:
  epochs: 80
  batch_size: 2             # ÇOK KÜÇÜK
  grad_accum_steps: 16      # etkili batch=32
loss:
  camouflage_aware:
    enabled: true
    focal: { enabled: true }
    boundary: { enabled: false }   # GT mask yok
    consistency: { enabled: true }
```

### `configs/kaggle_p100.yaml`
```yaml
defaults: [base.yaml]
model: { img_size: 640 }
training:
  batch_size: 16
  grad_accum_steps: 2
  epochs: 80
```

### `configs/base.yaml` öne çıkan ayarlar
- Optimizer: AdamW, lr=1e-3, weight_decay=0.05
- Scheduler: cosine, warmup_epochs=3, min_lr=1e-6
- AMP: true (mixed precision)
- EMA decay: 0.9999
- num_classes: 6 (M4-SAR sınıfları)
- Optical channels: 3, SAR channels: 2
- CMAFM heads: [4, 4, 8], window_size: 8
- Loss weights: box=7.5, cls=0.5, dfl=1.5
- CAL: λ_focal=1.0, λ_bound=0.3, λ_consist=0.2

---

## 8. Sonraki Mantıklı Adımlar (öncelik sırası)

### Acil
1. **Kaggle 401 çözmek** — 2 ihtimal:
   - (a) Browser UI'dan manuel zip upload (`kaggle.com/datasets → New Dataset`)
   - (b) Hesap trust score için 24-48 saat bekle, tekrar dene
2. **Colab'da klasör yapısı düzelt:**
   ```bash
   find /content/sar-optical-fusion/data -maxdepth 5 -type d
   # Çıktıya göre rename ya da symlink
   ```
3. **Best checkpoint seç** — `runs/` altında en yüksek epoch:
   ```powershell
   Get-ChildItem .\runs -Recurse -Filter *.pt | Sort LastWriteTime -Desc
   ```

### Orta vade
4. **cal_focal=0 bug'ını düzelt** — `camouflage_aware.py` debug
5. **eval.py ile mAP ölç** — loss yatay olabilir ama mAP iyi olabilir
6. **Boundary loss aktif et** — bbox'tan dummy mask üret
7. **Resume desteğini train.py'a geri ekle** (user kaldırmış, kullanışlı)

### Tez & Yayın
8. **Ablation deneyleri** — `scripts/run_ablations.py` (10 deney, ~30 saat T4)
9. **Stres testleri** — `src/datasets/augmentation/stress.py` 10 preset
10. **In-the-wild test seti** — 200-500 manuel etiketleme
11. **SİU 2026 bildirisi** — extended abstract Mart 2026
12. **Tez yazımı** — Hafta 14 (zaman çizelgesinde)

---

## 9. Sohbetin Çıkardığı Tribal Knowledge

### Kullanıcı tarafı yapılan değişiklikler (KORU)
- `src/train.py` — kullanıcı `dict_merge` + defaults YAML inheritance ekledi, resume/time_budget kaldırıldı. **Geri alınmamalı.**
- `configs/multimodal_full.yaml` — img_size=416, batch=2 (RAM kısıtlı)
- `configs/kaggle_p100.yaml` — kullanıcı kendi yarattı
- `scripts/download_m4sar.py` ve `scripts/generate_demo_dataset.py` — kullanıcı eklemiş
- `notebooks/kaggle_train.py` — kullanıcı .ipynb yerine .py tercih etmiş
- `src/losses/detection_loss.py` line 229 → kullanıcı `_prepare_targets` generator bug'ını düzeltti

### Cowork mount kısıtları
- Bu Claude oturumunda yalnızca `bitirme projem` mount'u var
- `bitirme projem deneme` mount'u "unsupervised mode" ile bloke
- Yeni Claude konuşmasında interaktif onay → mount alınabilir

### Computer-use kısıtları (Windows)
- Terminal/PowerShell tier "click" → tıklayabilir, **yazamaz**
- Browser (Chrome) tier "read" → görür, **dokunamaz**
- File Explorer tier "full" → tam kontrol
- Notepad/diğer apps → tam kontrol

### Kaggle gotcha'ları
- `dataset-metadata.json` UTF-8 BOM olursa parse hatası → BOM'suz yazılmalı
- `--dir-mode zip` tüm klasör hierarşisini zip'liyor (.git dahil) → ayrı upload klasörü gerek
- Phone verification yetmez, hesap trust score gecikmesi olabilir
- Token telefon doğrulamadan ÖNCE üretilirse geçersiz, yeniden expire+create gerek

---

## 10. Hızlı Komut Referansı

### Lokal eğitim (kullanıcının train.py'siyle)
```powershell
cd "C:\Users\roham\Documents\bitirme projem deneme"
python -m src.train --config configs/multimodal_full.yaml
```

### Kaggle pack + upload (manuel UI)
```powershell
cd "C:\Users\roham\Documents\bitirme projem deneme"
python scripts/pack_for_kaggle.py
# kaggle.com → Datasets → New Dataset → drag-drop sar-optical-fusion-code.zip
```

### Colab klasör fix (içinde çalıştır)
```python
import os
# Önce ne var bak
!find /content/sar-optical-fusion/data -maxdepth 5 -type d
# Sonra rename ya da symlink — gerçek yapıya göre
```

### Sync (deneme → bitirme projem)
```powershell
robocopy "C:\Users\roham\Documents\bitirme projem deneme" "C:\Users\roham\Documents\bitirme projem" /E /XO /XD .git __pycache__ runs data .venv
```

### Eğitim durumu kontrol
```powershell
Get-ChildItem .\runs -Recurse -Filter *.pt | Sort LastWriteTime -Desc | Select Name, Length, LastWriteTime
Get-Content train_log_gpu_final.txt -Tail 30 -Encoding Unicode
```

---

## 11. İlgili Belgeler ve Linkler

- **GitHub repo (planlanan):** github.com/roham/sar-optical-fusion
- **Kaggle hesap:** kaggle.com/rohamrasouli
- **Kaggle dataset (yüklendiyse):** kaggle.com/datasets/rohamrasouli/sar-optical-fusion-code
- **Kaggle kernel (yüklendiyse):** kaggle.com/code/rohamrasouli/sar-optical-fusion-train
- **M4-SAR resmî:** github.com/wangbingdi/M4-SAR
- **SARDet-100K:** Li et al. NeurIPS 2024

---

## 12. Eğer Yeni Bir Konuşma Açılırsa

İlk mesaja şunu yapıştır:

> Selam Claude, bitirme projemde devam ediyoruz. Bana
> `C:\Users\roham\Documents\bitirme projem deneme` klasörüne erişim ver.
> Bu klasörde `CLAUDE.md` dosyası var, önce onu oku — tüm bağlam orada.
> Sonra şu an çözmem gereken sorun: [GÜNCEL SORUN BURAYA].

---

**Son güncelleme:** 4 Mayıs 2026
**Sürüm:** 1.1
**Hazırlayan:** Claude (önceki konuşma + 4 May güncelleme)

---

## 13. GÜNCEL DURUM — `runs/final (1).pt` (4 May 2026)

> **Bu bölüm 4 Mayıs 2026'da Claude tarafından eklendi. CLAUDE.md'nin önceki
> bölümlerindeki "Kaggle 401 blocker'da" durumu artık GEÇERLİ DEĞİL — eğitim
> Kaggle'da koştu ve `runs/final (1).pt` checkpoint'i hazır.**

### Checkpoint özeti

| Alan | Değer |
|------|-------|
| **Dosya** | `runs/final (1).pt` (~246 MB) |
| **Tarih** | 4 May 2026 01:04 |
| **Epoch** | 37 (38 planlanmıştan) → cosine LR sıfıra inmiş, eğitim _de facto_ bitmiş |
| **Effective batch** | 256 (16 × 16 grad accum) |
| **img_size** | 416 |
| **Final loss** | 5.677 (CAL açıkken — eski 0.6 değerleri box-only baseline'dı) |
| **Total optimizer steps** | 8322 |
| **Last LR** | 1e-6 (cosine schedule taban) |
| **Source** | Kaggle GPU; veri `/kaggle/input/datasets/wchao0601/m4-sar/M4-SAR/M4-SAR` |
| **CMAFM** | enabled, heads=[4,4,8], window=8, drop_path=0.1 |
| **CAL focal** | enabled (γ_base=2.0, β=1.5, λ=1.0) |
| **CAL boundary** | disabled (mask yok hâlâ) |
| **CAL consistency** | enabled (λ=0.2, warmup 5 epoch) |
| **EMA** | decay 0.9999 |
| **Backbone freeze** | İlk 10 epoch optik+SAR encoder donmuş |

### Bu, projeyi şu noktaya getirdi

- ✅ **Kaggle entegrasyonu çalışıyor** (401 sorunu aşıldı, dataset yüklenmiş, kernel koşmuş)
- ✅ **Tam multimodal model 38 epoch eğitildi** (CMAFM + CAL + mosaic + mixup)
- ⚠️ **mAP henüz ölçülmedi** — final loss 5.7'nin iyi mi kötü mü olduğunu söylemek için `eval.py` çalıştırılmalı
- ⚠️ **5 baseline + ablation deneyleri YOK** — tezde karşılaştırma için gerekli
- ⚠️ **cal_focal=0 bug'ı bu run'da düzeldi mi bilinmiyor** — train_log_kaggle yok elimde

### Bir sonraki adım

Sıradaki en yüksek değer eylem: **`runs/final (1).pt` üzerinde `eval.py` çalıştırıp mAP@50 ve mAP@50-95 ölçmek.** Sayı tutuyorsa baseline'lar + ablation'a geçilir; tutmuyorsa ya 1-2 epoch fine-tune ya da config diagnozu gerekir.

Roadmap detayı için `ROADMAP.md`'ye bak.
