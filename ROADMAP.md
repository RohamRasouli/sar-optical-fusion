# Proje İlerleme Yol Haritası — SAR + Optik Füzyon Bitirme Projesi

> Son güncelleme: **4 Mayıs 2026** · Hazırlayan: Claude · Hedef tarih: tez teslimi (Hazırlanmakta)
> Bu doküman `CLAUDE.md`'nin bir uzantısıdır. Önce CLAUDE.md'yi oku, bağlam orada.

---

## 1. ŞU AN NEREDEYİZ (snapshot)

```
   Plan & Strateji   Kod Tabanı       Eğitim          Değerlendirme    Tez Yazımı
   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐    ┌──────────┐
   │  100%    │ ──▶ │  100%    │ ──▶ │   85%    │ ──▶ │   5%     │ ──▶│   0%     │
   │  ✅ DONE  │     │  ✅ DONE  │     │  ▣▣▣▣▢   │     │  ▢▢▢▢▢   │    │  ▢▢▢▢▢   │
   └──────────┘     └──────────┘     └──────────┘     └──────────┘    └──────────┘
        ▲                ▲                ▲                ▲                ▲
        │                │                │                │                │
   2 PDF rapor       5400+ satır     final(1).pt       0 metrik       0 bölüm
   (37+25 sayfa)     33 dosya        37/38 epoch       (acil!)        yazıldı
```

**Tek cümleyle:** Tam multimodal model Kaggle'da 38 epoch eğitildi (`runs/final (1).pt`), CMAFM ve CAL açık. Ama **mAP henüz ölçülmedi**, baseline yok, ablation yok, tez yazılmadı. Sonraki en kritik 24 saat: **eval çalıştır, sayıyı gör.**

---

## 2. CHECKPOINT DURUM RAPORU — `runs/final (1).pt`

| Boyut | Detay |
|-------|-------|
| Dosya | `runs/final (1).pt` (~246 MB, 4 May 01:04) |
| Epoch | **37/38** — cosine schedule taban LR'ye inmiş, eğitim de facto tamam |
| Loss | 5.677 (CAL+CMAFM açıkken — bu sayı tek başına anlamsız, mAP'a bakmamız gerek) |
| Effective batch | 256 (gerçek 16 × grad_accum 16) |
| img_size | 416 (Kaggle T4/P100 RAM limiti) |
| Total opt steps | 8322 |
| Last LR | 1e-6 (taban) |
| Veri | M4-SAR (Kaggle dataset: `wchao0601/m4-sar`) |
| Yapı | DualEncoder + 3× CMAFM (heads 4/4/8, win=8) + PAN-FPN + decoupled head |
| Loss | CIoU + DFL + BCE + DynamicFocal(γ=2,β=1.5) + Consistency(λ=0.2). Boundary KAPALI |
| EMA | 0.9999 (varsa EMA weights'i de eval'da kullanmak lazım) |
| Backbone freeze | İlk 10 epoch optik+SAR encoder donuk → fusion ve head önce ısındı |

**Yorum:** Konfigürasyon temiz, CAL bileşenlerinin çoğu açık, eğitim normal şekilde sonlandı. Loss 5.7 çok şey söylemiyor çünkü 6 farklı bileşenin toplamı; gerçek mantra **mAP**.

---

## 3. ACİL — ÖNÜMÜZDEKİ 1 HAFTA (Hafta 1)

> Hedef: Sayıları görmek, kötüyse hızlı düzeltmek, baseline'ları başlatmak.

### Gün 1-2: Değerlendirme (BLOCKER)

```
[ ] 1.  eval.py'yi runs/final (1).pt üzerinde çalıştır
        komut: python -m src.eval --checkpoint "runs/final (1).pt" --config configs/kaggle_p100.yaml
        çıktı: mAP@50, mAP@50-95, sınıf bazlı AP

[ ] 2.  EMA ağırlıkları varsa onlarla da eval et — genelde +1-2 puan getirir
        (checkpoint'te 'ema_state' var mı bak)

[ ] 3.  Tek görüntüde predict.py'yi test et — gerçekten bbox üretiyor mu?
        komut: python -m src.predict --optical sample.jpg --sar sample.tif --checkpoint "runs/final (1).pt"

[ ] 4.  Sonucu CLAUDE.md'ye yaz, baseline'lara karşılaştırma planı yap
```

**Karar noktası:** mAP@50 hedefimiz **≥ %75**.
- ≥ 75 → Plan A: ablation'a geç (sıradan).
- 60-75 → Plan B: 5-10 epoch fine-tune (img_size=640, daha düşük LR=1e-5).
- < 60 → Plan C: cal_focal bug debug + LR/loss-weight diagnozu, baştan 20 epoch.

### Gün 3-5: Baseline'lar (TEZ İÇİN ZORUNLU)

```
[ ] 5.  optical_only baseline (sadece optik kol, CMAFM yok)
[ ] 6.  sar_only baseline (sadece SAR kol)
[ ] 7.  concat early fusion (en basit baseline — tezdeki "biz daha iyiyiz" iddiası için)
[ ] 8.  late fusion (iki ayrı dedektör + score-level merge)
[ ] 9.  single-scale attention (CMAFM'in 1 yerine 3 ölçek olmasının değerini kanıtlar)
```

→ Hepsi `scripts/run_ablations.py` ile otomatize edilmiş; yine **Kaggle'da** koştur. Her biri ~6-8 saat T4'te. Toplam 30-40 saat → 4-5 Kaggle session (haftalık 30 saat GPU kotası mümkün).

### Gün 6-7: Hızlı Görselleştirme

```
[ ] 10. Tahmin örnekleri: clean / cloud_medium / night_dark / camo_only — 4 sahne
        (src/utils/visualization.py + src/datasets/augmentation/stress.py)
[ ] 11. CMAFM gating histogramı: σ_opt vs σ_sar dağılımları
        (bulutlu sahnede SAR ağır basıyor mu? → tezde KÜÇÜK ALTIN)
[ ] 12. Attention heatmap overlay: 3 ölçek için 3 sahne = 9 görsel
[ ] 13. Loss eğrileri (Kaggle log dosyalarından train_log_kaggle.txt)
```

---

## 4. ORTA VADE — HAFTA 2-3

### Hafta 2: Ablation & Stres Testleri

```
[ ] 14. CAL bileşen ablation:
        - A: tam (CIoU+DFL+BCE+Focal+Consistency)
        - B: -Consistency
        - C: -Focal
        - D: -hiçbiri (vanilla)
        Tablo: 4 satır × 3 metrik (mAP@50, mAP@50-95, FPS)

[ ] 15. CMAFM ablation:
        - 3 ölçek ✓ (tam)
        - 1 ölçek (sadece 1/16)
        - Window-attention yerine full-attention (yavaş ama kıyas)
        - Gating kapalı (σ=1 sabit)

[ ] 16. Stres test matrisi: 10 preset × 3 model varyantı = 30 koşu
        (sadece eval, eğitim YOK — hızlı)
        Çıktı: tezde KAPSAYICI bir tablo
```

### Hafta 3: cal_focal & Boundary Düzelt + Re-train

```
[ ] 17. cal_focal=0 bug'ı bu son run'da gerçekten düzelmiş mi kontrol et
        (Kaggle log'una bak; eğer hâlâ 0'sa src/losses/camouflage_aware.py debug)

[ ] 18. Boundary loss aktif et: bbox'tan dikdörtgen mask üret
        src/datasets/m4_sar.py'de bbox→mask dönüşümü ekle
        configs/kaggle_p100.yaml: boundary.enabled = true

[ ] 19. 20 epoch ek fine-tune (sadece head + fusion donsun encoder)
        amaç: boundary loss'un etkisini görmek + mAP'i +1-2 puan zorlamak
```

---

## 5. TEZ HAZIRLIK — HAFTA 4-8

### Hafta 4: In-the-wild Test Seti

```
[ ] 20. Sentinel Hub'dan 200 ham çift indir (Türkiye sahili / Antalya / Bartın)
        Aynı bbox'tan optik (S2) + SAR (S1) eşleştir, manuel etiketle
        (Roboflow / CVAT — 2 gün, 4-5 sınıf yeter)

[ ] 21. final (1).pt + 5 baseline'ı bu sette eval et → "real-world generalization"
        bölüm tezde alkış toplar
```

### Hafta 5-6: Tez Yazımı (Bölüm bölüm)

```
[ ] 22. Bölüm 1: Giriş (4-5 sayfa)        — problem, motivation, ikili kullanım
[ ] 23. Bölüm 2: Literatür (8-10 sayfa)    — SAR detection, multimodal fusion, attention
[ ] 24. Bölüm 3: Yöntem (12-15 sayfa)      — DualEncoder, CMAFM, CAL — denklemler + figürler
[ ] 25. Bölüm 4: Veri (4-5 sayfa)          — M4-SAR, augmentation, in-the-wild
[ ] 26. Bölüm 5: Deneyler (10-12 sayfa)    — baseline tablosu, ablation, stres test, niteliksel
[ ] 27. Bölüm 6: Sonuç (3 sayfa)           — katkı özeti, sınırlamalar, gelecek iş
[ ] 28. Ek A: Hiperparametre tablosu, Ek B: Reproduksiyon adımları
```

→ Toplam ~50 sayfa hedef. `Bitirme_Yol_Haritasi.pdf` (37 sayfa) iskelet sağlıyor zaten.

### Hafta 7-8: Sunum + Demo

```
[ ] 29. Tez sunumu (15-20 slayt) — savunma için
[ ] 30. Streamlit demo deploy (HuggingFace Spaces ya da Render)
        — jüriye canlı göster
[ ] 31. GitHub repo public et + README'ye sonuçları + demo linki ekle
[ ] 32. SİU 2026 / IEEE Sinyal İşleme Kurultayı için extended abstract (4 sayfa)
```

---

## 6. RİSK MATRİSİ

| Risk | Olasılık | Etki | Azaltım |
|------|----------|------|---------|
| mAP@50 < 60% (zayıf model) | Orta | Yüksek | Plan B/C — ek fine-tune, cal_focal debug, hyperparam search |
| Kaggle GPU kotası dolar (haftalık 30h) | Yüksek | Orta | Colab Pro yedek, ablation'ları sıraya koy, gece koştur |
| In-the-wild etiketleme yetiştirilmez | Orta | Orta | Plan B: SARDet-100K test seti yeter, in-the-wild "future work"a düşer |
| Boundary loss ek %1-2 getirmez | Düşük | Düşük | Tezde bug olarak değil, "boundary mask sınırlı geliştirme" diye yaz |
| cal_focal bug hâlâ var | Bilinmiyor | Yüksek | Hafta 1 günde 1 priorite — log'a bak, gerekirse 2 saatlik debug session |
| Tez son hafta yetiştirilmez | Düşük (eğer disiplinli yazılırsa) | Çok Yüksek | Hafta 5'ten itibaren her gün 2-3 saat yazma — paragraf bazlı |

---

## 7. KARAR AĞACI — Hafta 1 Sonu

```
                    ┌──────────────────┐
                    │  eval.py çalıştı │
                    │  mAP@50 = ?      │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    mAP ≥ 75            60 ≤ mAP < 75         mAP < 60
         │                   │                   │
    ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐
    │ PLAN A   │        │ PLAN B   │        │ PLAN C   │
    │ Ablation │        │ Fine-tune│        │ Debug    │
    │ + tez    │        │ 5-10 ep  │        │ + retrain│
    │ direk    │        │ img=640  │        │ 20 epoch │
    └──────────┘        └──────────┘        └──────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    Hafta 2'ye         Hafta 1.5'a        Hafta 2.5'a
    geç (planlı)       1-2 gün gecikme    4-5 gün gecikme
```

---

## 8. METRİK HEDEFLERİ (tezde "biz başardık" sayıları)

| Model | mAP@50 | mAP@50-95 | FPS (T4) | Tez claim |
|-------|--------|-----------|----------|-----------|
| YOLOv8 (optik) baseline | 75 | 50 | 60 | "Optik tek başına yetmiyor — bulutlu/gece düşer" |
| YOLOv8 (SAR) baseline | 70 | 45 | 65 | "SAR tek başına yetmiyor — düşük kontrast" |
| Concat early fusion | 78 | 53 | 55 | "Naif füzyon iyileştiriyor ama optimal değil" |
| Late fusion | 76 | 51 | 30 | "İki dedektör pahalı, kazanım az" |
| **Bizim (CMAFM+CAL)** | **82** | **58** | **40** | **"Ölçek-bilinçli füzyon + kamuflaj-bilinçli loss → SOTA"** |

→ Bu sayıların tutması için Hafta 1-2 sıkı koşmak lazım. mAP@50-95 hedefi en zor olanı; 50-95 5 puan altta kalsa bile "bizim modelimiz daha iyi" iddia edilebilir, çünkü kıyas baseline'lar aynı veri/aynı augmentation'la eğitildi.

---

## 9. HEMEN ŞİMDİ NE YAPILACAK (sırayla)

1. **Bu dosyayı oku** ✓ (oluşturuluyor)
2. **`runs/final (1).pt` üzerinde eval.py çalıştır** — komut yukarıda § 3 Gün 1
3. **Kaggle log'unu indir** (varsa `train_log_kaggle.txt`) — cal_focal=0 hâlâ var mı bak
4. **Sonucu Claude'a (yani bana) söyle** — Plan A/B/C kararı verelim, sıradaki adımı planlayalım

Sayılar tutmazsa kötü değil — hâlâ 13 hafta tez teslime varsa rahatlıkla iki tur eğitim daha sığar. Önemli olan **bilmek**, körlemesine ablation başlatmak değil.

---

**TL;DR:** Eğitim çalıştı, sayıyı henüz görmedik. Önce `eval.py`. Sonra baseline'lar. Sonra tez. Disiplinli koşarsan 8-10 hafta içinde her şey biter.
