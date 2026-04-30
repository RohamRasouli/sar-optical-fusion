# Mimari Detay

## Tam Boru Hattı

```
            ┌──────────────────────────────────────────────┐
            │  Optical (3, H, W)              SAR (2, H, W) │
            └──────────────┬─────────────┬─────────────────┘
                           ▼             ▼
                ┌──────────────┐  ┌──────────────┐
                │ Optic Encoder│  │ SAR Encoder  │   <- DualStreamEncoder
                │ CSPDarknet   │  │ CSPDarknet   │
                └──────┬───────┘  └──────┬───────┘
                       │                  │
        ┌──────────────┼──────────────────┼──────────────┐
        │       3 ölçek özellikleri     3 ölçek özellikleri │
        │       [128, 256, 512]ch      [128, 256, 512]ch │
        └──────────────┬─────────┬───────┴───────────────┘
                       ▼         ▼
                ┌──────────────────────────┐
                │  MultiScaleCMAFM         │   <- 3× CMAFM bloğu
                │  - Cross-attention iki   │      her ölçekte ayrı
                │    yönlü                 │
                │  - Window-tabanlı        │
                │  - Sigmoid gating        │
                └────────────┬─────────────┘
                             │
                       3 fused özellik
                             │
                             ▼
                ┌──────────────────────────┐
                │  PAN-FPN Neck            │
                │  - Top-down + Bottom-up  │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  Decoupled Detection Head│
                │  - cls branch (BCE)      │
                │  - reg branch (DFL)      │
                │  - anchor-free          │
                └────────────┬─────────────┘
                             │
                             ▼
                  Bounding boxes + skorlar


    Ek (sadece eğitimde, consistency loss için):
        Optic feats ──▶ aux_neck_opt ──▶ aux_head_opt
        SAR feats   ──▶ aux_neck_sar ──▶ aux_head_sar
```

## Parametre Bütçesi (varsayılan ayar — base.yaml)

| Bileşen          | Parametre | Tahmini FLOPs (640×640) |
|------------------|----------:|------------------------:|
| Optic Encoder    |  ~5.0 M   |  ~12 GFLOPs            |
| SAR Encoder      |  ~5.0 M   |  ~12 GFLOPs            |
| 3× CMAFM         |  ~1.8 M   |   ~3 GFLOPs            |
| PAN-FPN Neck     |  ~2.0 M   |   ~5 GFLOPs            |
| Detection Head   |  ~2.1 M   |   ~6 GFLOPs            |
| Aux Heads (eğt.) |  ~4.0 M   |   ~10 GFLOPs (eğt only) |
| **Toplam (inf)** | **~16 M** | **~38 GFLOPs**         |

YOLOv8-s (~11M) ile YOLOv8-m (~25M) arasında konumlanır.

## CMAFM Matematiği

Verilen iki özellik haritası `F_opt`, `F_sar ∈ R^(C×H×W)`:

```
1) Q, K, V projeksiyonu:
   Q_opt = Conv1x1_q_opt(F_opt)
   K_opt = Conv1x1_k_opt(F_opt)
   V_opt = Conv1x1_v_opt(F_opt)
   (SAR için simetrik)

2) Pencere bölme: w × w pencerelere böl

3) İki yönlü çapraz dikkat (her pencerede):
   F_o2s = MultiHead(Q_opt, K_sar, V_sar)
   F_s2o = MultiHead(Q_sar, K_opt, V_opt)

4) Sigmoid gating:
   sigma_opt = sigmoid(Conv1x1(Concat(F_opt, F_o2s)))
   sigma_sar = sigmoid(Conv1x1(Concat(F_sar, F_s2o)))

5) Birleştirme:
   F_opt' = F_opt + sigma_opt * F_o2s
   F_sar' = F_sar + sigma_sar * F_s2o
   F_fused = Conv1x1(Concat(F_opt', F_sar'))
```

## CamouflageAware Loss

```
L_total = w_box * L_CIoU + w_cls * L_BCE + w_dfl * L_DFL
       + λ_focal   * L_DynamicFocal
       + λ_bound   * L_BoundaryAware       (opsiyonel — GT mask gerek)
       + λ_consist * L_CrossModalConsistency

L_DynamicFocal:
  γ_dynamic = γ_base · (1 + β · D(x))
  D(x): 1 - max(softmax(model(x)))   (model belirsizliği)

L_BoundaryAware:
  Sobel(pred) - Sobel(GT) L1 hatası, sınır şeridinde

L_CrossModalConsistency:
  KL(P_fused || stop_grad(P_opt)) + KL(P_fused || stop_grad(P_sar))
```

## Eğitim İpuçları

- **Donmuş backbone**: İlk 10 epoch için `freeze.optical_backbone_epochs: 10` ile pre-trained ağırlıkları koru, sonra unfreeze
- **Mixed precision**: Bellek tasarrufu için `amp: true`, ~%40 daha az VRAM
- **Gradient accumulation**: Küçük batch'ler için `grad_accum_steps: 2` ile etkili batch=32 simüle edilir
- **EMA**: Ultralytics standartı; `ema.decay: 0.9999`
- **Warmup**: İlk 3 epoch lineer lr artışı, sonra cosine annealing
