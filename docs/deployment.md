# Dağıtım (Deployment) Rehberi

## TRL ve Hedef Cihazlar

| Cihaz                  | Çıkarım hızı (640×640) | Bellek    | Güç tüketimi |
|------------------------|-----------------------:|----------:|-------------:|
| RTX 3090 (PyTorch fp32) | ~60 FPS                | 4 GB      | 350 W        |
| RTX 3090 (TRT fp16)     | ~180 FPS               | 2 GB      | 350 W        |
| Jetson Orin Nano 8GB    | ~25 FPS (TRT fp16)      | 4 GB      | 15 W         |
| Jetson Orin NX 16GB     | ~45 FPS (TRT fp16)      | 8 GB      | 25 W         |
| Jetson Xavier NX        | ~18 FPS (TRT INT8)      | 6 GB      | 20 W         |

## ONNX Export

```bash
python -m src.export --checkpoint runs/final.pt --format onnx --output model.onnx
```

Bu, `dynamic batch` destekli ONNX dosyası üretir. CPU üzerinde onnxruntime ile
test edilebilir:

```python
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"optical": opt_np, "sar": sar_np})
```

## Jetson Orin'e Kurulum

### 1. JetPack 5.1.2+ kurulu olmalı

Jetson Orin'inde `sudo apt show nvidia-jetpack` ile kontrol et.

### 2. PyTorch'lu container indir

```bash
# NVIDIA NGC'den L4T uyumlu container
docker pull nvcr.io/nvidia/l4t-pytorch:r35.4.1-pth2.1-py3
```

### 3. Modeli ONNX olarak Jetson'a aktar

Host bilgisayarda:
```bash
python -m src.export --checkpoint runs/final.pt --format onnx --output model.onnx
scp model.onnx jetson@jetson-ip:/home/jetson/
```

### 4. Jetson'da TensorRT engine üret

```bash
# Jetson'da
trtexec --onnx=model.onnx --saveEngine=model.engine \
    --fp16 --workspace=4096 \
    --minShapes=optical:1x3x640x640,sar:1x2x640x640 \
    --optShapes=optical:1x3x640x640,sar:1x2x640x640 \
    --maxShapes=optical:4x3x640x640,sar:4x2x640x640
```

### 5. INT8 nicemleme (en hızlı)

```bash
# Calibration verisi gerekir (~500 örnek temsili görüntü)
trtexec --onnx=model.onnx --saveEngine=model_int8.engine \
    --int8 --calib=calibration.bin --workspace=4096 \
    [shape parametreleri yukarıdaki gibi]
```

INT8 ile mAP düşüşü tipik %0.5-1, hız 2-3 kat artar.

## Docker Imajı

```dockerfile
# Dockerfile.jetson
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime

RUN apt-get update && apt-get install -y \
    python3-pip libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    fastapi==0.104 \
    uvicorn[standard]==0.24 \
    opencv-python==4.8.1.78 \
    numpy==1.24.4 \
    rasterio==1.3.8 \
    pillow==10.0.0

COPY model.engine /app/model.engine
COPY src/ /app/src/
WORKDIR /app

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build:
```bash
docker build -f Dockerfile.jetson -t sar-fusion:jetson .
docker run --runtime nvidia -p 8080:8080 sar-fusion:jetson
```

## REST API Servisi

`src/api.py` (örnek FastAPI servisi — kendi durumuna göre genişlet):

```python
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

@app.post("/detect")
async def detect(optical: UploadFile = File(...), sar: UploadFile = File(...)):
    # Resimleri oku
    opt_bytes = await optical.read()
    sar_bytes = await sar.read()
    # ... preprocessing ...
    # ... model inference ...
    return {
        "detections": [
            {"class": "ship", "score": 0.92, "bbox": [120, 340, 180, 410]},
            # ...
        ]
    }
```

## Operasyonel Boru Hattı

```
   [Sentinel-1/2 alımı]
          ↓
   ┌──────────────────┐
   │ Pre-processing   │  GDAL ile co-registration, tile bölme
   └────────┬─────────┘
            ↓
   ┌──────────────────┐
   │ Inference (TRT)  │  Bizim model
   └────────┬─────────┘
            ↓
   ┌──────────────────┐
   │ Post-processing  │  NMS + coğrafi koordinat dönüşümü (lat/lon)
   └────────┬─────────┘
            ↓
   ┌──────────────────┐
   │ Görselleştirme   │  Web UI (Leaflet) + REST API
   │  + Operatör UI   │  Operatör son kararı verir (HITL)
   └──────────────────┘
```

## Güvenlik ve Etik

- Tüm tahminler güven skoru ile birlikte sunulur
- Operatör son kararı verir (Human-In-The-Loop)
- Sivil hedeflere (hastane, okul) flag mekanizması eklenebilir
- Model gizlilik: ONNX dosyası dağıtılırken obfuscation/encryption düşünülebilir
