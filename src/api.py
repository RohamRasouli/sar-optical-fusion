"""FastAPI servisi — model inference için REST endpoint.

Çalıştırma:
    pip install fastapi uvicorn[standard] python-multipart
    uvicorn src.api:app --host 0.0.0.0 --port 8080

Endpoint'ler:
    GET  /health       — sağlık kontrolü
    POST /detect       — optik+SAR çiftiyle tahmin
    GET  /info         — model bilgisi
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse


# ============================================================
# Global model state
# ============================================================

class ModelState:
    model = None
    config = None
    device = "cpu"
    class_names = ["aircraft", "ship", "vehicle", "bridge", "storage", "oiltank"]


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App başlangıcında modeli yükle."""
    import torch

    ckpt_path = os.getenv("SAR_FUSION_CKPT", "runs/final.pt")
    if not Path(ckpt_path).exists():
        print(f"  ! Checkpoint bulunamadı: {ckpt_path}")
        print(f"  ! Servis 'no model loaded' modunda başlıyor")
        yield
        return

    from .export import load_model_from_checkpoint

    try:
        print(f"Model yükleniyor: {ckpt_path}")
        state.model, state.config = load_model_from_checkpoint(ckpt_path)
        state.device = "cuda" if torch.cuda.is_available() else "cpu"
        state.model.to(state.device).eval()
        print(f"  ✓ Model hazır ({state.device})")
    except Exception as e:
        print(f"  ! Model yüklenemedi: {e}")

    yield

    # Cleanup
    print("API kapatılıyor.")


app = FastAPI(
    title="SAR + Optik Multimodal Tespit API",
    description="Sentinel-1 SAR ve Sentinel-2 optik görüntülerinde hedef tespiti.",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "device": state.device,
    }


@app.get("/info")
async def info():
    if state.model is None:
        raise HTTPException(503, "Model yüklü değil")
    return {
        "num_classes": state.config["model"]["num_classes"],
        "class_names": state.class_names,
        "img_size": state.config["model"]["img_size"],
        "optical_channels": state.config["model"]["channels"]["optical"],
        "sar_channels": state.config["model"]["channels"]["sar"],
        "device": state.device,
    }


@app.post("/detect")
async def detect(
    optical: UploadFile = File(..., description="Optik (RGB) görüntü"),
    sar: UploadFile = File(..., description="SAR görüntü (npy ya da tif)"),
    conf_threshold: float = 0.25,
    nms_iou: float = 0.6,
):
    """Optik+SAR çiftinden hedef tespit et."""
    if state.model is None:
        raise HTTPException(503, "Model yüklü değil")

    # Geçici dosyalara yaz
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=Path(optical.filename).suffix, delete=False) as f:
        f.write(await optical.read())
        opt_path = f.name
    with tempfile.NamedTemporaryFile(suffix=Path(sar.filename).suffix, delete=False) as f:
        f.write(await sar.read())
        sar_path = f.name

    try:
        from .predict import predict_single

        detections = predict_single(
            state.model, opt_path, sar_path,
            img_size=state.config["model"]["img_size"],
            device=state.device,
            conf_thr=conf_threshold,
            nms_iou=nms_iou,
        )

        # Sınıf isimlerini ekle
        for d in detections:
            d["class_name"] = state.class_names[d["class"]] \
                if 0 <= d["class"] < len(state.class_names) else f"class_{d['class']}"

        return JSONResponse({
            "num_detections": len(detections),
            "detections": detections,
            "model_version": "0.1.0",
        })

    except Exception as e:
        raise HTTPException(500, f"Inference hatası: {e}")

    finally:
        for p in [opt_path, sar_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


@app.post("/detect_geo")
async def detect_geo(
    optical: UploadFile = File(...),
    sar: UploadFile = File(...),
    bounds_n: float = 0.0,
    bounds_s: float = 0.0,
    bounds_e: float = 0.0,
    bounds_w: float = 0.0,
    conf_threshold: float = 0.25,
):
    """Coğrafi sınırları olan görüntüde tespit yap, GeoJSON dön.

    Args:
        bounds_n, bounds_s, bounds_e, bounds_w: WGS84 lat/lon sınırları
    """
    # Önce normal tespit
    response = await detect(optical, sar, conf_threshold=conf_threshold)
    data = response.body
    import json
    detections = json.loads(data)["detections"]

    # GeoJSON üret
    img_size = state.config["model"]["img_size"]
    features = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        # Piksel -> normalize -> coğrafi
        nx1 = x1 / img_size
        nx2 = x2 / img_size
        ny1 = y1 / img_size
        ny2 = y2 / img_size
        lon1 = bounds_w + nx1 * (bounds_e - bounds_w)
        lon2 = bounds_w + nx2 * (bounds_e - bounds_w)
        lat1 = bounds_n - ny1 * (bounds_n - bounds_s)
        lat2 = bounds_n - ny2 * (bounds_n - bounds_s)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon1, lat1], [lon2, lat1],
                    [lon2, lat2], [lon1, lat2],
                    [lon1, lat1],
                ]],
            },
            "properties": {
                "class": d["class"],
                "class_name": d.get("class_name", ""),
                "score": d["score"],
            },
        })

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
