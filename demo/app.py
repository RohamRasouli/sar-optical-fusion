"""Streamlit Demo UI — paydaşlara/operatörlere göstermek için.

Çalıştırma:
    pip install streamlit
    streamlit run demo/app.py

Özellikler:
  - Optik + SAR çiftini sürükle-bırak yükleme
  - Model tahmini (bbox + sınıf + skor)
  - Modalite gating haritası görselleştirme (CMAFM σ_opt, σ_sar)
  - Stres senaryoları (bulut, gece, kamuflaj) önizleme
  - Sonuçları JSON/GeoJSON olarak indirme
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="SAR + Optik Multimodal Tespit",
    page_icon="🛰️",
    layout="wide",
)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Ayarlar")
st.sidebar.markdown("### Model")
checkpoint_path = st.sidebar.text_input(
    "Checkpoint yolu",
    value="runs/final (1).pt",
    help="Eğitilmiş model dosyası",
)
device_option = st.sidebar.selectbox("Cihaz", ["cuda", "cpu"], index=0)
conf_threshold = st.sidebar.slider("Güven eşiği", 0.0, 1.0, 0.25, 0.05)
nms_iou = st.sidebar.slider("NMS IoU", 0.1, 0.9, 0.6, 0.05)

st.sidebar.markdown("### Stres Senaryosu")
stress_preset = st.sidebar.selectbox(
    "Önişleme stres",
    ["clean", "cloud_light", "cloud_medium", "cloud_heavy",
     "night_light", "night_dark", "camo_only",
     "cloud_camo", "night_camo", "all_combined"],
    index=0,
)

# ============================================================
# Ana sayfa
# ============================================================

st.title("🛰️ SAR + Optik Multimodal Hedef Tespiti")
st.markdown("""
**Çift akımlı çapraz-modal füzyon ağı** ile düşük görünürlüklü ve kamuflajlı
hedefleri tespit eden bir derin öğrenme demosu.
Sentinel-1 SAR ve Sentinel-2 optik görüntü çiftleri kabul eder.
""")

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("📷 Optik (Sentinel-2 RGB)")
    optical_file = st.file_uploader(
        "Görüntü dosyası (PNG, JPG, TIF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="optical",
    )
    if optical_file:
        from PIL import Image
        img = Image.open(optical_file).convert("RGB")
        st.image(img, use_column_width=True)
        st.caption(f"Boyut: {img.size}")

with col_r:
    st.subheader("📡 SAR (Sentinel-1 VV+VH)")
    sar_file = st.file_uploader(
        "Görüntü dosyası (TIF, NPY, PNG, JPG)",
        type=["tif", "tiff", "npy", "png", "jpg", "jpeg"],
        key="sar",
    )
    if sar_file:
        if sar_file.name.endswith(".npy"):
            arr = np.load(sar_file)
            if arr.ndim == 3 and arr.shape[0] in (1, 2):
                disp = arr[0] if arr.shape[0] == 1 else arr.mean(0)
            else:
                disp = arr
            disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
            st.image(disp, use_column_width=True, clamp=True)
        else:
            from PIL import Image
            img = Image.open(sar_file)
            st.image(img, use_column_width=True)

# ============================================================
# Inference
# ============================================================

st.markdown("---")

if optical_file and sar_file:
    if st.button("🎯 Tespit Çalıştır", type="primary", use_container_width=True):
        with st.spinner("Model çalışıyor..."):
            try:
                import torch
                from src.predict import predict_single
                from src.export import load_model_from_checkpoint

                import tempfile
                tmp_dir = Path(tempfile.mkdtemp(prefix="sar_fusion_demo_"))
                tmp_dir.mkdir(exist_ok=True)
                opt_path = tmp_dir / optical_file.name
                sar_path = tmp_dir / sar_file.name
                with open(opt_path, "wb") as f:
                    f.write(optical_file.getbuffer())
                with open(sar_path, "wb") as f:
                    f.write(sar_file.getbuffer())

                model, cfg = load_model_from_checkpoint(checkpoint_path)
                device = device_option if torch.cuda.is_available() else "cpu"
                model.to(device)

                detections = predict_single(
                    model, str(opt_path), str(sar_path),
                    img_size=cfg["model"]["img_size"], device=device,
                    conf_thr=conf_threshold, nms_iou=nms_iou,
                )

                st.success(f"✓ {len(detections)} hedef tespit edildi")

                # Görselleştirme
                from src.utils.visualization import draw_predictions

                col_pred, col_info = st.columns([2, 1])
                with col_pred:
                    st.subheader("Tahminler")
                    fig = draw_predictions(
                        np.array(Image.open(opt_path)), detections,
                        class_names=["uçak", "gemi", "araç", "köprü", "depo", "tank"],
                        score_threshold=conf_threshold,
                    )
                    if fig:
                        st.pyplot(fig)

                with col_info:
                    st.subheader("Detaylar")
                    st.dataframe(
                        [
                            {
                                "Sınıf": d["class"],
                                "Skor": f"{d['score']:.3f}",
                                "Bbox": [round(x, 1) for x in d["bbox"]],
                            }
                            for d in detections
                        ],
                        use_container_width=True,
                    )

                # JSON indirme
                st.download_button(
                    "📥 Tahminleri JSON olarak indir",
                    data=json.dumps({"detections": detections}, indent=2),
                    file_name="detections.json",
                    mime="application/json",
                )

            except FileNotFoundError as e:
                st.error(f"Checkpoint bulunamadı: {e}")
            except Exception as e:
                st.error(f"Hata: {e}")
                st.exception(e)
else:
    st.info("Hem optik hem de SAR dosyalarını yükleyin.")

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption(
    "Geliştirici: Roham R. Kerahroudi · "
    "Sakarya Üniversitesi Bilgisayar Müh. · "
    "Danışman: Prof. Dr. Cemil Öz"
)
