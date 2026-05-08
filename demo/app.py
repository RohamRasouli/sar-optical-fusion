"""SAR + Optik Füzyon — Taktik Hedef Tespit Sistemi (Demo)

Çalıştırma:
    streamlit run demo/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="FUSION-1 | Taktik Hedef Tespit",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tema CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

/* ── Zemin: derin siyah + ızgara doku ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #020507 !important;
    background-image:
        linear-gradient(rgba(0,255,100,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,100,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    color: #5aff8a !important;
}

/* Üst bant tarama animasyonu */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(0,255,100,0) 20%,
        rgba(0,255,100,0.9) 50%,
        rgba(0,255,100,0) 80%,
        transparent 100%);
    animation: topscan 4s ease-in-out infinite;
    z-index: 9999;
}
@keyframes topscan {
    0%   { transform: translateX(-100%); opacity: 0.6; }
    100% { transform: translateX(100%);  opacity: 0.6; }
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #03080e !important;
    border-right: 1px solid rgba(0,255,100,0.12) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #3aaa60 !important; font-family: 'Share Tech Mono', monospace !important; }

/* Tüm yazılar */
p, span, div, li, td, th { font-family: 'Share Tech Mono', monospace !important; }

/* ── Header ── */
.hdr {
    background: linear-gradient(180deg, #030f06 0%, #020507 100%);
    border: 1px solid rgba(0,255,100,0.2);
    border-top: 2px solid #00ff64;
    padding: 22px 32px 18px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hdr::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,255,100,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hdr-eyebrow {
    font-size: 9px; letter-spacing: 4px; color: #2a6640;
    text-transform: uppercase; margin-bottom: 6px;
}
.hdr-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 30px; font-weight: 900;
    color: #00ff64;
    letter-spacing: 6px;
    text-shadow: 0 0 30px rgba(0,255,100,0.5), 0 0 60px rgba(0,255,100,0.2);
    margin: 0 0 6px;
}
.hdr-sub {
    font-size: 10px; letter-spacing: 3px; color: #2a6640;
}
.badge {
    display: inline-block;
    border: 1px solid rgba(0,255,100,0.3);
    padding: 2px 10px;
    font-size: 9px; letter-spacing: 3px;
    color: #00cc50; margin-right: 10px; margin-bottom: 8px;
    text-transform: uppercase;
}

/* ── Metrik kartlar ── */
.mc {
    background: linear-gradient(160deg, #030f07 0%, #020508 100%);
    border: 1px solid rgba(0,255,100,0.15);
    border-radius: 2px;
    padding: 18px 16px 14px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.mc::before {
    content: "";
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 60%; height: 1px;
    background: linear-gradient(90deg, transparent, #00ff64, transparent);
}
.mc-val {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 38px; font-weight: 900;
    color: #00ff64; line-height: 1;
    text-shadow: 0 0 20px rgba(0,255,100,0.4);
}
.mc-lbl {
    font-size: 8px; letter-spacing: 3px; color: #1a5530;
    text-transform: uppercase; margin-top: 8px;
}
.mc.red .mc-val  { color: #ff4040; text-shadow: 0 0 20px rgba(255,64,64,0.5); }
.mc.red::before  { background: linear-gradient(90deg, transparent, #ff4040, transparent); }
.mc.cyan .mc-val { color: #00e5ff; text-shadow: 0 0 20px rgba(0,229,255,0.4); }
.mc.cyan::before { background: linear-gradient(90deg, transparent, #00e5ff, transparent); }
.mc.amber .mc-val{ color: #ffaa00; text-shadow: 0 0 20px rgba(255,170,0,0.4); }
.mc.amber::before{ background: linear-gradient(90deg, transparent, #ffaa00, transparent); }

/* ── Panel başlıkları ── */
.panel-hdr {
    font-size: 9px; letter-spacing: 4px; color: #00ff64;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,255,100,0.15);
    padding-bottom: 8px; margin-bottom: 10px;
}

/* ── Tespit tablosu ── */
.dtbl { width:100%; border-collapse:collapse; }
.dtbl th {
    font-size: 8px; letter-spacing: 3px; color: #1a5530;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,255,100,0.12);
    padding: 10px 14px; text-align: left;
}
.dtbl td {
    font-size: 12px; color: #3aaa60;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(0,255,100,0.05);
}
.dtbl tr:hover td { background: rgba(0,255,100,0.03); }
.t-hi  { color: #ff4040 !important; font-weight: bold; }
.t-med { color: #ffaa00 !important; font-weight: bold; }
.t-lo  { color: #00ff64 !important; }

/* ── Buton ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #031a09 0%, #052b10 100%) !important;
    color: #00ff64 !important;
    border: 1px solid rgba(0,255,100,0.4) !important;
    border-radius: 2px !important;
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 5px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 16px !important;
    text-shadow: 0 0 12px rgba(0,255,100,0.6) !important;
    box-shadow: inset 0 0 20px rgba(0,255,100,0.05) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #052b10 0%, #0a4a1a 100%) !important;
    border-color: rgba(0,255,100,0.7) !important;
    box-shadow: 0 0 25px rgba(0,255,100,0.25), inset 0 0 30px rgba(0,255,100,0.08) !important;
}
div[data-testid="stButton"] > button:disabled {
    opacity: 0.25 !important;
    cursor: not-allowed !important;
}

/* ── File uploader — minimal temiz görünüm ── */
[data-testid="stFileUploader"] { background: transparent !important; }

/* Dropzone dış kapsayıcı */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,255,100,0.02) !important;
    border: 1px dashed rgba(0,255,100,0.3) !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0,255,100,0.6) !important;
    background: rgba(0,255,100,0.04) !important;
}

/* İçteki icon + "Drag and drop" text bloğunu gizle — sadece buton kalsın */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}

/* Upload butonu — tam genişlik */
[data-testid="stFileUploaderDropzone"] button {
    width: 100% !important;
    background: #031a09 !important;
    color: #00ff64 !important;
    border: 1px solid rgba(0,255,100,0.35) !important;
    border-radius: 3px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 3px !important;
    padding: 10px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #052b10 !important;
    border-color: rgba(0,255,100,0.7) !important;
    box-shadow: 0 0 12px rgba(0,255,100,0.15) !important;
}
/* Butondaki material ikon span'ını gizle — sadece label kalır */
[data-testid="stFileUploaderDropzone"] button span:first-child {
    display: none !important;
}

/* Yüklenen dosya satırı */
[data-testid="stFileUploaderFile"] {
    background: rgba(0,255,100,0.04) !important;
    border: 1px solid rgba(0,255,100,0.15) !important;
    border-radius: 3px !important;
    padding: 6px 10px !important;
    margin-top: 6px !important;
}
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p {
    color: #00ff64 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}
/* Silme butonu */
[data-testid="stFileUploaderDeleteBtn"] button {
    color: #ff4040 !important;
    background: transparent !important;
    border: none !important;
    width: auto !important;
    padding: 2px 6px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #00ff64 !important;
}
[data-testid="stSlider"] * { color: #3aaa60 !important; }

/* ── Selectbox / text input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background: #030f07 !important;
    border-color: rgba(0,255,100,0.2) !important;
    color: #3aaa60 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] span { color: #3aaa60 !important; }

/* ── Status bar ── */
.status-bar {
    font-size: 9px; letter-spacing: 2px; color: #1a5530;
    border-top: 1px solid rgba(0,255,100,0.1);
    padding: 10px 0 4px; margin-top: 30px;
    text-transform: uppercase;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #020507; }
::-webkit-scrollbar-thumb { background: rgba(0,255,100,0.2); }

/* Streamlit varsayılan beyazları kapat */
h1,h2,h3,h4,h5,h6 { color: #00ff64 !important; font-family: 'Orbitron', sans-serif !important; }
[data-testid="stMarkdownContainer"] p { color: #3aaa60 !important; }
.stCaption p { color: #1a5530 !important; font-size: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Sabitler ──────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["UÇAK", "GEMİ", "ARAÇ", "KÖPRÜ", "DEPO", "TANK"]
CLASS_ICONS  = ["✈",    "⛴",    "🚗",   "🌉",    "🏭",   "🎯"]
CLASS_COLORS = [
    (255, 80,  80),   # uçak   — kırmızı
    (80,  200, 255),  # gemi   — mavi
    (255, 200, 50),   # araç   — sarı
    (160, 100, 255),  # köprü  — mor
    (255, 140, 50),   # depo   — turuncu
    (80,  255, 130),  # tank   — yeşil
]


def threat_level(score: float) -> str:
    if score >= 0.70: return "HIGH"
    if score >= 0.40: return "MED"
    return "LOW"


def draw_boxes_pil(img_np: np.ndarray, detections: list,
                   src_size: int, conf_thr: float) -> np.ndarray:
    """PIL ile bbox çiz. src_size: modelin çıkardığı koordinat uzayı (640)."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = img_np.shape[:2]
    sx, sy = w / src_size, h / src_size
    pil = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil, "RGBA")

    for d in detections:
        if d["score"] < conf_thr:
            continue
        x1, y1, x2, y2 = [c for c in d["bbox"]]
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        cls = d["class"] % len(CLASS_COLORS)
        r, g, b = CLASS_COLORS[cls]
        # Glow efekti — kalın yarı saydam dış
        draw.rectangle([x1-2, y1-2, x2+2, y2+2], outline=(r, g, b, 80), width=4)
        # Ana kutu
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 230), width=2)
        # Köşe işaretleri (taktik crosshair stili)
        l = min(14, (x2 - x1) / 3)
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([cx, cy, cx + dx*l, cy], fill=(r, g, b, 255), width=2)
            draw.line([cx, cy, cx, cy + dy*l], fill=(r, g, b, 255), width=2)
        # Etiket
        label = f"{CLASS_ICONS[cls]} {CLASS_NAMES[cls]}  {d['score']:.2f}"
        lx, ly = x1, max(y1 - 18, 2)
        draw.rectangle([lx, ly, lx + len(label)*7 + 6, ly + 16],
                        fill=(r, g, b, 200))
        draw.text((lx + 3, ly + 2), label, fill=(0, 0, 0, 255))

    return np.array(pil).astype(np.float32) / 255.0


# ── Başlık ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">
    <span class="badge">UNCLASSIFIED // DEMO</span>
    <span class="badge">BUILD 2.1</span>
    <span class="badge">CMAFM ARCH</span>
    <span class="badge">M4-SAR · 56K</span>
  </div>
  <div class="hdr-title">◈ FUSION-1 TACTICAL DETECTION SYSTEM</div>
  <div class="hdr-sub">
    DUAL-STREAM CROSS-MODAL ATTENTION NETWORK &nbsp;·&nbsp;
    SENTINEL-1 / SENTINEL-2 FUSION &nbsp;·&nbsp;
    SAR + OPTICAL MULTIMODAL INFERENCE
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("```\n■ SİSTEM KONFİGÜRASYONU\n```")

    checkpoint_path = st.text_input(
        "CHECKPOINT",
        value="runs/final.pt",
        help="Model ağırlık dosyası",
    )
    device_option = st.selectbox("CIHAZ", ["cuda", "cpu"])
    st.markdown("---")
    st.markdown("```\n■ ALGILAMA PARAMETRELERİ\n```")
    conf_threshold = st.slider("GÜVEN EŞİĞİ", 0.0, 1.0, 0.25, 0.05)
    nms_iou        = st.slider("NMS IoU", 0.1, 0.9, 0.6, 0.05)
    img_size_ui    = st.selectbox("GİRDİ BOYUTU", [640, 416, 512], index=0)
    st.markdown("---")
    st.markdown("```\n■ GÖRSELLEŞTİRME\n```")
    show_sar_boxes  = st.checkbox("SAR üzerine bbox göster", value=True)
    show_opt_boxes  = st.checkbox("Optik üzerine bbox göster", value=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Courier New;font-size:9px;color:#2a5a3a;letter-spacing:1px;line-height:1.8'>
    SAKARYA ÜNİVERSİTESİ<br>
    BİLGİSAYAR MÜHENDİSLİĞİ<br>
    BİTİRME PROJESİ 2026<br><br>
    ÖĞRENCİ: R. RASOULI<br>
    DANIŞMAN: PROF. DR. C. ÖZ
    </div>
    """, unsafe_allow_html=True)

# ── Metrik kartları (başlangıç) ───────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
metrics_placeholder = {
    "total": m1.empty(),
    "conf":  m2.empty(),
    "class": m3.empty(),
    "time":  m4.empty(),
}

def render_metrics(total="-", conf="-", cls="-", ms="-", alert=False):
    metrics_placeholder["total"].markdown(f"""
    <div class="mc {'red' if alert else ''}">
        <div class="mc-val">{total}</div>
        <div class="mc-lbl">HEDEF TESPİT</div>
    </div>""", unsafe_allow_html=True)
    metrics_placeholder["conf"].markdown(f"""
    <div class="mc amber">
        <div class="mc-val">{conf}</div>
        <div class="mc-lbl">ORT. GÜVEN</div>
    </div>""", unsafe_allow_html=True)
    metrics_placeholder["class"].markdown(f"""
    <div class="mc cyan">
        <div class="mc-val">{cls}</div>
        <div class="mc-lbl">SINIF SAYISI</div>
    </div>""", unsafe_allow_html=True)
    metrics_placeholder["time"].markdown(f"""
    <div class="mc">
        <div class="mc-val">{ms}</div>
        <div class="mc-lbl">İŞLEM ms</div>
    </div>""", unsafe_allow_html=True)

render_metrics()

st.markdown("<br>", unsafe_allow_html=True)

# ── Görüntü yükleme ───────────────────────────────────────────────────────────
col_opt, col_sar = st.columns(2)

with col_opt:
    st.markdown('<div class="panel-hdr">▸ OPTİK KANAL — SENTINEL-2 RGB</div>',
                unsafe_allow_html=True)
    optical_file = st.file_uploader(
        "OPTİK GÖRÜNTÜ", type=["png","jpg","jpeg","tif","tiff"],
        key="optical", label_visibility="collapsed")
    opt_preview = st.empty()
    if optical_file:
        from PIL import Image
        img_opt = Image.open(optical_file).convert("RGB")
        opt_preview.image(img_opt, use_container_width=True,
                          caption=f"RES: {img_opt.size[0]}×{img_opt.size[1]}px")

with col_sar:
    st.markdown('<div class="panel-hdr">▸ SAR KANAL — SENTINEL-1 VV/VH</div>',
                unsafe_allow_html=True)
    sar_file = st.file_uploader(
        "SAR GÖRÜNTÜ", type=["tif","tiff","npy","png","jpg","jpeg"],
        key="sar", label_visibility="collapsed")
    sar_preview = st.empty()
    if sar_file:
        from PIL import Image as PILImage
        if sar_file.name.endswith(".npy"):
            arr = np.load(sar_file)
            disp = arr[0] if (arr.ndim==3 and arr.shape[0] in (1,2)) else arr
            disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
            sar_preview.image(disp, use_container_width=True, clamp=True,
                              caption="SAR (VV kanal)")
        else:
            img_sar = PILImage.open(sar_file).convert("RGB")
            sar_preview.image(img_sar, use_container_width=True,
                              caption=f"RES: {img_sar.size[0]}×{img_sar.size[1]}px")

# ── Tarama butonu ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
scan_col, _ = st.columns([3, 1])
with scan_col:
    scan_btn = st.button(
        "◈  HEDEF TARAMASI BAŞLAT  ◈",
        use_container_width=True,
        disabled=not (optical_file and sar_file),
    )

if not (optical_file and sar_file):
    st.markdown(
        '<p style="font-family:Courier New;font-size:11px;color:#2a5a3a;'
        'letter-spacing:2px;text-align:center">[ OPTİK VE SAR GÖRÜNTÜSÜ YÜKLEYİN ]</p>',
        unsafe_allow_html=True)

# ── Inference ─────────────────────────────────────────────────────────────────
if scan_btn:
    status_ph = st.empty()
    status_ph.markdown(
        '<p style="font-family:Courier New;color:#4ade80;font-size:11px;'
        'letter-spacing:3px;animation:scanline 1s infinite">⣿ TARAMA DEVAM EDİYOR...</p>',
        unsafe_allow_html=True)

    try:
        import tempfile
        import torch
        from PIL import Image
        from src.export import load_model_from_checkpoint
        from src.predict import predict_single

        # Dosyaları diske yaz
        tmp = Path(tempfile.mkdtemp(prefix="fusion_demo_"))
        opt_path = tmp / f"opt_{optical_file.name}"
        sar_path = tmp / f"sar_{sar_file.name}"
        opt_path.write_bytes(optical_file.getbuffer())
        sar_path.write_bytes(sar_file.getbuffer())

        # Model yükle
        device = device_option if torch.cuda.is_available() else "cpu"
        model, cfg = load_model_from_checkpoint(checkpoint_path)
        model.to(device)
        effective_img_size = cfg["model"].get("img_size", img_size_ui)

        # Inference
        t0 = time.perf_counter()
        detections = predict_single(
            model, str(opt_path), str(sar_path),
            img_size=effective_img_size, device=device,
            conf_thr=conf_threshold, nms_iou=nms_iou,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        status_ph.empty()

        # ── Metrik güncelle ────────────────────────────────────────────────
        n = len(detections)
        avg_conf = f"{np.mean([d['score'] for d in detections]):.2f}" if n else "-"
        n_cls = len({d['class'] for d in detections})
        render_metrics(
            total=str(n),
            conf=avg_conf,
            cls=str(n_cls) if n else "-",
            ms=str(elapsed_ms),
            alert=(n > 0),
        )

        # ── Görseller: bbox'ları her iki görüntüye çiz ────────────────────
        img_opt_np = np.array(Image.open(opt_path).convert("RGB")).astype(np.float32) / 255.0

        if sar_file.name.endswith(".npy"):
            arr = np.load(str(sar_path))
            disp = arr[0] if (arr.ndim==3 and arr.shape[0] in (1,2)) else arr
            disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
            img_sar_np = np.stack([disp]*3, axis=-1).astype(np.float32)
        else:
            img_sar_np = np.array(Image.open(sar_path).convert("RGB")).astype(np.float32) / 255.0

        if n > 0 and show_opt_boxes:
            img_opt_out = draw_boxes_pil(img_opt_np, detections, effective_img_size, conf_threshold)
        else:
            img_opt_out = img_opt_np

        if n > 0 and show_sar_boxes:
            img_sar_out = draw_boxes_pil(img_sar_np, detections, effective_img_size, conf_threshold)
        else:
            img_sar_out = img_sar_np

        opt_preview.image(img_opt_out, use_container_width=True,
                          caption=f"OPTİK — {n} hedef")
        sar_preview.image(img_sar_out, use_container_width=True,
                          caption=f"SAR — {n} hedef")

        # ── Tespit tablosu ────────────────────────────────────────────────
        if n > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-hdr" style="margin-top:24px">◈ TESPİT RAPORU</div>',
                unsafe_allow_html=True)

            rows = ""
            for i, d in enumerate(sorted(detections, key=lambda x: -x["score"])):
                cls   = d["class"] % len(CLASS_NAMES)
                name  = CLASS_NAMES[cls]
                icon  = CLASS_ICONS[cls]
                score = d["score"]
                tl    = threat_level(score)
                tl_cls = {"HIGH":"t-hi","MED":"t-med","LOW":"t-lo"}[tl]
                x1,y1,x2,y2 = [f"{v:.0f}" for v in d["bbox"]]
                r,g,b = CLASS_COLORS[cls]
                bar_w = int(score * 90)
                rows += f"""
                <tr>
                  <td style="color:rgba({r},{g},{b},1);letter-spacing:2px">{icon}&nbsp;{name}</td>
                  <td>
                    <div style="background:rgba(0,255,100,0.05);border:1px solid rgba(0,255,100,0.1);
                         border-radius:1px;width:100px;height:6px;overflow:hidden;display:inline-block;
                         vertical-align:middle;margin-right:8px">
                      <div style="background:rgba({r},{g},{b},0.85);width:{bar_w}px;height:6px"></div>
                    </div>{score:.3f}
                  </td>
                  <td class="{tl_cls}" style="letter-spacing:2px">▲ {tl}</td>
                  <td style="color:#1a5530">[{x1},{y1}] → [{x2},{y2}]</td>
                </tr>"""

            st.markdown(f"""
            <table class="dtbl">
              <tr>
                <th>SINIF</th><th>GÜVEN SKORU</th><th>TEHDİT</th><th>KOORDİNAT (px)</th>
              </tr>
              {rows}
            </table>""", unsafe_allow_html=True)

            # JSON indirme
            st.markdown("<br>", unsafe_allow_html=True)
            dl_col, _ = st.columns([2, 3])
            with dl_col:
                st.download_button(
                    "⬇  RAPORU JSON OLARAK İNDİR",
                    data=json.dumps({"detections": detections,
                                     "elapsed_ms": elapsed_ms}, indent=2),
                    file_name="fusion1_detections.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.markdown(
                '<p style="font-size:11px;color:#1a5530;letter-spacing:3px;'
                'text-align:center;margin:30px 0;text-transform:uppercase">'
                '[ TARAMA TAMAMLANDI — HEDEF TESPİT EDİLEMEDİ ]</p>',
                unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"Checkpoint bulunamadı: {e}")
    except Exception as e:
        st.error(f"Hata: {e}")
        st.exception(e)

# ── Status bar ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="status-bar">
  SİSTEM: ÇEVRIMIÇI &nbsp;|&nbsp; MODEL: CMAFM DUAL-STREAM &nbsp;|&nbsp;
  VERİ: M4-SAR (56K TRAIN) &nbsp;|&nbsp; SAKARYAÜNİVERSİTESİ BİLGİSAYAR MÜH.
</div>
""", unsafe_allow_html=True)
