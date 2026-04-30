"""
SAR-Optical Fusion — Kaggle P100 GPU Eğitim Notebook'u
======================================================
Bu scripti Kaggle Notebook'ta çalıştırın.

KURULUM ADIMLARI (Kaggle'da):
1. kaggle.com → "New Notebook" → "File" → "Upload Notebook" ile bu dosyayı yükleyin
   VEYA yeni bir notebook açıp aşağıdaki kodu hücrelere yapıştırın.
2. Sağ panelden: Settings → Accelerator → "GPU P100" seçin
3. "Add Data" → "m4-sar-dataset" (kendi Kaggle veri setinizi) ekleyin
4. "Run All" ile çalıştırın

NOT: Kaggle'da haftada 30 saat ücretsiz GPU kullanabilirsiniz.
     P100 (16GB VRAM) ile batch_size=16, img_size=640 kullanıyoruz.
     Bu, GTX 1650'ye kıyasla yaklaşık 8x daha hızlıdır.
"""

# ============================================================
# HÜCRE 1: Repo'yu klonla ve bağımlılıkları kur
# ============================================================
import subprocess, sys, os

# Repo'yu klonla
if not os.path.exists("/kaggle/working/sar-optical-fusion"):
    subprocess.run(["git", "clone", "https://github.com/RohamRasouli/sar-optical-fusion.git",
                    "/kaggle/working/sar-optical-fusion"], check=True)

os.chdir("/kaggle/working/sar-optical-fusion")

# Bağımlılıkları kur (torch zaten Kaggle'da yüklü)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rasterio", "opencv-python-headless", "pyyaml"], check=True)

print("✅ Repo klonlandı ve bağımlılıklar kuruldu.")

# ============================================================
# HÜCRE 2: Veri setini bağla
# ============================================================
import glob
import shutil

# Kaggle'ın veri seti dizinini bul
KAGGLE_INPUT = "/kaggle/input"
DATASET_DIR = None

# /kaggle/input içindeki tüm klasörleri tara (isminin ne olduğuna bakılmaksızın)
if os.path.exists(KAGGLE_INPUT):
    for item in os.listdir(KAGGLE_INPUT):
        item_path = os.path.join(KAGGLE_INPUT, item)
        if not os.path.isdir(item_path): continue
            
        subdirs = [d.lower() for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
        if "images" in subdirs or "optical" in subdirs or "sar" in subdirs or "labels" in subdirs:
            DATASET_DIR = item_path
            break
            
        # Alt klasörleri kontrol et (bazen zip içeriği bir klasör içine açılır)
        for subitem in os.listdir(item_path):
            subitem_path = os.path.join(item_path, subitem)
            if not os.path.isdir(subitem_path): continue
            subsubdirs = [d.lower() for d in os.listdir(subitem_path) if os.path.isdir(os.path.join(subitem_path, d))]
            if "images" in subsubdirs or "optical" in subsubdirs or "sar" in subsubdirs or "labels" in subsubdirs:
                DATASET_DIR = subitem_path
                break
        if DATASET_DIR: break

if not DATASET_DIR:
    print("❌ HATA: Veri seti bulunamadı!")
    print(f"Şu anki Kaggle Input klasörleri: {os.listdir(KAGGLE_INPUT) if os.path.exists(KAGGLE_INPUT) else 'Yok'}")
    print("Lütfen sağ üstteki 'Add Data' butonuna tıklayıp M4-SAR veri setini eklediğinizden emin olun.")
    sys.exit(1)
else:
    print(f"📂 Veri seti bulundu: {DATASET_DIR}")
    
    # 48 GB'lık veriyi kopyalarsak Kaggle'ın 20 GB'lık disk limiti dolar.
    # Bu yüzden sadece bir kısayol (symlink) oluşturuyoruz.
    DATA_ROOT = "/kaggle/working/sar-optical-fusion/data/m4_sar"
    
    # Varsa eskisini sil
    if os.path.exists(DATA_ROOT) or os.path.islink(DATA_ROOT):
        if os.path.islink(DATA_ROOT):
            os.unlink(DATA_ROOT)
        else:
            shutil.rmtree(DATA_ROOT)
            
    # Ana klasörün (data/) var olduğundan emin ol
    os.makedirs(os.path.dirname(DATA_ROOT), exist_ok=True)
            
    # Kısayol oluştur
    os.symlink(DATASET_DIR, DATA_ROOT)
    print(f"🔗 Veri seti bağlandı: {DATA_ROOT} -> {DATASET_DIR}")

# ============================================================
# HÜCRE 3: GPU Kontrolü
# ============================================================
import torch
print(f"\n🖥️  PyTorch: {torch.__version__}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  GPU bulunamadı! Sağ panelden GPU T4 seçtiğinizden emin olun.")

# ============================================================
# HÜCRE 4: Eğitimi Başlat
# ============================================================
print("\n🚀 Eğitim başlıyor (Kaggle P100 — batch_size=16, img_size=640)...\n")

# Kaggle P100 konfigürasyonu ile eğitimi başlat
env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/sar-optical-fusion" + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

subprocess.run([
    sys.executable, "-m", "src.train",
    "--config", "configs/kaggle_p100.yaml",
], env=env, check=True)

print("\n✅ Eğitim tamamlandı!")

# ============================================================
# HÜCRE 5: Sonuçları kaydet
# ============================================================
import shutil

# En iyi modeli Kaggle output'a kopyala
output_dir = "/kaggle/working/sar-optical-fusion/runs"
if os.path.exists(os.path.join(output_dir, "best.pt")):
    shutil.copy(os.path.join(output_dir, "best.pt"), "/kaggle/working/best.pt")
    print("📦 best.pt → /kaggle/working/best.pt (indirilebilir)")

if os.path.exists(os.path.join(output_dir, "final.pt")):
    shutil.copy(os.path.join(output_dir, "final.pt"), "/kaggle/working/final.pt")
    print("📦 final.pt → /kaggle/working/final.pt (indirilebilir)")

print("\n🎉 Tüm model dosyaları Kaggle Output sekmesinden indirilebilir!")
