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
# HÜCRE 2: Veri setini bağla (symlink)
# ============================================================
import glob

# Kaggle'ın veri seti dizinini bul
# Veri setinizin Kaggle'daki adına göre bu yolu güncellemeniz gerekebilir
KAGGLE_INPUT = "/kaggle/input"
possible_dirs = glob.glob(f"{KAGGLE_INPUT}/m4-sar*") + glob.glob(f"{KAGGLE_INPUT}/M4*")
if not possible_dirs:
    possible_dirs = glob.glob(f"{KAGGLE_INPUT}/*")
    print(f"⚠️  Mevcut veri setleri: {possible_dirs}")
    print("    Lütfen sağ panelden M4-SAR veri setinizi ekleyin.")
else:
    DATASET_DIR = possible_dirs[0]
    print(f"📂 Veri seti bulundu: {DATASET_DIR}")

    # Projenin beklediği yapıyı kur
    DATA_ROOT = "/kaggle/working/sar-optical-fusion/data/m4_sar"
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Veri seti içindeki klasörleri keşfet ve bağla
    for subdir in ["optical", "sar", "labels"]:
        target = os.path.join(DATA_ROOT, subdir)
        if os.path.exists(target):
            continue
        # Kaggle input içinde ara
        source = None
        for candidate in [
            os.path.join(DATASET_DIR, subdir),
            os.path.join(DATASET_DIR, "m4_sar", subdir),
            os.path.join(DATASET_DIR, "data", "m4_sar", subdir),
        ]:
            if os.path.exists(candidate):
                source = candidate
                break
        if source:
            os.symlink(source, target)
            print(f"  ✅ {subdir}/ → {source}")
        else:
            print(f"  ⚠️  {subdir}/ bulunamadı, manuel yol gerekebilir")

    # Doğrula
    for split in ["train", "val", "test"]:
        opt_dir = os.path.join(DATA_ROOT, "optical", split)
        if os.path.exists(opt_dir):
            count = len(os.listdir(opt_dir))
            print(f"  📸 {split}: {count} optik görüntü")

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
