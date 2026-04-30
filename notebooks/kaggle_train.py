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
DATASET_DIR = KAGGLE_INPUT

# Kaggle Input klasöründe herhangi bir veri var mı kontrol et
has_data = False
if os.path.exists(KAGGLE_INPUT):
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        if len(files) > 0:
            has_data = True
            break

if not has_data:
    print("❌ HATA: Kaggle Input klasöründe hiç dosya bulunamadı!")
    print("Lütfen sağ üstteki 'Add Data' butonuna tıklayıp 'wchao0601/m4-sar' veri setini ekleyin.")
    sys.exit(1)

print(f"📂 Veri seti Kaggle Input dizininden okunacak.")

# 48 GB'lık veriyi kopyalarsak Kaggle'ın 20 GB'lık disk limiti dolar.
# Bu yüzden sadece KAGGLE_INPUT'u projeye bağlıyoruz.
# M4SARDataset sınıfı kendi içinde klasörleri otomatik (rglob ile) bulacak.
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
print(f"🔗 Kaggle Input dizini bağlandı: {DATA_ROOT} -> {DATASET_DIR}")

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
# Subprocess yerine doğrudan fonksiyon çağrısı yapıyoruz (Kaggle loglarının anlık görünmesi için en güvenli yol)
from src.train import main as train_main
import sys

# Argümanları simüle et
sys.argv = ["src.train", "--config", "configs/kaggle_p100.yaml"]

print("\n🚀 Eğitim süreci başlatılıyor (Direct Process Mode)...\n")
try:
    train_main()
except Exception as e:
    print(f"\n❌ Eğitim sırasında hata oluştu: {e}")
    import traceback
    traceback.print_exc()

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
