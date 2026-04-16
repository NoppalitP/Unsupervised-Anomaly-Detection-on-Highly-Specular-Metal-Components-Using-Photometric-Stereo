# =============================================================================
# 📦 COLAB SETUP CELL — วางไว้ Cell แรกของ Notebook
# ให้ environment ตรงกับ defect_vision conda env บน Windows
# =============================================================================

# ── 1. ตรวจสอบ GPU ─────────────────────────────────────────────────────────
import subprocess, sys

gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(gpu_info.stdout if gpu_info.returncode == 0 else "⚠️ No GPU detected")

import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"Device  : {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

# ── 2. ติดตั้ง packages ให้ตรงกับ local env ────────────────────────────────
# อัปโหลด requirements_colab.txt ไปที่ /content/ ก่อน แล้วรัน cell นี้
# หรือถ้าอยู่ใน Google Drive ปรับ path ด้านล่าง

REQUIREMENTS_PATH = "/content/requirements_colab.txt"
# REQUIREMENTS_PATH = "/content/drive/MyDrive/IAD/requirements_colab.txt"

print("\n📦 Installing packages …")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS_PATH],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])   # แสดง error 3000 ตัวสุดท้าย
else:
    print("✅ All packages installed.")

# ── 3. ตรวจสอบ package versions หลัก ───────────────────────────────────────
print("\n🔍 Verifying key packages …")
checks = [
    ("torch",        "2"),
    ("torchvision",  "0"),
    ("anomalib",     "2"),
    ("timm",         "1"),
    ("cv2",          "4"),
    ("numpy",        "2"),
    ("sklearn",      "1"),
    ("pandas",       "2"),
    ("lightning",    "2"),
]

import importlib
all_ok = True
for pkg_import, min_major in checks:
    try:
        mod = importlib.import_module(pkg_import)
        ver = getattr(mod, "__version__", "?")
        major = int(ver.split(".")[0])
        ok = major >= int(min_major)
        status = "✅" if ok else "⚠️"
        print(f"  {status}  {pkg_import:<15} {ver}")
        if not ok:
            all_ok = False
    except ImportError:
        print(f"  ❌  {pkg_import:<15} NOT FOUND")
        all_ok = False

print("\n✅ Environment ready!" if all_ok else "\n⚠️ Some packages may need attention.")

# ── 4. Mount Google Drive (optional) ────────────────────────────────────────
# ลบ comment ด้านล่างถ้าต้องการ mount Drive
# from google.colab import drive
# drive.mount("/content/drive")
# print("✅ Google Drive mounted.")
