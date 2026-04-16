import random
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# =============================================================================
# ตั้งค่าคอนฟิกูเรชัน (Configuration)
# =============================================================================
RAW_DIR = Path("D:/IAD/data_scan/dataset/raw_captures")
CALIB_NPY = Path("D:/IAD/data_scan/dataset/light_directions_12.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """
    โหลดเมทริกซ์ทิศทางแสง (L) และสุ่มดึงโฟลเดอร์ภาพตัวอย่างที่เปื้อนตำหนิ (Defect)
    """
    # 1. โหลดหรือสร้าง L Matrix
    if CALIB_NPY.exists():
        L = np.load(CALIB_NPY)
    else:
        print("ไม่พบไฟล์ Calibrate ทำการสร้าง L Matrix แบบจำลองเชิงทฤษฎี...")
        slant = np.radians(45)
        L = np.zeros((12, 3), dtype=np.float32)
        for i in range(12):
            az = 2. * np.pi * i / 12
            L[i, 0] = np.cos(az) * np.sin(slant)
            L[i, 1] = np.sin(az) * np.sin(slant)
            L[i, 2] = np.cos(slant)
            
    # 2. ค้นหาโฟลเดอร์ที่คาดว่าจะมี Defect
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ {RAW_DIR}")
        
    folders = [f for f in RAW_DIR.iterdir() if f.is_dir() and "good" not in f.name.lower()]
    if not folders:
        raise ValueError("ไม่พบโฟลเดอร์ภาพ Defect ในโฟลเดอร์ที่ระบุ")
        
    target_folder = random.choice(folders)
    print(f"กำลังประมวลผลชุดภาพตัวอย่างจาก: {target_folder.name}")
    
    # 3. โหลดภาพแสงเงาทั้ง 12 มุม
    img_paths = sorted(target_folder.glob("light_*.png"))
    if len(img_paths) != 12:
        print(f"คำเตือน: พบภาพจำนวน {len(img_paths)} ภาพ (คาดหวัง 12 ภาพ)")
        
    images = [cv2.imread(str(p)) for p in img_paths]
    if any(img is None for img in images):
        raise IOError("เกิดข้อผิดพลาดในการอ่านไฟล์ภาพบางไฟล์")
        
    return L.astype(np.float32), images

def build_object_mask(images: List[np.ndarray], h: int, w: int) -> np.ndarray:
    stack    = np.array(images, dtype=np.float32)
    robust   = np.percentile(stack, 80, axis=0).astype(np.uint8)
    gray     = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.medianBlur(gray, 5)
    _, mask  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k        = np.ones((5, 5), np.uint8)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    cnts, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final    = np.zeros((h, w), dtype=np.uint8)
    if cnts:
        cv2.drawContours(final, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return final > 0

def compute_ps(L_matrix, images):
    """
    คำนวณ Photometric Stereo (WLS) เพื่อสกัด Albedo และ Normal Map
    """
    # แปลงภาพทั้งหมดให้เป็น Grayscale สำหรับคำนวณ PS
    grays = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images], dtype=np.float32)
    N_lights, H, W = grays.shape
    
    # สร้าง Object Mask ด้วยกระบวนการ Morphology & Contours
    mask = build_object_mask(images, H, W)
    
    I_valid = grays[:, mask].T  # (P, N_lights)
    P = I_valid.shape[0]
    
    if P == 0:
        raise ValueError("ไม่พบพิกเซลของวัตถุเลย (Mask ว่างเปล่า)")

    # แก้สมการ Weighted Least Squares บน GPU
    with torch.no_grad():
        L_t = torch.tensor(L_matrix, dtype=torch.float32, device=DEVICE)
        I_t = torch.tensor(I_valid, dtype=torch.float32, device=DEVICE)
        
        L_px = L_t.unsqueeze(0).expand(P, -1, -1)
        # เติม Tikhonov Regularization (1e-5) ลงในแนวทแยงมุมป้องกัน Singular Matrix
        A = torch.bmm(L_px.transpose(1, 2), L_px) + (torch.eye(3, device=DEVICE) * 1e-5)
        B = torch.bmm(L_px.transpose(1, 2), I_t.unsqueeze(-1))
        
        N_raw = torch.linalg.solve(A, B).squeeze(-1)
        albedo = torch.linalg.norm(N_raw, dim=1, keepdim=True)
        N_unit = (N_raw / albedo.clamp(min=1e-5)).cpu().numpy()
        albedo = albedo.cpu().numpy().squeeze()
    
    # สร้าง Blank Maps
    albedo_map = np.zeros((H, W), dtype=np.float32)
    nx_map = np.full((H, W), 0.0, dtype=np.float32)
    ny_map = np.full((H, W), 0.0, dtype=np.float32)
    nz_map = np.full((H, W), 1.0, dtype=np.float32) # พื้นหลังชี้เข้าหากล้อง (Z=1)
    
    # เติมข้อมูลลงใน Map
    albedo_map[mask] = albedo
    nx_map[mask] = N_unit[:, 0]
    ny_map[mask] = N_unit[:, 1]
    nz_map[mask] = N_unit[:, 2]
    
    # แปลง Normal Vector ให้อยู่ในสเกล RGB (0 ถึง 1) สำหรับการแสดงผลภาพ
    rgb_normal = np.stack([
        (nx_map + 1) / 2, 
        (ny_map + 1) / 2, 
        (nz_map + 1) / 2
    ], axis=-1)
    
    return images[0], albedo_map, nx_map, rgb_normal

class AutoCropper:
    def __init__(self, padding: int = 15, output_size: int = 512, crop_offset: int = 12):
        self.padding     = padding
        self.output_size = output_size
        self.crop_offset = crop_offset

    def find_bbox(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        stack   = np.array(images, dtype=np.float32)
        robust  = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray    = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k       = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k)
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        box  = cv2.boxPoints(rect).astype(np.float32)
        return self._order_points(box)

    def crop_and_resize(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        rect = bbox.copy()
        o    = self.crop_offset
        rect[0] += [ o,  o]
        rect[1] += [-o,  o]
        rect[2] += [-o, -o]
        rect[3] += [ o, -o]
        s   = self.output_size - 1
        dst = np.array([[0,0],[s,0],[s,s],[0,s]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.output_size, self.output_size),
                                   flags=cv2.INTER_AREA)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        out  = np.zeros((4,2), dtype=np.float32)
        s    = pts.sum(axis=1)
        out[0] = pts[np.argmin(s)]
        out[2] = pts[np.argmax(s)]
        diff   = np.diff(pts, axis=1).ravel()
        out[1] = pts[np.argmin(diff)]
        out[3] = pts[np.argmax(diff)]
        return out

def main():
    L, images = load_data()
    raw_img, albedo_map, nx_map, rgb_normal = compute_ps(L, images)
    
    # แทนที่การครอบกึ่งกลางแบบเดิมด้วย AutoCropper
    cropper = AutoCropper(output_size=512, crop_offset=12)
    bbox = cropper.find_bbox(images)
    
    if bbox is None:
        raise RuntimeError("AutoCropper ไม่พบวัตถุ (Contour) ในภาพ")
        
    raw_crop = cropper.crop_and_resize(raw_img, bbox)
    albedo_crop = cropper.crop_and_resize(albedo_map, bbox)
    nx_crop = cropper.crop_and_resize(nx_map, bbox)
    rgb_crop = cropper.crop_and_resize(rgb_normal, bbox)
    
    # =========================================================================
    # การแสดงผลพล็อต (Matplotlib) สไตล์งานวิจัย (Academic Style)
    # =========================================================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    fig.patch.set_facecolor('white')
    
    # (a) Raw Image
    axes[0].imshow(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Raw Image (with Glare)", fontsize=16, fontname="serif", pad=12)
    
    # (b) Albedo Map (ความสว่างผิววัตถุ)
    vmax_albedo = np.percentile(albedo_crop, 99) # ป้องกันค่าโดด (Outliers) ทำให้ภาพมืดเกินไป
    axes[1].imshow(albedo_crop, cmap='gray', vmin=0, vmax=vmax_albedo)
    axes[1].set_title("(b) Albedo Map", fontsize=16, fontname="serif", pad=12)
    
    # (c) Normal X (แสดงความนูนเว้าแนวนอนชัดเจน)
    # ใช้ coolwarm cmap โดยมี center ที่ 0 เพื่อแบ่งแยกด้านซ้าย/ขวา
    axes[2].imshow(nx_crop, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title("(c) Normal X (Horizontal Gradients)", fontsize=16, fontname="serif", pad=12)
    
    # (d) 3D Normal Map
    axes[3].imshow(rgb_crop)
    axes[3].set_title("(d) 3D Normal Map (RGB)", fontsize=16, fontname="serif", pad=12)
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    
    # บันทึกไฟล์ภาพ
    out_filename = "photometric_results_fig2.png"
    plt.savefig(out_filename, dpi=300, bbox_inches='tight', transparent=False)
    print(f"เซฟรูปภาพสำเร็จ! ไฟล์ชื่อ: {out_filename}")

if __name__ == "__main__":
    main()