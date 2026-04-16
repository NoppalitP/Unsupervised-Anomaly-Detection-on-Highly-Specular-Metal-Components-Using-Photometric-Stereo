import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from typing import List, Optional

# =============================================================================
# ตั้งค่าโฟลเดอร์และดึง L_matrix
# =============================================================================
RAW_DIR = Path("D:/IAD/data_scan/dataset/raw_captures")
CALIB_NPY = Path("D:/IAD/data_scan/dataset/light_directions_12.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    # 1. โหลด L Matrix
    if CALIB_NPY.exists():
        L = np.load(CALIB_NPY)
    else:
        print("ไม่พบไฟล์ Calib สร้าง L Matrix จำลอง...")
        slant = np.radians(45)
        L = np.zeros((12, 3), dtype=np.float32)
        for i in range(12):
            az = 2. * np.pi * i / 12
            L[i, 0] = np.cos(az) * np.sin(slant)
            L[i, 1] = np.sin(az) * np.sin(slant)
            L[i, 2] = np.cos(slant)
            
    # 2. สุ่มหาโฟลเดอร์ Defect 1 โฟลเดอร์
    folders = [f for f in RAW_DIR.iterdir() if f.is_dir() and "good" not in f.name.lower()]
    target_folder = random.choice(folders)
    print(f"กำลังประมวลผลโฟลเดอร์: {target_folder.name}")
    
    # 3. โหลดภาพ 12 มุม
    img_paths = sorted(target_folder.glob("light_*.png"))
    images = [cv2.imread(str(p)) for p in img_paths]
    
    return L.astype(np.float32), images

def compute_ps(L_matrix, images):
    # แปลงภาพเป็น Gray Stack
    grays = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images], dtype=np.float32)
    N_lights, H, W = grays.shape
    
    # สร้าง Mask
    robust = np.percentile(grays, 80, axis=0).astype(np.uint8)
    _, mask = cv2.threshold(cv2.medianBlur(robust, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask > 0
    
    I_valid = grays[:, mask].T  # (P, N_lights)
    P = I_valid.shape[0]
    
    # WLS Solver (ดึง Albedo ออกมาด้วย)
    L_t = torch.tensor(L_matrix, dtype=torch.float32, device=DEVICE)
    I_t = torch.tensor(I_valid, dtype=torch.float32, device=DEVICE)
    
    L_px = L_t.unsqueeze(0).expand(P, -1, -1)
    A = torch.bmm(L_px.transpose(1, 2), L_px) + (torch.eye(3, device=DEVICE) * 1e-5)
    B = torch.bmm(L_px.transpose(1, 2), I_t.unsqueeze(-1))
    
    N_raw = torch.linalg.solve(A, B).squeeze(-1)
    albedo = torch.linalg.norm(N_raw, dim=1, keepdim=True)  # <-- นี่คือ Albedo (ความสว่างพื้นผิว)
    N_unit = (N_raw / albedo.clamp(min=1e-5)).cpu().numpy()
    albedo = albedo.cpu().numpy().squeeze()
    
    # สร้าง Map คืนกลับไปที่ขนาดรูปเดิม
    albedo_map = np.zeros((H, W), dtype=np.float32)
    nx_map = np.full((H, W), 0.0, dtype=np.float32)
    ny_map = np.full((H, W), 0.0, dtype=np.float32)
    nz_map = np.full((H, W), 1.0, dtype=np.float32)
    
    albedo_map[mask] = albedo
    nx_map[mask] = N_unit[:, 0]
    ny_map[mask] = N_unit[:, 1]
    nz_map[mask] = N_unit[:, 2]
    
    # แปลง Normal Map เป็น RGB สำหรับวาดรูป (X=R, Y=G, Z=B)
    rgb_normal = np.stack([
        (nx_map + 1) / 2, 
        (ny_map + 1) / 2, 
        (nz_map + 1) / 2
    ], axis=-1)
    
    return images[0], albedo_map, nx_map, rgb_normal

# =============================================================================
# 2. AUTO CROPPER
# =============================================================================
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
    
    # ใช้งาน AutoCropper แทนการตัดแบบกึ่งกลาง
    cropper = AutoCropper(output_size=512, crop_offset=12)
    bbox = cropper.find_bbox(images)
    
    if bbox is not None:
        print("AutoCropper: ตรวจพบพิกัดชิ้นงาน กำลังทำ Perspective Warp...")
        raw_crop = cropper.crop_and_resize(raw_img, bbox)
        albedo_crop = cropper.crop_and_resize(albedo_map, bbox)
        nx_crop = cropper.crop_and_resize(nx_map, bbox)
        rgb_crop = cropper.crop_and_resize(rgb_normal, bbox)
    else:
        print("AutoCropper: ไม่พบพิกัดชิ้นงาน สลับไปใช้โหมด Center Crop...")
        H, W = raw_img.shape[:2]
        size = min(H, W)
        cy, cx = H//2, W//2
        raw_crop = raw_img[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
        albedo_crop = albedo_map[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
        nx_crop = nx_map[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
        rgb_crop = rgb_normal[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    
    # วาดรูป (1 แถว 4 คอลัมน์) ความละเอียด 300 DPI
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    fig.patch.set_facecolor('white')
    
    # ถ้า raw_crop เป็นทศนิยม (float32) จะโดนเปลี่ยนกลับเป็น uint8 ตอนวาดรูป
    if raw_crop.dtype == np.float32:
        raw_crop = raw_crop.astype(np.uint8)
        
    axes[0].imshow(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Raw Image (2D)", fontsize=16)
    
    axes[1].imshow(albedo_crop, cmap='gray', vmax=np.percentile(albedo_crop, 99))
    axes[1].set_title("(b) Albedo Map", fontsize=16)
    
    axes[2].imshow(nx_crop, cmap='coolwarm')
    axes[2].set_title("(c) Normal X (Horizontal Gradients)", fontsize=16)
    
    axes[3].imshow(rgb_crop)
    axes[3].set_title("(d) Normal Map (3D)", fontsize=16)
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    output_filename = "photometric_results_autocrop.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"เซฟรูปภาพสำเร็จ! ไฟล์ชื่อ: {output_filename}")

if __name__ == "__main__":
    main()