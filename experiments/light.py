import numpy as np
import math

# ================= ตั้งค่าฮาร์ดแวร์ของคุณที่นี่ =================
NUM_LEDS = 24        # จำนวนหลอดไฟ (เช่น 12, 16, 24)
RADIUS   = 50.0      # รัศมี (R) หน่วย mm (ระยะจากเลนส์ถึงหลอดไฟ)
HEIGHT   = 150.0     # ความสูง (H) หน่วย mm (ระยะจากหลอดไฟถึงวัตถุ)
START_ANGLE = -90    # องศาของหลอดที่ 0 (ปกติหลอดแรกมักอยู่ข้างบน หรือขวา ลองไล่ดูครับ)
# -90 = เริ่มที่ทิศ 12 นาฬิกา (บน)
# 0   = เริ่มที่ทิศ 3 นาฬิกา (ขวา)
# ==========================================================

light_vectors = []

print(f"Generating vectors for {NUM_LEDS} LEDs...")
print(f"Geometry: Radius={RADIUS}, Height={HEIGHT}")

for i in range(NUM_LEDS):
    # 1. หามุมของหลอดนี้ (แปลงเป็น Radians เพื่อใช้ในสูตร sin/cos)
    angle_deg = START_ANGLE + (i * (360.0 / NUM_LEDS))
    angle_rad = math.radians(angle_deg)
    
    # 2. คำนวณพิกัด (x, y, z)
    lx = RADIUS * math.cos(angle_rad)
    ly = RADIUS * math.sin(angle_rad) # บางระบบอาจต้องใส่ - (ลบ) ถ้าแกน Y กลับหัว
    lz = HEIGHT
    
    # 3. Normalization (ทำให้ความยาวเวกเตอร์ = 1)
    length = math.sqrt(lx**2 + ly**2 + lz**2)
    
    norm_x = lx / length
    norm_y = ly / length
    norm_z = lz / length
    
    # เก็บค่า
    light_vectors.append([norm_x, norm_y, norm_z])
    
    print(f"LED {i+1}: [{norm_x:.4f}, {norm_y:.4f}, {norm_z:.4f}]")

# 4. บันทึกเป็นไฟล์ .txt (รูปแบบมาตรฐาน: 1 บรรทัดต่อ 1 รูป)
np.savetxt('my_lights.txt', np.array(light_vectors), fmt='%.6f')
print("\n✅ Saved to 'my_lights.txt' successfully!")