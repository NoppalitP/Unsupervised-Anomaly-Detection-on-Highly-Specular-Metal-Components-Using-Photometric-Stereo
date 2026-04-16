import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# 1. โหลดไฟล์รูปภาพที่อัปโหลด
# (a) รูปโมเดล 3D CAD จาก Fusion 360
img_a = mpimg.imread('ASB.png')
# (b) รูปถ่ายรอยขีดข่วนจำลอง (Scratches)
img_b = mpimg.imread('scratch.png')
# (c) รูปถ่ายรอยบุบจำลอง (Dents)
img_c = mpimg.imread('dent.png')

# 2. สร้าง Figure และ GridSpec สำหรับจัดวางรูปภาพ
fig = plt.figure(figsize=(12, 4.5))
gs = GridSpec(1, 3, figure=fig, wspace=0.1)

# 3. แสดงรูปภาพในแต่ละช่อง
# ช่อง (a)
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img_a)
ax1.axis('off')
ax1.text(0.5, -0.15, '(a) Experimental Setup (3D CAD Model)', 
         transform=ax1.transAxes, ha='center', va='top', fontsize=11)

# ช่อง (b)
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img_b)
ax2.axis('off')
ax2.text(0.5, -0.15, '(b) Simulated Scratches', 
         transform=ax2.transAxes, ha='center', va='top', fontsize=11)

# ช่อง (c)
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(img_c)
ax3.axis('off')
ax3.text(0.5, -0.15, '(c) Simulated Dents', 
         transform=ax3.transAxes, ha='center', va='top', fontsize=11)


# 5. ปรับระยะขอบและบันทึกไฟล์
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)
plt.savefig('Experimental_Setup_and_Defects_Figure2.png', dpi=300, bbox_inches='tight')

print("บันทึกไฟล์รูปภาพ Experimental_Setup_and_Defects_Figure2.png เรียบร้อยแล้ว")