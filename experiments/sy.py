import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# หมายเหตุ: ในการใช้งานจริง แนะนำให้นำไฟล์ภาพ image.png (เข็ม) และ image.png (ลูกบอล)
# ไปลบพื้นหลัง (ลบลายไม้) ออกให้เป็นพื้นหลังสีขาวก่อน แล้วค่อยนำมาเข้าสคริปต์นี้ครับ
# สมมติว่าไฟล์ถูกตั้งชื่อใหม่เป็น 'needle_clean.png' และ 'ball_clean.png'

try:
    # โหลดไฟล์รูปภาพ (ให้เปลี่ยนชื่อไฟล์ตามที่คุณเซฟไว้)
    img_needle = mpimg.imread('needle.png') # รูปเข็ม
    img_ball = mpimg.imread('ball.png')   # รูปลูกบอล (ใช้ชื่อไฟล์จริงของคุณ)
except FileNotFoundError:
    print("กรุณาตรวจสอบชื่อไฟล์รูปภาพให้ถูกต้อง")
    exit()

# สร้าง Figure พื้นหลังสีขาว
fig = plt.figure(figsize=(8, 2), dpi=300)
fig.patch.set_facecolor('white')
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

# (a) รูปเข็ม NIPRO
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img_needle)
ax1.axis('off')
ax1.set_title("(a) Blunt-tip Needle (Ø 0.3 mm)", fontsize=12, pad=10)

# (b) รูปลูกบอลสเตนเลส
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img_ball)
ax2.axis('off')
ax2.set_title("(b) 304 Stainless Steel Ball (Ø 1.25 inch)", fontsize=12, pad=10)

plt.tight_layout()

# บันทึกไฟล์รูปภาพใหม่
output_name = 'Defect_Tools_Figure.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"สร้างรูปภาพสำเร็จ! บันทึกเป็นชื่อ: {output_name}")
print("นำรูปนี้ไปแทรกใน Word ใต้หัวข้อ 2.1.1 ได้เลยครับ")