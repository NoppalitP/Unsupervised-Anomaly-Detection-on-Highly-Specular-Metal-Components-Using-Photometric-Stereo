#include <Adafruit_NeoPixel.h>

#define PIN            6   // ขาที่ต่อกับ Data In ของ NeoPixel
#define NUMPIXELS      24  // จำนวน LED ใน Ring
#define BRIGHTNESS     255 // ความสว่าง (0-255) สว่างสุด 100%

Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10); // ⚡ ทริคความเร็ว: ลดเวลา Timeout ของ Serial ลงเหลือ 10ms
  
  pixels.begin();
  pixels.setBrightness(BRIGHTNESS);
  clearLEDs();
  pixels.show(); // ปิดไฟทุกดวงตอนเริ่มต้น
  Serial.println("READY"); 
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // 1. คำสั่งปิดไฟทั้งหมด: "OFF"
    if (command == "OFF") {
      clearLEDs();
      pixels.show();
      Serial.println("OK: ALL OFF");
    }
    // 2. คำสั่งจัดกลุ่มไฟ: "N<จำนวนภาพทั้งหมด>_<สเต็ปที่>" (เช่น N8_1)
    else if (command.startsWith("N")) {
      int separatorIdx = command.indexOf('_');
      
      if (separatorIdx > 0) {
        // ดึงค่า จำนวนภาพทั้งหมด และ สเต็ปปัจจุบัน ออกมา
        int total_images = command.substring(1, separatorIdx).toInt();
        int step = command.substring(separatorIdx + 1).toInt();

        // ตรวจสอบความถูกต้อง (ต้องหาร 24 ลงตัว และสเต็ปต้องไม่เกินจำนวนภาพ)
        if (total_images > 0 && 24 % total_images == 0 && step >= 1 && step <= total_images) {
          
          int group_size = 24 / total_images;         // จำนวนดวงไฟที่ต้องเปิดพร้อมกัน (เช่น 24/8 = 3 ดวง)
          int start_led = (step - 1) * group_size;    // หาตำแหน่งดวงแรกของกลุ่ม

          clearLEDs(); // ล้างค่าสีเดิมก่อน
          
          // วนลูปเปิดไฟตามจำนวนกลุ่ม
          for (int i = 0; i < group_size; i++) {
            // ใช้ % NUMPIXELS เผื่อป้องกัน Index เกิน (Safety)
            pixels.setPixelColor((start_led + i) % NUMPIXELS, pixels.Color(255, 255, 255));
          }
          
          pixels.show(); // สั่งให้ไฟติดพร้อมกันทันที
          
          // ส่ง OK กลับไปให้ Python ถ่ายรูปได้เลย
          Serial.print("OK: MODE ");
          Serial.print(total_images);
          Serial.print(" STEP ");
          Serial.println(step);
        } else {
          Serial.println("ERR: INVALID PARAMS");
        }
      }
    }
  }
}

// ฟังก์ชันสำหรับเคลียร์ Buffer ไฟเป็นสีดำ (ยังไม่โชว์จนกว่าจะสั่ง pixels.show())
void clearLEDs() {
  for(int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(0, 0, 0));
  }
}