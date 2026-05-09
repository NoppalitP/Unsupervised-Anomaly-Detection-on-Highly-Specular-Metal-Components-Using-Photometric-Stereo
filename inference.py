import os
import sys
import time
import cv2
import serial
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Import necessary components from your pipeline
from src.ps_benchmark import (
    CFG, set_seed, PhotometricStereoSolver, AutoCropper, 
    BackboneExtractor, PaDiM, load_L_matrix, build_theoretical_L, _MEAN, _STD
)

# ---------------------------------------------------------
# 1. Jetson Camera Setup
# ---------------------------------------------------------
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

# ---------------------------------------------------------
# 2. Inference Application Class
# ---------------------------------------------------------
class IADInferenceApp:
    def __init__(self, calib_npy_path, model_checkpoint_path, threshold=0.6):
        print("\n=== Initializing IAD Inference System ===")
        set_seed(42)
        self.device = CFG.DEVICE
        self.threshold = threshold
        
        # Load Light Matrix
        if os.path.exists(calib_npy_path):
            self.L_matrix = load_L_matrix(calib_npy_path)
            print("✅ Loaded light calibration matrix.")
        else:
            self.L_matrix = build_theoretical_L(12, 45.0)
            print("⚠️ Using theoretical light matrix.")

        # Initialize PS Solver and Cropper
        self.solver = PhotometricStereoSolver(
            L_matrix=self.L_matrix, 
            drop_dark=0,    # Best config from grid search!
            drop_bright=0,  # Best config from grid search!
            output_mode="after",
            device=self.device
        )
        self.cropper = AutoCropper(output_size=256, crop_offset=12)
        print("✅ Initialized Photometric Stereo Solver.")

        # Initialize Model & Load Weights
        print("⏳ Loading PaDiM model & Backbone...")
        self.extractor = BackboneExtractor(backbone_name="convnext_tiny", device=self.device)
        self.model = PaDiM(self.extractor)
        
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
            
        ckpt = torch.load(model_checkpoint_path, map_location=self.device)
        self.model.mu = ckpt["mu"].to(self.device)
        self.model.inv = ckpt["inv"].to(self.device)
        print("✅ Model loaded successfully!")

        # Image Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD)
        ])

    def capture_images(self, mode_groups=12):
        images = []
        arduino_port = '/dev/ttyCH341USB0'
        try:
            arduino = serial.Serial(arduino_port, 115200, timeout=2)
            time.sleep(2)
            arduino.reset_input_buffer()
        except Exception as e:
            print(f"❌ Arduino Error: {e}")
            return None

        camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not camera.isOpened():
            print("❌ Camera Error.")
            return None

        try:
            print(f"\n📸 Capturing {mode_groups} lighting angles...")
            for i in range(1, mode_groups + 1):
                arduino.write(f"N{mode_groups}_{i}\n".encode())
                arduino.readline() 
                time.sleep(0.3) 
                
                for _ in range(5): camera.read() # clear buffer
                ret, frame = camera.read()
                
                if ret:
                    images.append(frame)
                    cv2.imshow("Capture Preview", frame)
                    cv2.waitKey(10)
                else:
                    print(f"⚠️ Capture failed at angle {i}")
                    
        finally:
            arduino.write(b"OFF\n")
            camera.release()
            cv2.destroyAllWindows()
            arduino.close()
            
        return images if len(images) == mode_groups else None

    def run_inference(self, images):
        print("\n⚙️ Processing 3D Normal Map...")
        t0 = time.time()
        
        # 1. Photometric Stereo
        ps_map = self.solver.solve(images)
        
        # 2. Auto Crop
        bbox = self.cropper.find_bbox(images)
        if bbox is None:
            print("❌ Failed to find object contour.")
            return None
        processed_img = self.cropper.crop_and_resize(ps_map, bbox)
        
        # 3. Model Inference
        print("🧠 Running AI Anomaly Detection...")
        tensor_img = self.transform(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        
        # Get raw Mahalanobis map from PaDiM
        raw_map = self.model.score_map(tensor_img)
        
        # Max score is the image-level anomaly score
        anomaly_score = float(raw_map.max())
        is_defect = anomaly_score > self.threshold
        
        t1 = time.time()
        print(f"⏱️ Total Inference Time: {(t1-t0)*1000:.1f} ms")
        
        self.display_result(processed_img, raw_map, anomaly_score, is_defect)
        
    def display_result(self, orig_img, raw_map, score, is_defect):
        """Displays the result with a nice bounding box and heatmap overlay"""
        # Smooth and normalize heatmap
        smoothed = gaussian_filter(raw_map, sigma=CFG.VIZ_SIGMA)
        norm_map = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
        
        heat_uint8 = (norm_map * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        
        # Resize original image to match heatmap size (224x224)
        orig_resized = cv2.resize(orig_img, (224, 224))
        
        # Overlay
        overlay = cv2.addWeighted(orig_resized, 0.5, heat_color, 0.5, 0)
        
        # Add text
        status_text = "DEFECT (NG)" if is_defect else "PASS (OK)"
        color = (0, 0, 255) if is_defect else (0, 255, 0) # BGR
        
        cv2.putText(overlay, f"{status_text} - Score: {score:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        # Show final result side by side
        final_display = np.hstack((orig_resized, overlay))
        cv2.imshow("IAD Inspection Result", final_display)
        print(f"\n👉 Result: {status_text} (Score: {score:.4f})")
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define paths
    calib_path = "D:/IAD/data_scan/dataset/light_directions_12.npy" # Replace with your actual path on Jetson
    model_path = "checkpoints_after/PaDiM_convnext_tiny.pt"
    
    # You might need to adjust this threshold based on your actual data distribution!
    # A score higher than this threshold is flagged as a Defect.
    decision_threshold = 0.6 
    
    app = IADInferenceApp(calib_path, model_path, threshold=decision_threshold)
    
    # Run the sequence
    captured_images = app.capture_images(mode_groups=12)
    if captured_images:
        app.run_inference(captured_images)
