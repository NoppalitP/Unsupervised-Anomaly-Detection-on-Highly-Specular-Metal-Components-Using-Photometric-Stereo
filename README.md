# Unsupervised Anomaly Detection on Highly Specular Metal Components Using Photometric Stereo

This project implements an end-to-end **Industrial Anomaly Detection (IAD)** pipeline designed to solve the challenges of inspecting highly specular metal surfaces. Human visual inspection often suffers from low repeatability (66.83%) and high miss rates (30-40%) (Stallard *et al.*, 2018). This project automates the process to achieve **Zero-Defect Manufacturing** by combining computational imaging with lightweight deep learning.

## 🌟 Key Methodologies

To address the core limitations of traditional inspection, this project integrates three main solutions:

1. **Acquisition (The Eye) - Photometric Stereo (PS):**
   Highly reflective metal surfaces cause blind spots (specular highlights) and hide defects under shadows. We implement **Photometric Stereo** (Woodham, 1980) by capturing multiple images from a fixed camera viewpoint while varying the light direction. The algorithm fuses these images to extract the **Surface Normal Map** (revealing 3D shape, dents, and scratches) and the **Albedo Map** (revealing true surface color, stains, and rust), effectively neutralizing distracting reflections.

2. **Augmentation (The Simulator) - Synthetic Defect Generation:**
   In real-world manufacturing, defect data is extremely rare (severe data imbalance). To train effective models using only "good" samples, we utilize synthetic anomaly generation techniques such as **CutPaste** (Li *et al.*, 2021) and **NSA** (Schlüter *et al.*, 2023) to simulate realistic defects (scratches, dents) during training.

3. **Inference (The Speed) - Lightweight Deep Learning (EfficientAD):**
   Industrial applications require high-speed, low-cost inference (ROI & Cost Efficiency). We benchmark several Unsupervised Anomaly Detection (UAD) models and focus on lightweight architectures like **EfficientAD** (Batzner *et al.*, 2024). Utilizing a Student-Teacher knowledge distillation approach, the model achieves millisecond-level latency while maintaining high accuracy (AUROC).

## 🚀 Features
- **Photometric Stereo Pipeline**: GPU-accelerated Weighted Least Squares (WLS) solver with outlier rejection to estimate surface normals and albedo.
- **Anomaly Detection**: Includes SOTA algorithms such as PatchCore, PaDiM, DRAEM, SimpleNet, and EfficientAD.
- **Dataset Management**: Automated cropping and conversion tools to build MVTec-style datasets from raw PS captures.
- **Hardware Integration**: Includes 3D CAD models for the lighting dome and Arduino firmware for LED synchronization.

## 📂 Project Structure
The repository has been streamlined for efficiency:

*   `main.py`: The consolidated End-to-End Pipeline (PS Solver → AutoCropper → Dataset Builder → UAD Benchmark).
*   `iad_benchmark.ipynb`: Interactive Jupyter Notebook for research and visualization.
*   `tests/`: Unit tests for verifying mathematical and image processing logic.
*   `experiments/`: Plotting scripts for research papers (`plot_*.py`) and hardware utilities.
*   `docs/`: Research papers, thesis sections, and project documentation.
*   `3D/`: CAD models for the physical experimental setup.
*   `arduino/`: Firmware for controlling the LED light source array.

## 🛠️ Setup & Installation

### Environment
You can set up the required Python environment using the provided `environment.yml`.

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate defect_vision
```

## 📖 Usage
Run the complete pipeline (PS → Dataset → Training → Evaluation) using `main.py`:

```bash
python main.py ^
    --raw_dir D:\IAD\data_scan\dataset\raw_captures ^
    --out_dir mvtec_dataset ^
    --calib_npy D:\IAD\data_scan\dataset\light_directions_12.npy ^
    --output_mode after ^
    --visualize ^
    --viz_dir heatmaps
```

*Use `--skip_build` if your MVTec dataset is already prepared.*

## 📝 References & Citation
- Woodham, R. J. (1980). *Photometric method for determining surface orientation from multiple images.*
- Stallard *et al.* (2018). *Uncertainty in Manual Visual Inspection.* Journal of Manufacturing Systems.
- Li *et al.* (2021). *CutPaste: Self-Supervised Learning for Anomaly Detection and Localization.* CVPR.
- Schlüter *et al.* (2023). *Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization.*
- Batzner *et al.* (2024). *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.* WACV.

If you use this work in your research, please refer to the documents in the `docs/` folder for citation details.
