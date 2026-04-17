# Unsupervised Anomaly Detection on Highly Specular Metal Components Using Photometric Stereo

This project implements a modular, production-ready **Industrial Anomaly Detection (IAD)** pipeline designed to inspect highly specular metal surfaces. By combining **Photometric Stereo (PS)** with State-of-the-Art (SOTA) Unsupervised Anomaly Detection (UAD) models, the system effectively neutralizes distracting specular highlights and reveals subtle surface defects like scratches, dents, and stains.

## 🌟 Key Methodologies

1.  **Acquisition - Photometric Stereo (PS):** Decouples surface geometry (Normal Map) from appearance (Albedo Map) using a multi-light setup and a GPU-accelerated WLS solver with outlier rejection.
2.  **Detection - UAD Benchmarking:** Evaluates 5 architectural families (PatchCore, PaDiM, SuperSimpleNet, CAE, DRAEM) to identify the most robust solution for industrial deployment.
3.  **100% Deterministic & Reproducible:** The pipeline enforces strict random seeding across Python, NumPy, PyTorch, and CuDNN, ensuring exact reproducibility for academic research and production validation.
4.  **Metadata Logging:** Automatically generates detailed JSON logs of all hyperparameters and configurations to ensure experimental rigor and transparency.

## 📊 Benchmark Results (Before vs. After PS)

The following results demonstrate the significant performance gain achieved by using Photometric Stereo Normal Maps compared to raw single-light images. **All results are 100% reproducible.**

| Model | **Before PS (Raw Image)** | **After PS (Normal Map)** | **Improvement (Gain)** |
| :--- | :---: | :---: | :---: |
| **PaDiM** | 0.9310 | **0.9881** | **+5.71%** |
| **PatchCore** | 0.9310 | **0.9833** | **+5.23%** |
| **SuperSimpleNet** | 0.9214 | **0.9524** | **+3.10%** |
| **DRAEM** | 0.5119 | **0.7500** | **+23.81%** |
| **CAE (Baseline)** | 0.5524 | **0.7357** | **+18.33%** |

*Note: AUROC scores calculated on a custom dataset of specular metal components.*

## 📂 Modular Project Structure

The project follows a "Separation of Concerns" (SoC) architecture:

```text
src/
├── config/     # Dataclass-based Configuration Management
├── core/       # Physics/Math (Photometric Stereo WLS Solver)
├── data/       # Dataset Building, AutoCropping, and Data Loaders
├── models/     # Modular UAD Model Wrappers (PatchCore, DRAEM, etc.)
├── utils/      # Visualization (Anomaly Heatmaps) and Logging
└── pipeline.py # The Main Orchestrator
main.py         # Entry Point (CLI Argument Parsing)
```

## 🛠️ Setup & Installation

**Environment Setup:**
```bash
conda env create -f environment.yml
conda activate defect_vision
```

## 📖 Usage

### 1. Run "After PS" Benchmark (Primary)
Processes raw captures into Normal Maps, evaluates the models, and saves checkpoints:
```powershell
python main.py `
    --raw_dir D:\IAD\data_scan\dataset\raw_captures `
    --out_dir mvtec_dataset_after `
    --output_mode after `
    --output_csv benchmark_results_after.csv `
    --visualize `
    --viz_dir heatmaps_after `
    --save_models `
    --models_dir checkpoints_after
```

### 2. Run "Before PS" Comparison
Evaluates models directly on masked raw images:
```powershell
python main.py `
    --raw_dir D:\IAD\data_scan\dataset\raw_captures `
    --out_dir mvtec_dataset_before `
    --output_mode before `
    --output_csv benchmark_results_before.csv `
    --visualize `
    --viz_dir heatmaps_before `
    --save_models `
    --models_dir checkpoints_before
```

### 3. Reproducibility
Every run automatically generates a `.json` metadata file (e.g., `benchmark_results_after.json`) containing:
*   Backbone architecture and feature dimensions.
*   Hyperparameters (Learning rates, epochs, coreset ratios).
*   PS Solver settings (Drop dark/bright counts, lambda).
*   Random Seed configuration.

## 📝 References
- Woodham, R. J. (1980). *Photometric method for determining surface orientation from multiple images.*
- Stallard *et al.* (2018). *Uncertainty in Manual Visual Inspection.*
- Batzner *et al.* (2024). *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.*
