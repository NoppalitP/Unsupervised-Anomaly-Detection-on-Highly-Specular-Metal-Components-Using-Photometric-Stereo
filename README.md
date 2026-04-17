# Unsupervised Anomaly Detection on Highly Specular Metal Components Using Photometric Stereo

This repository contains the implementation of an end-to-end Industrial Anomaly Detection (IAD) pipeline specifically designed for the inspection of highly specular metal surfaces. By integrating Photometric Stereo (PS) with state-of-the-art Unsupervised Anomaly Detection (UAD) models, this framework systematically mitigates specular highlights to reveal subtle surface defects such as scratches, dents, and stains.

## Key Methodologies

1. **Acquisition - Photometric Stereo (PS):** 
   Decouples surface geometry (Normal Map) from appearance (Albedo Map) utilizing a multi-light setup and a GPU-accelerated Weighted Least Squares (WLS) solver, which includes an outlier rejection mechanism for robustness against specular noise.
2. **Detection - UAD Benchmarking:** 
   Evaluates five distinct architectural paradigms (PatchCore, PaDiM, SuperSimpleNet, CAE, and DRAEM) to establish the most robust methodology for industrial deployment.
3. **Deterministic Reproducibility:** 
   The pipeline enforces strict random seeding across the Python, NumPy, PyTorch, and CuDNN environments. This ensures 100% exact reproducibility, satisfying the rigorous standards required for academic research and empirical validation.
4. **Metadata Logging:** 
   Automatically serializes detailed JSON logs encompassing all hyperparameters and hardware configurations to guarantee experimental transparency and rigor.

## Benchmark Results (Before vs. After PS)

The following empirical results demonstrate the substantial performance improvement achieved by utilizing Photometric Stereo Normal Maps as opposed to raw single-light images. All reported metrics are 100% reproducible based on the provided pipeline.

| Model | Before PS (Raw Image) | After PS (Normal Map) | Improvement (Gain) |
| :--- | :---: | :---: | :---: |
| **PaDiM** | 0.9310 | **0.9881** | **+5.71%** |
| **PatchCore** | 0.9310 | **0.9833** | **+5.23%** |
| **SuperSimpleNet** | 0.9214 | **0.9524** | **+3.10%** |
| **DRAEM** | 0.5119 | **0.7500** | **+23.81%** |
| **CAE (Baseline)** | 0.5524 | **0.7357** | **+18.33%** |

*Note: Performance is quantified using Area Under the Receiver Operating Characteristic Curve (AUROC), evaluated on a proprietary dataset comprising specular metal components.*

## Modular Project Structure

The repository adheres to a strict Separation of Concerns (SoC) architecture to facilitate scalability and maintainability:

```text
src/
├── config/     # Dataclass-based Configuration Management
├── core/       # Physics and Mathematics (Photometric Stereo WLS Solver)
├── data/       # Dataset Generation, AutoCropping, and Dataloaders
├── models/     # Modular UAD Model Implementations (PatchCore, DRAEM, etc.)
├── utils/      # Analytical Utilities (Visualization, Heatmaps, Logging)
└── pipeline.py # Primary Orchestrator
main.py         # Pipeline Entry Point and CLI Argument Parsing
```

## Setup and Installation

**Environment Configuration:**
```bash
conda env create -f environment.yml
conda activate defect_vision
```

## Usage and Execution

### 1. Execute "After PS" Benchmark (Primary Pipeline)
This protocol processes raw captures into Normal Maps, evaluates the models, and serializes the state dictionaries (checkpoints):
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

### 2. Execute "Before PS" Benchmark (Baseline Comparison)
This protocol evaluates the models directly on masked raw images to establish a comparative baseline:
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

### 3. Verification of Reproducibility
Each execution automatically synthesizes a `.json` metadata ledger (e.g., `benchmark_results_after.json`). This ledger explicitly records:
*   Backbone architecture and intrinsic feature dimensions.
*   Model-specific hyperparameters (e.g., learning rates, epoch counts, coreset subsampling ratios).
*   Photometric Stereo computational parameters (e.g., regularization terms, intensity truncation bounds).
*   Global random seed configurations.

## References
- Woodham, R. J. (1980). *Photometric method for determining surface orientation from multiple images.* Optical Engineering, 19(1), 139-144.
- Stallard, C., et al. (2018). *Uncertainty in Manual Visual Inspection.* 
- Batzner, K., et al. (2024). *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.* Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
