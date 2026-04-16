# Unsupervised Anomaly Detection on Highly Specular Metal Components Using Photometric Stereo

This project implements **Industrial Anomaly Detection (IAD)** techniques specifically leveraging **Photometric Stereo (PS)**. By using multiple light directions with a fixed camera position, we can estimate surface normals and isolate defects (like scratches and dents) from surface texture and stains on highly specular metal surfaces.

## 🚀 Features
- **Photometric Stereo Pipeline**: Estimate surface normals, albedo, and depth maps.
- **Anomaly Detection**: Algorithms for identifying surface defects in industrial parts.
- **Dataset Management**: Tools for building and preprocessing MVTec-style datasets.
- **Benchmarking**: Scripts to evaluate performance against standard benchmarks.

## 📂 Project Structure
The repository is organized as follows:

*   `src/`: Core logic including the main `iad_main_pipeline.py`, `iad_dataset_generator.py`, and `iad_thesis_grid_search.py`.
*   `experiments/`: Plotting scripts for research papers (`plot_*.py`) and hardware utilities.
*   `docs/`: Research papers, thesis sections, and project documentation.
*   `image/`: Visualization results, defect heatmaps, and experimental setup photos.
*   `3D/`: CAD models for the physical experimental setup (casing, mounts).
*   `arduino/`: Firmware for controlling the light source array.
*   `archive/`: Historical versions of scripts for reference.

## 🛠️ Setup & Installation

### Environment
You can set up the required Python environment using the provided `environment.yml` (for Conda) or `iad_setup_colab.py` (for Google Colab).

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate iad-env
```

## 📖 Usage
1.  **Run End-to-End Benchmark**: Use `src/iad_main_pipeline.py` for the complete pipeline (PS → Dataset → Training → Evaluation).
2.  **Build Dataset**: Use `src/iad_dataset_generator.py` to prepare your raw captures in MVTec format.
3.  **Run Thesis Experiments**: Use `src/iad_thesis_grid_search.py` to run the 30-experiment grid search benchmark.
4.  **Generate Figures**: All figure generation scripts (e.g., `plot_iad_ps_results.py`, `plot_iad_setup_diagram.py`) are located in the `experiments/` folder.

## 📝 Citation
If you use this work in your research, please refer to the documents in the `docs/` folder for citation details.
