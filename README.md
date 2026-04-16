# Unsupervised Anomaly Detection on Highly Specular Metal Components Using Photometric Stereo

This project implements **Industrial Anomaly Detection (IAD)** techniques specifically leveraging **Photometric Stereo (PS)**. By using multiple light directions with a fixed camera position, we can estimate surface normals and isolate defects (like scratches and dents) from surface texture and stains on highly specular metal surfaces.

## 🚀 Features
- **Photometric Stereo Pipeline**: Estimate surface normals, albedo, and depth maps.
- **Anomaly Detection**: Algorithms for identifying surface defects in industrial parts.
- **Dataset Management**: Tools for building and preprocessing MVTec-style datasets.
- **Benchmarking**: Scripts to evaluate performance against standard benchmarks.

## 📂 Project Structure
The repository is organized as follows:

*   `src/`: Core logic including the main Photometric Stereo implementation and training scripts.
*   `experiments/`: Benchmarking scripts and figure generation for research papers.
*   `docs/`: Research papers, thesis sections, and project documentation.
*   `image/`: Visualization results, defect heatmaps, and experimental setup photos.
*   `3D/`: CAD models for the physical experimental setup (casing, mounts).
*   `arduino/`: Firmware for controlling the light source array.
*   `archive/`: Historical versions of scripts for reference.

## 🛠️ Setup & Installation

### Environment
You can set up the required Python environment using the provided `environment.yml` (for Conda) or `requirements_colab.txt` (for pip/Google Colab).

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate iad-env
```

**Using Pip:**
```bash
pip install -r requirements_colab.txt
```

## 📖 Usage
1.  **Build Dataset**: Use `src/build_dataset.py` to prepare your raw captures.
2.  **Run Photometric Stereo**: Execute `src/photometric_stereo.py` to process images and generate normal maps.
3.  **Train Anomaly Detection**: Use `src/train.ipynb` or `src/thesis_experiment_full.py` to train detection models.

## 📝 Citation
If you use this work in your research, please refer to the documents in the `docs/` folder for citation details.
