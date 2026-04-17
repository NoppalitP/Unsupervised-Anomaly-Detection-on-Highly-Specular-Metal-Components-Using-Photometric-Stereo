from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
import platform
import torch

def _get_default_workers() -> int:
    """Returns optimal num_workers based on OS to avoid Windows DataLoader issues."""
    if platform.system() == "Windows":
        return 0
    return min(4, os.cpu_count() or 1)

@dataclass
class PSConfig:
    drop_dark: int = 0
    drop_bright: int = 0
    lambda_reg: float = 1e-5
    output_mode: str = "after"
    before_light_idx: int = 0

@dataclass
class DataConfig:
    raw_dir: Path
    out_dir: Path
    calib_npy: Optional[Path] = None
    n_lights: int = 12
    slant_deg: float = 45.0
    output_size: int = 256
    crop_offset: int = 12
    train_ratio: float = 0.8
    batch_size: int = 16
    num_workers: int = field(default_factory=_get_default_workers)
    seed: int = 42

@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Model Specific
    pc_coreset: float = 0.10
    padim_reg: float = 0.01
    ssn_proj: int = 256
    ssn_lr: float = 1e-3
    ssn_epochs: int = 150
    ssn_noise: float = 0.15
    ssn_alpha: float = 0.5
    cae_lr: float = 1e-3
    cae_epochs: int = 50
    cae_latent: int = 256
    draem_lr: float = 1e-4
    draem_epochs: int = 50
    draem_noise: float = 0.15

@dataclass
class AppConfig:
    ps: PSConfig
    data: DataConfig
    model: ModelConfig
    visualize: bool = False
    viz_dir: Path = Path("heatmaps")
    output_csv: str = "benchmark_results.csv"
    save_models: bool = False
    models_dir: Path = Path("checkpoints")
