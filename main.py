import argparse
import logging
import sys
import random
import numpy as np
import torch
from pathlib import Path

from src.config.config import AppConfig, PSConfig, DataConfig, ModelConfig
from src.pipeline import IADPipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("IAD")

import os

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser(description="Industrial Anomaly Detection Pipeline")
    
    # Data Paths
    p.add_argument("--raw_dir", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--calib_npy", type=Path)
    
    # Execution Flags
    p.add_argument("--skip_build", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--viz_dir", type=Path, default=Path("heatmaps"))
    p.add_argument("--output_csv", default="benchmark_results.csv")
    p.add_argument("--save_models", action="store_true")
    p.add_argument("--models_dir", type=Path, default=Path("checkpoints"))
    
    # PS Settings
    p.add_argument("--output_mode", default="after", choices=["after", "before"])
    
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    # Initialize Config Objects
    ps_cfg = PSConfig(output_mode=args.output_mode)
    data_cfg = DataConfig(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        calib_npy=args.calib_npy
    )
    model_cfg = ModelConfig() # Default settings

    app_cfg = AppConfig(
        ps=ps_cfg,
        data=data_cfg,
        model=model_cfg,
        visualize=args.visualize,
        viz_dir=args.viz_dir,
        output_csv=args.output_csv,
        save_models=args.save_models,
        models_dir=args.models_dir
    )

    # Initialize and Run Pipeline
    pipeline = IADPipeline(app_cfg)
    pipeline.run(skip_build=args.skip_build)

if __name__ == "__main__":
    main()
