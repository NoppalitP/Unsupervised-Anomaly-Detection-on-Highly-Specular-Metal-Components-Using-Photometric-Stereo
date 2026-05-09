import logging
import time
import json
import numpy as np
import pandas as pd
from dataclasses import asdict
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from src.config.config import AppConfig
from src.core.solver import PhotometricStereoSolver
from src.data.builder import AutoCropper, MVTecDatasetBuilder, MVTecDataset
from src.models.iad_models import BackboneExtractor, PatchCore, PaDiM, SuperSimpleNet, CAEModel, DRAEMModel
from src.utils.viz import visualize_heatmaps

log = logging.getLogger(__name__)

class IADPipeline:
    """The orchestrator for the full IAD benchmarking suite with Metadata Logging."""
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup()

    def _setup(self):
        # 1. Load Calibration
        if self.config.data.calib_npy and self.config.data.calib_npy.exists():
            L = np.load(self.config.data.calib_npy)
        else:
            log.warning("Calibration file missing. Building theoretical L matrix.")
            L = self._build_theoretical_L()

        # 2. Components
        self.solver = PhotometricStereoSolver(L, self.config.ps, self.config.model.device)
        self.cropper = AutoCropper(self.config.data)
        self.builder = MVTecDatasetBuilder(self.solver, self.cropper, self.config.data)

    def _build_theoretical_L(self):
        n = self.config.data.n_lights
        slant = np.radians(self.config.data.slant_deg)
        L = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            az = 2.0 * np.pi * i / n
            L[i, 0] = np.cos(az) * np.sin(slant)
            L[i, 1] = np.sin(az) * np.sin(slant)
            L[i, 2] = np.cos(slant)
        return L

    def save_metadata(self):
        """Saves experimental configurations to address Reviewer reproducibility concerns."""
        meta_path = Path(self.config.output_csv).with_suffix(".json")
        
        # Deep convert dataclass to dict and fix non-serializable objects
        def _fix_types(obj):
            if isinstance(obj, dict):
                return {k: _fix_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_fix_types(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            # Use type name check to avoid dependency issues if needed, or just ensure torch is accessible
            elif "torch.device" in str(type(obj)):
                return str(obj)
            return obj

        meta_dict = _fix_types(asdict(self.config))
        with open(meta_path, 'w') as f:
            json.dump(meta_dict, f, indent=4)
        log.info(f"Experimental metadata saved to {meta_path}")

    def run(self, skip_build: bool = False):
        if not skip_build:
            self.builder.build()

        self.save_metadata()

        # Data Loading
        _MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        tf = transforms.Compose([
            transforms.Resize(self.config.data.output_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD)
        ])
        train_dl = DataLoader(MVTecDataset(self.config.data.out_dir, "train", tf), 
                              batch_size=self.config.data.batch_size, 
                              shuffle=True,
                              num_workers=self.config.data.num_workers,
                              pin_memory=True if self.config.data.num_workers > 0 else False)
        test_dl = DataLoader(MVTecDataset(self.config.data.out_dir, "test", tf), 
                             batch_size=self.config.data.batch_size,
                             num_workers=self.config.data.num_workers,
                             pin_memory=True if self.config.data.num_workers > 0 else False)

        # Extractor
        extractor = BackboneExtractor(self.config.model)
        
        # Benchmarking
        models_to_test = {
            "PatchCore": PatchCore(extractor, self.config.model),
            "PaDiM": PaDiM(extractor, self.config.model),
            "SuperSimpleNet": SuperSimpleNet(extractor, self.config.model),
            "CAE": CAEModel(self.config.model),
            "DRAEM": DRAEMModel(self.config.model)
        }
        
        results = []
        for name, model in models_to_test.items():
            log.info(f"--- Benchmarking {name} ---")
            
            # Ensure perfect reproducibility for each model independent of execution order
            import random
            import os
            seed = self.config.data.seed
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            t0 = time.time()
            if hasattr(model, "fit"): model.fit(train_dl)
            t_train = time.time() - t0
            
            t1 = time.time()
            sc, lb, mp = model.predict(test_dl)
            t_infer = time.time() - t1
            
            auroc = roc_auc_score(lb, sc)
            log.info(f"{name} AUROC: {auroc:.4f}")
            
            results.append({
                "Model": name,
                "Backbone": self.config.model.backbone,
                "AUROC": round(auroc, 4),
                "Train_Time": round(t_train, 2),
                "Infer_Time": round(t_infer, 2)
            })
            
            if self.config.visualize:
                visualize_heatmaps(name, test_dl, mp, lb, self.config.viz_dir)
                
            if self.config.save_models:
                self.config.models_dir.mkdir(parents=True, exist_ok=True)
                model_path = self.config.models_dir / f"{name}_{self.config.model.backbone}.pt"
                if hasattr(model, "save"):
                    model.save(model_path)
                elif isinstance(model, torch.nn.Module):
                    torch.save(model.state_dict(), model_path)
                log.info(f"Model checkpoint saved to {model_path}")

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(self.config.output_csv, index=False)
        log.info(f"Benchmark summary saved to {self.config.output_csv}")
        return df
