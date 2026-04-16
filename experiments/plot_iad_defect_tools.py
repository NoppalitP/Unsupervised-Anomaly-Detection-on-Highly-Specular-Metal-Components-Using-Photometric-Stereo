# =============================================================================
# generate_killer_figure.py
#
# Automatically generates and visualizes "Before vs After" comparisons
# for ALL defect images using Photometric Stereo Anomaly Detection.
# 
# Pipeline:
#   1. Train PaDiM on 2D Before images (mvtec_dataset_before/train/good)
#   2. Train PaDiM on 3D After Normal Maps (mvtec_dataset_after/train/good)
#   3. Iterate over ALL matching defect images in the test set.
#   4. Calculate the spatial anomaly score map for both models.
#   5. Generate and save a 2x3 Matplotlib figure (300 DPI) for each defect.
# =============================================================================

import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED        = 42
    IMG_SIZE    = 256
    CROP_SIZE   = 224
    BATCH_SIZE  = 16
    NUM_WORKERS = 2
    PADIM_REG   = 0.01  # Stability for CPU MKL pinv
    MAX_DIMS    = 512   # Random dimensionality reduction for PaDiM

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# DATASET & UTILS
# =============================================================================
def _make_tf() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.CenterCrop(CFG.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

class TrainGoodDataset(Dataset):
    """Loads only the 'train/good' folder for PaDiM fitting."""
    def __init__(self, dataset_dir: Path):
        self.transform = _make_tf()
        self.samples = sorted((dataset_dir / "train" / "good").glob("*.png"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.samples[idx]
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise IOError(f"Could not read {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(img_rgb)
        return self.transform(pil), path.name

def load_single_image(path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Returns the preprocessed tensor and the cropped raw RGB image for visualization."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise IOError(f"Could not read {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Process for visualization (matching center crop)
    pil = PILImage.fromarray(img_rgb)
    pil_resized = pil.resize((CFG.IMG_SIZE, CFG.IMG_SIZE), PILImage.BILINEAR)
    left = (CFG.IMG_SIZE - CFG.CROP_SIZE) // 2
    top = (CFG.IMG_SIZE - CFG.CROP_SIZE) // 2
    pil_cropped = pil_resized.crop((left, top, left + CFG.CROP_SIZE, top + CFG.CROP_SIZE))
    img_vis = np.array(pil_cropped)

    # Process for model
    tensor = _make_tf()(pil).unsqueeze(0).to(CFG.DEVICE)
    return tensor, img_vis

# =============================================================================
# MODELS
# =============================================================================
class BackboneExtractor(nn.Module):
    """Frozen ConvNeXt-Tiny preserving 14x14 spatial dimensions."""
    def __init__(self, device: torch.device = CFG.DEVICE) -> None:
        super().__init__()
        net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for p in net.parameters():
            p.requires_grad_(False)
        net.eval().to(device)

        self.net = net
        self.device = device
        self._hooks: Dict[str, torch.Tensor] = {}
        self._handles = []

        for name in ["features.4", "features.6"]:
            mod = self._get_mod(name)
            self._handles.append(mod.register_forward_hook(self._hook(name)))

        self.feat_dim = self._probe()

    def _get_mod(self, name: str) -> nn.Module:
        m = self.net
        for p in name.split("."):
            m = getattr(m, p)
        return m

    def _hook(self, name: str):
        def fn(_, __, out):
            self._hooks[name] = out.detach()
        return fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._hooks.clear()
        _ = self.net(x.to(self.device))
        parts = [F.adaptive_avg_pool2d(v, (14, 14)) for v in self._hooks.values()]
        return F.normalize(torch.cat(parts, dim=1), dim=1) # (B, C, 14, 14)

    def _probe(self) -> int:
        dummy = torch.zeros(1, 3, CFG.CROP_SIZE, CFG.CROP_SIZE, device=self.device)
        with torch.no_grad():
            return self.forward(dummy).shape[1]

class PaDiM:
    """Per-patch Multivariate Gaussian anomaly detection."""
    def __init__(self, extractor: BackboneExtractor) -> None:
        self.ext = extractor
        self.mu: Optional[torch.Tensor] = None
        self.cov_inv: Optional[torch.Tensor] = None
        self.idx: Optional[torch.Tensor] = None

    def fit(self, loader: DataLoader) -> None:
        feats = []
        for x, _ in tqdm(loader, leave=False, desc="Extracting Train Features"):
            with torch.no_grad():
                f = self.ext(x).cpu()
            feats.append(f)
            
        X = torch.cat(feats, 0).float() # (N, C, 14, 14)
        D = X.shape[1]
        
        if D > CFG.MAX_DIMS:
            torch.manual_seed(42)
            self.idx = torch.randperm(D)[:CFG.MAX_DIMS]
            X = X[:, self.idx]
        else:
            self.idx = torch.arange(D)
            
        X = X.permute(0, 2, 3, 1) # (N, 14, 14, C)
        N, H, W, C = X.shape
        
        mu = X.mean(0) # (14, 14, C)
        Xc = X - mu    # (N, 14, 14, C)
        
        cov = torch.einsum('nhwc,nhwd->hwcd', Xc, Xc) / max(N - 1, 1)
        I = torch.eye(C, dtype=cov.dtype, device=cov.device).view(1, 1, C, C).expand(H, W, C, C)
        cov = cov + I * CFG.PADIM_REG
        
        self.mu = mu
        # Using pseudo-inverse to guarantee stability and avoid lu_solve errors
        self.cov_inv = torch.linalg.pinv(cov)

    @torch.no_grad()
    def predict_single(self, x: torch.Tensor) -> np.ndarray:
        """Takes a (1, C, H, W) tensor, returns a (14, 14) anomaly map."""
        mu  = self.mu.to(self.ext.device)
        inv = self.cov_inv.to(self.ext.device)
        
        f = self.ext(x)
        if self.idx is not None:
            f = f[:, self.idx.to(self.ext.device)]
            
        f = f.permute(0, 2, 3, 1) # (1, 14, 14, C)
        diff = f - mu # (1, 14, 14, C)
        
        diff_u = diff.unsqueeze(-2) # (1, 14, 14, 1, C)
        m = torch.matmul(diff_u, inv.unsqueeze(0)) 
        m = torch.matmul(m, diff_u.transpose(-1, -2)).squeeze(-1).squeeze(-1) # (1, 14, 14)
        
        sc_map = torch.sqrt(m.clamp(min=1e-6)).squeeze(0).cpu().numpy()
        return sc_map

# =============================================================================
# VISUALIZATION UTILS
# =============================================================================
def process_heatmap(hmap: np.ndarray, target_size: int = CFG.CROP_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Resizes, Blurs, and Normalizes a 14x14 heatmap to target size."""
    hmap_resized = cv2.resize(hmap, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    hmap_smoothed = cv2.GaussianBlur(hmap_resized, (7, 7), 0)
    
    h_min, h_max = hmap_smoothed.min(), hmap_smoothed.max()
    hmap_norm = (hmap_smoothed - h_min) / (h_max - h_min + 1e-8)
    
    heatmap_color = cv2.applyColorMap((hmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    return hmap_norm, heatmap_color

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    set_seed(CFG.SEED)
    
    dir_before = Path("mvtec_dataset_before")
    dir_after  = Path("mvtec_dataset_after")
    
    out_dir = Path("comparison_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dir_before.exists() or not dir_after.exists():
        log.error("Dataset directories not found. Please ensure 'mvtec_dataset_before' and 'mvtec_dataset_after' exist.")
        return

    # 1. Initialize Extractor
    log.info("Initializing Feature Extractor...")
    extractor = BackboneExtractor()

    # 2. Train PaDiM Before
    log.info("Training PaDiM on 'Before' Dataset (2D)...")
    ds_train_before = TrainGoodDataset(dir_before)
    dl_train_before = DataLoader(ds_train_before, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)
    padim_before = PaDiM(extractor)
    padim_before.fit(dl_train_before)

    # 3. Train PaDiM After
    log.info("Training PaDiM on 'After' Dataset (3D)...")
    ds_train_after = TrainGoodDataset(dir_after)
    dl_train_after = DataLoader(ds_train_after, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)
    padim_after = PaDiM(extractor)
    padim_after.fit(dl_train_after)

    # 4. Find all defect images that exist in both datasets
    defect_paths_before = list((dir_before / "test").rglob("*.png"))
    defect_paths_before = [p for p in defect_paths_before if p.parent.name != "good"]
    
    paired_defects = []
    for p_bef in defect_paths_before:
        # Construct corresponding path in the 'after' directory
        rel_path = p_bef.relative_to(dir_before)
        p_aft = dir_after / rel_path
        if p_aft.exists():
            paired_defects.append((p_bef, p_aft))

    if not paired_defects:
        log.error("No matching defect images found between 'before' and 'after' test sets.")
        return

    log.info(f"Found {len(paired_defects)} paired defect images. Generating all comparisons...")

    # 5. Generate and save figures for ALL defect images
    for p_bef, p_aft in tqdm(paired_defects, desc="Evaluating and Saving Pairs"):
        t_bef, img_bef = load_single_image(p_bef)
        t_aft, img_aft = load_single_image(p_aft)

        raw_map_bef = padim_before.predict_single(t_bef)
        raw_map_aft = padim_after.predict_single(t_aft)

        # Normalize maps
        norm_bef, color_bef = process_heatmap(raw_map_bef)
        norm_aft, color_aft = process_heatmap(raw_map_aft)

        # Contrast Metric just for information/titles
        contrast_score = np.mean(norm_bef) - np.mean(norm_aft)

        # 6. Generate Publication-Ready Figure
        overlay_bef = cv2.addWeighted(img_bef, 0.4, color_bef, 0.6, 0)
        overlay_aft = cv2.addWeighted(img_aft, 0.4, color_aft, 0.6, 0)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
        fig.patch.set_facecolor('white')
        
        defect_type = p_bef.parent.name
        fig.suptitle(f"Defect: {defect_type} | Image: {p_bef.name} | Contrast Score: {contrast_score:.4f}", fontsize=18)

        # Row 1: Before
        axes[0, 0].imshow(img_bef);   axes[0, 0].set_title("2D (Before PS) - Raw Image", fontsize=16)
        axes[0, 1].imshow(color_bef); axes[0, 1].set_title("2D PaDiM Heatmap", fontsize=16)
        axes[0, 2].imshow(overlay_bef); axes[0, 2].set_title("2D Overlay (False Alarms)", fontsize=16)

        # Row 2: After
        axes[1, 0].imshow(img_aft);   axes[1, 0].set_title("3D (After PS) - Normal Map", fontsize=16)
        axes[1, 1].imshow(color_aft); axes[1, 1].set_title("3D PaDiM Heatmap", fontsize=16)
        axes[1, 2].imshow(overlay_aft); axes[1, 2].set_title("3D Overlay (Clean Localization)", fontsize=16)

        # Clean up axes
        for ax in axes.flatten():
            ax.axis("off")

        plt.tight_layout()
        
        # Save each image uniquely
        out_filename = f"{defect_type}_{p_bef.stem}_comparison.png"
        out_filepath = out_dir / out_filename
        plt.savefig(out_filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    log.info(f"Success! Saved {len(paired_defects)} visual comparisons to the '{out_dir}' directory.")

if __name__ == "__main__":
    main()