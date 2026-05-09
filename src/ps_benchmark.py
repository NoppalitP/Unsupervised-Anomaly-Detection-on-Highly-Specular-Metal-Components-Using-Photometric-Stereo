# =============================================================================
# ps_benchmark.py
#
# End-to-End Pipeline: Photometric Stereo → A/B Benchmark
# ─────────────────────────────────────────────────────────
# Stage 1 │ PS Solver      : raw light_*.png → "before" or "after" image
# Stage 2 │ AutoCropper    : perspective-warp ROI to fixed square
# Stage 3 │ MVTec Builder  : write train/test folder structure
# Stage 4 │ Benchmark      : evaluate 5 UAD models × 2 representations
#
# Models (5 architectural families)
# ──────────────────────────────────
#  1. PatchCore       Memory-bank nearest-neighbour   (Feature-based)
#  2. PaDiM           Per-patch Multivariate Gaussian (Statistical)
#  3. SuperSimpleNet  Discriminator + GRL + synthetic hard anomalies
#  4. CAE             Convolutional AutoEncoder reconstruction baseline
#  5. DRAEM           DRAEM-style reconstruction + spatial discriminator
#
# Localization (NEW)
# ──────────────────
#  All 5 models now produce (H, W) spatial anomaly maps.
#  Image-level score = max over map (preserves AUROC ranking).
#  --visualize saves {model}_localization_heatmap.png (one defect sample).
#
# Usage (Windows CMD):
#   python ps_benchmark.py ^
#       --raw_dir   D:\IAD\data_scan\dataset\raw_captures ^
#       --out_dir   mvtec_dataset ^
#       --calib_npy D:\IAD\data_scan\dataset\light_directions_12.npy ^
#       --n_lights  12  --slant_deg 45 ^
#       --drop_dark 0   --drop_bright 0 ^
#       --output_mode after ^
#       --output_size 256 ^
#       --train_ratio 0.8 --seed 42 ^
#       --backbone convnext_tiny ^
#       --batch_size 16 --ssn_epochs 30 --ae_epochs 50 ^
#       --output_csv results.csv ^
#       --visualize --viz_dir heatmaps/
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# 0. GLOBAL CONFIG & SEED
# =============================================================================
class CFG:
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED         = 42
    IMG_SIZE     = 256
    CROP_SIZE    = 224
    BATCH_SIZE   = 16
    NUM_WORKERS  = 2
    SPATIAL_GRID = 14   # backbone feature maps pooled to G×G for patch scoring

    PC_CORESET   = 0.10
    PADIM_REG    = 1e-4
    PADIM_DIMS   = 512

    SSN_PROJ     = 256
    SSN_LR       = 1e-3
    SSN_EPOCHS   = 150
    SSN_NOISE    = 0.15
    SSN_ALPHA    = 0.5

    CAE_LR       = 1e-3
    CAE_EPOCHS   = 50
    CAE_LATENT   = 256

    DRAEM_LR     = 1e-4
    DRAEM_EPOCHS = 50
    DRAEM_NOISE  = 0.15

    VIZ_SIGMA    = 4
    VIZ_ALPHA    = 0.45


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. PHOTOMETRIC STEREO SOLVER  (unchanged from original)
# =============================================================================
class PhotometricStereoSolver:
    def __init__(self, L_matrix, drop_dark=2, drop_bright=5,
                 lambda_reg=1e-5, output_mode="after",
                 before_light_idx=0, device=None):
        if L_matrix.ndim != 2 or L_matrix.shape[1] != 3:
            raise ValueError("L_matrix must be shape (N_lights, 3).")
        if output_mode not in ("after", "before"):
            raise ValueError(f"output_mode must be after/before, got {output_mode}.")
        self.L_matrix         = L_matrix.astype(np.float32)
        self.n_lights         = L_matrix.shape[0]
        self.drop_dark        = drop_dark
        self.drop_bright      = drop_bright
        self.lambda_reg       = lambda_reg
        self.output_mode      = output_mode
        self.before_light_idx = before_light_idx
        self.device           = device or CFG.DEVICE
        log.info("PSolver | lights=%d | mode=%s | device=%s",
                 self.n_lights, output_mode, self.device)

    def solve(self, images):
        if len(images) != self.n_lights:
            raise ValueError(f"Expected {self.n_lights} images, got {len(images)}.")
        gray_stack = self._to_gray_stack(images)
        h, w       = gray_stack.shape[1], gray_stack.shape[2]
        valid_mask = self._build_object_mask(images, h, w)
        if self.output_mode == "before":
            idx           = self.before_light_idx % len(images)
            single_masked = images[idx].copy()
            single_masked[~valid_mask] = 0
            return single_masked
        I_valid  = gray_stack[:, valid_mask].T
        n_pixels = I_valid.shape[0]
        if n_pixels == 0:
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[..., :2] = 128
            return out
        W      = self._build_weight_mask(I_valid)
        N_unit = self._wls_solve(I_valid, W, n_pixels)
        nx_map = np.full((h, w), 128, dtype=np.uint8)
        ny_map = np.full((h, w), 128, dtype=np.uint8)
        nz_map = np.zeros((h, w), dtype=np.uint8)
        nx_map[valid_mask] = ((N_unit[:, 0] + 1.) / 2. * 255.).astype(np.uint8)
        ny_map[valid_mask] = ((N_unit[:, 1] + 1.) / 2. * 255.).astype(np.uint8)
        nz_map[valid_mask] = ((N_unit[:, 2] + 1.) / 2. * 255.).astype(np.uint8)
        return np.stack([nz_map, ny_map, nx_map], axis=-1)

    def _to_gray_stack(self, images):
        return np.array([
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if img.ndim == 3 else img.astype(np.float32)
            for img in images], dtype=np.float32)

    def _build_object_mask(self, images, h, w):
        stack   = np.array(images, dtype=np.float32)
        robust  = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray    = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k       = np.ones((5, 5), np.uint8)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final   = np.zeros((h, w), dtype=np.uint8)
        if cnts:
            cv2.drawContours(final, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        return final > 0

    def _build_weight_mask(self, I):
        P, N = I.shape
        idx  = np.argsort(I, axis=1)
        W    = np.ones((P, N), dtype=np.float32)
        px   = np.arange(P)
        for i in range(self.drop_dark):
            W[px, idx[:, i]] = 0.
        for i in range(N - self.drop_bright, N):
            W[px, idx[:, i]] = 0.
        return W

    def _wls_solve(self, I, W, n_pixels):
        dev   = self.device
        L_t   = torch.tensor(self.L_matrix, dtype=torch.float32, device=dev)
        W_t   = torch.tensor(W,             dtype=torch.float32, device=dev)
        I_t   = torch.tensor(I,             dtype=torch.float32, device=dev)
        I_eye = torch.eye(3, dtype=torch.float32, device=dev) * self.lambda_reg
        if dev.type == "cuda":
            torch.cuda.synchronize()
        L_px  = L_t.unsqueeze(0).expand(n_pixels, -1, -1)
        L_W   = W_t.unsqueeze(-1) * L_px
        A     = torch.bmm(L_W.transpose(1, 2), L_px) + I_eye
        B     = torch.bmm(L_W.transpose(1, 2), I_t.unsqueeze(-1))
        N_raw = torch.linalg.solve(A, B).squeeze(-1)
        norm  = torch.linalg.norm(N_raw, dim=1, keepdim=True).clamp(min=self.lambda_reg)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (N_raw / norm).cpu().numpy()


# =============================================================================
# 2. AUTO CROPPER  (unchanged)
# =============================================================================
class AutoCropper:
    def __init__(self, padding=15, output_size=512, crop_offset=12):
        self.padding = padding; self.output_size = output_size; self.crop_offset = crop_offset

    def find_bbox(self, images):
        stack   = np.array(images, dtype=np.float32)
        robust  = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray    = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        return self._order_points(cv2.boxPoints(rect).astype(np.float32))

    def crop_and_resize(self, img, bbox):
        rect = bbox.copy(); o = self.crop_offset
        rect[0] += [ o, o]; rect[1] += [-o, o]; rect[2] += [-o,-o]; rect[3] += [ o,-o]
        s = self.output_size - 1
        dst = np.array([[0,0],[s,0],[s,s],[0,s]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.output_size, self.output_size), flags=cv2.INTER_AREA)

    @staticmethod
    def _order_points(pts):
        out = np.zeros((4,2), dtype=np.float32); s = pts.sum(axis=1)
        out[0] = pts[np.argmin(s)]; out[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1).ravel()
        out[1] = pts[np.argmin(diff)]; out[3] = pts[np.argmax(diff)]
        return out


# =============================================================================
# 3. MVTEC DATASET BUILDER  (unchanged)
# =============================================================================
class MVTecDatasetBuilder:
    _FOLDER_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")

    def __init__(self, raw_dir, out_dir, solver, cropper, train_ratio=0.8, seed=42):
        self.raw_dir = Path(raw_dir); self.out_dir = Path(out_dir)
        self.solver = solver; self.cropper = cropper
        self.train_ratio = train_ratio; self.rng = random.Random(seed)
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"raw_dir not found: {self.raw_dir}")

    def build(self):
        samples = self._discover()
        if not samples:
            log.error("No valid folders in '%s'.", self.raw_dir); return
        grp: Dict[str, List[Path]] = {}
        for folder, cls in samples:
            grp.setdefault(cls, []).append(folder)
        for cls, folders in grp.items():
            self.rng.shuffle(folders)
        queue = []
        for cls, folders in grp.items():
            if cls == "good":
                n = max(1, int(len(folders) * self.train_ratio))
                for i, f in enumerate(folders):
                    queue.append((f, cls, "train" if i < n else "test"))
            else:
                for f in folders: queue.append((f, cls, "test"))
        ok = skipped = 0
        for folder, cls, split in tqdm(queue, desc="Building dataset"):
            try: self._process_one(folder, cls, split); ok += 1
            except Exception as exc:
                log.warning("SKIP '%s' — %s", folder.name, exc); skipped += 1
        log.info("Done. ok=%d | skipped=%d → %s", ok, skipped, self.out_dir)

    def _discover(self):
        out = []
        for e in sorted(self.raw_dir.iterdir()):
            if e.is_dir():
                m = self._FOLDER_RE.match(e.name)
                if m: out.append((e, m.group(1).lower().strip()))
        return out

    def _process_one(self, folder, cls, split):
        img_paths = sorted(folder.glob("light_*.png"))
        if not img_paths: raise FileNotFoundError(f"No light_*.png in {folder}")
        images = []
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None: raise IOError(f"Failed to read {p}")
            images.append(img)
        if len(images) != self.solver.n_lights:
            raise ValueError(f"'{folder.name}' has {len(images)} images, expected {self.solver.n_lights}.")
        ps_map = self.solver.solve(images)
        bbox   = self.cropper.find_bbox(images)
        if bbox is None: raise RuntimeError("AutoCropper found no contour.")
        output_img = self.cropper.crop_and_resize(ps_map, bbox)
        filename   = folder.name + ".png"
        save_path  = self.out_dir / ("train" if split == "train" else "test") / cls / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), output_img)
        if cls != "good":
            gt = self.out_dir / "ground_truth" / cls / filename
            gt.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(gt), np.zeros((self.cropper.output_size, self.cropper.output_size), dtype=np.uint8))


# =============================================================================
# 4. PYTORCH DATASET & DATALOADER
#    CHANGED: __getitem__ now also returns the file path (str) for visualization
# =============================================================================
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _make_tf(augment: bool = False) -> transforms.Compose:
    ops: list = [transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE))]
    if augment:
        ops += [transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    ops += [transforms.CenterCrop(CFG.CROP_SIZE), transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


class MVTecFlatDataset(Dataset):
    """Returns (tensor, label, path_str) — path enables loading original image for viz."""
    def __init__(self, root: Path, split: str, transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        split_dir = root / split
        if not split_dir.exists(): return
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir(): continue
            label = 0 if cls_dir.name.lower() == "good" else 1
            for p in sorted(cls_dir.glob("*.png")):
                self.samples.append((p, label))

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(PILImage.fromarray(img_rgb)), label, str(path)


def build_loaders(dataset_root: Path, batch_size=CFG.BATCH_SIZE,
                  num_workers=CFG.NUM_WORKERS) -> Tuple[DataLoader, DataLoader]:
    train_ds = MVTecFlatDataset(dataset_root, "train", _make_tf(augment=True))
    test_ds  = MVTecFlatDataset(dataset_root, "test",  _make_tf(augment=False))
    log.info("Dataset | train=%d | test=%d", len(train_ds), len(test_ds))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, test_dl


# =============================================================================
# 5. BACKBONE EXTRACTOR  (REDESIGNED — spatial-aware)
# =============================================================================
class BackboneExtractor(nn.Module):
    """
    Frozen backbone with TWO operating modes:

    spatial_forward(x) → (B, C_total, G, G)
        Each hooked stage is independently pooled to SPATIAL_GRID×SPATIAL_GRID
        then channel-concatenated.  Preserves spatial structure for patch-level
        anomaly localization (PatchCore, PaDiM, SuperSimpleNet).

    forward(x)         → (B, D)  L2-normalised flat vector
        Collapses spatial dims via flatten for the SuperSimpleNet MLP head.
        Kept for backward compatibility.

    Attributes exposed:
        spatial_dim : C_total  (channels in spatial map)
        feat_dim    : D = C_total * G * G  (flat dim)
    """

    _LAYER_MAP = {
        "convnext_tiny":   (["features.4", "features.6"],
                            models.convnext_tiny,
                            models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "resnet50":        (["layer2", "layer3"],
                            models.resnet50,
                            models.ResNet50_Weights.IMAGENET1K_V1),
        "efficientnet_b4": (["features.4", "features.6"],
                            models.efficientnet_b4,
                            models.EfficientNet_B4_Weights.IMAGENET1K_V1),
    }

    def __init__(self, backbone_name: str = "convnext_tiny",
                 device: torch.device = CFG.DEVICE) -> None:
        super().__init__()
        if backbone_name not in self._LAYER_MAP:
            backbone_name = "convnext_tiny"
            log.warning("Unknown backbone — falling back to convnext_tiny.")

        layers, factory, weights = self._LAYER_MAP[backbone_name]
        net = factory(weights=weights)
        for p in net.parameters():
            p.requires_grad_(False)
        net.eval().to(device)

        self.net      = net
        self.device   = device
        self._hooks: Dict[str, torch.Tensor] = {}
        self._handles = []
        for name in layers:
            mod = self._get_mod(name)
            self._handles.append(mod.register_forward_hook(self._hook(name)))

        self.spatial_dim, self.feat_dim = self._probe()
        log.info("Backbone=%s | spatial_dim(C)=%d | feat_dim(flat)=%d",
                 backbone_name, self.spatial_dim, self.feat_dim)

    def _get_mod(self, name: str) -> nn.Module:
        m = self.net
        for p in name.split("."): m = getattr(m, p)
        return m

    def _hook(self, name: str):
        def fn(_, __, out): self._hooks[name] = out.detach()
        return fn

    @torch.no_grad()
    def spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, 3, H, W) → (B, C_total, G, G)
        Each backbone stage pooled to SPATIAL_GRID × SPATIAL_GRID,
        then concatenated along channel dim.
        """
        self._hooks.clear()
        _ = self.net(x.to(self.device))
        G     = CFG.SPATIAL_GRID
        parts = [F.adaptive_avg_pool2d(v, (G, G)) for v in self._hooks.values()]
        return torch.cat(parts, dim=1)          # (B, C_total, G, G)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, D) L2-normalised flat vector."""
        sp = self.spatial_forward(x)            # (B, C, G, G)
        return F.normalize(sp.flatten(1), dim=1)

    def _probe(self) -> Tuple[int, int]:
        dummy = torch.zeros(1, 3, CFG.CROP_SIZE, CFG.CROP_SIZE, device=self.device)
        sp    = self.spatial_forward(dummy)     # (1, C, G, G)
        flat  = self.forward(dummy)             # (1, D)
        return int(sp.shape[1]), int(flat.shape[1])

    def remove_hooks(self) -> None:
        for h in self._handles: h.remove()


# =============================================================================
# 6A. PATCHCORE  (spatial patch-level memory bank)
# =============================================================================
class PatchCore:
    """
    Spatial PatchCore
    ─────────────────
    fit()    : (B, C, G, G) → flatten to (B*G*G, C) patches → coreset bank.
    predict(): per-patch min-dist → score map (G,G) → upsample → max = image score.
    score_map(): (CROP_SIZE, CROP_SIZE) float32 anomaly map for visualization.
    """
    def __init__(self, extractor: BackboneExtractor, coreset_ratio=CFG.PC_CORESET):
        self.ext   = extractor
        self.ratio = coreset_ratio
        self.bank: Optional[torch.Tensor] = None
        self.G     = CFG.SPATIAL_GRID

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        sp = self.ext.spatial_forward(x)
        B, C, G, _ = sp.shape
        return sp.permute(0, 2, 3, 1).reshape(B * G * G, C)

    def fit(self, loader: DataLoader) -> None:
        log.info("PatchCore: building spatial memory bank …")
        patches = [self._to_patches(x).cpu() for x, *_ in tqdm(loader, leave=False, desc="  PC fit")]
        bank = torch.cat(patches, 0)
        k    = max(1, int(len(bank) * self.ratio))
        self.bank = self._coreset(bank, k)
        log.info("PatchCore: bank=%d patches", len(self.bank))

    def _coreset(self, patches: torch.Tensor, k: int) -> torch.Tensor:
        N = len(patches)
        if k >= N: return patches
        sel   = [random.randint(0, N - 1)]
        dists = torch.cdist(patches[sel], patches).squeeze(0)
        for _ in range(1, k):
            i = int(torch.argmax(dists).item()); sel.append(i)
            dists = torch.min(dists, torch.cdist(patches[i:i+1], patches).squeeze(0))
        return patches[sel]

    def _patch_dists(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, G, G) min-distance score map on CPU."""
        bank = self.bank.to(self.ext.device)
        sp   = self.ext.spatial_forward(x)
        B, C, G, _ = sp.shape
        patches = sp.permute(0, 2, 3, 1).reshape(B * G * G, C)
        chunk = 512; mins = []
        for i in range(0, len(patches), chunk):
            d = torch.cdist(patches[i:i+chunk], bank)
            mins.append(d.min(1).values)
        return torch.cat(mins, 0).reshape(B, G, G).cpu()

    def save(self, path: str):
        torch.save({"bank": self.bank.cpu() if self.bank is not None else None}, path)

    def load(self, path: str, device=None):
        ckpt = torch.load(path, map_location=device or self.ext.device)
        self.bank = ckpt["bank"].to(device or self.ext.device)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        scores, labels = [], []
        for x, y, *_ in tqdm(loader, leave=False, desc="  PC infer"):
            dm = self._patch_dists(x)                              # (B, G, G)
            dm_up = F.interpolate(dm.unsqueeze(1),
                                  size=(CFG.CROP_SIZE, CFG.CROP_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(1)
            scores.append(dm_up.flatten(1).max(1).values.numpy())
            labels.append(y.numpy())
        return np.concatenate(scores), np.concatenate(labels)

    @torch.no_grad()
    def score_map(self, x: torch.Tensor) -> np.ndarray:
        dm = self._patch_dists(x)                                  # (1, G, G)
        out = F.interpolate(dm.unsqueeze(1),
                            size=(CFG.CROP_SIZE, CFG.CROP_SIZE),
                            mode="bilinear", align_corners=False)
        return out.squeeze().numpy()


# =============================================================================
# 6B. PADIM  (spatial per-patch Gaussian)
# =============================================================================
class PaDiM:
    """
    Spatial PaDiM
    ─────────────
    fit()    : per-position mean + precision matrix from (N, G*G, D) training feats.
    predict(): Mahalanobis distance per patch → score map → max = image score.
    """
    def __init__(self, extractor: BackboneExtractor, max_dims=CFG.PADIM_DIMS):
        self.ext      = extractor
        self.max_dims = max_dims
        self.G        = CFG.SPATIAL_GRID
        self.means:   Optional[torch.Tensor] = None
        self.cov_inv: Optional[torch.Tensor] = None
        self.idx:     Optional[torch.Tensor] = None

    def fit(self, loader: DataLoader) -> None:
        log.info("PaDiM: accumulating spatial patch statistics …")
        all_feats = []
        for x, *_ in tqdm(loader, leave=False, desc="  PaDiM fit"):
            sp = self.ext.spatial_forward(x).cpu()
            B, C, G, _ = sp.shape
            all_feats.append(sp.permute(0, 2, 3, 1).reshape(B, G * G, C))
        X = torch.cat(all_feats, 0).float()            # (N, G*G, C)
        N, HW, C = X.shape
        D = min(self.max_dims, C)
        torch.manual_seed(42)
        self.idx = torch.randperm(C)[:D]
        X = X[:, :, self.idx]                          # (N, G*G, D)
        self.means = X.mean(0)                         # (G*G, D)
        self.cov_inv = torch.zeros(HW, D, D)
        for pos in tqdm(range(HW), leave=False, desc="  PaDiM cov"):
            x_pos = X[:, pos, :]
            diff  = x_pos - self.means[pos]
            cov   = (diff.T @ diff) / max(N - 1, 1)
            cov.diagonal().add_(CFG.PADIM_REG)
            self.cov_inv[pos] = torch.linalg.pinv(cov)
        log.info("PaDiM: fitted | G=%d | D_red=%d (from C=%d)", self.G, D, C)

    def _maha_map(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, G, G) Mahalanobis score map."""
        dev  = self.ext.device
        mu   = self.means.to(dev)
        cinv = self.cov_inv.to(dev)
        G    = self.G
        sp   = self.ext.spatial_forward(x)
        B, C, _, _ = sp.shape
        feats = sp[:, self.idx.to(dev)].permute(0, 2, 3, 1).reshape(B, G * G, -1)
        diff  = feats - mu.unsqueeze(0)
        maha  = torch.einsum("bpd,pdq,bpq->bp", diff, cinv, diff
                             ).clamp(min=0).sqrt()
        return maha.reshape(B, G, G).cpu()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        scores, labels = [], []
        for x, y, *_ in tqdm(loader, leave=False, desc="  PaDiM infer"):
            mm = self._maha_map(x)
            mm_up = F.interpolate(mm.unsqueeze(1),
                                  size=(CFG.CROP_SIZE, CFG.CROP_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(1)
            scores.append(mm_up.flatten(1).max(1).values.numpy())
            labels.append(y.numpy())
        return np.concatenate(scores), np.concatenate(labels)

    @torch.no_grad()
    def score_map(self, x: torch.Tensor) -> np.ndarray:
        mm  = self._maha_map(x)
        out = F.interpolate(mm.unsqueeze(1),
                            size=(CFG.CROP_SIZE, CFG.CROP_SIZE),
                            mode="bilinear", align_corners=False)
        return out.squeeze().numpy()


# =============================================================================
# 6C. SUPERSIMPLENET  (spatial patch discriminator)
# =============================================================================
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.alpha * g, None


class SuperSimpleNet(nn.Module):
    """
    Spatial SuperSimpleNet
    ──────────────────────
    MLP trained on (B*G*G, C) spatial patches.
    Normal patches → 0, hard-anomaly patches → 1.
    score_map() reshapes logits to (G,G) and upsamples to (CROP_SIZE, CROP_SIZE).
    """
    def __init__(self, spatial_dim: int, device=CFG.DEVICE,
                 proj_dim=CFG.SSN_PROJ, lr=CFG.SSN_LR, epochs=CFG.SSN_EPOCHS,
                 noise=CFG.SSN_NOISE, alpha=CFG.SSN_ALPHA):
        super().__init__()
        self.device = device; self.epochs = epochs
        self.noise  = noise;  self.alpha  = alpha
        self.G      = CFG.SPATIAL_GRID
        self.head   = nn.Sequential(
            nn.Linear(spatial_dim, proj_dim * 2), nn.BatchNorm1d(proj_dim * 2), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(proj_dim * 2, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(),
            nn.Linear(proj_dim, 1),
        ).to(device)
        self.opt  = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
        self.sch  = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)
        self.loss = nn.BCEWithLogitsLoss()

    def fit(self, extractor: BackboneExtractor, loader: DataLoader) -> None:
        log.info("SSN: training %d epochs on spatial patches …", self.epochs)
        G = self.G; self.head.train()
        for epoch in range(self.epochs):
            ep_loss = 0.
            for x, *_ in loader:
                with torch.no_grad():
                    sp = extractor.spatial_forward(x)
                B, C, _, _ = sp.shape
                fc = sp.permute(0, 2, 3, 1).reshape(B * G * G, C)
                if B > 1:
                    idx_img = torch.randperm(B, device=self.device)
                    fc_shuf = sp[idx_img].permute(0, 2, 3, 1).reshape(B * G * G, C)
                    mixed   = 0.5 * fc + 0.5 * fc_shuf
                else:
                    mixed = fc
                fn  = mixed + torch.randn_like(fc) * fc.std() * self.noise
                fa  = torch.cat([fc, fn], 0)
                lbl = torch.cat([torch.zeros(B*G*G), torch.ones(B*G*G)]).to(self.device)
                logit = self.head(_GRL.apply(fa, self.alpha)).squeeze(-1)
                loss  = self.loss(logit, lbl)
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.head.parameters(), 1.)
                self.opt.step(); ep_loss += loss.item()
            self.sch.step()
            if (epoch + 1) % 10 == 0:
                log.info("  SSN epoch %d/%d | loss=%.4f", epoch+1, self.epochs,
                         ep_loss / max(len(loader), 1))
        self.head.eval()

    @torch.no_grad()
    def _patch_scores(self, extractor, x) -> torch.Tensor:
        G  = self.G
        sp = extractor.spatial_forward(x)
        B, C, _, _ = sp.shape
        patches = sp.permute(0, 2, 3, 1).reshape(B * G * G, C)
        logits  = self.head(_GRL.apply(patches, 0.)).squeeze(-1)
        return torch.sigmoid(logits).reshape(B, G, G).cpu()

    @torch.no_grad()
    def predict(self, extractor, loader) -> Tuple[np.ndarray, np.ndarray]:
        self.head.eval(); scores, labels = [], []
        for x, y, *_ in tqdm(loader, leave=False, desc="  SSN infer"):
            sc = self._patch_scores(extractor, x)
            sc_up = F.interpolate(sc.unsqueeze(1),
                                  size=(CFG.CROP_SIZE, CFG.CROP_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(1)
            scores.append(sc_up.flatten(1).max(1).values.numpy()); labels.append(y.numpy())
        return np.concatenate(scores), np.concatenate(labels)

    @torch.no_grad()
    def score_map(self, extractor, x) -> np.ndarray:
        self.head.eval()
        sc  = self._patch_scores(extractor, x)
        out = F.interpolate(sc.unsqueeze(1),
                            size=(CFG.CROP_SIZE, CFG.CROP_SIZE),
                            mode="bilinear", align_corners=False)
        return out.squeeze().numpy()


# =============================================================================
# 6D. CAE  (pixel-level MSE map)
# =============================================================================
class CAE(nn.Module):
    def __init__(self, in_ch=3, latent=CFG.CAE_LATENT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32,  4,2,1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,    64,  4,2,1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,    128, 4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,   256, 4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, latent,4,2,1), nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1),    nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64, 4,2,1),    nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4,2,1),    nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32,in_ch,4,2,1),   nn.Sigmoid(),
        )
    def forward(self, x): return self.dec(self.enc(x))


class CAEModel:
    """
    CAE Reconstruction Baseline
    ───────────────────────────
    score_map(): pixel-level MSE map (H, W) averaged over RGB channels.
    image score = max over spatial map.
    Thesis note: on "before" input, AE memorises glare → poor AUROC.
    """
    def __init__(self, device=CFG.DEVICE, lr=CFG.CAE_LR, epochs=CFG.CAE_EPOCHS):
        self.device=device; self.epochs=epochs
        self.net = CAE().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)

    @staticmethod
    def _denorm(x):
        mean=torch.tensor(_MEAN,device=x.device).view(1,3,1,1)
        std =torch.tensor(_STD, device=x.device).view(1,3,1,1)
        return (x*std+mean).clamp(0,1)

    def fit(self, loader: DataLoader) -> None:
        log.info("CAE: training %d epochs …", self.epochs)
        self.net.train()
        for epoch in range(self.epochs):
            ep_loss=0.
            for x, *_ in loader:
                x01=self._denorm(x.to(self.device)); rec=self.net(x01)
                loss=F.mse_loss(rec,x01)
                self.opt.zero_grad(); loss.backward(); self.opt.step(); ep_loss+=loss.item()
            self.sch.step()
            if (epoch+1)%10==0:
                log.info("  CAE epoch %d/%d | loss=%.5f",epoch+1,self.epochs,ep_loss/max(len(loader),1))
        self.net.eval()

    @torch.no_grad()
    def _err_map(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W) pixel-level MSE map (mean over RGB channels)."""
        x01=self._denorm(x.to(self.device)); rec=self.net(x01)
        return F.mse_loss(rec, x01, reduction="none").mean(dim=1)   # (B, H, W)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.net.eval(); scores, labels = [], []
        for x, y, *_ in tqdm(loader, leave=False, desc="  CAE infer"):
            err = self._err_map(x)
            scores.append(err.flatten(1).max(1).values.cpu().numpy())
            labels.append(y.numpy())
        return np.concatenate(scores), np.concatenate(labels)

    @torch.no_grad()
    def score_map(self, x: torch.Tensor) -> np.ndarray:
        return self._err_map(x).squeeze().cpu().numpy()


# =============================================================================
# 6E. DRAEM  (spatial discriminator — full-resolution output)
# =============================================================================
class _DRAEMRecon(nn.Module):
    @staticmethod
    def _blk(ci,co):
        return nn.Sequential(nn.Conv2d(ci,co,3,1,1),nn.BatchNorm2d(co),nn.ReLU(True),
                             nn.Conv2d(co,co,3,1,1),nn.BatchNorm2d(co),nn.ReLU(True))
    def __init__(self, in_ch=3):
        super().__init__()
        self.e1=self._blk(in_ch,32); self.e2=self._blk(32,64); self.e3=self._blk(64,128)
        self.pool=nn.MaxPool2d(2); self.bot=self._blk(128,256)
        self.up3=nn.ConvTranspose2d(256,128,2,2); self.d3=self._blk(256,128)
        self.up2=nn.ConvTranspose2d(128,64,2,2);  self.d2=self._blk(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,2);   self.d1=self._blk(64,32)
        self.out=nn.Conv2d(32,in_ch,1)
    def forward(self, x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2))
        b=self.bot(self.pool(e3))
        d=self.d3(torch.cat([self.up3(b),e3],1))
        d=self.d2(torch.cat([self.up2(d),e2],1))
        d=self.d1(torch.cat([self.up1(d),e1],1))
        return torch.sigmoid(self.out(d))


class _DRAEMDisc(nn.Module):
    """
    Spatial discriminator — outputs (B, 1, H, W) pixel-level anomaly logits.
    AdaptiveAvgPool removed; all convolutions use padding=1 to preserve spatial dims.
    """
    def __init__(self, in_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,32,3,1,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,1,1),    nn.ReLU(True),
            nn.Conv2d(64,128,3,1,1),   nn.ReLU(True),
            nn.Conv2d(128,1,1),         # (B, 1, H, W) raw logits
        )
    def forward(self, x): return self.net(x)


class DRAEMModel:
    """
    DRAEM with spatial discriminator
    ─────────────────────────────────
    score_map(): (H, W) spatial anomaly probability for visualization.
    image score = max over spatial map.
    """
    def __init__(self, device=CFG.DEVICE, lr=CFG.DRAEM_LR,
                 epochs=CFG.DRAEM_EPOCHS, noise_std=CFG.DRAEM_NOISE):
        self.device=device; self.epochs=epochs; self.noise_std=noise_std
        self.recon=_DRAEMRecon().to(device); self.disc=_DRAEMDisc().to(device)
        params=list(self.recon.parameters())+list(self.disc.parameters())
        self.opt=torch.optim.Adam(params,lr=lr)
        self.sch=torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=epochs)
        self.bce=nn.BCEWithLogitsLoss()

    @staticmethod
    def _denorm(x):
        mean=torch.tensor(_MEAN,device=x.device).view(1,3,1,1)
        std =torch.tensor(_STD, device=x.device).view(1,3,1,1)
        return (x*std+mean).clamp(0,1)

    def _augment(self, x):
        B,C,H,W=x.shape; aug=x.clone(); noise=torch.randn_like(x)*self.noise_std
        for b in range(B):
            r1,r2=sorted(random.sample(range(H),2)); c1,c2=sorted(random.sample(range(W),2))
            aug[b,:,r1:r2,c1:c2]=(aug[b,:,r1:r2,c1:c2]+noise[b,:,r1:r2,c1:c2]).clamp(0,1)
        return aug

    def fit(self, loader: DataLoader) -> None:
        log.info("DRAEM: training %d epochs (spatial disc) …", self.epochs)
        self.recon.train(); self.disc.train()
        for epoch in range(self.epochs):
            ep_loss=0.
            for x, *_ in loader:
                x=x.to(self.device); x01=self._denorm(x); aug=self._augment(x01)
                rec=self.recon(aug); loss_rec=F.mse_loss(rec,x01)
                B,C,H,W=x01.shape
                pair=torch.cat([torch.cat([x01,rec.detach()],1),torch.cat([aug,rec.detach()],1)],0)
                lbl =torch.cat([torch.zeros(B,1,H,W),torch.ones(B,1,H,W)],0).to(self.device)
                loss_disc=self.bce(self.disc(pair),lbl)
                loss=loss_rec+loss_disc
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.recon.parameters())+list(self.disc.parameters()),1.)
                self.opt.step(); ep_loss+=loss.item()
            self.sch.step()
            if (epoch+1)%10==0:
                log.info("  DRAEM epoch %d/%d | loss=%.4f",epoch+1,self.epochs,ep_loss/max(len(loader),1))
        self.recon.eval(); self.disc.eval()

    @torch.no_grad()
    def _spatial_map(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W) spatial anomaly probability."""
        x01=self._denorm(x.to(self.device)); rec=self.recon(x01)
        return torch.sigmoid(self.disc(torch.cat([x01,rec],1))).squeeze(1)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.recon.eval(); self.disc.eval(); scores,labels=[],[]
        for x,y,*_ in tqdm(loader, leave=False, desc="  DRAEM infer"):
            sc=self._spatial_map(x)
            scores.append(sc.flatten(1).max(1).values.cpu().numpy()); labels.append(y.numpy())
        return np.concatenate(scores),np.concatenate(labels)

    @torch.no_grad()
    def score_map(self, x: torch.Tensor) -> np.ndarray:
        return self._spatial_map(x).squeeze().cpu().numpy()


# =============================================================================
# 7. METRICS
# =============================================================================
def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(labels)) < 2:
        log.warning("Only one class — metrics undefined."); return 0., 0.
    auroc = roc_auc_score(labels, scores); best_f1 = 0.
    for t in np.percentile(scores, np.linspace(0, 100, 300)):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1 = f1
    return float(auroc), float(best_f1)


# =============================================================================
# 8. HEATMAP VISUALISATION  (NEW)
# =============================================================================
def visualize_heatmap(
    model_name: str,
    model,
    test_dl: DataLoader,
    extractor: Optional[BackboneExtractor],
    output_dir: str = ".",
    sigma: float = CFG.VIZ_SIGMA,
    alpha: float = CFG.VIZ_ALPHA,
    img_size: int = CFG.CROP_SIZE,
) -> None:
    """
    Pick one random defect sample (label==1) from test_dl.
    Compute spatial anomaly map → smooth → Jet overlay.
    Save {model_name}_localization_heatmap.png:
        Panel 1: Original RGB
        Panel 2: Anomaly Heatmap (Jet colormap)
        Panel 3: Overlay (alpha blend)
    """
    # ── Collect defect samples ────────────────────────────────────────────
    defect_pool: List[Tuple[torch.Tensor, str]] = []
    for batch in test_dl:
        xs, ys, paths = batch[0], batch[1], batch[2]
        for i, lbl in enumerate(ys):
            if int(lbl.item()) == 1:
                defect_pool.append((xs[i:i+1], paths[i]))
        if len(defect_pool) >= 5:    # collect a small pool then stop
            break

    if not defect_pool:
        log.warning("[%s] No defect samples — skipping heatmap.", model_name)
        return

    x_sample, img_path = random.choice(defect_pool)
    x_sample = x_sample.to(CFG.DEVICE)

    # ── Compute raw anomaly map ───────────────────────────────────────────
    try:
        with torch.no_grad():
            if model_name == "PatchCore":
                raw_map = model.score_map(x_sample)
            elif model_name == "PaDiM":
                raw_map = model.score_map(x_sample)
            elif model_name == "SuperSimpleNet":
                raw_map = model.score_map(extractor, x_sample)
            elif model_name == "CAE":
                raw_map = model.score_map(x_sample)
            elif model_name == "DRAEM":
                raw_map = model.score_map(x_sample)
            else:
                log.warning("Unknown model for heatmap: %s", model_name); return
    except Exception as e:
        log.warning("[%s] score_map() failed: %s", model_name, e); return

    # ── Resize map to img_size × img_size if needed ───────────────────────
    if raw_map.shape != (img_size, img_size):
        t = torch.tensor(raw_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(img_size, img_size),
                          mode="bilinear", align_corners=False)
        raw_map = t.squeeze().numpy()

    # ── Gaussian smoothing ────────────────────────────────────────────────
    smoothed = gaussian_filter(raw_map.astype(np.float32), sigma=sigma)
    mn, mx   = smoothed.min(), smoothed.max()
    norm_map = (smoothed - mn) / (mx - mn + 1e-8)

    # ── Load original RGB image ───────────────────────────────────────────
    if os.path.exists(str(img_path)):
        orig_bgr = cv2.imread(str(img_path))
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = cv2.resize(orig_rgb, (img_size, img_size))
    else:
        mean_a = np.array(_MEAN).reshape(3,1,1)
        std_a  = np.array(_STD).reshape(3,1,1)
        orig_np  = (x_sample.squeeze().cpu().numpy() * std_a + mean_a).clip(0,1)
        orig_rgb = (np.transpose(orig_np,(1,2,0))*255).astype(np.uint8)

    # ── Jet colormap & overlay ────────────────────────────────────────────
    heat_uint8 = (norm_map * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb   = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay    = (orig_rgb.astype(np.float32)*(1-alpha)
                 +heat_rgb.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)

    # ── Save figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{model_name} — Anomaly Localization Heatmap",
                 fontsize=16, fontweight="bold")
    axes[0].imshow(orig_rgb);  axes[0].set_title("Original Image"); axes[0].axis("off")
    im = axes[1].imshow(norm_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Anomaly Heatmap"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay (α={alpha:.2f})"); axes[2].axis("off")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{model_name}_localization_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("[%s] Heatmap saved → %s", model_name, out_path)


# =============================================================================
# 9. BENCHMARK RUNNER
# =============================================================================
def run_benchmark(
    train_dl: DataLoader,
    test_dl:  DataLoader,
    extractor: BackboneExtractor,
    visualize: bool = False,
    viz_dir: str = ".",
) -> pd.DataFrame:
    results: List[Dict] = []

    def _record(name, scores, labels, t_train, t_infer):
        auroc, f1 = compute_metrics(scores, labels)
        log.info("  %-18s AUROC=%.4f | F1=%.4f | train=%.1fs | infer=%.1fs",
                 name, auroc, f1, t_train, t_infer)
        results.append(dict(Model=name, AUROC=round(auroc,4), F1_Score=round(f1,4),
                            Train_s=round(t_train,1), Infer_s=round(t_infer,1)))

    def _viz(name, model):
        if visualize:
            try:
                visualize_heatmap(name, model, test_dl,
                                  extractor=extractor, output_dir=viz_dir)
            except Exception as e:
                log.warning("[%s] Heatmap failed: %s", name, e)

    def _free():
        if CFG.DEVICE.type == "cuda": torch.cuda.empty_cache()

    # 1. PatchCore
    log.info("─── PatchCore ───")
    pc = PatchCore(extractor)
    t0=time.time(); pc.fit(train_dl);             t_tr=time.time()-t0
    t0=time.time(); sc,lb=pc.predict(test_dl);   t_in=time.time()-t0
    _record("PatchCore",sc,lb,t_tr,t_in); _viz("PatchCore",pc); del pc; _free()

    # 2. PaDiM
    log.info("─── PaDiM ───")
    pd_m = PaDiM(extractor)
    t0=time.time(); pd_m.fit(train_dl);           t_tr=time.time()-t0
    t0=time.time(); sc,lb=pd_m.predict(test_dl); t_in=time.time()-t0
    _record("PaDiM",sc,lb,t_tr,t_in); _viz("PaDiM",pd_m); del pd_m; _free()

    # 3. SuperSimpleNet
    log.info("─── SuperSimpleNet ───")
    ssn = SuperSimpleNet(extractor.spatial_dim, device=extractor.device, epochs=CFG.SSN_EPOCHS)
    t0=time.time(); ssn.fit(extractor, train_dl);              t_tr=time.time()-t0
    t0=time.time(); sc,lb=ssn.predict(extractor, test_dl);    t_in=time.time()-t0
    _record("SuperSimpleNet",sc,lb,t_tr,t_in); _viz("SuperSimpleNet",ssn); del ssn; _free()

    # 4. CAE
    log.info("─── CAE (reconstruction baseline) ───")
    cae = CAEModel(device=extractor.device, epochs=CFG.CAE_EPOCHS)
    t0=time.time(); cae.fit(train_dl);            t_tr=time.time()-t0
    t0=time.time(); sc,lb=cae.predict(test_dl);  t_in=time.time()-t0
    _record("CAE",sc,lb,t_tr,t_in); _viz("CAE",cae); del cae; _free()

    # 5. DRAEM
    log.info("─── DRAEM ───")
    draem = DRAEMModel(device=extractor.device, epochs=CFG.DRAEM_EPOCHS)
    t0=time.time(); draem.fit(train_dl);            t_tr=time.time()-t0
    t0=time.time(); sc,lb=draem.predict(test_dl);  t_in=time.time()-t0
    _record("DRAEM",sc,lb,t_tr,t_in); _viz("DRAEM",draem); del draem; _free()

    return pd.DataFrame(results)


# =============================================================================
# 10. PRETTY PRINT
# =============================================================================
def print_results(df: pd.DataFrame, output_mode: str) -> None:
    rep  = "After PS — Normal Map (Nx,Ny,Nz)" if output_mode=="after" \
           else "Before PS — Raw Single Light Image"
    line = "="*68
    print(f"\n{line}\n  BENCHMARK RESULTS  |  Input: {rep}\n{line}")
    print(df.to_string(index=False))
    print(line)
    best    = df.loc[df["AUROC"].idxmax()]
    best_f1 = df.loc[df["F1_Score"].idxmax()]
    print(f"  Best AUROC: {best['Model']} ({best['AUROC']:.4f})")
    print(f"  Best F1:    {best_f1['Model']} ({best_f1['F1_Score']:.4f})")
    print(line)


# =============================================================================
# 11. LIGHT MATRIX UTILITIES
# =============================================================================
def build_theoretical_L(n_lights: int, slant_deg: float = 45.) -> np.ndarray:
    slant = np.radians(slant_deg)
    L = np.zeros((n_lights, 3), dtype=np.float32)
    for i in range(n_lights):
        az=2.*np.pi*i/n_lights
        L[i,0]=np.cos(az)*np.sin(slant); L[i,1]=np.sin(az)*np.sin(slant); L[i,2]=np.cos(slant)
    return L


def load_L_matrix(path: str) -> np.ndarray:
    """Robust loader: standard .npy, pickled, 0-d object, raw float32 binary."""
    path = str(path)
    for allow_p in (False, True):
        try:
            raw = np.load(path, allow_pickle=allow_p)
            if raw.ndim == 0: raw = raw.item()
            L = np.array(raw, dtype=np.float32)
            if L.ndim == 2 and L.shape[1] == 3: return L
        except Exception: pass
    try:
        raw = np.frombuffer(open(path,"rb").read(), dtype=np.float32)
        if raw.size % 3 == 0: return raw.reshape(-1, 3)
    except Exception: pass
    raise ValueError(
        f"Cannot load L matrix from '{path}'. "
        "Save with: np.save('light_directions.npy', L_matrix)  # shape (N,3)")


# =============================================================================
# 12. CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PS pipeline + 5 UAD models with spatial anomaly localization.")
    p.add_argument("--raw_dir",          default="raw_captures")
    p.add_argument("--out_dir",          default="mvtec_dataset")
    p.add_argument("--calib_npy",        default=None)
    p.add_argument("--n_lights",         type=int,   default=12)
    p.add_argument("--slant_deg",        type=float, default=45.)
    p.add_argument("--drop_dark",        type=int,   default=2)
    p.add_argument("--drop_bright",      type=int,   default=5)
    p.add_argument("--crop_offset",      type=int,   default=12)
    p.add_argument("--output_size",      type=int,   default=256)
    p.add_argument("--train_ratio",      type=float, default=0.8)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--output_mode",      default="after", choices=["after","before"])
    p.add_argument("--before_light_idx", type=int,   default=0)
    p.add_argument("--skip_build",       action="store_true")
    p.add_argument("--backbone",         default="convnext_tiny",
                   choices=["convnext_tiny","resnet50","efficientnet_b4"])
    p.add_argument("--batch_size",       type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--ssn_epochs",       type=int, default=CFG.SSN_EPOCHS)
    p.add_argument("--ae_epochs",        type=int, default=CFG.CAE_EPOCHS,
                   help="Epochs for CAE and DRAEM.")
    p.add_argument("--output_csv",       default="benchmark_results.csv")
    # Visualization (NEW)
    p.add_argument("--visualize",        action="store_true",
                   help="Save pixel-level heatmap for one defect sample per model.")
    p.add_argument("--viz_dir",          default=".",
                   help="Directory to save heatmap PNGs.")
    return p.parse_args()


# =============================================================================
# 13. MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    CFG.BATCH_SIZE   = args.batch_size
    CFG.SSN_EPOCHS   = args.ssn_epochs
    CFG.CAE_EPOCHS   = args.ae_epochs
    CFG.DRAEM_EPOCHS = args.ae_epochs

    log.info("Device    : %s", CFG.DEVICE)
    log.info("Mode      : %s", args.output_mode)
    log.info("Visualize : %s", args.visualize)

    if args.calib_npy and Path(args.calib_npy).exists():
        log.info("Loading calibrated L matrix from '%s'.", args.calib_npy)
        L_matrix = load_L_matrix(args.calib_npy)
    else:
        if args.calib_npy: log.warning("Calib file not found — using theoretical L.")
        L_matrix = build_theoretical_L(args.n_lights, args.slant_deg)
    log.info("L_matrix shape: %s", L_matrix.shape)

    if not args.skip_build:
        solver  = PhotometricStereoSolver(
            L_matrix=L_matrix, drop_dark=args.drop_dark, drop_bright=args.drop_bright,
            output_mode=args.output_mode, before_light_idx=args.before_light_idx)
        cropper = AutoCropper(output_size=args.output_size, crop_offset=args.crop_offset)
        MVTecDatasetBuilder(
            raw_dir=args.raw_dir, out_dir=args.out_dir,
            solver=solver, cropper=cropper,
            train_ratio=args.train_ratio, seed=args.seed).build()
    else:
        log.info("Skipping dataset build (--skip_build).")

    dataset_root = Path(args.out_dir)
    if not dataset_root.exists():
        log.error("Dataset root '%s' not found.", dataset_root); return

    train_dl, test_dl = build_loaders(dataset_root, batch_size=args.batch_size)
    if len(train_dl.dataset) == 0 or len(test_dl.dataset) == 0:
        log.error("Empty dataset — check %s.", dataset_root); return

    extractor = BackboneExtractor(backbone_name=args.backbone, device=CFG.DEVICE)

    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)

    log.info("\n%s\nStarting benchmark …\n%s", "="*60, "="*60)
    df = run_benchmark(train_dl, test_dl, extractor,
                       visualize=args.visualize, viz_dir=args.viz_dir)
    extractor.remove_hooks()

    print_results(df, args.output_mode)
    df.to_csv(args.output_csv, index=False)
    log.info("Results → '%s'", args.output_csv)
    if args.visualize:
        log.info("Heatmaps → '%s/'", args.viz_dir)


if __name__ == "__main__":
    main()
