# =============================================================================
# ps_benchmark.py
#
# End-to-End Pipeline: Photometric Stereo → A/B Benchmark (With Localization)
# ─────────────────────────────────────────────────────────
# Stage 1 │ PS Solver      : raw light_*.png → "before" or "after" image
# Stage 2 │ AutoCropper    : perspective-warp ROI to fixed square
# Stage 3 │ MVTec Builder  : write train/test folder structure
# Stage 4 │ Benchmark      : evaluate 5 UAD models × 2 representations
# Stage 5 │ Visualization  : Output localization heatmaps for anomaly sites
#
# Models (5 architectural families)
# ──────────────────────────────────
#  1. PatchCore       Memory-bank nearest-neighbour (Feature-based)
#  2. PaDiM           Per-patch Multivariate Gaussian (Statistical)
#  3. SuperSimpleNet  1x1 Conv Discriminator + GRL + synthetic hard anomalies
#  4. CAE             Convolutional AutoEncoder reconstruction baseline
#  5. DRAEM           DRAEM-style reconstruction + FCN discriminator (SOTA)
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
#       --output_csv results.csv --visualize
# =============================================================================

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ── logging ───────────────────────────────────────────────────────────────────
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
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED        = 42
    IMG_SIZE    = 256       # resize before crop
    CROP_SIZE   = 224       # centre-crop for backbone
    BATCH_SIZE  = 16
    NUM_WORKERS = 2

    # ── PatchCore ─────────────────────────────────────────
    PC_CORESET  = 0.10      # keep 10 % of train patches

    # ── PaDiM ─────────────────────────────────────────────
    PADIM_REG   = 0.01      # covariance regularisation (increased for stability)

    # ── SuperSimpleNet ────────────────────────────────────
    SSN_PROJ    = 256
    SSN_LR      = 1e-3
    SSN_EPOCHS  = 150
    SSN_NOISE   = 0.15
    SSN_ALPHA   = 0.5       # GRL lambda

    # ── CAE ───────────────────────────────────────────────
    CAE_LR      = 1e-3
    CAE_EPOCHS  = 50
    CAE_LATENT  = 256

    # ── DRAEM ─────────────────────────────────────────────
    DRAEM_LR    = 1e-4
    DRAEM_EPOCHS = 50
    DRAEM_NOISE  = 0.15     # synthetic texture noise std


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. PHOTOMETRIC STEREO SOLVER
# =============================================================================
class PhotometricStereoSolver:
    def __init__(
        self,
        L_matrix: np.ndarray,
        drop_dark: int = 2,
        drop_bright: int = 5,
        lambda_reg: float = 1e-5,
        output_mode: str = "after",
        before_light_idx: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if L_matrix.ndim != 2 or L_matrix.shape[1] != 3:
            raise ValueError("L_matrix must be shape (N_lights, 3).")
        if output_mode not in ("after", "before"):
            raise ValueError(f"output_mode must be 'after' or 'before', got '{output_mode}'.")

        self.L_matrix        = L_matrix.astype(np.float32)
        self.n_lights        = L_matrix.shape[0]
        self.drop_dark       = drop_dark
        self.drop_bright     = drop_bright
        self.lambda_reg      = lambda_reg
        self.output_mode     = output_mode
        self.before_light_idx = before_light_idx
        self.device          = device or CFG.DEVICE

        log.info(
            "PhotometricStereoSolver | lights=%d | drop_dark=%d | drop_bright=%d | "
            "output_mode=%s | before_light_idx=%d | device=%s",
            self.n_lights, drop_dark, drop_bright,
            output_mode, before_light_idx, self.device,
        )

    def solve(self, images: List[np.ndarray]) -> np.ndarray:
        if len(images) != self.n_lights:
            raise ValueError(f"Expected {self.n_lights} images, got {len(images)}.")

        gray_stack  = self._to_gray_stack(images)
        h, w        = gray_stack.shape[1], gray_stack.shape[2]
        valid_mask  = self._build_object_mask(images, h, w)

        if self.output_mode == "before":
            idx           = self.before_light_idx % len(images)
            single_masked = images[idx].copy()
            single_masked[~valid_mask] = 0
            return single_masked

        I_valid  = gray_stack[:, valid_mask].T
        n_pixels = I_valid.shape[0]

        if n_pixels == 0:
            log.warning("No valid pixels — returning blank normal map.")
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[..., :2] = 128
            return out

        W      = self._build_weight_mask(I_valid)
        N_unit = self._wls_solve(I_valid, W, n_pixels)

        nx_map = np.full((h, w), 128, dtype=np.uint8)
        ny_map = np.full((h, w), 128, dtype=np.uint8)
        nz_map = np.zeros((h, w),   dtype=np.uint8)

        nx_map[valid_mask] = ((N_unit[:, 0] + 1.) / 2. * 255.).astype(np.uint8)
        ny_map[valid_mask] = ((N_unit[:, 1] + 1.) / 2. * 255.).astype(np.uint8)
        nz_map[valid_mask] = ((N_unit[:, 2] + 1.) / 2. * 255.).astype(np.uint8)

        return np.stack([nz_map, ny_map, nx_map], axis=-1)

    def _to_gray_stack(self, images: List[np.ndarray]) -> np.ndarray:
        grays = []
        for img in images:
            if img is None:
                raise ValueError("None image in list.")
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) \
                if img.ndim == 3 else img.astype(np.float32)
            grays.append(g)
        return np.array(grays, dtype=np.float32)

    def _build_object_mask(self, images: List[np.ndarray], h: int, w: int) -> np.ndarray:
        stack    = np.array(images, dtype=np.float32)
        robust   = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray     = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.medianBlur(gray, 5)
        _, mask  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k        = np.ones((5, 5), np.uint8)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        cnts, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final    = np.zeros((h, w), dtype=np.uint8)
        if cnts:
            cv2.drawContours(final, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        return final > 0

    def _build_weight_mask(self, I: np.ndarray) -> np.ndarray:
        P, N   = I.shape
        idx    = np.argsort(I, axis=1)
        W      = np.ones((P, N), dtype=np.float32)
        px     = np.arange(P)
        for i in range(self.drop_dark):
            W[px, idx[:, i]] = 0.
        for i in range(N - self.drop_bright, N):
            W[px, idx[:, i]] = 0.
        return W

    def _wls_solve(self, I: np.ndarray, W: np.ndarray, n_pixels: int) -> np.ndarray:
        dev   = self.device
        L_t   = torch.tensor(self.L_matrix, dtype=torch.float32, device=dev)
        W_t   = torch.tensor(W,             dtype=torch.float32, device=dev)
        I_t   = torch.tensor(I,             dtype=torch.float32, device=dev)
        I_eye = torch.eye(3, dtype=torch.float32, device=dev) * self.lambda_reg

        if dev.type == "cuda":
            _ = torch.bmm(torch.randn(4,3,3,device=dev), torch.randn(4,3,1,device=dev))
            torch.cuda.synchronize()

        L_px  = L_t.unsqueeze(0).expand(n_pixels, -1, -1)
        L_W   = W_t.unsqueeze(-1) * L_px
        A     = torch.bmm(L_W.transpose(1,2), L_px) + I_eye
        B     = torch.bmm(L_W.transpose(1,2), I_t.unsqueeze(-1))
        N_raw = torch.linalg.solve(A, B).squeeze(-1)
        norm  = torch.linalg.norm(N_raw, dim=1, keepdim=True).clamp(min=self.lambda_reg)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (N_raw / norm).cpu().numpy()


# =============================================================================
# 2. AUTO CROPPER
# =============================================================================
class AutoCropper:
    def __init__(self, padding: int = 15, output_size: int = 512, crop_offset: int = 12):
        self.padding     = padding
        self.output_size = output_size
        self.crop_offset = crop_offset

    def find_bbox(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        stack   = np.array(images, dtype=np.float32)
        robust  = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray    = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k       = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k)
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        box  = cv2.boxPoints(rect).astype(np.float32)
        return self._order_points(box)

    def crop_and_resize(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        rect = bbox.copy()
        o    = self.crop_offset
        rect[0] += [ o,  o]
        rect[1] += [-o,  o]
        rect[2] += [-o, -o]
        rect[3] += [ o, -o]
        s   = self.output_size - 1
        dst = np.array([[0,0],[s,0],[s,s],[0,s]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.output_size, self.output_size),
                                   flags=cv2.INTER_AREA)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        out  = np.zeros((4,2), dtype=np.float32)
        s    = pts.sum(axis=1)
        out[0] = pts[np.argmin(s)]
        out[2] = pts[np.argmax(s)]
        diff   = np.diff(pts, axis=1).ravel()
        out[1] = pts[np.argmin(diff)]
        out[3] = pts[np.argmax(diff)]
        return out


# =============================================================================
# 3. MVTEC DATASET BUILDER
# =============================================================================
class MVTecDatasetBuilder:
    _FOLDER_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")

    def __init__(
        self, raw_dir: str | Path, out_dir: str | Path,
        solver: PhotometricStereoSolver, cropper: AutoCropper,
        train_ratio: float = 0.8, seed: int = 42,
    ) -> None:
        self.raw_dir     = Path(raw_dir)
        self.out_dir     = Path(out_dir)
        self.solver      = solver
        self.cropper     = cropper
        self.train_ratio = train_ratio
        self.rng         = random.Random(seed)
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"raw_dir not found: {self.raw_dir}")

    def build(self) -> None:
        samples = self._discover()
        if not samples:
            log.error("No valid capture folders in '%s'.", self.raw_dir)
            return

        class_groups: Dict[str, List[Path]] = {}
        for folder, cls in samples:
            class_groups.setdefault(cls, []).append(folder)
        for cls, folders in class_groups.items():
            self.rng.shuffle(folders)
            log.info("Class '%s': %d sample(s).", cls, len(folders))

        queue: List[Tuple[Path, str, str]] = []
        for cls, folders in class_groups.items():
            if cls == "good":
                n_train = max(1, int(len(folders) * self.train_ratio))
                for i, f in enumerate(folders):
                    queue.append((f, cls, "train" if i < n_train else "test"))
            else:
                for f in folders:
                    queue.append((f, cls, "test"))

        ok = skipped = 0
        for folder, cls, split in tqdm(queue, desc="Building dataset", unit="sample"):
            try:
                self._process_one(folder, cls, split)
                ok += 1
            except Exception as exc:
                log.warning("SKIP '%s' — %s: %s", folder.name, type(exc).__name__, exc)
                skipped += 1

        log.info("Done. Processed=%d | Skipped=%d → %s", ok, skipped, self.out_dir.resolve())

    def _discover(self) -> List[Tuple[Path, str]]:
        out = []
        for entry in sorted(self.raw_dir.iterdir()):
            if not entry.is_dir():
                continue
            m = self._FOLDER_RE.match(entry.name)
            if m:
                out.append((entry, m.group(1).lower().strip()))
        return out

    def _process_one(self, folder: Path, cls: str, split: str) -> None:
        img_paths = sorted(folder.glob("light_*.png"))
        if not img_paths:
            raise FileNotFoundError(f"No light_*.png in {folder}")
        images = []
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                raise IOError(f"Failed to read {p}")
            images.append(img)
        
        ps_map = self.solver.solve(images)
        bbox   = self.cropper.find_bbox(images)
        if bbox is None:
            raise RuntimeError("AutoCropper found no contour.")
        output_img = self.cropper.crop_and_resize(ps_map, bbox)

        filename = folder.name + ".png"
        save_path = self.out_dir / ("train" if split == "train" else "test") / cls / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), output_img)

        if cls != "good":
            gt = self.out_dir / "ground_truth" / cls / filename
            gt.parent.mkdir(parents=True, exist_ok=True)
            s = self.cropper.output_size
            cv2.imwrite(str(gt), np.zeros((s, s), dtype=np.uint8))


# =============================================================================
# 4. PYTORCH DATASET & DATALOADER
# =============================================================================
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _make_tf(augment: bool = False) -> transforms.Compose:
    ops: list = [transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        transforms.CenterCrop(CFG.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
    return transforms.Compose(ops)


class MVTecFlatDataset(Dataset):
    def __init__(self, root: Path, split: str, transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        split_dir = root / split
        if not split_dir.exists():
            return
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = 0 if cls_dir.name.lower() == "good" else 1
            for p in sorted(cls_dir.glob("*.png")):
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil     = PILImage.fromarray(img_rgb)
        return self.transform(pil), label


def build_loaders(
    dataset_root: Path, batch_size: int = CFG.BATCH_SIZE, num_workers: int = CFG.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MVTecFlatDataset(dataset_root, "train", _make_tf(augment=True))
    test_ds  = MVTecFlatDataset(dataset_root, "test",  _make_tf(augment=False))
    log.info("Dataset | train=%d | test=%d", len(train_ds), len(test_ds))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, test_dl


# =============================================================================
# 5. SHARED BACKBONE
# =============================================================================
class BackboneExtractor(nn.Module):
    """
    Frozen feature extractor retaining spatial (H, W) structure 
    by enforcing a standard adaptive size (14x14) across hooks.
    """
    _LAYER_MAP = {
        "convnext_tiny":   (["features.4", "features.6"],  models.convnext_tiny,
                            models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "resnet50":        (["layer2",     "layer3"],       models.resnet50,
                            models.ResNet50_Weights.IMAGENET1K_V1),
        "efficientnet_b4": (["features.4", "features.6"],  models.efficientnet_b4,
                            models.EfficientNet_B4_Weights.IMAGENET1K_V1),
    }

    def __init__(self, backbone_name: str = "convnext_tiny",
                 device: torch.device = CFG.DEVICE) -> None:
        super().__init__()
        if backbone_name not in self._LAYER_MAP:
            backbone_name = "convnext_tiny"
        layers, factory, weights = self._LAYER_MAP[backbone_name]
        net = factory(weights=weights)
        for p in net.parameters():
            p.requires_grad_(False)
        net.eval().to(device)

        self.net     = net
        self.device  = device
        self._hooks: Dict[str, torch.Tensor] = {}
        self._handles = []

        for name in layers:
            mod = self._get_mod(name)
            self._handles.append(mod.register_forward_hook(self._hook(name)))

        self.feat_dim = self._probe()
        log.info("Backbone=%s | Spatial Feat_Dim=%d (14x14)", backbone_name, self.feat_dim)

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
        
        # ⭐ FIX: Extract with spatial dimensions (14x14)
        parts = [F.adaptive_avg_pool2d(v, (14, 14)) for v in self._hooks.values()]
        return F.normalize(torch.cat(parts, dim=1), dim=1) # (B, C, 14, 14)

    def _probe(self) -> int:
        dummy = torch.zeros(1, 3, CFG.CROP_SIZE, CFG.CROP_SIZE, device=self.device)
        with torch.no_grad():
            return self.forward(dummy).shape[1]

    def remove_hooks(self) -> None:
        for h in self._handles: h.remove()


# =============================================================================
# 6A. PATCHCORE (Spatial / Pixel-Level)
# =============================================================================
class PatchCore:
    def __init__(self, extractor: BackboneExtractor,
                 coreset_ratio: float = CFG.PC_CORESET) -> None:
        self.ext    = extractor
        self.ratio  = coreset_ratio
        self.bank: Optional[torch.Tensor] = None

    def fit(self, loader: DataLoader) -> None:
        log.info("PatchCore: building memory bank …")
        feats = []
        for x, _ in tqdm(loader, leave=False):
            with torch.no_grad():
                f = self.ext(x) # (B, C, 14, 14)
            # Permute & Reshape to (B*14*14, C)
            f_flat = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu()
            feats.append(f_flat)
        
        all_f = torch.cat(feats, 0)
        k     = max(1, int(len(all_f) * self.ratio))
        self.bank = self._coreset(all_f, k)
        log.info("PatchCore: bank size=%d patches", len(self.bank))

    def _coreset(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        N = len(feats)
        if k >= N: return feats
        sel   = [random.randint(0, N-1)]
        dists = torch.cdist(feats[sel], feats).squeeze(0)
        for _ in range(1, k):
            i   = torch.argmax(dists).item()
            sel.append(int(i))
            dists = torch.min(dists, torch.cdist(feats[i:i+1], feats).squeeze(0))
        return feats[sel]

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bank   = self.bank.to(self.ext.device)
        scores, labels, maps = [], [], []
        for x, y in tqdm(loader, leave=False):
            f  = self.ext(x) # (B, C, 14, 14)
            B, C, H, W = f.shape
            f_flat = f.permute(0, 2, 3, 1).reshape(-1, C)
            
            # Distance mapping to memory bank
            dist = torch.cdist(f_flat, bank).min(1).values # (B*196)
            sc_map = dist.view(B, H, W).cpu().numpy()
            img_score = sc_map.max(axis=(1,2))
            
            scores.append(img_score)
            labels.append(y.numpy())
            maps.append(sc_map)
        return np.concatenate(scores), np.concatenate(labels), np.concatenate(maps, axis=0)


# =============================================================================
# 6B. PADIM (Spatial / Pixel-Level)
# =============================================================================
class PaDiM:
    def __init__(self, extractor: BackboneExtractor, max_dims: int = 512) -> None:
        self.ext = extractor
        self.max_dims = max_dims
        self.mu:      Optional[torch.Tensor] = None
        self.cov_inv: Optional[torch.Tensor] = None
        self.idx:     Optional[torch.Tensor] = None

    def fit(self, loader: DataLoader) -> None:
        log.info("PaDiM: fitting Spatial Gaussians …")
        feats = []
        for x, _ in tqdm(loader, leave=False):
            with torch.no_grad():
                f = self.ext(x).cpu()
            feats.append(f)
            
        X = torch.cat(feats, 0).float() # (N, C, 14, 14)
        D = X.shape[1]
        
        if D > self.max_dims:
            torch.manual_seed(42)
            self.idx = torch.randperm(D)[:self.max_dims]
            X = X[:, self.idx]
        else:
            self.idx = torch.arange(D)
            
        X = X.permute(0, 2, 3, 1) # (N, 14, 14, C)
        N, H, W, C = X.shape
        
        mu = X.mean(0) # (14, 14, C)
        Xc = X - mu    # (N, 14, 14, C)
        
        # Efficient covariance calculation (avoid OOM for large N)
        cov = torch.einsum('nhwc,nhwd->hwcd', Xc, Xc) / max(N - 1, 1)
        # Add regularization on the diagonal for each pixel
        I = torch.eye(C, dtype=cov.dtype, device=cov.device).view(1, 1, C, C).expand(H, W, C, C)
        cov = cov + I * CFG.PADIM_REG
        
        self.mu      = mu
        # Fix for CPU MKL / singular matrix crashes: use pseudo-inverse (pinv)
        self.cov_inv = torch.linalg.pinv(cov)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu  = self.mu.to(self.ext.device)
        inv = self.cov_inv.to(self.ext.device)
        scores, labels, maps = [], [], []
        for x, y in tqdm(loader, leave=False):
            f = self.ext(x)
            if self.idx is not None:
                f = f[:, self.idx.to(self.ext.device)]
                
            f = f.permute(0, 2, 3, 1) # (B, 14, 14, C)
            diff = f - mu # (B, 14, 14, C)
            
            # Mahalanobis distance per pixel
            diff_u = diff.unsqueeze(-2) # (B, 14, 14, 1, C)
            m = torch.matmul(diff_u, inv.unsqueeze(0)) # (B, 14, 14, 1, C)
            m = torch.matmul(m, diff_u.transpose(-1, -2)).squeeze(-1).squeeze(-1) # (B, 14, 14)
            sc_map = torch.sqrt(m.clamp(min=1e-6)).cpu().numpy()
            
            img_score = sc_map.max(axis=(1,2))
            scores.append(img_score); labels.append(y.numpy()); maps.append(sc_map)
        return np.concatenate(scores), np.concatenate(labels), np.concatenate(maps, axis=0)


# =============================================================================
# 6C. SUPERSIMPLENET (Spatial / Pixel-Level)
# =============================================================================
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


class SuperSimpleNet(nn.Module):
    def __init__(self, feat_dim: int, device: torch.device = CFG.DEVICE,
                 proj_dim: int = CFG.SSN_PROJ, lr: float = CFG.SSN_LR,
                 epochs: int = CFG.SSN_EPOCHS, noise: float = CFG.SSN_NOISE,
                 alpha: float = CFG.SSN_ALPHA) -> None:
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.noise  = noise
        self.alpha  = alpha
        
        # ⭐ FIX: Changed to 1x1 Convolutions for fully-convolutional dense predictions
        self.head   = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim * 2, 1), nn.BatchNorm2d(proj_dim * 2), nn.GELU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(proj_dim * 2, proj_dim, 1), nn.BatchNorm2d(proj_dim), nn.GELU(),
            nn.Conv2d(proj_dim, 1, 1),
        ).to(device)
        self.opt  = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
        self.sch  = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)
        self.loss = nn.BCEWithLogitsLoss()

    def fit(self, extractor: BackboneExtractor, loader: DataLoader) -> None:
        log.info("SuperSimpleNet: training for %d epochs …", self.epochs)
        self.head.train()
        for epoch in range(self.epochs):
            ep_loss = 0.
            for x, _ in loader:
                B = x.size(0)
                with torch.no_grad():
                    fc = extractor(x) # (B, C, 14, 14)
                
                if B > 1:
                    idx = torch.randperm(B, device=self.device)
                    mixed = 0.5 * fc + 0.5 * fc[idx]
                else:
                    mixed = fc
                
                fn  = mixed + torch.randn_like(fc) * fc.std() * self.noise
                fa  = torch.cat([fc, fn], 0)
                
                # Spatial Targets: Clean = 0, Synthetic Noise = 1
                zeros = torch.zeros(B, 1, fc.shape[2], fc.shape[3])
                ones  = torch.ones(B, 1, fc.shape[2], fc.shape[3])
                lbl   = torch.cat([zeros, ones], 0).to(self.device)
                
                fa_grl = _GRL.apply(fa, self.alpha)
                logit  = self.head(fa_grl) # (2B, 1, 14, 14)
                loss   = self.loss(logit, lbl)
                
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.head.parameters(), 1.)
                self.opt.step()
                ep_loss += loss.item()
            self.sch.step()
            if (epoch + 1) % 10 == 0:
                log.info("  SSN epoch %d/%d | loss=%.4f", epoch+1, self.epochs,
                         ep_loss / max(len(loader), 1))
        self.head.eval()

    @torch.no_grad()
    def predict(self, extractor: BackboneExtractor,
                loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.head.eval()
        scores, labels, maps = [], [], []
        for x, y in tqdm(loader, leave=False):
            f     = extractor(x)
            fg    = _GRL.apply(f, 0.)           
            sc_map = torch.sigmoid(self.head(fg).squeeze(1)).cpu().numpy() # (B, 14, 14)
            img_score = sc_map.max(axis=(1,2))
            
            scores.append(img_score); labels.append(y.numpy()); maps.append(sc_map)
        return np.concatenate(scores), np.concatenate(labels), np.concatenate(maps, axis=0)


# =============================================================================
# 6D. CONVOLUTIONAL AUTOENCODER (CAE)
# =============================================================================
class CAE(nn.Module):
    def __init__(self, in_ch: int = 3, latent: int = CFG.CAE_LATENT) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,    64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,    128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,   256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256,   latent, 4, 2, 1), nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1),    nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, 2, 1),    nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32,  in_ch, 4, 2, 1),  nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))

class CAEModel:
    def __init__(self, device: torch.device = CFG.DEVICE,
                 lr: float = CFG.CAE_LR, epochs: int = CFG.CAE_EPOCHS) -> None:
        self.device = device
        self.epochs = epochs
        self.net    = CAE().to(device)
        self.opt    = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.sch    = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)

    def fit(self, loader: DataLoader) -> None:
        log.info("CAE: training for %d epochs …", self.epochs)
        self.net.train()
        for epoch in range(self.epochs):
            ep_loss = 0.
            for x, _ in loader:
                x   = x.to(self.device)
                rec = self.net(x)
                loss = F.mse_loss(rec, x)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                ep_loss += loss.item()
            self.sch.step()
            if (epoch + 1) % 10 == 0:
                log.info("  CAE epoch %d/%d | loss=%.5f", epoch+1, self.epochs,
                         ep_loss / max(len(loader), 1))
        self.net.eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.net.eval()
        scores, labels, maps = [], [], []
        for x, y in tqdm(loader, leave=False):
            x   = x.to(self.device)
            rec = self.net(x)
            
            # MSE pixel map -> (B, 224, 224)
            err_map = F.mse_loss(rec, x, reduction="none").mean(dim=1)
            img_score = err_map.view(err_map.size(0), -1).max(dim=1).values.cpu().numpy()
            
            scores.append(img_score); labels.append(y.numpy()); maps.append(err_map.cpu().numpy())
        return np.concatenate(scores), np.concatenate(labels), np.concatenate(maps, axis=0)


# =============================================================================
# 6E. DRAEM (Spatial / Pixel-Level)
# =============================================================================
class _DRAEMRecon(nn.Module):
    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.e1 = self._blk(in_ch, 32)
        self.e2 = self._blk(32, 64)
        self.e3 = self._blk(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bot = self._blk(128, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3   = self._blk(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64,  2, 2)
        self.d2   = self._blk(128, 64)
        self.up1  = nn.ConvTranspose2d(64,  32,  2, 2)
        self.d1   = self._blk(64,  32)
        self.out  = nn.Conv2d(32, in_ch, 1)

    @staticmethod
    def _blk(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, 1, 1), nn.BatchNorm2d(co), nn.ReLU(True),
            nn.Conv2d(co, co, 3, 1, 1), nn.BatchNorm2d(co), nn.ReLU(True),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d  = self.d3(torch.cat([self.up3(b), e3], 1))
        d  = self.d2(torch.cat([self.up2(d), e2], 1))
        d  = self.d1(torch.cat([self.up1(d), e1], 1))
        return torch.sigmoid(self.out(d))

class _DRAEMDisc(nn.Module):
    """Fully Convolutional Discriminator for pixel-level anomaly mapping."""
    def __init__(self, in_ch: int = 6) -> None: 
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32,  3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,    64,  3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,    128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,   1,   3, 1, 1),
        )

    def forward(self, x):
        out = self.net(x)
        # Interpolate back to original input resolution (224x224)
        return F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)


class DRAEMModel:
    def __init__(self, device: torch.device = CFG.DEVICE,
                 lr: float = CFG.DRAEM_LR, epochs: int = CFG.DRAEM_EPOCHS,
                 noise_std: float = CFG.DRAEM_NOISE) -> None:
        self.device    = device
        self.epochs    = epochs
        self.noise_std = noise_std
        self.recon     = _DRAEMRecon().to(device)
        self.disc      = _DRAEMDisc().to(device)
        params = list(self.recon.parameters()) + list(self.disc.parameters())
        self.opt  = torch.optim.Adam(params, lr=lr)
        self.sch  = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs)
        self.bce  = nn.BCEWithLogitsLoss()

    def _augment(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x) * self.noise_std
        B, C, H, W = x.shape
        aug = x.clone()
        mask = torch.zeros((B, 1, H, W), device=x.device)
        
        for b in range(B):
            aw, ah = random.randint(20, min(60, W//2)), random.randint(20, min(60, H//2))
            r1 = random.randint(0, max(0, H - ah - 1))
            c1 = random.randint(0, max(0, W - aw - 1))
            aug[b, :, r1:r1+ah, c1:c1+aw] = (aug[b, :, r1:r1+ah, c1:c1+aw] 
                                             + noise[b, :, r1:r1+ah, c1:c1+aw]).clamp(0, 1)
            mask[b, 0, r1:r1+ah, c1:c1+aw] = 1.0
        return aug, mask

    def fit(self, loader: DataLoader) -> None:
        log.info("DRAEM: training for %d epochs …", self.epochs)
        self.recon.train(); self.disc.train()
        for epoch in range(self.epochs):
            ep_loss = 0.
            for x, _ in loader:
                x   = x.to(self.device)
                x01 = self._denorm(x)
                aug, mask_aug = self._augment(x01)

                rec  = self.recon(aug)
                loss_rec = F.mse_loss(rec, x01)

                clean_pair = torch.cat([x01, rec.detach()], 1)
                aug_pair   = torch.cat([aug, rec.detach()], 1)
                pair       = torch.cat([clean_pair, aug_pair], 0)
                
                # Clean targets are 0 everywhere, augmented targets use the generated mask
                mask_clean = torch.zeros_like(mask_aug)
                lbl        = torch.cat([mask_clean, mask_aug], 0)
                
                loss_disc = self.bce(self.disc(pair), lbl)
                loss = loss_rec + loss_disc
                
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.recon.parameters()) + list(self.disc.parameters()), 1.)
                self.opt.step()
                ep_loss += loss.item()
            self.sch.step()
            if (epoch + 1) % 10 == 0:
                log.info("  DRAEM epoch %d/%d | loss=%.4f", epoch+1, self.epochs,
                         ep_loss / max(len(loader), 1))
        self.recon.eval(); self.disc.eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.recon.eval(); self.disc.eval()
        scores, labels, maps = [], [], []
        for x, y in tqdm(loader, leave=False):
            x   = x.to(self.device)
            x01 = self._denorm(x)
            rec = self.recon(x01)
            pair = torch.cat([x01, rec], 1)
            
            sc_map = torch.sigmoid(self.disc(pair)).squeeze(1).cpu().numpy() # (B, 224, 224)
            img_score = sc_map.max(axis=(1,2))
            
            scores.append(img_score); labels.append(y.numpy()); maps.append(sc_map)
        return np.concatenate(scores), np.concatenate(labels), np.concatenate(maps, axis=0)

    @staticmethod
    def _denorm(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(_MEAN, device=x.device).view(1,3,1,1)
        std  = torch.tensor(_STD,  device=x.device).view(1,3,1,1)
        return (x * std + mean).clamp(0, 1)


# =============================================================================
# 7. METRICS & VISUALIZATION
# =============================================================================
def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(labels)) < 2:
        log.warning("Only one class in test set — metrics undefined.")
        return 0., 0.
    auroc   = roc_auc_score(labels, scores)
    best_f1 = 0.
    for t in np.percentile(scores, np.linspace(0, 100, 300)):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return float(auroc), float(best_f1)


def visualize_heatmaps(model_name: str, test_dl: DataLoader, 
                       spatial_maps: np.ndarray, labels: np.ndarray) -> None:
    """Extracts a random defect sample, interpolates the heatmap, applies jet colormap and overlays."""
    defect_indices = np.where(labels == 1)[0]
    if len(defect_indices) == 0:
        log.warning("Visualization Skipped: No defect images found in test set.")
        return

    # Randomly pick 1 defect image
    idx = np.random.choice(defect_indices)
    hmap = spatial_maps[idx]
    
    # Retrieve & Denormalize original image
    img_tensor, _ = test_dl.dataset[idx]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array(_MEAN); std = np.array(_STD)
    img_np = np.clip(img_np * std + mean, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    # Process Heatmap (Interpolation + Blur + MinMaxNorm)
    h, w = img_uint8.shape[:2]
    hmap_resized = cv2.resize(hmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hmap_smoothed = cv2.GaussianBlur(hmap_resized, (7, 7), 0)
    
    h_min, h_max = hmap_smoothed.min(), hmap_smoothed.max()
    hmap_norm = (hmap_smoothed - h_min) / (h_max - h_min + 1e-8)
    
    # Apply Colormap
    heatmap_color = cv2.applyColorMap((hmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay Map onto the Image
    overlay = cv2.addWeighted(img_uint8, 0.4, heatmap_color, 0.6, 0)
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_uint8); axs[0].set_title("Original Image (Defect)")
    axs[1].imshow(heatmap_color); axs[1].set_title("Anomaly Heatmap (Jet)")
    axs[2].imshow(overlay); axs[2].set_title("Overlay Localization")
    
    for ax in axs: ax.axis("off")
    plt.suptitle(f"{model_name} - Anomaly Localization Map", fontsize=16)
    plt.tight_layout()
    
    out_name = f"{model_name.lower().replace(' ', '_')}_localization_heatmap.png"
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  └── Saved visualization → '%s'", out_name)


# =============================================================================
# 8. BENCHMARK RUNNER
# =============================================================================
def run_benchmark(
    train_dl: DataLoader,
    test_dl:  DataLoader,
    extractor: BackboneExtractor,
    visualize: bool = False,
) -> pd.DataFrame:
    results: List[Dict] = []

    def _record(name, scores, labels, maps, t_train, t_infer):
        auroc, f1 = compute_metrics(scores, labels)
        log.info("  %-18s AUROC=%.4f | F1=%.4f | train=%.1fs | infer=%.1fs",
                 name, auroc, f1, t_train, t_infer)
        results.append(dict(Model=name, AUROC=round(auroc,4),
                            F1_Score=round(f1,4),
                            Train_s=round(t_train,1), Infer_s=round(t_infer,1)))
        if visualize:
            visualize_heatmaps(name, test_dl, maps, labels)

    # ── 1. PatchCore ──────────────────────────────────────────────────────
    log.info("─── PatchCore ───")
    pc = PatchCore(extractor)
    t0 = time.time(); pc.fit(train_dl);       t_tr = time.time() - t0
    t0 = time.time(); sc, lb, mp = pc.predict(test_dl); t_in = time.time() - t0
    _record("PatchCore", sc, lb, mp, t_tr, t_in)

    # ── 2. PaDiM ──────────────────────────────────────────────────────────
    log.info("─── PaDiM ───")
    pd_m = PaDiM(extractor)
    t0 = time.time(); pd_m.fit(train_dl);       t_tr = time.time() - t0
    t0 = time.time(); sc, lb, mp = pd_m.predict(test_dl); t_in = time.time() - t0
    _record("PaDiM", sc, lb, mp, t_tr, t_in)

    # ── 3. SuperSimpleNet ─────────────────────────────────────────────────
    log.info("─── SuperSimpleNet ───")
    ssn = SuperSimpleNet(extractor.feat_dim, device=extractor.device, epochs=CFG.SSN_EPOCHS)
    t0 = time.time(); ssn.fit(extractor, train_dl);       t_tr = time.time() - t0
    t0 = time.time(); sc, lb, mp = ssn.predict(extractor, test_dl); t_in = time.time() - t0
    _record("SuperSimpleNet", sc, lb, mp, t_tr, t_in)

    # ── 4. CAE ────────────────────────────────────────────────────────────
    log.info("─── CAE (reconstruction baseline) ───")
    cae = CAEModel(device=extractor.device, epochs=CFG.CAE_EPOCHS)
    t0 = time.time(); cae.fit(train_dl);       t_tr = time.time() - t0
    t0 = time.time(); sc, lb, mp = cae.predict(test_dl); t_in = time.time() - t0
    _record("CAE", sc, lb, mp, t_tr, t_in)

    # ── 5. DRAEM ──────────────────────────────────────────────────────────
    log.info("─── DRAEM ───")
    draem = DRAEMModel(device=extractor.device, epochs=CFG.DRAEM_EPOCHS)
    t0 = time.time(); draem.fit(train_dl);       t_tr = time.time() - t0
    t0 = time.time(); sc, lb, mp = draem.predict(test_dl); t_in = time.time() - t0
    _record("DRAEM", sc, lb, mp, t_tr, t_in)

    return pd.DataFrame(results)


# =============================================================================
# 9. PRETTY PRINT
# =============================================================================
def print_results(df: pd.DataFrame, output_mode: str) -> None:
    rep = "After PS — Normal Map (Nx,Ny,Nz)" if output_mode == "after" \
          else "Before PS — Raw Single Light Image"
    line = "=" * 68
    print(f"\n{line}")
    print(f"  BENCHMARK RESULTS  |  Input: {rep}")
    print(line)
    print(df.to_string(index=False))
    print(line)
    best = df.loc[df["AUROC"].idxmax()]
    print(f"  Best AUROC: {best['Model']} ({best['AUROC']:.4f})")
    best_f1 = df.loc[df["F1_Score"].idxmax()]
    print(f"  Best F1:    {best_f1['Model']} ({best_f1['F1_Score']:.4f})")
    print(line)


# =============================================================================
# 10. LIGHT MATRIX UTILITIES
# =============================================================================
def build_theoretical_L(n_lights: int, slant_deg: float = 45.) -> np.ndarray:
    slant = np.radians(slant_deg)
    L = np.zeros((n_lights, 3), dtype=np.float32)
    for i in range(n_lights):
        az      = 2. * np.pi * i / n_lights
        L[i, 0] = np.cos(az) * np.sin(slant)
        L[i, 1] = np.sin(az) * np.sin(slant)
        L[i, 2] = np.cos(slant)
    return L

def load_L_matrix(path: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0: raw = raw.item()
    L = np.array(raw, dtype=np.float32)
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError(f"L matrix must be shape (N,3), got {L.shape}.")
    return L


# =============================================================================
# 11. CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PS Pipeline: Benchmark + Localization.")
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
    p.add_argument("--output_mode",      default="after", choices=["after", "before"])
    p.add_argument("--before_light_idx", type=int,   default=0)
    p.add_argument("--skip_build",       action="store_true")
    
    # ── Stage 4: benchmark ────────────────────────────────────────────────
    p.add_argument("--backbone",    default="convnext_tiny",
                   choices=["convnext_tiny", "resnet50", "efficientnet_b4"])
    p.add_argument("--batch_size",  type=int,   default=CFG.BATCH_SIZE)
    p.add_argument("--ssn_epochs",  type=int,   default=CFG.SSN_EPOCHS)
    p.add_argument("--ae_epochs",   type=int,   default=CFG.CAE_EPOCHS)
    p.add_argument("--output_csv",  default="benchmark_results.csv")
    
    # ── Stage 5: visualization ────────────────────────────────────────────
    p.add_argument("--visualize",   action="store_true", 
                   help="Generate and save Anomaly Localization Heatmaps for each model.")
    return p.parse_args()


# =============================================================================
# 12. MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    CFG.BATCH_SIZE   = args.batch_size
    CFG.SSN_EPOCHS   = args.ssn_epochs
    CFG.CAE_EPOCHS   = args.ae_epochs
    CFG.DRAEM_EPOCHS = args.ae_epochs

    log.info("Device : %s", CFG.DEVICE)
    log.info("Mode   : %s", args.output_mode)

    if args.calib_npy and Path(args.calib_npy).exists():
        L_matrix = load_L_matrix(args.calib_npy)
    else:
        L_matrix = build_theoretical_L(args.n_lights, args.slant_deg)

    if not args.skip_build:
        solver  = PhotometricStereoSolver(
            L_matrix=L_matrix, drop_dark=args.drop_dark, drop_bright=args.drop_bright,
            output_mode=args.output_mode, before_light_idx=args.before_light_idx,
        )
        cropper = AutoCropper(output_size=args.output_size, crop_offset=args.crop_offset)
        builder = MVTecDatasetBuilder(
            raw_dir=args.raw_dir, out_dir=args.out_dir, solver=solver, 
            cropper=cropper, train_ratio=args.train_ratio, seed=args.seed,
        )
        builder.build()
    else:
        log.info("Skipping dataset build (--skip_build).")

    dataset_root = Path(args.out_dir)
    if not dataset_root.exists():
        log.error("Dataset root '%s' does not exist.", dataset_root)
        return

    train_dl, test_dl = build_loaders(dataset_root, batch_size=args.batch_size)
    if len(train_dl.dataset) == 0 or len(test_dl.dataset) == 0:
        log.error("Dataset loaders failed. Data missing.")
        return

    extractor = BackboneExtractor(backbone_name=args.backbone, device=CFG.DEVICE)

    log.info("\n%s\nStarting benchmark …\n%s", "="*60, "="*60)
    df = run_benchmark(train_dl, test_dl, extractor, visualize=args.visualize)
    extractor.remove_hooks()

    print_results(df, args.output_mode)
    df.to_csv(args.output_csv, index=False)
    log.info("Results saved → '%s'", args.output_csv)

if __name__ == "__main__":
    main()