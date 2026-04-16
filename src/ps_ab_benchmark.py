# =============================================================================
# ps_ab_benchmark.py  —  v2
#
# End-to-End A/B Testing Benchmark: Before vs. After Photometric Stereo
# for Industrial Anomaly Detection on Specular Metal Surfaces
#
# ┌─────────────────────────────────────────────────────────────┐
# │  Representations                                            │
# │  [A] Before PS  → Single raw light image (BGR, 3-ch)        │
# │  [B] After  PS  → WLS Normal Map (Nx, Ny, Nz), 0-255        │
# ├─────────────────────────────────────────────────────────────┤
# │  Models (5 architectural families)                          │
# │  1. PatchCore      — Feature / Memory-Bank                  │
# │  2. PaDiM          — Statistical / Distribution             │
# │  3. SuperSimpleNet — Discriminator + GRL + Synthetic noise  │
# │  4. CAE            — Convolutional AutoEncoder (baseline)   │
# │  5. DRAEM          — Reconstruction + Discriminator (SOTA)  │
# ├─────────────────────────────────────────────────────────────┤
# │  Metrics: AUROC, F1-Score (at optimal threshold)            │
# └─────────────────────────────────────────────────────────────┘
#
# Usage (Windows CMD):
#   python ps_ab_benchmark.py ^
#       --data_root D:\IAD\data_scan\dataset\raw_captures ^
#       --n_lights 12 ^
#       --calib_npy D:\IAD\data_scan\dataset\light_directions_12.npy ^
#       --before_light_idx 0 ^
#       --output_csv results.csv
#
# Usage (Linux / Colab):
#   python ps_ab_benchmark.py \
#       --data_root /content/raw_captures \
#       --n_lights 12 \
#       --calib_npy /content/light_directions_12.npy \
#       --before_light_idx 0 \
#       --output_csv results.csv
#
# Data folder structure:
#   data_root/
#   ├── train/
#   │   └── good/
#   │       └── 20260305_103000_good/   ← light_01.png … light_N.png
#   └── test/
#       ├── good/   ...
#       └── scratch/
#           └── 20260305_110500_scratch/
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# =============================================================================
# 1. CONFIG & SEED
# =============================================================================
class Config:
    # ── Device ────────────────────────────────────────────────
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Backbone (shared by PatchCore, PaDiM, SuperSimpleNet) ─
    BACKBONE_LAYERS: List[str] = ["features.4", "features.6"]

    # ── PS Solver ─────────────────────────────────────────────
    DROP_DARK: int    = 0
    DROP_BRIGHT: int  = 0
    LAMBDA_REG: float = 1e-5

    # ── Image ─────────────────────────────────────────────────
    IMG_SIZE: int  = 256    # resize all images to this before crop
    CROP_SIZE: int = 224    # center-crop fed to backbone / reconstruction models

    # ── DataLoader ────────────────────────────────────────────
    BATCH_SIZE: int  = 16
    NUM_WORKERS: int = 2

    # ── PatchCore ─────────────────────────────────────────────
    PC_CORESET_RATIO: float = 0.10

    # ── SuperSimpleNet ─────────────────────────────────────────
    SSN_PROJ_DIM: int   = 256
    SSN_LR: float       = 1e-3
    SSN_EPOCHS: int     = 20
    SSN_NOISE_STD: float = 0.15

    # ── CAE (Convolutional AutoEncoder) ───────────────────────
    CAE_LATENT: int  = 128
    CAE_LR: float    = 1e-3
    CAE_EPOCHS: int  = 30

    # ── DRAEM ─────────────────────────────────────────────────
    DRAEM_LR: float     = 1e-4
    DRAEM_EPOCHS: int   = 30
    DRAEM_NOISE_STD: float = 0.15  # Gaussian noise for synthetic anomalies
    DRAEM_BLEND_ALPHA: float = 0.4 # blending weight for anomaly region

    # ── Seed ──────────────────────────────────────────────────
    SEED: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. PHOTOMETRIC STEREO DATA PROCESSOR
# =============================================================================
class PSDataProcessor:
    """
    Converts N raw multi-light images → TWO representations:

    [A] Before PS  — Single raw light image (index=before_light_idx).
                     Retains illumination artefacts & specular glare.
                     Output: (H, W, 3) uint8 BGR.

    [B] After  PS  — WLS Photometric Stereo Normal Map (Nx, Ny, Nz).
                     Removes lighting/glare; encodes surface geometry.
                     Output: (H, W, 3) uint8 RGB.
    """

    def __init__(
        self,
        L_matrix: np.ndarray,
        drop_dark: int   = Config.DROP_DARK,
        drop_bright: int = Config.DROP_BRIGHT,
        lambda_reg: float = Config.LAMBDA_REG,
        before_light_idx: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if L_matrix.ndim != 2 or L_matrix.shape[1] != 3:
            raise ValueError("L_matrix must be shape (N_lights, 3).")
        self.L            = L_matrix.astype(np.float32)
        self.n_lights     = L_matrix.shape[0]
        self.drop_dark    = drop_dark
        self.drop_bright  = drop_bright
        self.lambda_reg   = lambda_reg
        self.before_light_idx = before_light_idx
        self.device       = device or Config.DEVICE

    # ── [A] Before PS ────────────────────────────────────────
    def make_before(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Pick a single raw light image by index.
        Zero out pixels outside the object mask so background = black.
        Returns (H, W, 3) uint8 BGR.
        """
        idx    = self.before_light_idx % len(images)
        single = images[idx].copy()                              # (H,W,3) BGR
        mask   = self._build_object_mask(images,
                                         single.shape[0], single.shape[1])
        single[~mask] = 0
        return single

    # ── [B] After PS ─────────────────────────────────────────
    def make_after(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Robust WLS Photometric Stereo → Normal Map (Nx, Ny, Nz) in [0,255].
        Returns (H, W, 3) uint8 RGB  (R=Nx, G=Ny, B=Nz).
        """
        h, w       = images[0].shape[:2]
        gray_stack = self._to_gray_stack(images)                 # (N, H, W)
        mask       = self._build_object_mask(images, h, w)       # (H,W) bool

        I_valid  = gray_stack[:, mask].T                         # (P, N)
        n_pixels = I_valid.shape[0]

        if n_pixels == 0:
            log.warning("No valid pixels — returning neutral normal map.")
            out = np.full((h, w, 3), 128, dtype=np.uint8)
            out[..., 2] = 0
            return out

        W      = self._build_weight_mask(I_valid)                # (P, N)
        N_unit = self._wls_solve(I_valid, W, n_pixels)           # (P, 3)

        nx_map = np.full((h, w), 128, dtype=np.uint8)
        ny_map = np.full((h, w), 128, dtype=np.uint8)
        nz_map = np.zeros((h, w),  dtype=np.uint8)

        nx_map[mask] = ((N_unit[:, 0] + 1.0) / 2.0 * 255.0).astype(np.uint8)
        ny_map[mask] = ((N_unit[:, 1] + 1.0) / 2.0 * 255.0).astype(np.uint8)
        nz_map[mask] = ((N_unit[:, 2] + 1.0) / 2.0 * 255.0).astype(np.uint8)

        return np.stack([nx_map, ny_map, nz_map], axis=-1)       # (H,W,3) RGB

    # ── Private helpers ──────────────────────────────────────
    def _to_gray_stack(self, images: List[np.ndarray]) -> np.ndarray:
        grays = []
        for img in images:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) \
                if img.ndim == 3 else img.astype(np.float32)
            grays.append(g)
        return np.array(grays, dtype=np.float32)

    def _build_object_mask(
        self, images: List[np.ndarray], h: int, w: int
    ) -> np.ndarray:
        stack   = np.array(images, dtype=np.float32)
        robust  = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray    = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        _, mask = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k       = np.ones((5, 5), np.uint8)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        final   = np.zeros((h, w), dtype=np.uint8)
        if cnts:
            cv2.drawContours(final, [max(cnts, key=cv2.contourArea)],
                             -1, 255, cv2.FILLED)
        return final > 0

    def _build_weight_mask(self, I: np.ndarray) -> np.ndarray:
        P, N   = I.shape
        idx    = np.argsort(I, axis=1)
        W      = np.ones((P, N), dtype=np.float32)
        px     = np.arange(P)
        for i in range(self.drop_dark):
            W[px, idx[:, i]] = 0.0
        for i in range(N - self.drop_bright, N):
            W[px, idx[:, i]] = 0.0
        return W

    def _wls_solve(
        self,
        I: np.ndarray,
        W: np.ndarray,
        n_pixels: int,
    ) -> np.ndarray:
        dev   = self.device
        L_t   = torch.tensor(self.L, dtype=torch.float32, device=dev)
        W_t   = torch.tensor(W,      dtype=torch.float32, device=dev)
        I_t   = torch.tensor(I,      dtype=torch.float32, device=dev)
        I_eye = torch.eye(3, dtype=torch.float32, device=dev) * self.lambda_reg

        if dev.type == "cuda":
            _ = torch.bmm(torch.randn(4, 3, 3, device=dev),
                          torch.randn(4, 3, 1, device=dev))
            torch.cuda.synchronize()

        L_px  = L_t.unsqueeze(0).expand(n_pixels, -1, -1)       # (P,N,3) view
        L_W   = W_t.unsqueeze(-1) * L_px                        # (P,N,3)
        A     = torch.bmm(L_W.transpose(1, 2), L_px) + I_eye    # (P,3,3)
        B     = torch.bmm(L_W.transpose(1, 2), I_t.unsqueeze(-1))# (P,3,1)
        N_raw = torch.linalg.solve(A, B).squeeze(-1)             # (P,3)
        norm  = torch.linalg.norm(N_raw, dim=1, keepdim=True).clamp(min=self.lambda_reg)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        return (N_raw / norm).cpu().numpy()                      # (P,3)


# =============================================================================
# 3. DUAL DATASET & DATALOADERS
# =============================================================================
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _make_transform(augment: bool = False) -> transforms.Compose:
    ops: list = [transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        transforms.CenterCrop(Config.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def _make_recon_transform(augment: bool = False) -> transforms.Compose:
    """
    Transform for reconstruction models (CAE, DRAEM).
    No ImageNet normalisation — pixel range stays [0, 1].
    """
    ops: list = [transforms.Resize((Config.CROP_SIZE, Config.CROP_SIZE))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ]
    ops += [transforms.ToTensor()]   # → [0,1]
    return transforms.Compose(ops)


class PSMultiLightDataset(Dataset):
    """
    Yields (img_before, img_after, label) for each multi-light capture folder.

    img_before : backbone-normalised tensor  (3, H, W)
    img_after  : backbone-normalised tensor  (3, H, W)
    label      : 0 = good, 1 = defect
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        processor: PSDataProcessor,
        backbone_tf: transforms.Compose,
        recon_tf: transforms.Compose,
        cache: bool = True,
    ) -> None:
        from PIL import Image as _PIL
        self._PIL       = _PIL
        self.samples    = samples
        self.processor  = processor
        self.b_tf       = backbone_tf
        self.r_tf       = recon_tf
        self.cache      = cache
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns
        -------
        before_bb  : (3,H,W) backbone-normalised  — for PatchCore/PaDiM/SSN
        after_bb   : (3,H,W) backbone-normalised
        before_rc  : (3,H,W) [0,1]               — for CAE / DRAEM
        after_rc   : (3,H,W) [0,1]
        label      : int
        """
        folder, label = self.samples[idx]

        if self.cache and idx in self._cache:
            before_np, after_np = self._cache[idx]
        else:
            img_paths = sorted(folder.glob("light_*.png"))
            if not img_paths:
                raise FileNotFoundError(f"No light_*.png in {folder}")
            images = [cv2.imread(str(p)) for p in img_paths]
            if any(im is None for im in images):
                raise IOError(f"Failed to read images in {folder}")

            before_np = self.processor.make_before(images)   # (H,W,3) BGR
            after_np  = self.processor.make_after(images)    # (H,W,3) RGB

            if self.cache:
                self._cache[idx] = (before_np, after_np)

        # before is BGR from cv2 → convert to RGB for PIL
        before_rgb = cv2.cvtColor(before_np, cv2.COLOR_BGR2RGB)
        after_rgb  = after_np  # already RGB

        before_pil = self._PIL.fromarray(before_rgb)
        after_pil  = self._PIL.fromarray(after_rgb)

        before_bb = self.b_tf(before_pil)
        after_bb  = self.b_tf(after_pil)
        before_rc = self.r_tf(before_pil)
        after_rc  = self.r_tf(after_pil)

        return before_bb, after_bb, before_rc, after_rc, label


def discover_samples(data_root: Path, split: str) -> List[Tuple[Path, int]]:
    """
    Walk data_root/<split>/<class>/<sample_folder>/
    label: 0 = good, 1 = defect
    """
    split_dir = data_root / split
    if not split_dir.exists():
        log.warning("Split dir not found: %s", split_dir)
        return []
    samples: List[Tuple[Path, int]] = []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = 0 if cls_dir.name.lower() == "good" else 1
        for s in sorted(cls_dir.iterdir()):
            if s.is_dir() and any(s.glob("light_*.png")):
                samples.append((s, label))
    return samples


def build_dataloaders(
    data_root: Path,
    processor: PSDataProcessor,
) -> Tuple[DataLoader, DataLoader]:
    bb_train = _make_transform(augment=True)
    bb_test  = _make_transform(augment=False)
    rc_train = _make_recon_transform(augment=True)
    rc_test  = _make_recon_transform(augment=False)

    train_s = discover_samples(data_root, "train")
    test_s  = discover_samples(data_root, "test")
    log.info("Train: %d samples | Test: %d samples", len(train_s), len(test_s))

    train_ds = PSMultiLightDataset(train_s, processor, bb_train, rc_train, cache=True)
    test_ds  = PSMultiLightDataset(test_s,  processor, bb_test,  rc_test,  cache=True)

    kw = dict(num_workers=Config.NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE,
                              shuffle=False, **kw)
    return train_loader, test_loader


# =============================================================================
# 4. SHARED BACKBONE
# =============================================================================
class ConvNeXtFeatureExtractor(nn.Module):
    """
    Frozen ConvNeXt-Tiny backbone.
    Hooks on two intermediate stages → concatenated, L2-normalised feature vector.
    """

    def __init__(self, device: torch.device = Config.DEVICE) -> None:
        super().__init__()
        weights  = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = models.convnext_tiny(weights=weights)
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval().to(device)

        self.backbone = backbone
        self.device   = device
        self._hooks: Dict[str, torch.Tensor] = {}
        self._handles = []

        for name in Config.BACKBONE_LAYERS:
            mod    = self._get_submodule(name)
            handle = mod.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _get_submodule(self, name: str) -> nn.Module:
        mod = self.backbone
        for p in name.split("."):
            mod = getattr(mod, p)
        return mod

    def _make_hook(self, name: str):
        def hook(_, __, out):
            self._hooks[name] = out.detach()
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._hooks.clear()
        _ = self.backbone(x.to(self.device))
        feats = [F.adaptive_avg_pool2d(v, (7, 7)).flatten(1)
                 for v in self._hooks.values()]
        return F.normalize(torch.cat(feats, dim=1), dim=1)

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()


def probe_feature_dim(ext: ConvNeXtFeatureExtractor) -> int:
    dummy = torch.zeros(1, 3, Config.CROP_SIZE, Config.CROP_SIZE,
                        device=ext.device)
    with torch.no_grad():
        return ext(dummy).shape[1]


# =============================================================================
# 5A. PatchCore  (Memory-Bank)
# =============================================================================
class PatchCoreModel:
    """
    PatchCore (Roth et al. 2022).
    Training : extract features → greedy k-centre coreset subsampling.
    Inference: anomaly score = min L2 distance to memory bank.
    """

    def __init__(self, extractor: ConvNeXtFeatureExtractor) -> None:
        self.extractor   = extractor
        self.memory_bank: Optional[torch.Tensor] = None

    def fit(self, loader: DataLoader, key: str) -> None:
        log.info("  PatchCore: extracting features [%s] …", key)
        feats = []
        for batch in tqdm(loader, desc="  PatchCore fit", leave=False):
            imgs = batch[0] if key == "before" else batch[1]
            feats.append(self.extractor(imgs).cpu())
        all_f = torch.cat(feats, dim=0)
        n_keep = max(1, int(len(all_f) * Config.PC_CORESET_RATIO))
        self.memory_bank = self._greedy_coreset(all_f, n_keep)
        log.info("  PatchCore: bank size = %d", len(self.memory_bank))

    def _greedy_coreset(self, f: torch.Tensor, k: int) -> torch.Tensor:
        N = len(f)
        if k >= N:
            return f
        sel   = [random.randint(0, N - 1)]
        dists = torch.cdist(f[sel], f).squeeze(0)
        for _ in range(1, k):
            idx = int(torch.argmax(dists).item())
            sel.append(idx)
            dists = torch.min(dists, torch.cdist(f[idx:idx+1], f).squeeze(0))
        return f[sel]

    @torch.no_grad()
    def predict(self, loader: DataLoader, key: str) -> Tuple[np.ndarray, np.ndarray]:
        bank = self.memory_bank.to(self.extractor.device)
        scores, labels = [], []
        for batch in tqdm(loader, desc="  PatchCore predict", leave=False):
            imgs  = batch[0] if key == "before" else batch[1]
            feats = self.extractor(imgs)
            score = torch.cdist(feats, bank).min(dim=1).values.cpu().numpy()
            scores.append(score);  labels.append(batch[4].numpy())
        return np.concatenate(scores), np.concatenate(labels)


# =============================================================================
# 5B. PaDiM  (Statistical)
# =============================================================================
class PaDiMModel:
    """
    PaDiM (Defard et al. 2021).
    Training : fit multivariate Gaussian on training features.
    Inference: Mahalanobis distance score.
    """

    def __init__(self, extractor: ConvNeXtFeatureExtractor) -> None:
        self.extractor = extractor
        self.mu: Optional[torch.Tensor]      = None
        self.cov_inv: Optional[torch.Tensor] = None

    def fit(self, loader: DataLoader, key: str) -> None:
        log.info("  PaDiM: accumulating statistics [%s] …", key)
        feats = []
        for batch in tqdm(loader, desc="  PaDiM fit", leave=False):
            imgs = batch[0] if key == "before" else batch[1]
            feats.append(self.extractor(imgs).cpu())
        X       = torch.cat(feats, dim=0).float()
        self.mu = X.mean(dim=0)
        Xc      = X - self.mu
        cov     = (Xc.T @ Xc) / (len(X) - 1)
        cov    += torch.eye(cov.shape[0]) * 1e-4
        self.cov_inv = torch.linalg.inv(cov)

    @torch.no_grad()
    def predict(self, loader: DataLoader, key: str) -> Tuple[np.ndarray, np.ndarray]:
        mu      = self.mu.to(self.extractor.device)
        cov_inv = self.cov_inv.to(self.extractor.device)
        scores, labels = [], []
        for batch in tqdm(loader, desc="  PaDiM predict", leave=False):
            imgs  = batch[0] if key == "before" else batch[1]
            feats = self.extractor(imgs)
            diff  = feats - mu
            m     = (diff @ cov_inv * diff).sum(dim=1).sqrt().cpu().numpy()
            scores.append(m);  labels.append(batch[4].numpy())
        return np.concatenate(scores), np.concatenate(labels)


# =============================================================================
# 5C. SuperSimpleNet  (Discriminator + GRL)
# =============================================================================
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj * 2), nn.BatchNorm1d(proj * 2), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(proj * 2, proj),   nn.BatchNorm1d(proj),     nn.GELU(),
            nn.Linear(proj, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SuperSimpleNet(nn.Module):
    """
    SuperSimpleNet (Batzner et al.).
    Frozen backbone → GRL → Projection head → BCE loss.
    Synthetic hard anomalies: Gaussian noise injected in feature space.
    """

    def __init__(self, extractor: ConvNeXtFeatureExtractor,
                 feature_dim: int) -> None:
        super().__init__()
        self.extractor = extractor
        self.head      = _ProjectionHead(feature_dim,
                                         Config.SSN_PROJ_DIM).to(extractor.device)
        self.device    = extractor.device
        self._trained  = False

    def fit(self, loader: DataLoader, key: str) -> None:
        log.info("  SuperSimpleNet: training [%s] …", key)
        opt  = torch.optim.AdamW(self.head.parameters(),
                                 lr=Config.SSN_LR, weight_decay=1e-4)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=Config.SSN_EPOCHS)
        crit = nn.BCEWithLogitsLoss()
        self.head.train()

        for ep in range(Config.SSN_EPOCHS):
            ep_loss = 0.0
            for batch in loader:
                imgs = (batch[0] if key == "before" else batch[1]).to(self.device)
                B    = imgs.size(0)
                with torch.no_grad():
                    f_clean = self.extractor(imgs)
                f_noisy = f_clean + torch.randn_like(f_clean) * Config.SSN_NOISE_STD
                f_all   = torch.cat([f_clean, f_noisy], dim=0)
                y_all   = torch.cat([torch.zeros(B, device=self.device),
                                     torch.ones(B,  device=self.device)])
                logit   = self.head(_GRL.apply(f_all, 0.5))
                loss    = crit(logit, y_all)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                opt.step();  ep_loss += loss.item()
            sch.step()
            if (ep + 1) % 5 == 0:
                log.info("    SSN ep %d/%d | loss=%.4f",
                         ep + 1, Config.SSN_EPOCHS, ep_loss / len(loader))

        self.head.eval()
        self._trained = True

    @torch.no_grad()
    def predict(self, loader: DataLoader, key: str) -> Tuple[np.ndarray, np.ndarray]:
        self.head.eval()
        scores, labels = [], []
        for batch in tqdm(loader, desc="  SSN predict", leave=False):
            imgs  = (batch[0] if key == "before" else batch[1]).to(self.device)
            feats = self.extractor(imgs)
            score = torch.sigmoid(self.head(_GRL.apply(feats, 0.0))).cpu().numpy()
            scores.append(score);  labels.append(batch[4].numpy())
        return np.concatenate(scores), np.concatenate(labels)


# =============================================================================
# 5D. CAE  (Convolutional AutoEncoder — 2D Glare Baseline)
# =============================================================================
class CAEEncoder(nn.Module):
    def __init__(self, latent: int = Config.CAE_LATENT) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            # 224 → 112
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            # 112 → 56
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            # 56 → 28
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            # 28 → 14
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            # 14 → 7
            nn.Conv2d(256, latent, 4, 2, 1), nn.BatchNorm2d(latent), nn.Tanh(),
        )
    def forward(self, x):
        return self.enc(x)


class CAEDecoder(nn.Module):
    def __init__(self, latent: int = Config.CAE_LATENT) -> None:
        super().__init__()
        self.dec = nn.Sequential(
            # 7 → 14
            nn.ConvTranspose2d(latent, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            # 14 → 28
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            # 28 → 56
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            # 56 → 112
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            # 112 → 224
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.dec(z)


class CAEModel(nn.Module):
    """
    Convolutional AutoEncoder baseline.

    Architectural role in benchmark
    ────────────────────────────────
    On specular metal images (Before PS), specular glare creates
    high-reconstruction-error regions that are NOT defects.
    CAE will show high false-positive rates on Before, while After
    (Normal Map, glare-free) should produce cleaner residuals.
    This directly quantifies the "2D Glare Issue" for the thesis.

    Training : minimise MSE on good samples only.
    Inference: anomaly score = mean pixel-wise MSE(input, reconstruction).
    """

    def __init__(self, device: torch.device = Config.DEVICE) -> None:
        super().__init__()
        self.enc     = CAEEncoder().to(device)
        self.dec     = CAEDecoder().to(device)
        self.device  = device
        self._trained = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))

    def fit(self, loader: DataLoader, key: str) -> None:
        log.info("  CAE: training [%s] …", key)
        opt  = torch.optim.Adam(self.parameters(), lr=Config.CAE_LR)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=Config.CAE_EPOCHS)
        crit = nn.MSELoss()
        self.train()

        for ep in range(Config.CAE_EPOCHS):
            ep_loss = 0.0
            for batch in loader:
                # Use reconstruction-normalised tensors (batch[2] / batch[3])
                imgs = (batch[2] if key == "before" else batch[3]).to(self.device)
                recon = self(imgs)
                loss  = crit(recon, imgs)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item()
            sch.step()
            if (ep + 1) % 10 == 0:
                log.info("    CAE ep %d/%d | loss=%.5f",
                         ep + 1, Config.CAE_EPOCHS, ep_loss / len(loader))

        self.eval()
        self._trained = True

    @torch.no_grad()
    def predict(self, loader: DataLoader, key: str) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        scores, labels = [], []
        for batch in tqdm(loader, desc="  CAE predict", leave=False):
            imgs  = (batch[2] if key == "before" else batch[3]).to(self.device)
            recon = self(imgs)
            # Per-sample mean reconstruction error
            score = F.mse_loss(recon, imgs, reduction="none") \
                      .mean(dim=[1, 2, 3]).cpu().numpy()
            scores.append(score);  labels.append(batch[4].numpy())
        return np.concatenate(scores), np.concatenate(labels)


# =============================================================================
# 5E. DRAEM  (Discriminative Reconstruction — SOTA)
# =============================================================================
class _UNetBlock(nn.Module):
    """Lightweight U-Net encoder-decoder block (shared by DRAEM recon net)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)


class DRAEMReconNet(nn.Module):
    """
    Lightweight U-Net reconstruction network.
    Learns to map augmented (noisy) images back to clean surface appearance.
    """
    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.e1 = _UNetBlock(3, 32)
        self.e2 = _UNetBlock(32, 64)
        self.e3 = _UNetBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bn = _UNetBlock(128, 256)
        # Decoder
        self.up3  = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3   = _UNetBlock(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2   = _UNetBlock(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1   = _UNetBlock(64, 32)
        self.out  = nn.Conv2d(32, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b  = self.bn(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class DRAEMDiscNet(nn.Module):
    """
    Lightweight discriminator that scores pixel-wise anomaly probability
    from (reconstructed, original) concatenated input.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1),        # pixel-wise logit
        )
    def forward(self, orig: torch.Tensor,
                recon: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([orig, recon], dim=1))  # (B,1,H,W)


class DRAEMModel(nn.Module):
    """
    DRAEM — Discriminatively-trained Reconstruction Anomaly Estimation Model.

    Architectural role in benchmark
    ────────────────────────────────
    Combines reconstruction (U-Net) with a discriminator that explicitly learns
    to separate real surface texture from synthetic anomaly artefacts.
    Serves as the SOTA reconstruction baseline. Expected to outperform plain CAE
    on both representations, but the After-PS gap vs Before-PS gap quantifies
    the advantage of 3D geometry information.

    Training strategy
    ─────────────────
    1. Add synthetic anomalies to input (Gaussian noise blended by alpha).
    2. Reconstruction net tries to restore clean surface.
    3. Discriminator learns from (clean_recon, noisy_input) pairs.
    4. Loss = λ_r * MSE(recon, clean) + λ_d * BCE(disc_out, anomaly_mask).

    Inference
    ─────────
    Score = mean of pixel-wise discriminator output (after sigmoid).
    """

    def __init__(self, device: torch.device = Config.DEVICE) -> None:
        super().__init__()
        self.recon_net = DRAEMReconNet().to(device)
        self.disc_net  = DRAEMDiscNet().to(device)
        self.device    = device
        self._trained  = False

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        recon = self.recon_net(x)
        disc  = self.disc_net(x, recon)
        return recon, disc

    def _make_synthetic_anomaly(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create hard synthetic anomalies by blending Gaussian noise patches.
        Returns (augmented_input, binary_anomaly_mask).
        """
        B, C, H, W = imgs.shape
        aug_imgs = imgs.clone()
        masks    = torch.zeros(B, 1, H, W, device=self.device)

        for b in range(B):
            # Random rectangular anomaly region
            h_start = random.randint(0, H // 2)
            w_start = random.randint(0, W // 2)
            h_size  = random.randint(H // 8, H // 3)
            w_size  = random.randint(W // 8, W // 3)
            h_end   = min(h_start + h_size, H)
            w_end   = min(w_start + w_size, W)

            noise   = torch.randn(C, h_end - h_start, w_end - w_start,
                                  device=self.device) * Config.DRAEM_NOISE_STD
            alpha   = Config.DRAEM_BLEND_ALPHA
            aug_imgs[b, :, h_start:h_end, w_start:w_end] = (
                (1 - alpha) * imgs[b, :, h_start:h_end, w_start:w_end] + alpha * noise
            ).clamp(0, 1)
            masks[b, 0, h_start:h_end, w_start:w_end] = 1.0

        return aug_imgs, masks

    def fit(self, loader: DataLoader, key: str) -> None:
        log.info("  DRAEM: training [%s] …", key)
        params  = list(self.recon_net.parameters()) + \
                  list(self.disc_net.parameters())
        opt     = torch.optim.Adam(params, lr=Config.DRAEM_LR)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=Config.DRAEM_EPOCHS)
        mse     = nn.MSELoss()
        bce     = nn.BCEWithLogitsLoss()
        self.train()

        for ep in range(Config.DRAEM_EPOCHS):
            ep_loss = 0.0
            for batch in loader:
                imgs_clean = (batch[2] if key == "before"
                              else batch[3]).to(self.device)
                aug_imgs, masks = self._make_synthetic_anomaly(imgs_clean)

                recon, disc_out = self(aug_imgs)

                # Reconstruction loss on clean target
                loss_r = mse(recon, imgs_clean)
                # Discriminator loss — predict anomaly mask
                loss_d = bce(disc_out, masks)
                loss   = loss_r + loss_d

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step();  ep_loss += loss.item()

            sch.step()
            if (ep + 1) % 10 == 0:
                log.info("    DRAEM ep %d/%d | loss=%.4f",
                         ep + 1, Config.DRAEM_EPOCHS, ep_loss / len(loader))

        self.eval()
        self._trained = True

    @torch.no_grad()
    def predict(self, loader: DataLoader, key: str) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        scores, labels = [], []
        for batch in tqdm(loader, desc="  DRAEM predict", leave=False):
            imgs  = (batch[2] if key == "before" else batch[3]).to(self.device)
            _, disc_out = self(imgs)
            # Image-level score = mean pixel-wise anomaly probability
            score = torch.sigmoid(disc_out).mean(dim=[1, 2, 3]).cpu().numpy()
            scores.append(score);  labels.append(batch[4].numpy())
        return np.concatenate(scores), np.concatenate(labels)


# =============================================================================
# 6. METRICS
# =============================================================================
def compute_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """AUROC + F1 at optimal threshold (max-F1 over score percentiles)."""
    if len(np.unique(labels)) < 2:
        log.warning("Only one class present — returning 0 for all metrics.")
        return 0.0, 0.0
    auroc = float(roc_auc_score(labels, scores))
    best_f1 = 0.0
    for t in np.percentile(scores, np.linspace(0, 100, 200)):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return auroc, float(best_f1)


# =============================================================================
# 7. EXPERIMENT LOOP
# =============================================================================
def run_benchmark(
    train_loader: DataLoader,
    test_loader:  DataLoader,
    extractor:    ConvNeXtFeatureExtractor,
    feature_dim:  int,
) -> pd.DataFrame:
    """
    Full 5-model × 2-representation grid.

    Model families
    ──────────────
    PatchCore      — Feature / Memory-Bank
    PaDiM          — Statistical / Gaussian Distribution
    SuperSimpleNet — Discriminator + GRL + Synthetic Noise
    CAE            — Reconstruction (baseline, exposes glare issue)
    DRAEM          — Reconstruction + Discriminator (SOTA)
    """
    results: List[Dict] = []
    dev = extractor.device

    MODEL_DEFS = [
        ("PatchCore",
         "Feature / Memory-Bank",
         lambda: PatchCoreModel(extractor)),
        ("PaDiM",
         "Statistical / Distribution",
         lambda: PaDiMModel(extractor)),
        ("SuperSimpleNet",
         "Discriminator + GRL",
         lambda: SuperSimpleNet(extractor, feature_dim)),
        ("CAE",
         "Reconstruction (2D glare baseline)",
         lambda: CAEModel(dev)),
        ("DRAEM",
         "Reconstruction + Discriminator (SOTA)",
         lambda: DRAEMModel(dev)),
    ]

    for key in ("before", "after"):
        label_str = ("Before PS — Single Light Image"
                     if key == "before"
                     else "After PS  — Normal Map (Nx,Ny,Nz)")
        log.info("\n%s\n=== INPUT: %s ===\n%s",
                 "=" * 68, label_str, "=" * 68)

        for model_name, family, build_fn in MODEL_DEFS:
            log.info("[%s] %s", model_name, label_str)
            model = build_fn()  # fresh instance each run
            t0 = time.time()

            model.fit(train_loader, key)
            scores, labels = model.predict(test_loader, key)
            auroc, f1 = compute_metrics(scores, labels)
            elapsed = round(time.time() - t0, 1)

            results.append({
                "Model":       model_name,
                "Family":      family,
                "Input_Type":  label_str.strip(),
                "AUROC":       round(auroc, 4),
                "F1_Score":    round(f1,    4),
                "Time_s":      elapsed,
            })
            log.info("  → AUROC=%.4f | F1=%.4f | %.1fs",
                     auroc, f1, elapsed)

            # Free GPU memory between runs
            if hasattr(model, "parameters"):
                del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return pd.DataFrame(results)


# =============================================================================
# 8. PRETTY PRINT
# =============================================================================
def print_results(df: pd.DataFrame) -> None:
    """
    Formatted benchmark table + per-model delta summary.
    """
    print("\n" + "=" * 80)
    print("  A/B BENCHMARK RESULTS")
    print("  Before PS (Single Raw Light)  vs  After PS (Normal Map Nx,Ny,Nz)")
    print("=" * 80)

    # Full table
    display_cols = ["Model", "Family", "Input_Type", "AUROC", "F1_Score", "Time_s"]
    print(df[display_cols].to_string(index=False))

    # Delta summary
    print("\n" + "-" * 68)
    print("  DELTA: After PS minus Before PS  (positive = PS helps)")
    print("-" * 68)
    for model in df["Model"].unique():
        sub     = df[df["Model"] == model]
        before  = sub[sub["Input_Type"].str.startswith("Before")].iloc[0]
        after   = sub[sub["Input_Type"].str.startswith("After")].iloc[0]
        d_a     = after["AUROC"]    - before["AUROC"]
        d_f     = after["F1_Score"] - before["F1_Score"]
        indicator = "✓ PS helps" if d_a > 0.01 else ("≈ neutral" if abs(d_a) <= 0.01 else "✗ PS hurts")
        print(f"  {model:<18} ΔAUROC={d_a:+.4f}  ΔF1={d_f:+.4f}   [{indicator}]")

    print("=" * 80)


# =============================================================================
# 9. LIGHT MATRIX UTILITIES
# =============================================================================
def build_theoretical_L(n_lights: int, slant_deg: float = 45.0) -> np.ndarray:
    slant = np.radians(slant_deg)
    L = np.zeros((n_lights, 3), dtype=np.float32)
    for i in range(n_lights):
        az      = 2.0 * np.pi * i / n_lights
        L[i, 0] = np.cos(az) * np.sin(slant)
        L[i, 1] = np.sin(az) * np.sin(slant)
        L[i, 2] = np.cos(slant)
    return L


def load_L_matrix(path: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    if raw.ndim == 0:
        raw = raw.item()
    L = np.array(raw, dtype=np.float32)
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError(
            f"L matrix must be (N_lights, 3), got {L.shape}."
        )
    return L


# =============================================================================
# 10. CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A/B Benchmark: 5 UAD models × Before/After Photometric Stereo"
    )
    p.add_argument("--data_root",         type=str, required=True)
    p.add_argument("--n_lights",          type=int,   default=12)
    p.add_argument("--slant_deg",         type=float, default=45.0)
    p.add_argument("--calib_npy",         type=str,   default=None)
    p.add_argument("--before_light_idx",  type=int,   default=0,
                   help="Light index for 'before' representation (default=0).")
    p.add_argument("--drop_dark",         type=int,   default=Config.DROP_DARK)
    p.add_argument("--drop_bright",       type=int,   default=Config.DROP_BRIGHT)
    p.add_argument("--batch_size",        type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--cae_epochs",        type=int,   default=Config.CAE_EPOCHS)
    p.add_argument("--draem_epochs",      type=int,   default=Config.DRAEM_EPOCHS)
    p.add_argument("--ssn_epochs",        type=int,   default=Config.SSN_EPOCHS)
    p.add_argument("--output_csv",        type=str,   default="ab_benchmark_results.csv")
    return p.parse_args()


# =============================================================================
# 11. MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    set_seed(Config.SEED)

    log.info("Device  : %s", Config.DEVICE)
    log.info("Backbone: ConvNeXt-Tiny  (shared by PatchCore / PaDiM / SSN)")

    # Override epoch counts from CLI
    Config.CAE_EPOCHS   = args.cae_epochs
    Config.DRAEM_EPOCHS = args.draem_epochs
    Config.SSN_EPOCHS   = args.ssn_epochs
    Config.BATCH_SIZE   = args.batch_size

    # ── Light matrix ──────────────────────────────────────────
    if args.calib_npy and Path(args.calib_npy).exists():
        log.info("Loading calibrated L matrix: %s", args.calib_npy)
        L_matrix = load_L_matrix(args.calib_npy)
    else:
        if args.calib_npy:
            log.warning("Calib file not found — using theoretical L matrix.")
        L_matrix = build_theoretical_L(args.n_lights, args.slant_deg)
    log.info("L matrix shape: %s", L_matrix.shape)

    # ── PS Processor ──────────────────────────────────────────
    processor = PSDataProcessor(
        L_matrix=L_matrix,
        drop_dark=args.drop_dark,
        drop_bright=args.drop_bright,
        before_light_idx=args.before_light_idx,
    )

    # ── DataLoaders ───────────────────────────────────────────
    data_root = Path(args.data_root)
    train_loader, test_loader = build_dataloaders(data_root, processor)

    if len(train_loader.dataset) == 0:
        log.error("No training samples found. Check --data_root.")
        return
    if len(test_loader.dataset) == 0:
        log.error("No test samples found. Check --data_root.")
        return

    # ── Backbone ──────────────────────────────────────────────
    extractor   = ConvNeXtFeatureExtractor(device=Config.DEVICE)
    feature_dim = probe_feature_dim(extractor)
    log.info("Backbone feature dim: %d", feature_dim)

    # ── Run benchmark ─────────────────────────────────────────
    results_df = run_benchmark(train_loader, test_loader, extractor, feature_dim)

    # ── Output ────────────────────────────────────────────────
    print_results(results_df)
    out = Path(args.output_csv)
    results_df.to_csv(out, index=False)
    log.info("Results saved → %s", out)

    extractor.remove_hooks()


if __name__ == "__main__":
    main()
