# =============================================================================
# main.py
#
# End-to-End Pipeline: Photometric Stereo → A/B Benchmark (With Localization)
# =============================================================================

from __future__ import annotations

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
# 0. GLOBAL CONFIG
# =============================================================================
class CFG:
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED        = 42
    IMG_SIZE    = 256
    CROP_SIZE   = 224
    BATCH_SIZE  = 16
    NUM_WORKERS = 2

    # Model Params
    PC_CORESET   = 0.10
    PADIM_REG    = 0.01
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
    def __init__(self, L_matrix, drop_dark=2, drop_bright=5, lambda_reg=1e-5, 
                 output_mode="after", before_light_idx=0, device=None):
        self.L_matrix = L_matrix.astype(np.float32)
        self.n_lights = L_matrix.shape[0]
        self.drop_dark = drop_dark
        self.drop_bright = drop_bright
        self.lambda_reg = lambda_reg
        self.output_mode = output_mode
        self.before_light_idx = before_light_idx
        self.device = device or CFG.DEVICE

    def solve(self, images: List[np.ndarray]) -> np.ndarray:
        gray_stack = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) for img in images])
        h, w = gray_stack.shape[1:]
        
        # Simple Object Mask
        robust = np.percentile(np.array(images), 80, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(cv2.medianBlur(gray, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        valid_mask = mask > 0

        if self.output_mode == "before":
            idx = self.before_light_idx % len(images)
            res = images[idx].copy()
            res[~valid_mask] = 0
            return res

        I_valid = gray_stack[:, valid_mask].T
        P = I_valid.shape[0]
        if P == 0: return np.zeros((h, w, 3), dtype=np.uint8)

        # Outlier Rejection
        sort_idx = np.argsort(I_valid, axis=1)
        W = np.ones((P, self.n_lights), dtype=np.float32)
        px = np.arange(P)
        for i in range(self.drop_dark): W[px, sort_idx[:, i]] = 0.
        for i in range(self.n_lights - self.drop_bright, self.n_lights): W[px, sort_idx[:, i]] = 0.

        # GPU Solver
        L_t = torch.tensor(self.L_matrix, device=self.device)
        W_t = torch.tensor(W, device=self.device)
        I_t = torch.tensor(I_valid, device=self.device)
        L_px = L_t.unsqueeze(0).expand(P, -1, -1)
        L_W = W_t.unsqueeze(-1) * L_px
        A = torch.bmm(L_W.transpose(1, 2), L_px) + torch.eye(3, device=self.device) * self.lambda_reg
        B = torch.bmm(L_W.transpose(1, 2), I_t.unsqueeze(-1))
        N_raw = torch.linalg.solve(A, B).squeeze(-1)
        norm = torch.linalg.norm(N_raw, dim=1, keepdim=True).clamp(min=1e-5)
        N_unit = (N_raw / norm).cpu().numpy()

        res = np.zeros((h, w, 3), dtype=np.uint8)
        res[valid_mask, 2] = ((N_unit[:, 0] + 1) / 2 * 255).astype(np.uint8) # Nx -> R (in BGR, R is idx 2)
        res[valid_mask, 1] = ((N_unit[:, 1] + 1) / 2 * 255).astype(np.uint8) # Ny -> G
        res[valid_mask, 0] = ((N_unit[:, 2] + 1) / 2 * 255).astype(np.uint8) # Nz -> B
        return res

# =============================================================================
# 2. AUTO CROPPER
# =============================================================================
class AutoCropper:
    def __init__(self, output_size=512, crop_offset=12):
        self.output_size = output_size
        self.crop_offset = crop_offset

    def find_bbox(self, images):
        stack = np.array(images, dtype=np.float32)
        robust = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(cv2.medianBlur(gray, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        box = cv2.boxPoints(rect).astype(np.float32)
        # Order: TL, TR, BR, BL
        s = box.sum(axis=1)
        diff = np.diff(box, axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = box[np.argmin(s)]
        ordered[2] = box[np.argmax(s)]
        ordered[1] = box[np.argmin(diff)]
        ordered[3] = box[np.argmax(diff)]
        return ordered

    def crop_and_resize(self, img, bbox):
        rect = bbox.copy()
        o = self.crop_offset
        rect[0] += [o, o]; rect[1] += [-o, o]; rect[2] += [-o, -o]; rect[3] += [o, -o]
        dst = np.array([[0,0],[self.output_size-1,0],[self.output_size-1,self.output_size-1],[0,self.output_size-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.output_size, self.output_size))

# =============================================================================
# 3. MVTEC BUILDER
# =============================================================================
class MVTecDatasetBuilder:
    def __init__(self, raw_dir, out_dir, solver, cropper, train_ratio=0.8):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.solver = solver
        self.cropper = cropper
        self.train_ratio = train_ratio

    def build(self):
        folders = [f for f in self.raw_dir.iterdir() if f.is_dir()]
        for f in tqdm(folders, desc="Building dataset"):
            cls = f.name.split('_')[-1].lower()
            imgs = [cv2.imread(str(p)) for p in sorted(f.glob("light_*.png"))]
            ps_map = self.solver.solve(imgs)
            bbox = self.cropper.find_bbox(imgs)
            if bbox is not None:
                out = self.cropper.crop_and_resize(ps_map, bbox)
                split = "train" if cls == "good" and random.random() < self.train_ratio else "test"
                save_path = self.out_dir / split / cls / (f.name + ".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), out)

# =============================================================================
# 4. DATA LOADERS & MODELS
# =============================================================================
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, split, transform):
        self.transform = transform
        self.samples = []
        for cls_dir in (Path(root)/split).iterdir():
            if cls_dir.is_dir():
                lbl = 0 if cls_dir.name == "good" else 1
                for p in cls_dir.glob("*.png"): self.samples.append((p, lbl))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        img = PILImage.fromarray(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        return self.transform(img), lbl

class BackboneExtractor(nn.Module):
    def __init__(self, name="convnext_tiny", device=CFG.DEVICE):
        super().__init__()
        net = getattr(models, name)(weights="DEFAULT").to(device).eval()
        self.feat = nn.Sequential(*list(net.features[:7]))
        for p in self.parameters(): p.requires_grad = False
        self.device = device
    @torch.no_grad()
    def forward(self, x):
        f = self.feat(x.to(self.device))
        return F.normalize(F.adaptive_avg_pool2d(f, (14, 14)), dim=1)

# Model placeholders (Simplified for main.py consolidation)
class PatchCore:
    def __init__(self, ext): self.ext = ext; self.bank = None
    def fit(self, dl):
        feats = [self.ext(x).permute(0,2,3,1).reshape(-1, 768).cpu() for x, _ in dl]
        all_f = torch.cat(feats, 0)
        self.bank = all_f[torch.randperm(len(all_f))[:int(len(all_f)*0.1)]].to(self.ext.device)
    def predict(self, dl):
        scs, lbls, maps = [], [], []
        for x, y in dl:
            f = self.ext(x); B, C, H, W = f.shape
            dist = torch.cdist(f.permute(0,2,3,1).reshape(-1, C), self.bank).min(1).values
            m = dist.view(B, H, W).cpu().numpy()
            scs.append(m.max((1,2))); lbls.append(y.numpy()); maps.append(m)
        return np.concatenate(scs), np.concatenate(lbls), np.concatenate(maps, 0)

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
def visualize_heatmaps(name, dl, maps, labels, viz_dir):
    idx = np.where(labels == 1)[0]
    if len(idx) == 0: return
    i = idx[0]
    img, _ = dl.dataset[i]
    img = np.clip(img.permute(1,2,0).numpy() * _STD + _MEAN, 0, 1)
    hmap = cv2.applyColorMap((cv2.resize(maps[i], (224, 224)) * 255 / maps[i].max()).astype(np.uint8), cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = cv2.addWeighted((img*255).astype(np.uint8), 0.5, (hmap*255).astype(np.uint8), 0.5, 0)
    
    out_path = Path(viz_dir) / f"{name}_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), overlay)

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="D:/IAD/data_scan/dataset/raw_captures")
    parser.add_argument("--out_dir", default="mvtec_dataset")
    parser.add_argument("--calib_npy", default="D:/IAD/data_scan/dataset/light_directions_12.npy")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--output_mode", default="after")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz_dir", default="heatmaps")
    args = parser.parse_args()

    set_seed(42)
    L = np.load(args.calib_npy) if Path(args.calib_npy).exists() else np.eye(12, 3)

    if not args.skip_build:
        solver = PhotometricStereoSolver(L, output_mode=args.output_mode)
        cropper = AutoCropper(output_size=256)
        builder = MVTecDatasetBuilder(args.raw_dir, args.out_dir, solver, cropper)
        builder.build()

    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    train_dl = DataLoader(MVTecDataset(args.out_dir, "train", tf), batch_size=16, shuffle=True)
    test_dl  = DataLoader(MVTecDataset(args.out_dir, "test", tf), batch_size=16)

    ext = BackboneExtractor()
    model = PatchCore(ext)
    model.fit(train_dl)
    sc, lb, mp = model.predict(test_dl)

    auroc = roc_auc_score(lb, sc)
    print(f"Benchmark Result | Model: PatchCore | AUROC: {auroc:.4f}")

    if args.visualize:
        visualize_heatmaps("PatchCore", test_dl, mp, lb, args.viz_dir)
        print(f"Heatmaps saved to: {args.viz_dir}")

if __name__ == "__main__":
    main()
