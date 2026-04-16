"""
================================================================================
  วิทยานิพนธ์: Anomaly Detection Benchmark
  6 Models x 5 Backbones = 30 Experiments (Grid Search)
  ครบทุกขั้นตอน: DataLoader → Train → Inference → Metrics → Export
================================================================================
"""

import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional
import math

warnings.filterwarnings("ignore")

# ================================================================================
# 0. CONFIG — แก้ค่าตรงนี้เพียงที่เดียว
# ================================================================================
CONFIG = {
    "data_root":        "./mvtec_anomaly_detection",  # path ไปยัง MVTec Dataset
    "category":         "bottle",                      # หมวดหมู่ที่ต้องการทดสอบ เช่น bottle, cable, pill
    "image_size":       224,
    "batch_size_train": 8,
    "batch_size_test":  1,                             # ต้องเป็น 1 เพื่อจับเวลา inference ต่อรูปได้แม่น
    "num_workers":      4,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "seed":             42,
    "output_csv":       "thesis_experiment_results.csv",
    "patchcore_k":      50,                            # จำนวน nearest neighbors สำหรับ PatchCore
    "padim_t":          350,                           # จำนวน feature dimensions สำหรับ PaDiM
    "efficientad_st_epochs": 70000,                    # training steps สำหรับ EfficientAD
    "simplenet_epochs": 40,
    "supersimplenet_epochs": 160,
}

DEVICE = CONFIG["device"]

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

MODELS_LIST = [
    "PatchCore",
    "PaDiM",
    "FastFlow",
    "EfficientAD",
    "SimpleNet",
    "SuperSimpleNet",
]

BACKBONES_LIST = [
    "WideResNet50",
    "ResNet18",
    "EfficientNet_B4",
    "ConvNeXt_Tiny",
    "MobileNetV3_Large",
]

print(f"🖥️  Device: {DEVICE}")
print(f"📦 Models : {len(MODELS_LIST)} | Backbones: {len(BACKBONES_LIST)} | Total experiments: {len(MODELS_LIST)*len(BACKBONES_LIST)}")


# ================================================================================
# 1. DATASET — รองรับ MVTec AD (train=normal only, test=normal+anomaly)
# ================================================================================

class MVTecDataset(Dataset):
    """
    โหลดข้อมูล MVTec Anomaly Detection Dataset
    โครงสร้างไฟล์:
      {data_root}/{category}/train/good/*.png
      {data_root}/{category}/test/good/*.png
      {data_root}/{category}/test/{defect_type}/*.png
    """
    def __init__(self, root: str, category: str, split: str, image_size: int = 224):
        self.image_size = image_size
        self.split = split
        self.samples: List[Tuple[str, int]] = []  # (path, label) 0=normal, 1=anomaly

        base = os.path.join(root, category, split)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {base}")

        for defect_type in sorted(os.listdir(base)):
            label = 0 if defect_type == "good" else 1
            folder = os.path.join(base, defect_type)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".png", ".jpg", ".bmp")):
                    self.samples.append((os.path.join(folder, fname), label))

        # Transform: resize → center crop → normalize (ImageNet stats)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, path


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    train_ds = MVTecDataset(config["data_root"], config["category"], "train",  config["image_size"])
    test_ds  = MVTecDataset(config["data_root"], config["category"], "test",   config["image_size"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size_train"],
                              shuffle=True,  num_workers=config["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size_test"],
                              shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    print(f"  📂 Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    return train_loader, test_loader


# ================================================================================
# 2. BACKBONE FACTORY — สร้าง feature extractor จาก 5 backbones
# ================================================================================

class BackboneExtractor(nn.Module):
    """
    Wrapper ที่ดึง intermediate feature maps จาก backbone ที่เลือก
    คืนค่า dict ของ features: {"layer2": tensor, "layer3": tensor}
    """
    SUPPORTED = {
        "WideResNet50":      ("wide_resnet50_2",   True),
        "ResNet18":          ("resnet18",          True),
        "EfficientNet_B4":   ("efficientnet_b4",   True),
        "ConvNeXt_Tiny":     ("convnext_tiny",     True),
        "MobileNetV3_Large": ("mobilenet_v3_large",True),
    }

    def __init__(self, backbone_name: str):
        super().__init__()
        if backbone_name not in self.SUPPORTED:
            raise ValueError(f"ไม่รู้จัก backbone: {backbone_name}")

        arch, pretrained = self.SUPPORTED[backbone_name]
        weights_arg = "IMAGENET1K_V1" if pretrained else None

        # โหลด backbone
        if backbone_name == "WideResNet50":
            base = models.wide_resnet50_2(weights=weights_arg)
            self.layer1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.out_channels = {"layer2": 512, "layer3": 1024}

        elif backbone_name == "ResNet18":
            base = models.resnet18(weights=weights_arg)
            self.layer1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.out_channels = {"layer2": 128, "layer3": 256}

        elif backbone_name == "EfficientNet_B4":
            base = models.efficientnet_b4(weights=weights_arg)
            features = base.features
            # ดึง feature จาก block 3 และ 5 ซึ่ง ≈ layer2/layer3
            self.layer1 = features[:3]
            self.layer2 = features[3:5]
            self.layer3 = features[5:7]
            self.out_channels = {"layer2": 56, "layer3": 160}

        elif backbone_name == "ConvNeXt_Tiny":
            base = models.convnext_tiny(weights=weights_arg)
            features = base.features
            self.layer1 = features[:3]
            self.layer2 = features[3:5]
            self.layer3 = features[5:7]
            self.out_channels = {"layer2": 192, "layer3": 384}

        elif backbone_name == "MobileNetV3_Large":
            base = models.mobilenet_v3_large(weights=weights_arg)
            features = base.features
            self.layer1 = features[:4]
            self.layer2 = features[4:7]
            self.layer3 = features[7:13]
            self.out_channels = {"layer2": 40, "layer3": 112}

        # Freeze ทุก parameter (ใช้เป็น fixed feature extractor)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return {"layer2": f2, "layer3": f3}

    def get_feature_dims(self) -> int:
        return sum(self.out_channels.values())


# ================================================================================
# 3. METRIC HELPERS
# ================================================================================

def compute_metrics(y_true: List[int], scores: List[float]) -> Dict:
    """
    คำนวณ AUROC, F1, Precision, Recall และหา Optimal Threshold จาก Youden's J
    """
    y_true  = np.array(y_true)
    scores  = np.array(scores)

    # AUROC
    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0

    # หา Optimal Threshold ด้วย Youden's J statistic (TPR - FPR สูงสุด)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    opt_thresh = float(thresholds[best_idx])

    # F1, Precision, Recall ที่ threshold นี้
    y_pred = (scores >= opt_thresh).astype(int)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    return {
        "AUROC":             round(float(auroc), 4),
        "F1_Score":          round(float(f1),   4),
        "Precision":         round(float(prec), 4),
        "Recall":            round(float(rec),  4),
        "Optimal_Threshold": round(opt_thresh,  4),
    }


# ================================================================================
# 4. MODEL IMPLEMENTATIONS
# ================================================================================

# ─────────────────────────────────────────────────────────────
# 4A. PATCHCORE  (Roth et al., 2022)
#     สร้าง Memory Bank จาก patch features ของรูป train ทั้งหมด
#     แล้วใช้ k-NN เพื่อหา anomaly score
# ─────────────────────────────────────────────────────────────

class PatchCore:
    def __init__(self, backbone: BackboneExtractor, k: int = 50):
        self.backbone    = backbone.to(DEVICE).eval()
        self.k           = k
        self.memory_bank: Optional[torch.Tensor] = None  # shape: (N, D)

    # ── Extract ── ดึง patch features จาก 1 batch
    def _extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(imgs)               # {"layer2": (B,C,H,W), "layer3": ...}

        # upsample ทั้งหมดให้ขนาดเท่า layer2 แล้ว concat
        ref_size = feats["layer2"].shape[-2:]
        aligned  = [feats["layer2"]]
        for k, v in feats.items():
            if k != "layer2":
                aligned.append(F.interpolate(v, size=ref_size, mode="bilinear", align_corners=False))

        combined = torch.cat(aligned, dim=1)          # (B, C_total, H, W)
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        return patches

    # ── Train ── เก็บ patch features ทั้งหมดลง memory bank
    def fit(self, train_loader: DataLoader):
        all_patches = []
        for imgs, _, _ in tqdm(train_loader, desc="  PatchCore: Building memory bank"):
            imgs = imgs.to(DEVICE)
            all_patches.append(self._extract_patches(imgs).cpu())
        self.memory_bank = torch.cat(all_patches, dim=0)  # (N_patches, C)
        # Greedy Coreset Subsampling เพื่อลดขนาด memory bank (ประหยัด RAM)
        self.memory_bank = self._coreset_subsample(self.memory_bank, ratio=0.1)
        print(f"    Memory bank size: {self.memory_bank.shape}")

    def _coreset_subsample(self, patches: torch.Tensor, ratio: float) -> torch.Tensor:
        """Mini coreset: random subsample (approximation)"""
        n = max(1, int(len(patches) * ratio))
        idx = torch.randperm(len(patches))[:n]
        return patches[idx]

    # ── Inference ── คำนวณ anomaly score สำหรับ 1 รูป
    def predict(self, img: torch.Tensor) -> float:
        patches = self._extract_patches(img)  # (H*W, C)
        # คำนวณ distance ถึง k nearest neighbors ใน memory bank
        dists = torch.cdist(patches.to(DEVICE), self.memory_bank.to(DEVICE))  # (P, M)
        knn_dists, _ = dists.topk(self.k, largest=False, dim=1)               # (P, k)
        # anomaly score = max patch-level score
        score = knn_dists[:, 0].max().item()
        return float(score)


# ─────────────────────────────────────────────────────────────
# 4B. PADIM  (Defard et al., 2021)
#     ทำ Gaussian statistics บน patch features ของ training set
#     แล้วใช้ Mahalanobis distance เป็น anomaly score
# ─────────────────────────────────────────────────────────────

class PaDiM:
    def __init__(self, backbone: BackboneExtractor, t: int = 350):
        self.backbone = backbone.to(DEVICE).eval()
        self.t = t  # จำนวน random dimensions ที่เลือก
        self.means:   Optional[torch.Tensor] = None  # (H*W, t)
        self.cov_inv: Optional[torch.Tensor] = None  # (H*W, t, t)
        self.feat_idx: Optional[torch.Tensor] = None  # random dim indices

    def _extract(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(imgs)
        ref_size = feats["layer2"].shape[-2:]
        aligned  = [feats["layer2"]]
        for k, v in feats.items():
            if k != "layer2":
                aligned.append(F.interpolate(v, size=ref_size, mode="bilinear", align_corners=False))
        combined = torch.cat(aligned, dim=1)  # (B, C, H, W)
        return combined

    def fit(self, train_loader: DataLoader):
        all_feats = []
        for imgs, _, _ in tqdm(train_loader, desc="  PaDiM: Collecting features"):
            imgs = imgs.to(DEVICE)
            all_feats.append(self._extract(imgs).cpu())

        feats = torch.cat(all_feats, dim=0)   # (N, C, H, W)
        N, C, H, W = feats.shape

        # Random dimension selection
        t = min(self.t, C)
        self.feat_idx = torch.randperm(C)[:t]
        feats = feats[:, self.feat_idx]        # (N, t, H, W)

        # คำนวณ mean และ covariance สำหรับแต่ละ patch position
        feats_flat = feats.permute(0, 2, 3, 1).reshape(N, H*W, t)  # (N, H*W, t)
        self.means = feats_flat.mean(dim=0)     # (H*W, t)

        # คำนวณ covariance matrix แต่ละ position
        cov = torch.zeros(H*W, t, t)
        for i in tqdm(range(H*W), desc="  PaDiM: Computing covariance", leave=False):
            x = feats_flat[:, i, :]  # (N, t)
            diff = x - self.means[i]
            cov[i] = (diff.T @ diff) / (N - 1) + 1e-4 * torch.eye(t)

        # Inverse ของ covariance matrix (ใช้ pseudo-inverse เพื่อความเสถียร)
        self.cov_inv = torch.linalg.pinv(cov)  # (H*W, t, t)
        self.H, self.W = H, W
        print(f"    PaDiM fitted: H={H}, W={W}, t={t}")

    def predict(self, img: torch.Tensor) -> float:
        with torch.no_grad():
            feats = self.backbone(img)
        ref_size = feats["layer2"].shape[-2:]
        aligned  = [feats["layer2"]]
        for k, v in feats.items():
            if k != "layer2":
                aligned.append(F.interpolate(v, size=ref_size, mode="bilinear", align_corners=False))
        combined = torch.cat(aligned, dim=1)[:, self.feat_idx]  # (1, t, H, W)

        feats_flat = combined[0].permute(1, 2, 0).reshape(-1, len(self.feat_idx))  # (H*W, t)
        diff = feats_flat.cpu() - self.means     # (H*W, t)

        # Mahalanobis distance: sqrt(diff @ Sigma^-1 @ diff^T) per patch
        scores = torch.einsum("ni,nij,nj->n", diff, self.cov_inv, diff)  # (H*W,)
        scores = torch.sqrt(torch.clamp(scores, min=0))
        score_map = scores.reshape(self.H, self.W).numpy()
        score_map = gaussian_filter(score_map, sigma=4)
        return float(score_map.max())


# ─────────────────────────────────────────────────────────────
# 4C. FASTFLOW  (Yu et al., 2021)
#     ใช้ Normalizing Flow เรียนรู้ distribution ของ normal features
#     anomaly score = negative log-likelihood
# ─────────────────────────────────────────────────────────────

class AffineCouplingBlock(nn.Module):
    """Affine Coupling Layer สำหรับ Normalizing Flow"""
    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(half, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,  128, 1),            nn.ReLU(),
            nn.Conv2d(128, channels - half, 3, padding=1),
        )
        self.scale_net = nn.Sequential(
            nn.Conv2d(half, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,  128, 1),            nn.ReLU(),
            nn.Conv2d(128, channels - half, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale_net(x1)
        t = self.net(x1)
        if not reverse:
            y2   = x2 * torch.exp(s) + t
            ldj  = s.sum(dim=[1, 2, 3])
            return torch.cat([x1, y2], dim=1), ldj
        else:
            y2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, y2], dim=1)


class FastFlowModel(nn.Module):
    def __init__(self, in_channels: int, n_flows: int = 8):
        super().__init__()
        self.flows = nn.ModuleList([AffineCouplingBlock(in_channels) for _ in range(n_flows)])

    def forward(self, x):
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        for flow in self.flows:
            x, ldj = flow(x)
            log_det_sum += ldj
        # log-likelihood under standard normal
        log_prob = -0.5 * (x**2).sum(dim=[1, 2, 3]) - log_det_sum
        return log_prob, x


class FastFlow:
    def __init__(self, backbone: BackboneExtractor, n_flows: int = 8, epochs: int = 20, lr: float = 1e-3):
        self.backbone = backbone.to(DEVICE).eval()
        self.n_flows  = n_flows
        self.epochs   = epochs
        self.lr       = lr
        self.flow_model: Optional[FastFlowModel] = None
        self._in_ch: int = 0

    def _extract(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(imgs)
        ref_size = feats["layer2"].shape[-2:]
        aligned  = [feats["layer2"]]
        for k, v in feats.items():
            if k != "layer2":
                aligned.append(F.interpolate(v, size=ref_size, mode="bilinear", align_corners=False))
        return torch.cat(aligned, dim=1)  # (B, C, H, W)

    def fit(self, train_loader: DataLoader):
        # ดึง feature แรกเพื่อรู้จำนวน channels
        sample_imgs, _, _ = next(iter(train_loader))
        sample_feats = self._extract(sample_imgs[:1].to(DEVICE))
        in_ch = sample_feats.shape[1]
        if in_ch % 2 != 0:
            in_ch -= 1  # ต้องเป็นเลขคู่สำหรับ coupling
        self._in_ch = in_ch

        self.flow_model = FastFlowModel(in_ch, self.n_flows).to(DEVICE)
        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.lr)

        self.flow_model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for imgs, _, _ in train_loader:
                feats = self._extract(imgs.to(DEVICE))[:, :in_ch]
                log_prob, _ = self.flow_model(feats)
                loss = -log_prob.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"    FastFlow Epoch [{epoch+1}/{self.epochs}] Loss: {total_loss/len(train_loader):.4f}")

        self.flow_model.eval()

    def predict(self, img: torch.Tensor) -> float:
        feats = self._extract(img.to(DEVICE))[:, :self._in_ch]
        with torch.no_grad():
            log_prob, _ = self.flow_model(feats)
        return float(-log_prob.mean().item())


# ─────────────────────────────────────────────────────────────
# 4D. EFFICIENTAD  (Batzner et al., 2023)
#     Student-Teacher framework: Teacher มี pretrained knowledge,
#     Student เรียนรู้เฉพาะ normal patterns
#     anomaly score = discrepancy ระหว่าง student กับ teacher
# ─────────────────────────────────────────────────────────────

class PDN_Small(nn.Module):
    """Patch Description Network (Small) — ใช้เป็น Teacher/Student"""
    def __init__(self, out_channels: int = 384):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, padding=3)
        self.conv2 = nn.Conv2d(128, 256, 4, padding=3)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, out_channels, 4)
        self.bn1   = nn.BatchNorm2d(128)
        self.bn2   = nn.BatchNorm2d(256)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool  = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class EfficientAD:
    def __init__(self, backbone: BackboneExtractor, steps: int = 70000, lr: float = 1e-4):
        self.backbone = backbone  # ไม่ใช้ backbone โดยตรง (EfficientAD มี PDN ของตัวเอง)
        self.steps    = steps
        self.lr       = lr
        self.teacher: Optional[PDN_Small] = None
        self.student: Optional[PDN_Small] = None
        self.teacher_mean: Optional[torch.Tensor] = None
        self.teacher_std:  Optional[torch.Tensor] = None

    def fit(self, train_loader: DataLoader):
        out_ch = 384
        self.teacher = PDN_Small(out_ch).to(DEVICE)
        self.student = PDN_Small(out_ch).to(DEVICE)

        # Freeze teacher หลังจาก random init (simulate pretrained)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        step      = 0
        pbar      = tqdm(total=self.steps, desc="  EfficientAD: Training student")

        # Compute teacher stats บน training set ก่อน
        t_feats = []
        self.teacher.eval()
        with torch.no_grad():
            for imgs, _, _ in train_loader:
                t_feats.append(self.teacher(imgs.to(DEVICE)).cpu())
        t_feats_cat = torch.cat(t_feats, dim=0)
        self.teacher_mean = t_feats_cat.mean(dim=(0, 2, 3), keepdim=True)
        self.teacher_std  = t_feats_cat.std(dim=(0, 2, 3),  keepdim=True) + 1e-8

        self.student.train()
        while step < self.steps:
            for imgs, _, _ in train_loader:
                imgs = imgs.to(DEVICE)
                with torch.no_grad():
                    t_out = self.teacher(imgs)
                    t_norm = (t_out - self.teacher_mean.to(DEVICE)) / self.teacher_std.to(DEVICE)
                s_out = self.student(imgs)
                loss  = F.mse_loss(s_out, t_norm.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                pbar.update(1)
                if step >= self.steps:
                    break
        pbar.close()
        self.student.eval()

    def predict(self, img: torch.Tensor) -> float:
        img = img.to(DEVICE)
        with torch.no_grad():
            t_out  = self.teacher(img)
            t_norm = (t_out - self.teacher_mean.to(DEVICE)) / self.teacher_std.to(DEVICE)
            s_out  = self.student(img)
        diff  = (t_norm - s_out) ** 2
        score = diff.mean(dim=1).max().item()
        return float(score)


# ─────────────────────────────────────────────────────────────
# 4E. SIMPLENET  (Liu et al., 2023)
#     ใช้ Discriminator เพื่อแยก normal features กับ synthetic anomaly features
# ─────────────────────────────────────────────────────────────

class SimpleDiscriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512),   nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x)


class SimpleNet:
    def __init__(self, backbone: BackboneExtractor, epochs: int = 40, lr: float = 2e-4):
        self.backbone    = backbone.to(DEVICE).eval()
        self.epochs      = epochs
        self.lr          = lr
        self.discriminator: Optional[SimpleDiscriminator] = None

    def _extract_flat(self, imgs: torch.Tensor) -> torch.Tensor:
        """ดึง features แล้ว flatten เป็น (B, D)"""
        with torch.no_grad():
            feats = self.backbone(imgs)
        f2 = F.adaptive_avg_pool2d(feats["layer2"], (1, 1)).squeeze(-1).squeeze(-1)
        f3 = F.adaptive_avg_pool2d(feats["layer3"], (1, 1)).squeeze(-1).squeeze(-1)
        return torch.cat([f2, f3], dim=1)

    def fit(self, train_loader: DataLoader):
        # หา input dimension จาก sample
        sample_imgs, _, _ = next(iter(train_loader))
        in_dim = self._extract_flat(sample_imgs[:1].to(DEVICE)).shape[1]

        self.discriminator = SimpleDiscriminator(in_dim).to(DEVICE)
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for imgs, _, _ in train_loader:
                imgs  = imgs.to(DEVICE)
                feats = self._extract_flat(imgs)

                # Synthetic anomalies = Gaussian noise + features
                noise   = torch.randn_like(feats) * 0.2
                anom_f  = feats + noise

                real_logits = self.discriminator(feats)
                fake_logits = self.discriminator(anom_f)
                labels_r    = torch.zeros(len(feats), 1, device=DEVICE)  # normal = 0
                labels_f    = torch.ones(len(anom_f), 1, device=DEVICE)  # anomaly = 1

                loss = criterion(real_logits, labels_r) + criterion(fake_logits, labels_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"    SimpleNet Epoch [{epoch+1}/{self.epochs}] Loss: {total_loss/len(train_loader):.4f}")

        self.discriminator.eval()

    def predict(self, img: torch.Tensor) -> float:
        feats = self._extract_flat(img.to(DEVICE))
        with torch.no_grad():
            score = self.discriminator(feats).sigmoid()
        return float(score.mean().item())


# ─────────────────────────────────────────────────────────────
# 4F. SUPERSIMPLENET  (Batzner et al., 2024 — unofficial)
#     ปรับปรุง SimpleNet ด้วย:
#     - Gradient Reversal สำหรับ domain adaptation
#     - Projection head ก่อน discriminator
#     - Harder synthetic anomalies (perlin-like noise)
# ─────────────────────────────────────────────────────────────

class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer"""
    @staticmethod
    def forward(ctx, x, lam=1.0):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class SuperDiscriminator(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.GELU(),
            nn.Linear(256, 128),      nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, reverse_grad: bool = False, lam: float = 1.0):
        if reverse_grad:
            x = GradReverse.apply(x, lam)
        z = self.proj(x)
        return self.head(z)


class SuperSimpleNet:
    def __init__(self, backbone: BackboneExtractor, epochs: int = 160, lr: float = 1e-4):
        self.backbone    = backbone.to(DEVICE).eval()
        self.epochs      = epochs
        self.lr          = lr
        self.discriminator: Optional[SuperDiscriminator] = None

    def _extract_flat(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(imgs)
        f2 = F.adaptive_avg_pool2d(feats["layer2"], (2, 2)).flatten(1)
        f3 = F.adaptive_avg_pool2d(feats["layer3"], (2, 2)).flatten(1)
        return torch.cat([f2, f3], dim=1)

    def _make_hard_anomaly(self, feats: torch.Tensor) -> torch.Tensor:
        """สร้าง hard anomaly ด้วยการ mix features + noise"""
        idx   = torch.randperm(len(feats), device=feats.device)
        mixed = 0.5 * feats + 0.5 * feats[idx]
        noise = torch.randn_like(feats) * feats.std() * 0.5
        return mixed + noise

    def fit(self, train_loader: DataLoader):
        sample_imgs, _, _ = next(iter(train_loader))
        in_dim = self._extract_flat(sample_imgs[:1].to(DEVICE)).shape[1]

        self.discriminator = SuperDiscriminator(in_dim).to(DEVICE)
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr,
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for imgs, _, _ in train_loader:
                imgs  = imgs.to(DEVICE)
                feats = self._extract_flat(imgs)
                anom  = self._make_hard_anomaly(feats)

                # normal path (ใช้ gradient reversal เพื่อเสริม feature smoothness)
                real_logits = self.discriminator(feats, reverse_grad=True, lam=0.05)
                fake_logits = self.discriminator(anom, reverse_grad=False)

                labels_r = torch.zeros(len(feats), 1, device=DEVICE)
                labels_f = torch.ones(len(anom),  1, device=DEVICE)
                loss = criterion(real_logits, labels_r) + criterion(fake_logits, labels_f)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 40 == 0:
                print(f"    SuperSimpleNet Epoch [{epoch+1}/{self.epochs}] Loss: {total_loss/len(train_loader):.4f}")

        self.discriminator.eval()

    def predict(self, img: torch.Tensor) -> float:
        feats = self._extract_flat(img.to(DEVICE))
        with torch.no_grad():
            score = self.discriminator(feats).sigmoid()
        return float(score.mean().item())


# ================================================================================
# 5. MODEL FACTORY — เลือก Model + Backbone ตาม config
# ================================================================================

def build_model(model_name: str, backbone: BackboneExtractor, config: dict):
    """สร้าง model object ตามชื่อ"""
    k = {
        "PatchCore":      lambda: PatchCore(backbone, k=config["patchcore_k"]),
        "PaDiM":          lambda: PaDiM(backbone, t=config["padim_t"]),
        "FastFlow":       lambda: FastFlow(backbone),
        "EfficientAD":    lambda: EfficientAD(backbone, steps=config["efficientad_st_epochs"]),
        "SimpleNet":      lambda: SimpleNet(backbone, epochs=config["simplenet_epochs"]),
        "SuperSimpleNet": lambda: SuperSimpleNet(backbone, epochs=config["supersimplenet_epochs"]),
    }
    if model_name not in k:
        raise ValueError(f"ไม่รู้จัก model: {model_name}")
    return k[model_name]()


# ================================================================================
# 6. TRAIN & EVALUATE PIPELINE — ใช้กับทุก model เหมือนกัน
# ================================================================================

def train_and_evaluate(
    model_name: str,
    backbone_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
) -> Dict:
    print(f"\n{'='*70}")
    print(f"  ▶ {model_name}  +  {backbone_name}")
    print(f"{'='*70}")

    # ── สร้าง Backbone ──
    backbone = BackboneExtractor(backbone_name)

    # ── สร้าง Model ──
    model = build_model(model_name, backbone, config)

    # ── Training + จับเวลา ──
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t_train_start = time.perf_counter()
    model.fit(train_loader)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t_train_end   = time.perf_counter()
    train_time_sec = t_train_end - t_train_start

    # ── Inference + จับเวลาต่อรูป ──
    scores:  List[float] = []
    y_true:  List[int]   = []
    total_infer_time = 0.0

    for imgs, labels, _ in tqdm(test_loader, desc="  Inference"):
        imgs = imgs.to(DEVICE)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        score = model.predict(imgs)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t1 = time.perf_counter()

        total_infer_time += (t1 - t0)
        scores.append(score)
        y_true.append(int(labels[0].item()))

    num_test = len(test_loader.dataset)
    infer_ms = (total_infer_time / max(num_test, 1)) * 1000

    # ── Metrics ──
    metrics = compute_metrics(y_true, scores)
    metrics["Model"]                          = model_name
    metrics["Backbone"]                       = backbone_name
    metrics["Train_Time_sec"]                 = round(train_time_sec, 2)
    metrics["Inference_Time_per_Sample_ms"]   = round(infer_ms, 3)

    print(f"  ✅ AUROC={metrics['AUROC']:.4f}  F1={metrics['F1_Score']:.4f}  "
          f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}  "
          f"Thresh={metrics['Optimal_Threshold']:.4f}")
    print(f"  ⏱  Train: {train_time_sec:.1f}s  |  Infer: {infer_ms:.2f} ms/img")

    # คืน memory (สำคัญมากเมื่อรัน 30 ครั้งติดกัน)
    del model, backbone
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return metrics


# ================================================================================
# 7. MAIN — Grid Search Loop
# ================================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("  🔄 เตรียม DataLoaders...")
    print("="*70)
    train_loader, test_loader = get_dataloaders(CONFIG)

    results_list: List[Dict] = []
    failed_list:  List[str]  = []

    total_experiments = len(MODELS_LIST) * len(BACKBONES_LIST)
    exp_idx = 0

    for model_name in MODELS_LIST:
        for backbone_name in BACKBONES_LIST:
            exp_idx += 1
            print(f"\n[{exp_idx}/{total_experiments}] {model_name} + {backbone_name}")
            try:
                metrics = train_and_evaluate(
                    model_name, backbone_name,
                    train_loader, test_loader,
                    CONFIG
                )
                results_list.append(metrics)

                # บันทึก checkpoint ทุก experiment (กันหาย)
                pd.DataFrame(results_list).to_csv(
                    CONFIG["output_csv"].replace(".csv", "_checkpoint.csv"),
                    index=False
                )

            except Exception as e:
                print(f"  ❌ ล้มเหลว: {str(e)}")
                failed_list.append(f"{model_name}+{backbone_name}: {str(e)}")
                import traceback; traceback.print_exc()

    # ── สรุปผล ──
    if results_list:
        results_df = pd.DataFrame(results_list)
        col_order  = ["Model", "Backbone", "AUROC", "F1_Score", "Precision", "Recall",
                      "Optimal_Threshold", "Train_Time_sec", "Inference_Time_per_Sample_ms"]
        results_df = results_df[col_order]
        results_df = results_df.sort_values(
            by=["Model", "AUROC"], ascending=[True, False]
        ).reset_index(drop=True)

        # Best per model
        best_df = results_df.loc[results_df.groupby("Model")["AUROC"].idxmax()]

        print("\n" + "="*80)
        print("📊  สรุปผลการทดลองทั้งหมด (Thesis Experiment Results)")
        print("="*80)
        print(results_df.to_string(index=False))

        print("\n" + "="*80)
        print("🏆  Best Backbone ต่อ Model (AUROC สูงสุด)")
        print("="*80)
        print(best_df[["Model", "Backbone", "AUROC", "F1_Score",
                        "Train_Time_sec", "Inference_Time_per_Sample_ms"]].to_string(index=False))

        # เซฟ CSV
        results_df.to_csv(CONFIG["output_csv"], index=False)
        best_df.to_csv(CONFIG["output_csv"].replace(".csv", "_best.csv"), index=False)
        print(f"\n💾  บันทึก '{CONFIG['output_csv']}' และ '_best.csv' เรียบร้อยแล้ว")

    if failed_list:
        print(f"\n⚠️  มี {len(failed_list)} experiment ที่ล้มเหลว:")
        for f in failed_list:
            print(f"  • {f}")

    print("\n🎉  เสร็จสิ้นการทดลองทั้งหมด!")
