import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.config.config import DataConfig
from src.core.solver import PhotometricStereoSolver

class AutoCropper:
    """Handles ROI extraction via perspective warp."""
    def __init__(self, config: DataConfig):
        self.config = config

    def find_bbox(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        stack = np.array(images, dtype=np.float32)
        robust = np.percentile(stack, 80, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(cv2.medianBlur(gray, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        box = cv2.boxPoints(rect).astype(np.float32)
        return self._order_points(box)

    def crop_and_resize(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        rect = bbox.copy()
        o = self.config.crop_offset
        rect[0] += [o, o]; rect[1] += [-o, o]; rect[2] += [-o, -o]; rect[3] += [o, -o]
        s = self.config.output_size - 1
        dst = np.array([[0,0],[s,0],[s,s],[0,s]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.config.output_size, self.config.output_size))

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        out = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
        out[0] = pts[np.argmin(s)]; out[2] = pts[np.argmax(s)]
        out[1] = pts[np.argmin(diff)]; out[3] = pts[np.argmax(diff)]
        return out

class MVTecDatasetBuilder:
    """Orchestrates PS solving and directory structured building."""
    def __init__(self, solver: PhotometricStereoSolver, cropper: AutoCropper, config: DataConfig):
        self.solver = solver
        self.cropper = cropper
        self.config = config

    def build(self) -> None:
        folders = sorted([f for f in self.config.raw_dir.iterdir() if f.is_dir()])
        for folder in tqdm(folders, desc="Building MVTec Dataset"):
            cls = folder.name.split('_')[-1].lower()
            imgs = [cv2.imread(str(p)) for p in sorted(folder.glob("light_*.png"))]
            ps_map = self.solver.solve(imgs)
            bbox = self.cropper.find_bbox(imgs)
            if bbox is not None:
                out = self.cropper.crop_and_resize(ps_map, bbox)
                split = "train" if cls == "good" and random.random() < self.config.train_ratio else "test"
                save_p = self.config.out_dir / split / cls / (folder.name + ".png")
                save_p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_p), out)

class MVTecDataset(Dataset):
    def __init__(self, root: Path, split: str, transform: transforms.Compose):
        self.transform = transform
        self.samples = []
        path = root / split
        if not path.exists(): return
        for cls_dir in sorted(path.iterdir()):
            if cls_dir.is_dir():
                lbl = 0 if cls_dir.name == "good" else 1
                for p in sorted(cls_dir.glob("*.png")): self.samples.append((p, lbl))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        img = Image.fromarray(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        return self.transform(img), lbl
