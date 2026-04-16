# =============================================================================
# build_dataset.py
#
# Photometric Stereo → MVTec AD Format Dataset Builder
# Refactored OOP pipeline for EfficientAD training on Google Colab
#
# Architecture:
#   PhotometricStereoSolver  →  AutoCropper  →  MVTecDatasetBuilder
#
# Usage (Google Colab / Windows):
#   python build_dataset.py \
#       --raw_dir raw_captures \
#       --out_dir mvtec_dataset \
#       --calib_npy light_directions.npy \
#       --slant_deg 45 \
#       --drop_dark 2 \
#       --drop_bright 5 \
#       --crop_offset 12 \
#       --output_size 512 \
#       --train_ratio 0.8 \
#       --seed 42 \
#       --output_mode after
#
# output_mode choices:
#   after  → WLS Photometric Stereo Normal Map (Nx, Ny, Nz)  [default]
#   before → Single raw light image (index set by --before_light_idx, default=0)
#            Use --before_light_idx 0 for first light, -1 for last, etc.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# 1. PhotometricStereoSolver
# =============================================================================
class PhotometricStereoSolver:
    """
    Solves the Lambertian Photometric Stereo equation:
        I = ρ (N · L)
    using GPU-accelerated Weighted Least Squares (WLS).

    Implements pixel-wise outlier rejection:
        - Drop the `drop_dark`   darkest  light readings  (shadow rejection)
        - Drop the `drop_bright` brightest light readings (specular rejection)

    Returns a 3-channel uint8 BGR normal map image.
    Convention matches photometric_stereo.py (valid pixels only, bg Nx/Ny=128, Nz=0).
    """

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
        """
        Parameters
        ----------
        L_matrix          : (N_lights, 3) float32 array of unit light direction vectors.
        drop_dark         : Number of darkest observations to zero-weight per pixel.
        drop_bright       : Number of brightest observations to zero-weight per pixel.
        lambda_reg        : Tikhonov regularisation added to A = LᵀWL + λI.
        output_mode       : "after"  → WLS Normal Map Nx,Ny,Nz (default)
                            "before" → Single raw light image (no PS math)
        before_light_idx  : Index of the light image to use when output_mode="before".
                            0 = first image (default), 1 = second, -1 = last, etc.
        device            : torch.device; defaults to CUDA if available, else CPU.
        """
        if L_matrix.ndim != 2 or L_matrix.shape[1] != 3:
            raise ValueError("L_matrix must be shape (N_lights, 3).")
        if output_mode not in ("after", "before"):
            raise ValueError(
                f"output_mode must be 'after' or 'before', got '{output_mode}'."
            )

        self.L_matrix = L_matrix.astype(np.float32)
        self.n_lights = L_matrix.shape[0]
        self.drop_dark = drop_dark
        self.drop_bright = drop_bright
        self.lambda_reg = lambda_reg
        self.output_mode = output_mode
        self.before_light_idx = before_light_idx
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(
            f"PhotometricStereoSolver | lights={self.n_lights} | "
            f"drop_dark={drop_dark} | drop_bright={drop_bright} | "
            f"output_mode={output_mode} | before_light_idx={before_light_idx} | "
            f"device={self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, images: list[np.ndarray]) -> np.ndarray:
            """
            Parameters
            ----------
            images : List of N_lights BGR or grayscale uint8 images, all same shape.

            Returns
            -------
            result : (H, W, 3) uint8 array — Normal Map in BGR order for cv2.imwrite.
                    Convention matches photometric_stereo.py:
                    - Valid pixels: RGB=(Nx,Ny,Nz) encoded to [0,255], stored as BGR.
                    - Background (Nx,Ny): 128  (= 0 in [-1,1] space → neutral)
                    - Background (Nz):     0   (same as photometric_stereo.py)
                    Flat surfaces appear classic bluish-purple (RGB: 128, 128, 255).
            """
            if len(images) != self.n_lights:
                raise ValueError(
                    f"Expected {self.n_lights} images, got {len(images)}."
                )

            # ── Convert to grayscale float stack ────────────────────────────
            gray_stack = self._to_gray_stack(images)          # (N, H, W) float32
            h, w = gray_stack.shape[1], gray_stack.shape[2]

            # ── Build object mask — same pipeline as photometric_stereo.py ──
            # P70 robust max → Grayscale → MedianBlur → Otsu → Morph → largest contour
            stack_uint8  = np.array(images, dtype=np.float32)
            robust_max   = np.percentile(stack_uint8, 80, axis=0).astype(np.uint8)
            gray_robust  = cv2.cvtColor(robust_max, cv2.COLOR_BGR2GRAY)
            blurred      = cv2.medianBlur(gray_robust, 5)
            _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel5      = np.ones((5, 5), np.uint8)
            mask_morph   = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel5)
            mask_morph   = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN,  kernel5)
            contours, _  = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_mask   = np.zeros((h, w), dtype=np.uint8)
            if contours:
                cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)],
                                 -1, 255, thickness=cv2.FILLED)
            valid_mask = final_mask > 0                                   # (H, W) bool

            # ── Extract only valid pixel intensities (same as photometric_stereo.py) ─
            I_valid      = gray_stack[:, valid_mask].T                    # (P, N)
            n_pixels     = I_valid.shape[0]

            if n_pixels == 0:
                log.warning("No valid pixels found — returning blank output.")
                result = np.zeros((h, w, 3), dtype=np.uint8)
                result[..., 0] = 128   # Nx → 128
                result[..., 1] = 128   # Ny → 128
                result[..., 2] = 0     # Nz → 0  (same as photometric_stereo.py)
                return result

            # ── Branch: "before" — single raw light image, no PS math ──────
            if self.output_mode == "before":
                # Pick one raw light image by index (default 0 = first capture).
                # Clamp index to valid range so -1 (last), etc. all work safely.
                idx = self.before_light_idx % len(images)
                single = images[idx]                                      # (H,W,3) BGR
                # Zero out background outside object mask
                single_masked = single.copy()
                single_masked[~valid_mask] = 0
                return single_masked                                      # (H,W,3) BGR

            # ── Branch: "after" — full WLS Photometric Stereo ─────────────────
            # ── Pixel-wise Outlier Rejection → weight_mask: (P, N) ──────────
            weight_mask_px = self._build_weight_mask(I_valid)             # (P, N) float32

            # ── GPU Weighted Least Squares ────────────────────────────────────
            N_unit, _ = self._wls_torch(I_valid, weight_mask_px, n_pixels)
            # N_unit : (P, 3) unit normal vectors [-1, 1]

            # ── Map back to 2D with background convention ─────────────────────
            # Convention from photometric_stereo.py:
            #   nx_display / ny_display background = 128
            #   nz_display background = 0
            nx_disp = ((N_unit[:, 0] + 1.0) / 2.0 * 255.0).astype(np.uint8)
            ny_disp = ((N_unit[:, 1] + 1.0) / 2.0 * 255.0).astype(np.uint8)
            nz_disp = ((N_unit[:, 2] + 1.0) / 2.0 * 255.0).astype(np.uint8)

            nx_map = np.full((h, w), 128, dtype=np.uint8)
            ny_map = np.full((h, w), 128, dtype=np.uint8)
            nz_map = np.zeros((h, w), dtype=np.uint8)            # bg = 0

            nx_map[valid_mask] = nx_disp
            ny_map[valid_mask] = ny_disp
            nz_map[valid_mask] = nz_disp

            # ── Stack as BGR (R=Nx, G=Ny, B=Nz → stored BGR: Nz,Ny,Nx) ──────
            result = np.stack([nz_map, ny_map, nx_map], axis=-1)          # (H,W,3) BGR
            return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _to_gray_stack(self, images: list[np.ndarray]) -> np.ndarray:
        """Convert list of BGR/gray images to (N, H, W) float32 stack."""
        gray_list = []
        for img in images:
            if img is None:
                raise ValueError("One of the input images is None (failed to load).")
            if img.ndim == 3:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                g = img.astype(np.float32)
            gray_list.append(g)
        return np.array(gray_list, dtype=np.float32)   # (N, H, W)

    def _build_weight_mask(self, I_valid: np.ndarray) -> np.ndarray:
        """
        Pixel-wise outlier rejection.

        For each pixel (row) sort intensities across lights;
        zero-weight the `drop_dark` darkest and `drop_bright` brightest.

        Parameters
        ----------
        I_valid : (P, N) float32

        Returns
        -------
        weight_mask : (P, N) float32  values in {0, 1}
        """
        n_pixels, n_lights = I_valid.shape
        sort_idx = np.argsort(I_valid, axis=1)          # (P, N)
        weight_mask = np.ones((n_pixels, n_lights), dtype=np.float32)
        idx_arr = np.arange(n_pixels)

        for i in range(self.drop_dark):
            weight_mask[idx_arr, sort_idx[:, i]] = 0.0

        for i in range(n_lights - self.drop_bright, n_lights):
            weight_mask[idx_arr, sort_idx[:, i]] = 0.0

        return weight_mask

    def _wls_torch(
        self,
        I_valid: np.ndarray,
        weight_mask: np.ndarray,
        n_pixels: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU Weighted Least Squares solver.

        Solves per pixel:  n = (LᵀWL + λI)⁻¹ Lᵀ W I

        Returns
        -------
        N_unit   : (P, 3) numpy float32 — unit normal vectors for valid pixels
        albedo   : (P,)   numpy uint8   — albedo magnitudes (0-255)
        """
        device = self.device
        n_lights = self.n_lights

        # Move to device
        L_t  = torch.tensor(self.L_matrix, dtype=torch.float32, device=device)  # (N, 3)
        W_t  = torch.tensor(weight_mask,   dtype=torch.float32, device=device)  # (P, N)
        I_t  = torch.tensor(I_valid,       dtype=torch.float32, device=device)  # (P, N)
        I_eye = torch.eye(3, dtype=torch.float32, device=device) * self.lambda_reg

        # ── GPU warm-up (avoid CUDA cold-start timing skew) ─────────────
        if device.type == "cuda":
            _ = torch.bmm(
                torch.randn(8, 3, 3, device=device),
                torch.randn(8, 3, 1, device=device),
            )
            torch.cuda.synchronize()

        # ── Build per-pixel L matrix: (P, N, 3) ─────────────────────────
        # expand() uses a view — no extra memory allocated
        L_px = L_t.unsqueeze(0).expand(n_pixels, -1, -1)   # (P, N, 3)

        # ── Apply weights: L_W = W[:, :, None] * L_px ───────────────────
        L_W = W_t.unsqueeze(-1) * L_px                     # (P, N, 3)

        # ── Normal equations ─────────────────────────────────────────────
        # A = LᵀWL + λI   shape (P, 3, 3)
        # B = LᵀWI        shape (P, 3, 1)
        A = torch.bmm(L_W.transpose(1, 2), L_px) + I_eye   # (P, 3, 3)
        B = torch.bmm(L_W.transpose(1, 2), I_t.unsqueeze(-1))  # (P, 3, 1)

        # ── Solve Ax = B ─────────────────────────────────────────────────
        N_raw = torch.linalg.solve(A, B).squeeze(-1)        # (P, 3)

        # ── Albedo = ||N_raw|| ───────────────────────────────────────────
        norm = torch.linalg.norm(N_raw, dim=1, keepdim=True)   # (P, 1)
        norm.clamp_(min=self.lambda_reg)

        # ── Unit normals ─────────────────────────────────────────────────
        N_unit_t = N_raw / norm                             # (P, 3)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # ── Back to CPU numpy ────────────────────────────────────────────
        N_unit = N_unit_t.cpu().numpy()                     # (P, 3)
        norm_np = norm.squeeze(-1).cpu().numpy()            # (P,)

        # Normalise albedo to [0, 255] uint8
        alb_min, alb_max = norm_np.min(), norm_np.max()
        if alb_max > alb_min:
            albedo = ((norm_np - alb_min) / (alb_max - alb_min) * 255.0).astype(np.uint8)
        else:
            albedo = np.zeros(n_pixels, dtype=np.uint8)

        return N_unit, albedo


# =============================================================================
# 2. AutoCropper
# =============================================================================
class AutoCropper:
    """
    Extracts the Region of Interest (ROI) from a stacked (Nx, Ny, Albedo) image
    using Perspective Warp on the minimum-area rotated rectangle.

    Pipeline
    --------
    1. Robust max (80th percentile across lights) → grayscale
    2. Median blur → Otsu thresholding
    3. Morphological Close + Open  (5×5 kernel)
    4. Largest contour → cv2.minAreaRect  (handles tilted objects)
    5. Shrink 4 corners inward by `crop_offset` px  (avoids blue border bleed)
    6. Perspective warp → square output at `output_size × output_size`
    """

    def __init__(
        self,
        padding: int = 15,
        output_size: int = 512,
        crop_offset: int = 12,
    ) -> None:
        self.padding = padding          # kept for API compatibility (not used in warp path)
        self.output_size = output_size
        self.crop_offset = crop_offset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_bbox(self, images: list[np.ndarray]) -> Optional[np.ndarray]:
        """
        Find the rotated bounding box of the object in the image set.

        Parameters
        ----------
        images : List of N BGR uint8 images.

        Returns
        -------
        box : (4, 2) float32 array of corner points from cv2.boxPoints,
              ordered [TL, TR, BR, BL].
        None if no contour is found.
        """
        # ── Robust max: 80th-percentile across all light images ──────────
        stack = np.array(images, dtype=np.float32)               # (N, H, W, 3)
        robust = np.percentile(stack, 80, axis=0).astype(np.uint8)  # (H, W, 3)
        gray = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)

        # ── Smooth + Otsu ────────────────────────────────────────────────
        blurred = cv2.medianBlur(gray, 5)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── Morphological clean-up ───────────────────────────────────────
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

        # ── Largest contour → minimum-area rotated rect ──────────────────
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)                          # (center, size, angle)
        box = cv2.boxPoints(rect).astype(np.float32)             # (4, 2)

        # ── Order corners: TL, TR, BR, BL ────────────────────────────────
        box = self._order_points(box)
        return box

    def crop_and_resize(
        self,
        ny_map: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """
        Perspective-warp the PS image using the 4-corner rotated bbox.

        The 4 corners are shrunk inward by `crop_offset` pixels before warping
        to avoid capturing border artefacts.

        Parameters
        ----------
        ny_map : (H, W) or (H, W, 3) uint8  — PS output [Nx | Ny | Albedo]
        bbox   : (4, 2) float32 ordered [TL, TR, BR, BL]

        Returns
        -------
        (output_size, output_size, 3) uint8
        """
        rect = bbox.copy()
        o = self.crop_offset

        # Shrink each corner inward (away from its respective edge)
        rect[0] += [ o,  o]   # TL → move right + down
        rect[1] += [-o,  o]   # TR → move left  + down
        rect[2] += [-o, -o]   # BR → move left  + up
        rect[3] += [ o, -o]   # BL → move right + up

        # Destination: perfect square
        s = self.output_size - 1
        dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            ny_map, M, (self.output_size, self.output_size),
            flags=cv2.INTER_AREA,
        )
        return warped

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

        Uses the sum-and-diff trick:
            TL has smallest sum  (x+y),  BR has largest sum.
            TR has smallest diff (x-y),  BL has largest diff.
        """
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]   # TL
        ordered[2] = pts[np.argmax(s)]   # BR
        diff = np.diff(pts, axis=1).ravel()
        ordered[1] = pts[np.argmin(diff)]  # TR
        ordered[3] = pts[np.argmax(diff)]  # BL
        return ordered


# =============================================================================
# 3. MVTecDatasetBuilder
# =============================================================================
class MVTecDatasetBuilder:
    """
    Traverses `raw_dir`, processes each sample through the PS solver and
    cropper, then writes results in MVTec AD directory format.

    Folder naming convention for raw captures:
        <timestamp>_<class_name>
        e.g.  20260305_103000_good
              20260305_110500_scratch

    Output structure
    ----------------
    mvtec_dataset/
    ├── train/
    │   └── good/
    │       └── *.png
    ├── test/
    │   ├── good/
    │   │   └── *.png
    │   └── scratch/
    │       └── *.png
    └── ground_truth/
        └── scratch/
            └── *.png   ← blank black 512×512 placeholder masks
    """

    # Regex: match folder names like  20260305_103000_good
    #        captures the class part after the last underscore-separated date+time block
    _FOLDER_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")

    def __init__(
        self,
        raw_dir: str | Path,
        out_dir: str | Path,
        solver: PhotometricStereoSolver,
        cropper: AutoCropper,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.solver = solver
        self.cropper = cropper
        self.train_ratio = train_ratio
        self.rng = random.Random(seed)

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"raw_dir not found: {self.raw_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> None:
        """
        Main entry point.  Iterates over all capture folders and builds
        the MVTec dataset.
        """
        sample_folders = self._discover_folders()

        if not sample_folders:
            log.error("No valid capture folders found in '%s'. Exiting.", self.raw_dir)
            return

        log.info("Found %d capture folder(s) to process.", len(sample_folders))

        # Group by class so we can compute train/test split per class
        class_groups: dict[str, list[Path]] = {}
        for folder, class_name in sample_folders:
            class_groups.setdefault(class_name, []).append(folder)

        # Shuffle + split `good` samples; all defects go to test
        for class_name, folders in class_groups.items():
            self.rng.shuffle(folders)
            log.info("Class '%s': %d sample(s).", class_name, len(folders))

        # Flatten into processing queue with destination info
        queue: list[Tuple[Path, str, str]] = []  # (folder, class_name, split)
        for class_name, folders in class_groups.items():
            if class_name == "good":
                n_train = max(1, int(len(folders) * self.train_ratio))
                for i, folder in enumerate(folders):
                    split = "train" if i < n_train else "test"
                    queue.append((folder, class_name, split))
            else:
                for folder in folders:
                    queue.append((folder, class_name, "test"))

        # ── Process ─────────────────────────────────────────────────────
        ok, skipped = 0, 0
        for folder, class_name, split in tqdm(queue, desc="Building dataset", unit="sample"):
            try:
                self._process_one(folder, class_name, split)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("SKIP '%s' — %s: %s", folder.name, type(exc).__name__, exc)
                skipped += 1

        log.info("Done. Processed: %d | Skipped: %d", ok, skipped)
        log.info("Dataset written to: %s", self.out_dir.resolve())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _discover_folders(self) -> list[Tuple[Path, str]]:
        """Return list of (folder_path, class_name) for valid capture dirs."""
        result = []
        for entry in sorted(self.raw_dir.iterdir()):
            if not entry.is_dir():
                continue
            m = self._FOLDER_RE.match(entry.name)
            if not m:
                log.debug("Skipping non-matching folder: %s", entry.name)
                continue
            class_name = m.group(1).lower().strip()
            result.append((entry, class_name))
        return result

    def _process_one(self, folder: Path, class_name: str, split: str) -> None:
        """
        Process a single capture folder:
          1. Load all light images
          2. Run PS solver
          3. Auto-crop
          4. Save output + ground-truth mask (if defect)
        """
        # ── Load images ──────────────────────────────────────────────────
        img_paths = sorted(folder.glob("light_*.png"))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No light_*.png images found in {folder}")

        images = []
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                raise IOError(f"cv2.imread failed for {p}")
            images.append(img)

        if len(images) != self.solver.n_lights:
            raise ValueError(
                f"Folder '{folder.name}' has {len(images)} images, "
                f"expected {self.solver.n_lights}."
            )

        # ── PS solve ─────────────────────────────────────────────────────
        ps_map = self.solver.solve(images)                    # (H, W, 3) uint8

        # ── Find crop bbox (uses raw images for robust masking) ──────────
        bbox = self.cropper.find_bbox(images)
        if bbox is None:
            raise RuntimeError("AutoCropper could not find a valid object contour.")

        # ── Crop & resize ─────────────────────────────────────────────────
        output_img = self.cropper.crop_and_resize(ps_map, bbox)    # (512, 512, 3)

        # ── Determine output path ─────────────────────────────────────────
        filename = folder.name + ".png"

        if split == "train":
            save_path = self.out_dir / "train" / class_name / filename
        else:
            save_path = self.out_dir / "test" / class_name / filename

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), output_img)

        # ── Ground-truth mask for defect classes ─────────────────────────
        if class_name != "good":
            gt_path = self.out_dir / "ground_truth" / class_name / filename
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            size = self.cropper.output_size
            blank_mask = np.zeros((size, size), dtype=np.uint8)
            cv2.imwrite(str(gt_path), blank_mask)


# =============================================================================
# Light Matrix Utilities
# =============================================================================
def build_theoretical_L_matrix(n_lights: int, slant_deg: float = 45.0) -> np.ndarray:
    """
    Generate a theoretical light direction matrix assuming lights are arranged
    evenly around a ring at a fixed slant angle.

    Returns (n_lights, 3) float32 unit vectors.
    """
    slant = np.radians(slant_deg)
    L = np.zeros((n_lights, 3), dtype=np.float32)
    for i in range(n_lights):
        azimuth = 2.0 * np.pi * i / n_lights
        L[i, 0] = np.cos(azimuth) * np.sin(slant)
        L[i, 1] = np.sin(azimuth) * np.sin(slant)
        L[i, 2] = np.cos(slant)
    return L


def load_L_matrix(path: str) -> np.ndarray:
    """Load a calibrated L matrix from a .npy file."""
    L = np.load(path).astype(np.float32)
    if L.ndim != 2 or L.shape[1] != 3:
        raise ValueError(f"Expected shape (N,3) in {path}, got {L.shape}")
    return L


# =============================================================================
# CLI Entry Point
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a MVTec AD format dataset from Photometric Stereo captures."
    )
    p.add_argument("--raw_dir",     type=str, default="raw_captures",
                   help="Root folder containing timestamped capture sub-folders.")
    p.add_argument("--out_dir",     type=str, default="mvtec_dataset",
                   help="Output root for the MVTec AD formatted dataset.")
    p.add_argument("--calib_npy",   type=str, default=None,
                   help="Path to calibrated light_directions.npy. "
                        "If omitted, a theoretical matrix is generated.")
    p.add_argument("--n_lights",    type=int, default=24,
                   help="Number of lights (used only when --calib_npy is omitted).")
    p.add_argument("--slant_deg",   type=float, default=45.0,
                   help="Slant angle in degrees for theoretical L matrix.")
    p.add_argument("--drop_dark",   type=int, default=2,
                   help="Number of darkest readings to reject per pixel.")
    p.add_argument("--drop_bright", type=int, default=5,
                   help="Number of brightest readings to reject per pixel.")
    p.add_argument("--crop_offset", type=int, default=12,
                   help="Pixels to shrink each corner inward before perspective warp "
                        "(avoids border colour bleed).")
    p.add_argument("--output_size", type=int, default=512,
                   help="Final square output resolution for EfficientAD.")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="Fraction of 'good' samples routed to train/good/.")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for reproducible train/test split.")
    p.add_argument("--output_mode", type=str, default="after",
                   choices=["after", "before"],
                   help="Output channel mode: "
                        "'after'  = WLS Normal Map (Nx,Ny,Nz) — after Photometric Stereo (default). "
                        "'before' = Single raw light image — before Photometric Stereo.")
    p.add_argument("--before_light_idx", type=int, default=0,
                   help="Index of the light image to use when --output_mode=before. "
                        "0 = first image (default), 1 = second, -1 = last.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Light matrix ─────────────────────────────────────────────────────
    if args.calib_npy and Path(args.calib_npy).exists():
        log.info("Loading calibrated L matrix from '%s'.", args.calib_npy)
        L_matrix = load_L_matrix(args.calib_npy)
    else:
        if args.calib_npy:
            log.warning(
                "Calibration file '%s' not found — using theoretical matrix.",
                args.calib_npy,
            )
        else:
            log.info("No --calib_npy provided — generating theoretical L matrix.")
        L_matrix = build_theoretical_L_matrix(args.n_lights, args.slant_deg)

    log.info("L matrix shape: %s | slant=%.1f°", L_matrix.shape, args.slant_deg)

    # ── Instantiate pipeline components ──────────────────────────────────
    solver = PhotometricStereoSolver(
        L_matrix=L_matrix,
        drop_dark=args.drop_dark,
        drop_bright=args.drop_bright,
        output_mode=args.output_mode,
        before_light_idx=args.before_light_idx,
    )

    cropper = AutoCropper(
        output_size=args.output_size,
        crop_offset=args.crop_offset,
    )

    builder = MVTecDatasetBuilder(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        solver=solver,
        cropper=cropper,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    builder.build()


if __name__ == "__main__":
    main()