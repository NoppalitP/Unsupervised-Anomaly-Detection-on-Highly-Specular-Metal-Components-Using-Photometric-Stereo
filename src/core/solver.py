import torch
import cv2
import numpy as np
from typing import List, Tuple
from src.config.config import PSConfig

class PhotometricStereoSolver:
    """Solves for surface normals and albedo using GPU-accelerated WLS."""
    def __init__(self, L_matrix: np.ndarray, config: PSConfig, device: torch.device):
        self.L_matrix = L_matrix.astype(np.float32)
        self.config = config
        self.device = device
        self.n_lights = L_matrix.shape[0]

    def solve(self, images: List[np.ndarray]) -> np.ndarray:
        """Processes images into a normal map (BGR)."""
        gray_stack = self._to_gray_stack(images)
        h, w = gray_stack.shape[1:]
        mask = self._get_mask(images)
        
        if self.config.output_mode == "before":
            return self._handle_before_mode(images, mask)

        I_valid = gray_stack[:, mask].T
        P = I_valid.shape[0]
        if P == 0: return np.zeros((h, w, 3), dtype=np.uint8)

        W = self._build_weight_mask(I_valid)
        N_unit = self._wls_solve(I_valid, W, P)

        res = np.zeros((h, w, 3), dtype=np.uint8)
        res[mask, 2] = ((N_unit[:, 0] + 1) / 2 * 255).astype(np.uint8) # Nx
        res[mask, 1] = ((N_unit[:, 1] + 1) / 2 * 255).astype(np.uint8) # Ny
        res[mask, 0] = ((N_unit[:, 2] + 1) / 2 * 255).astype(np.uint8) # Nz
        return res

    def _to_gray_stack(self, images: List[np.ndarray]) -> np.ndarray:
        return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) for img in images])

    def _get_mask(self, images: List[np.ndarray]) -> np.ndarray:
        robust = np.percentile(np.array(images), 80, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(robust, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(cv2.medianBlur(gray, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask > 0

    def _build_weight_mask(self, I: np.ndarray) -> np.ndarray:
        P, N = I.shape
        sort_idx = np.argsort(I, axis=1)
        W = np.ones((P, N), dtype=np.float32)
        px = np.arange(P)
        for i in range(self.config.drop_dark): W[px, sort_idx[:, i]] = 0.
        for i in range(N - self.config.drop_bright, N): W[px, sort_idx[:, i]] = 0.
        return W

    def _wls_solve(self, I: np.ndarray, W: np.ndarray, P: int) -> np.ndarray:
        L_t, W_t, I_t = [torch.tensor(x, device=self.device) for x in [self.L_matrix, W, I]]
        L_px = L_t.unsqueeze(0).expand(P, -1, -1)
        L_W = W_t.unsqueeze(-1) * L_px
        A = torch.bmm(L_W.transpose(1, 2), L_px) + torch.eye(3, device=self.device) * self.config.lambda_reg
        B = torch.bmm(L_W.transpose(1, 2), I_t.unsqueeze(-1))
        N_raw = torch.linalg.solve(A, B).squeeze(-1)
        norm = torch.linalg.norm(N_raw, dim=1, keepdim=True).clamp(min=1e-5)
        return (N_raw / norm).cpu().numpy()

    def _handle_before_mode(self, images, mask):
        idx = self.config.before_light_idx % len(images)
        res = images[idx].copy()
        res[~mask] = 0
        return res
