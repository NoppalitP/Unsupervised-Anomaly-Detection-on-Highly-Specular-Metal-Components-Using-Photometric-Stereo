import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from src.config.config import ModelConfig

class BackboneExtractor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        net = getattr(models, config.backbone)(weights="DEFAULT").to(config.device).eval()
        self.feat = nn.Sequential(*list(net.features[:7]))
        for p in self.parameters(): p.requires_grad = False
        self.device = config.device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feat(x.to(self.device))
        return F.normalize(F.adaptive_avg_pool2d(f, (14, 14)), dim=1)

class PatchCore:
    """Simplified PatchCore implementation."""
    def __init__(self, extractor: BackboneExtractor, config: ModelConfig):
        self.ext = extractor
        self.config = config
        self.bank: Optional[torch.Tensor] = None

    def fit(self, dl):
        feats = [self.ext(x).permute(0,2,3,1).reshape(-1, 768).cpu() for x, _ in dl]
        all_f = torch.cat(feats, 0)
        # Simple random coreset
        idx = torch.randperm(len(all_f))[:int(len(all_f) * self.config.coreset_ratio)]
        self.bank = all_f[idx].to(self.config.device)

    def predict(self, dl):
        scs, lbls, maps = [], [], []
        for x, y in dl:
            f = self.ext(x); B, C, H, W = f.shape
            dist = torch.cdist(f.permute(0,2,3,1).reshape(-1, C), self.bank).min(1).values
            m = dist.view(B, H, W).cpu().numpy()
            scs.append(m.max((1,2))); lbls.append(y.numpy()); maps.append(m)
        return np.concatenate(scs), np.concatenate(lbls), np.concatenate(maps, 0)
