import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import random
from tqdm import tqdm
from typing import Tuple, List, Optional
from src.config.config import ModelConfig

# ── Shared Backbone ───────────────────────────────────────────────────────────
class BackboneExtractor(nn.Module):
    _LAYER_MAP = {
        "convnext_tiny": (["features.4", "features.6"], models.convnext_tiny),
        "resnet50": (["layer2", "layer3"], models.resnet50),
        "efficientnet_b4": (["features.4", "features.6"], models.efficientnet_b4),
    }

    def __init__(self, config: ModelConfig):
        super().__init__()
        backbone_name = config.backbone
        layers, factory = self._LAYER_MAP.get(backbone_name, self._LAYER_MAP["convnext_tiny"])
        net = factory(weights="DEFAULT").to(config.device).eval()
        for p in net.parameters(): p.requires_grad = False
        self.net = net
        self.device = config.device
        self._hooks = {}
        self._handles = []

        for name in layers:
            mod = self._get_mod(name)
            self._handles.append(mod.register_forward_hook(self._hook(name)))
        
        # Probe feature dimension
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad(): self.feat_dim = self.forward(dummy).shape[1]

    def _get_mod(self, name: str):
        m = self.net
        for p in name.split("."): m = getattr(m, p)
        return m

    def _hook(self, name):
        def fn(_, __, out): self._hooks[name] = out.detach()
        return fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._hooks.clear()
        _ = self.net(x.to(self.device))
        parts = [F.adaptive_avg_pool2d(v, (14, 14)) for v in self._hooks.values()]
        return F.normalize(torch.cat(parts, dim=1), dim=1)

# ── 1. PatchCore ──────────────────────────────────────────────────────────────
class PatchCore:
    def __init__(self, extractor: BackboneExtractor, config: ModelConfig):
        self.ext = extractor; self.ratio = config.pc_coreset; self.bank = None

    def fit(self, dl):
        feats = [self.ext(x).permute(0,2,3,1).reshape(-1, self.ext.feat_dim).cpu() for x, _ in dl]
        all_f = torch.cat(feats, 0)
        k = max(1, int(len(all_f) * self.ratio))
        self.bank = self._coreset(all_f, k).to(self.ext.device)

    def _coreset(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        if k >= len(feats): return feats
        sel = [random.randint(0, len(feats)-1)]
        dists = torch.cdist(feats[sel], feats).squeeze(0)
        for _ in range(1, k):
            i = torch.argmax(dists).item()
            sel.append(int(i))
            dists = torch.min(dists, torch.cdist(feats[i:i+1], feats).squeeze(0))
        return feats[sel]

    @torch.no_grad()
    def predict(self, dl):
        scs, lbls, maps = [], [], []
        for x, y in tqdm(dl, leave=False):
            f = self.ext(x); B, C, H, W = f.shape
            dist = torch.cdist(f.permute(0,2,3,1).reshape(-1, C), self.bank).min(1).values
            m = dist.view(B, H, W).cpu().numpy()
            scs.append(m.max((1,2))); lbls.append(y.numpy()); maps.append(m)
        return np.concatenate(scs), np.concatenate(lbls), np.concatenate(maps, 0)

# ── 2. PaDiM ──────────────────────────────────────────────────────────────────
class PaDiM:
    def __init__(self, extractor: BackboneExtractor, config: ModelConfig):
        self.ext = extractor; self.reg = config.padim_reg; self.mu = None; self.inv = None

    def fit(self, dl):
        feats = torch.cat([self.ext(x).cpu() for x, _ in dl], 0) # (N, C, 14, 14)
        N, C, H, W = feats.shape
        X = feats.permute(0, 2, 3, 1) # (N, H, W, C)
        self.mu = X.mean(0)
        Xc = X - self.mu
        cov = torch.einsum('nhwc,nhwd->hwcd', Xc, Xc) / (N - 1)
        I = torch.eye(C).view(1, 1, C, C).expand(H, W, C, C)
        self.inv = torch.linalg.pinv(cov + I * self.reg)

    @torch.no_grad()
    def predict(self, dl):
        mu, inv = self.mu.to(self.ext.device), self.inv.to(self.ext.device)
        scs, lbls, maps = [], [], []
        for x, y in tqdm(dl, leave=False):
            f = self.ext(x).permute(0, 2, 3, 1) # (B, 14, 14, C)
            B, H, W, C = f.shape
            
            # Subtraction: (B, H, W, C)
            diff = f - mu
            
            # Mahalanobis per pixel: (diff^T * inv * diff)
            # We can use einsum for memory efficiency: 'bhwc,hwcd,bhwd->bhw'
            # diff: (B, H, W, C)
            # inv:  (H, W, C, C)
            # res = sum_c (sum_d (diff_bhwc * inv_hwcd * diff_bhwd))
            
            # Step 1: (B, H, W, D) = sum_c (diff_bhwc * inv_hwcd)
            tmp = torch.einsum('bhwc,hwcd->bhwd', diff, inv)
            # Step 2: (B, H, W) = sum_d (tmp_bhwd * diff_bhwd)
            dist_sq = torch.einsum('bhwd,bhwd->bhw', tmp, diff)
            
            m_np = torch.sqrt(dist_sq.clamp(min=1e-6)).cpu().numpy()
            scs.append(m_np.max((1,2))); lbls.append(y.numpy()); maps.append(m_np)
        return np.concatenate(scs), np.concatenate(lbls), np.concatenate(maps, 0)

    def save(self, path):
        torch.save({
            "mu": self.mu.cpu() if self.mu is not None else None,
            "inv": self.inv.cpu() if self.inv is not None else None
        }, path)

# ── 3. SuperSimpleNet ─────────────────────────────────────────────────────────
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.alpha * g, None

class SuperSimpleNet(nn.Module):
    def __init__(self, extractor: BackboneExtractor, config: ModelConfig):
        super().__init__()
        self.ext = extractor; self.cfg = config
        self.head = nn.Sequential(
            nn.Conv2d(extractor.feat_dim, config.ssn_proj*2, 1), nn.BatchNorm2d(config.ssn_proj*2), nn.GELU(),
            nn.Conv2d(config.ssn_proj*2, config.ssn_proj, 1), nn.BatchNorm2d(config.ssn_proj), nn.GELU(),
            nn.Conv2d(config.ssn_proj, 1, 1)
        ).to(extractor.device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=config.ssn_lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, dl):
        self.head.train()
        for _ in range(self.cfg.ssn_epochs):
            for x, _ in dl:
                with torch.no_grad(): fc = self.ext(x)
                B = x.size(0)
                fn = fc + torch.randn_like(fc) * fc.std() * self.cfg.ssn_noise
                fa = torch.cat([fc, fn], 0)
                lbl = torch.cat([torch.zeros(B,1,14,14), torch.ones(B,1,14,14)], 0).to(self.ext.device)
                logit = self.head(_GRL.apply(fa, self.cfg.ssn_alpha))
                loss = self.loss_fn(logit, lbl)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.head.eval()

    @torch.no_grad()
    def predict(self, dl):
        scs, lbls, maps = [], [], []
        for x, y in tqdm(dl, leave=False):
            f = self.ext(x)
            m = torch.sigmoid(self.head(f)).squeeze(1).cpu().numpy()
            scs.append(m.max((1,2))); lbls.append(y.numpy()); maps.append(m)
        return np.concatenate(scs), np.concatenate(lbls), np.concatenate(maps, 0)

# ── 4. CAE ────────────────────────────────────────────────────────────────────
class CAEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config; self.device = config.device
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        ).to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=config.cae_lr)

    def fit(self, dl):
        self.train()
        for _ in range(self.cfg.cae_epochs):
            for x, _ in dl:
                x = x.to(self.device)
                loss = F.mse_loss(self.net(x), x)
                self.opt.zero_grad(); loss.backward(); self.opt.step()

    @torch.no_grad()
    def predict(self, dl):
        self.eval()
        scs, lbls, maps = [], [], []
        for x, y in tqdm(dl, leave=False):
            x = x.to(self.device)
            rec = self.net(x)
            m = F.mse_loss(rec, x, reduction="none").mean(1).cpu().numpy()
            scs.append(m.max((1,2))); lbls.append(y.numpy()); maps.append(m)
        return np.concatenate(scs), np.concatenate(lbls), np.concatenate(maps, 0)

# ── 5. DRAEM ──────────────────────────────────────────────────────────────────
class _DRAEMRecon(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1), nn.BatchNorm2d(co), nn.ReLU(True))
        self.e1, self.e2, self.e3 = blk(3, 32), blk(32, 64), blk(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bot = blk(128, 256)
        self.up3, self.up2, self.up1 = nn.ConvTranspose2d(256, 128, 2, 2), nn.ConvTranspose2d(128, 64, 2, 2), nn.ConvTranspose2d(64, 32, 2, 2)
        self.d3, self.d2, self.d1 = blk(256, 128), blk(128, 64), blk(64, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        h1 = self.e1(x); h2 = self.e2(self.pool(h1)); h3 = self.e3(self.pool(h2))
        b = self.bot(self.pool(h3))
        o3 = self.d3(torch.cat([self.up3(b), h3], 1))
        o2 = self.d2(torch.cat([self.up2(o3), h2], 1))
        o1 = self.d1(torch.cat([self.up1(o2), h1], 1))
        return torch.sigmoid(self.out(o1))

class _DRAEMDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 1, 3, 1, 1)
        )

    def forward(self, x):
        return F.interpolate(self.net(x), size=(x.shape[2], x.shape[3]), mode='bilinear')

class DRAEMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config; self.device = config.device
        self.recon = _DRAEMRecon().to(self.device)
        self.disc = _DRAEMDisc().to(self.device)
        self.opt = torch.optim.Adam(list(self.recon.parameters()) + list(self.disc.parameters()), lr=config.draem_lr)

    def _augment(self, x):
        B, C, H, W = x.shape
        aug, mask = x.clone(), torch.zeros((B, 1, H, W), device=x.device)
        for b in range(B):
            if random.random() > 0.5:
                # Simple geometric noise patch
                h, w = random.randint(20, 50), random.randint(20, 50)
                y, x_p = random.randint(0, H-h), random.randint(0, W-w)
                aug[b, :, y:y+h, x_p:x_p+w] = (aug[b, :, y:y+h, x_p:x_p+w] + torch.randn((C, h, w), device=x.device)*self.cfg.draem_noise).clamp(0,1)
                mask[b, 0, y:y+h, x_p:x_p+w] = 1.0
        return aug, mask

    def fit(self, dl):
        self.train()
        for _ in range(self.cfg.draem_epochs):
            for x, _ in dl:
                x = x.to(self.device)
                # DRAEM expects 0-1 range (denormalized for internal processing)
                aug, mask = self._augment(x)
                rec = self.recon(aug)
                l_rec = F.mse_loss(rec, x)
                l_disc = F.binary_cross_entropy_with_logits(self.disc(torch.cat([aug, rec.detach()], 1)), mask)
                loss = l_rec + l_disc
                self.opt.zero_grad(); loss.backward(); self.opt.step()

    @torch.no_grad()
    def predict(self, dl):
        self.eval()
        scs, lbls, maps = [], [], []
        for x, y in tqdm(dl, leave=False):
            x = x.to(self.device)
            rec = self.recon(x)
            m = torch.sigmoid(self.disc(torch.cat([x, rec], 1))).squeeze(1).cpu().numpy()
            scs.append(m.max((1,2))); lbls.append(y.numpy()); maps.append(m)
        return np.concatenate(scs), np.concatenate(lbls), np.concatenate(maps, 0)
