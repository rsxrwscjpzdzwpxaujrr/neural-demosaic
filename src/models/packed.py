import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tent(k: int) -> torch.Tensor:
    t = 1 - (torch.arange(k, dtype=torch.float32) - (k - 1) / 2).abs() / ((k + 1) / 2)
    return torch.outer(t, t)


class ChromaBaseline(nn.Module):
    """Parameter-free demosaic: bilinear G, then bilinear chroma differences R-G, B-G."""

    def __init__(self, pattern: np.ndarray, k_g: int = 5, k_rb: int = 7):
        super().__init__()
        masks = torch.from_numpy(np.stack([(pattern == c) for c in range(3)]).astype(np.float32))
        self.register_buffer("masks", masks)  # (3, 6, 6)
        self.register_buffer("kern_g", _tent(k_g)[None, None])
        self.register_buffer("kern_rb", _tent(k_rb)[None, None])

    @staticmethod
    def _nconv(x, mask, kern):
        pad = kern.shape[-1] // 2
        num = F.conv2d(x * mask, kern, padding=pad)
        den = F.conv2d(mask.expand_as(x), kern, padding=pad)
        return num / den.clamp_min(1e-8)

    def forward(self, mosaic):  # (B, 1, H, W) -> (B, 3, H, W)
        _, _, H, W = mosaic.shape
        m = self.masks.repeat(1, H // 6 + 1, W // 6 + 1)[:, :H, :W].unsqueeze(1)  # (3, 1, H, W)
        g = self._nconv(mosaic, m[1], self.kern_g)
        r = g + self._nconv(mosaic - g, m[0], self.kern_rb)
        b = g + self._nconv(mosaic - g, m[2], self.kern_rb)
        return torch.cat([r, g, b], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class PackedXTransNet(nn.Module):
    """Pack-3 residual demosaicer. Input mosaic must be aligned to the canonical
    X-Trans pattern origin, with H and W divisible by 6."""

    def __init__(self, pattern: np.ndarray, width: int = 64, depth: int = 16):
        super().__init__()
        self.baseline = ChromaBaseline(pattern)
        self.stem = nn.Conv2d(10, width, 3, padding=1)  # 9 packed + 1 phase
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 27, 3, padding=1)

    def forward(self, mosaic):  # (B, 1, H, W) -> (B, 3, H, W)
        B, _, H, W = mosaic.shape
        assert H % 6 == 0 and W % 6 == 0, "H and W must be divisible by 6"

        x = F.pixel_unshuffle(mosaic, 3)  # (B, 9, H/3, W/3)
        yy = torch.arange(H // 3, device=x.device)
        xx = torch.arange(W // 3, device=x.device)
        phase = ((yy[:, None] + xx[None, :]) % 2).to(x.dtype)
        x = torch.cat([x, phase.expand(B, 1, -1, -1)], dim=1)

        x = self.head(self.body(self.stem(x)))
        residual = F.pixel_shuffle(x, 3)  # (B, 3, H, W)
        return self.baseline(mosaic) + residual
