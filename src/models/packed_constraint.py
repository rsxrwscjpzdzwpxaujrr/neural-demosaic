import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chroma_base import ChromaBaseline


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x))) * 0.25


class PackedConstraintXTransNet(nn.Module):
    """Pack-3 residual demosaicer. Input mosaic must be aligned to the canonical
    X-Trans pattern origin, with H and W divisible by 6. Estimates only the missing channels."""

    def __init__(self, pattern: np.ndarray, width: int = 64, depth: int = 16):
        super().__init__()
        self.baseline = ChromaBaseline(pattern)
        self.stem = nn.Conv2d(10, width, 3, padding=1)  # 9 packed + 1 phase
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 18, 3, padding=1)
        self.register_buffer('pattern', torch.from_numpy(pattern))

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, mosaic):  # (B, 1, H, W) -> (B, 3, H, W)
        B, _, H, W = mosaic.shape
        assert H % 6 == 0 and W % 6 == 0, "H and W must be divisible by 6"

        x = F.pixel_unshuffle(mosaic, 3)  # (B, 9, H/3, W/3)
        yy = torch.arange(H // 3, device=x.device)
        xx = torch.arange(W // 3, device=x.device)
        phase = ((yy[:, None] + xx[None, :]) % 2).to(x.dtype)
        x = torch.cat([x, phase.expand(B, 1, -1, -1)], dim=1)

        x = self.head(self.body(self.stem(x)))
        residual = F.pixel_shuffle(x, 3)  # (B, 2, H, W)

        c = torch.arange(3, device=x.device).view(1, 3, 1, 1)  # (1, 3, 1, 1)
        pattern = self.pattern.repeat(H // 6, W // 6)[None, None]  # (1, 1, H, W)
        src = (c - (c > pattern).long()).clamp(max=1)  # (1, 1, H, W)

        residual_3ch = residual.gather(1, src.expand(B, -1, -1, -1))  # (B, 3, H, W)

        out = torch.where(c == pattern, mosaic, residual_3ch + self.baseline(mosaic))  # (B, 3, H, W)

        return out
