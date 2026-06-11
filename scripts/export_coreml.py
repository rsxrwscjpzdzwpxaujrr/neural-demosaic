import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cfa import XTRANS_PATTERN
from src.models import PackedXTransNet

CKPT = "weights/packed_5050_3202.pt"
OUT = "weights/packed.mlpackage"
SIZE = 288
MARGIN = 12  # crop-discard ring for tiled inference; >=12 keeps tiled vs whole-image error below fp16 noise
WARMUP = 20
RUNS = 200


class ExportModel(torch.nn.Module):
    """Fixed-size wrapper: all shape-dependent tensors precomputed as buffers."""

    def __init__(self, net: PackedXTransNet, size: int, batch: int = 1):
        super().__init__()
        self.net = net
        bl = net.baseline
        m = bl.masks.repeat(1, size // 6, size // 6)[:, None]  # (3, 1, S, S)
        self.register_buffer("m", m)
        self.register_buffer("kg", bl.kern_g)
        self.register_buffer("krb", bl.kern_rb)
        pg, prb = bl.kern_g.shape[-1] // 2, bl.kern_rb.shape[-1] // 2
        self.pg, self.prb = pg, prb
        self.register_buffer("den_g", torch.nn.functional.conv2d(m[1:2], bl.kern_g, padding=pg).clamp_min(1e-8))
        self.register_buffer("den_r", torch.nn.functional.conv2d(m[0:1], bl.kern_rb, padding=prb).clamp_min(1e-8))
        self.register_buffer("den_b", torch.nn.functional.conv2d(m[2:3], bl.kern_rb, padding=prb).clamp_min(1e-8))

        s3 = size // 3
        yy, xx = torch.meshgrid(torch.arange(s3), torch.arange(s3), indexing="ij")
        phase = ((yy + xx) % 2).float()[None, None]
        self.register_buffer("phase", phase.expand(batch, -1, -1, -1).contiguous())

    def forward(self, mosaic):
        F = torch.nn.functional
        g = F.conv2d(mosaic * self.m[1], self.kg, padding=self.pg) / self.den_g
        r = g + F.conv2d((mosaic - g) * self.m[0], self.krb, padding=self.prb) / self.den_r
        b = g + F.conv2d((mosaic - g) * self.m[2], self.krb, padding=self.prb) / self.den_b
        baseline = torch.cat([r, g, b], dim=1)

        x = F.pixel_unshuffle(mosaic, 3)
        x = torch.cat([x, self.phase], dim=1)
        x = self.net.head(self.net.body(self.net.stem(x)))
        return baseline + F.pixel_shuffle(x, 3)


def export():
    import coremltools as ct

    model = PackedXTransNet(XTRANS_PATTERN, width=32, depth=2)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    wrapped = ExportModel(model, SIZE).eval()
    example = torch.rand(1, 1, SIZE, SIZE)
    with torch.no_grad():
        diff = (wrapped(example) - model(example)).abs().max().item()
    assert diff < 1e-5, f"export wrapper mismatch: {diff}"

    traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mosaic", shape=(1, 1, SIZE, SIZE), dtype=np.float16)],
        outputs=[ct.TensorType(name="rgb", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS26,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(OUT)
    print(f"saved {OUT}")
    return mlmodel


def bench():
    import coremltools as ct

    m = ct.models.MLModel(OUT, compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = {"mosaic": np.random.rand(1, 1, SIZE, SIZE).astype(np.float16)}

    for _ in range(WARMUP):
        m.predict(x)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        m.predict(x)
        times.append(time.perf_counter() - t0)

    ms = float(np.median(times)) * 1e3
    raw = SIZE * SIZE / ms / 1e3
    eff = (SIZE - 2 * MARGIN) ** 2 / ms / 1e3
    print(f"{SIZE}x{SIZE} fp16 on ANE: median {ms:.2f} ms  "
          f"({1e3 / ms:.0f} patches/s, raw {raw:.1f} MP/s, eff {eff:.1f} MP/s @ margin {MARGIN})")


if __name__ == "__main__":
    export()
    bench()
