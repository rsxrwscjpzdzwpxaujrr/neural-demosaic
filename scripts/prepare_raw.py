"""Build linear-RGB ground truth from raw files.

Superpixel-demosaic each raw (2x2 for Bayer, 3x3 for X-Trans), then Lanczos-downscale
further in linear space to suppress channel misregistration, aliasing and noise.
Output: float16 .npy files (H, W, 3), WB-applied linear RGB.
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import rawler_py
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cfa import XTRANS_PATTERN, BAYER_PATTERN
from scripts.inference import get_model_width_depth, run_tiled
from src.models import PackedXTransNet

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CKPT_PATH = Path("weights/")
RAW_EXTENSIONS = {".raf", ".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2"}


def decode_raw(path: Path) -> tuple[np.ndarray, int]:
    """Decode to WB'd linear CFA, cropped so the canonical pattern starts at (0, 0).
    Returns (cfa, period)."""
    with rawler_py.RawImage.open(str(path)) as img:
        print(f"  {img.make} {img.model} | {img.width}x{img.height}")

        if img.active_area is not None:
            left, top, w, h = img.active_area
        else:
            left, top, w, h = 0, 0, img.width, img.height
        print(f"  active {w}x{h} (top={top}, left={left})")
        raw_data = img.raw_data()[top:top + h, left:left + w].astype(np.float32)

        black = float(img.blacklevel["levels"][0])
        white = float(img.whitelevel[0])

        wb = np.array(img.wb_coeffs[:3], dtype=np.float32)
        if np.isnan(wb[1]) or wb[1] == 0:
            wb[1] = 1.0
        wb /= wb[1]

        period = 6 if len(img.cfa_pattern) == 36 else 2
        char_to_color = {"R": 0, "G": 1, "B": 2}
        pattern_grid = np.zeros((period, period), dtype=np.int32)
        for y in range(period):
            for x in range(period):
                char = img.cfa_pattern[((y + top) % period) * period + ((x + left) % period)]
                pattern_grid[y, x] = char_to_color[char]

        canonical = XTRANS_PATTERN if period == 6 else BAYER_PATTERN
        shift = (0, 0)
        for dy in range(period):
            for dx in range(period):
                if np.array_equal(np.roll(canonical, (-dy, -dx), axis=(0, 1)), pattern_grid):
                    shift = (dy, dx)
                    break

        cfa = np.clip((raw_data - black) / (white - black), 0.0, 1.2)

        tiled = np.tile(pattern_grid, ((h + period - 1) // period, (w + period - 1) // period))[:h, :w]
        cfa = cfa * wb[tiled]

        # Crop instead of pad: drop leading rows/cols so the pattern becomes canonical
        r0 = (period - shift[0]) % period
        c0 = (period - shift[1]) % period
        return cfa[r0:, c0:], period


def superpixel(cfa: np.ndarray, period: int) -> np.ndarray:
    """Collapse CFA blocks into RGB pixels by per-color averaging.
    Block is 3x3 for X-Trans (every 3x3 has 2R/5G/2B) and 2x2 for Bayer."""
    canonical = XTRANS_PATTERN if period == 6 else BAYER_PATTERN
    block = 3 if period == 6 else 2
    h = cfa.shape[0] - cfa.shape[0] % period
    w = cfa.shape[1] - cfa.shape[1] % period
    tiled = np.tile(canonical, (h // period, w // period))

    rgb = np.empty((h // block, w // block, 3), dtype=np.float32)
    for c in range(3):
        m = (tiled == c).astype(np.float32)
        num = (cfa[:h, :w] * m).reshape(h // block, block, w // block, block).sum(axis=(1, 3))
        den = m.reshape(h // block, block, w // block, block).sum(axis=(1, 3))
        rgb[..., c] = num / den
    return rgb


def downscale(rgb: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return rgb
    h = rgb.shape[0] - rgb.shape[0] % factor
    w = rgb.shape[1] - rgb.shape[1] % factor
    size = (w // factor, h // factor)
    planes = [
        np.asarray(Image.fromarray(rgb[:h, :w, c], mode="F").resize(size, Image.Resampling.BOX))
        for c in range(3)
    ]
    return np.stack(planes, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="raw files or directories")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/gt"))
    parser.add_argument("--downscale", type=lambda s: set(int(item) for item in s.split(',')), default=[2,3],
                        help="comma separated extra downscale factor after superpixel list (default 2,3)")
    parser.add_argument("--infer-with", type=str, default=None,
                        help="Name of the weights which are going to be used for inference, superpixel will be used if not set")
    args = parser.parse_args()

    inference = args.infer_with is not None

    files = []
    for inp in args.inputs:
        if inp.is_dir():
            files += sorted(p for p in inp.rglob("*") if p.suffix.lower() in RAW_EXTENSIONS)
        else:
            files.append(inp)
    if not files:
        sys.exit("no raw files found")

    if inference:
        model_file = CKPT_PATH / (args.infer_with + '.pt')
        model_dict = torch.load(model_file, map_location=DEVICE)
        width, depth = get_model_width_depth(model_dict)
        model = PackedXTransNet(XTRANS_PATTERN, width=width, depth=depth).to(DEVICE)
        model.load_state_dict(model_dict)
        model.half()
        model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        print(f"{path.name}:")

        try:
            cfa, period = decode_raw(path)
        except Exception as e:
            print(f"  error occured during decoding {e}")
            continue

        pre_rgb = None

        for factor in args.downscale:
            out_path = args.output / f"{path.stem}_x{factor}.npy"
            if out_path.exists():
                print(f"  skip {out_path} (exists)")
                continue

            if inference and period == 6:
                factor = factor * 2

            if pre_rgb is None:
                if inference and period == 6:
                    print(f"  Running inference on {path.stem}...")
                    pre_rgb = np.moveaxis(run_tiled(model, cfa, 0, 0), 0, -1)
                else:
                    pre_rgb = superpixel(cfa, period)

            rgb = downscale(pre_rgb, factor)
            rgb = np.maximum(rgb, 0.0)
            np.save(out_path, rgb.astype(np.float16))
            print(f"  -> {out_path} {rgb.shape[1]}x{rgb.shape[0]}, "
                  f"range [{rgb.min():.3f}, {rgb.max():.3f}]")


if __name__ == "__main__":
    main()
