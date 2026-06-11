"""Build linear-RGB ground truth from raw files.

Superpixel-demosaic each raw (2x2 for Bayer, 3x3 for X-Trans), then Lanczos-downscale
further in linear space to suppress channel misregistration, aliasing and noise.
Output: float16 .npy files (H, W, 3), WB-applied linear RGB.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rawler_py
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cfa import XTRANS_PATTERN, BAYER_PATTERN

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
        np.asarray(Image.fromarray(rgb[:h, :w, c], mode="F").resize(size, Image.LANCZOS))
        for c in range(3)
    ]
    return np.stack(planes, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="raw files or directories")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/gt"))
    parser.add_argument("--downscale", type=int, default=2,
                        help="extra downscale factor after superpixel (default 2)")
    args = parser.parse_args()

    files = []
    for inp in args.inputs:
        if inp.is_dir():
            files += sorted(p for p in inp.rglob("*") if p.suffix.lower() in RAW_EXTENSIONS)
        else:
            files.append(inp)
    if not files:
        sys.exit("no raw files found")

    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        out_path = args.output / f"{path.stem}.npy"
        if out_path.exists():
            print(f"skip {path.name} (exists)")
            continue
        print(f"{path.name}:")
        cfa, period = decode_raw(path)
        rgb = downscale(superpixel(cfa, period), args.downscale)
        rgb = np.maximum(rgb, 0.0)
        np.save(out_path, rgb.astype(np.float16))
        print(f"  -> {out_path} {rgb.shape[1]}x{rgb.shape[0]}, "
              f"range [{rgb.min():.3f}, {rgb.max():.3f}]")


if __name__ == "__main__":
    main()
