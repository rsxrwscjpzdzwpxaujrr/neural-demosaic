import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import apply_aces, apply_srgb_gamma


def npy_to_png(path: Path, out_path: Path) -> None:
    img = np.load(path).astype(np.float32)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"{path} is not an (H, W, 3) array (got shape {img.shape})")

    rgb = apply_aces(img)
    rgb = apply_srgb_gamma(rgb)

    rgb8 = np.round(rgb * 255.0).astype(np.uint8)
    Image.fromarray(rgb8, mode="RGB").save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help=".npy files or directories")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/gt_preview"))
    args = parser.parse_args()

    files = []
    for inp in args.inputs:
        if inp.is_dir():
            files += sorted(inp.glob("*.npy"))
        else:
            files.append(inp)
    if not files:
        sys.exit("no .npy files found")

    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        out_path = args.output / f"{path.stem}.png"
        npy_to_png(path, out_path)
        print(f"{path.name} -> {out_path}")


if __name__ == "__main__":
    main()
