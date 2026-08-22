import sys
from pathlib import Path

import numpy as np
import rawler_py
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.demosaic_opencl import MarkesteijnOpenCLDemosaicer
import os
os.environ['PYOPENCL_CTX'] = '0'

RAF_PATH = Path("samples/")
OUTPUT_PATH = Path("output/")
KERNEL_PATH = "src/kernels/demosaic_markesteijn.cl"


def apply_color_matrix(rgb: np.ndarray, xyz_to_cam: np.ndarray) -> np.ndarray:
    XYZ_TO_SRGB = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    srgb_to_xyz = np.linalg.inv(XYZ_TO_SRGB)
    srgb_to_cam = xyz_to_cam @ srgb_to_xyz
    row_sums = srgb_to_cam.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) > 1e-8, row_sums, 1.0)
    srgb_to_cam /= row_sums
    cam_to_srgb = np.linalg.inv(srgb_to_cam)
    h, w = rgb.shape[1], rgb.shape[2]
    return np.maximum(cam_to_srgb @ rgb.reshape(3, -1), 0).reshape(3, h, w)


def apply_aces(rgb: np.ndarray, exposure: float = 2.0) -> np.ndarray:
    x = rgb * exposure
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def apply_srgb_gamma(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)


def apply_orientation(rgb: np.ndarray, orientation: str) -> np.ndarray:
    mapping = {
        "Normal": (False, False, False),
        "ReverseYaw": (False, True, False),
        "Rotate180": (False, True, True),
        "ReverseRoll": (False, False, True),
        "ReversePitch": (True, True, False),
        "Rotate90": (True, False, False),
        "MirrorHorizontalAndRotate90": (True, False, True),
        "Rotate270": (True, True, True),
    }
    transpose, hflip, vflip = mapping.get(orientation, (False, False, False))
    if transpose:
        rgb = np.transpose(rgb, (0, 2, 1))
        if hflip:
            rgb = np.flip(rgb, axis=1)
        if vflip:
            rgb = np.flip(rgb, axis=2)
    else:
        if hflip:
            rgb = np.flip(rgb, axis=2)
        if vflip:
            rgb = np.flip(rgb, axis=1)
    return rgb


if __name__ == "__main__":
    demosaicer = MarkesteijnOpenCLDemosaicer(kernel_path=KERNEL_PATH)
    raf_files = sorted(RAF_PATH.glob("*"))

    for raf_file in raf_files:
        file = Path(OUTPUT_PATH / f"{raf_file.stem}_markesteijn.png")

        print(f"{raf_file}...")

        if file.exists():
            print(f"  file exists, skipping")
            continue

        with rawler_py.RawImage.open(str(raf_file)) as img:
            print(f"  {img.make} {img.model} | {img.width}x{img.height}")

            if img.active_area is not None:
                left, top, w, h = img.active_area
            else:
                left, top, w, h = 0, 0, img.width, img.height

            print(f"  active {w}x{h} (top={top}, left={left})")
            raw_data = img.raw_data()[top:top+h, left:left+w].astype(np.float32)

            black = float(img.blacklevel["levels"][0])
            white = float(img.whitelevel[0])

            wb = np.array(img.wb_coeffs[:3], dtype=np.float32)
            if np.isnan(wb[1]) or wb[1] == 0:
                wb[1] = 1.0
            wb /= wb[1]
            print(f"  WB (G=1): R={wb[0]:.3f} B={wb[2]:.3f}")

            period = 6 if len(img.cfa_pattern) == 36 else 2
            char_to_color = {"R": 0, "G": 1, "B": 2}
            pattern_grid = np.zeros((period, period), dtype=np.uint8)
            for y in range(period):
                for x in range(period):
                    char = img.cfa_pattern[((y + top) % period) * period + ((x + left) % period)]
                    pattern_grid[y, x] = char_to_color[char]

            cfa = np.clip((raw_data - black) / (white - black), 0.0, 1.2)

            tiled_colors = np.tile(pattern_grid, ((h + period - 1) // period, (w + period - 1) // period))[:h, :w]
            wb_mask = wb[tiled_colors]
            cfa = cfa * wb_mask

            print(f"  Running Markesteijn demosaic on {w}x{h}...")
            rgb_hwc = demosaicer.demosaic(raw_image=cfa, xtrans_pattern=pattern_grid, passes=1)
            rgb = np.transpose(rgb_hwc, (2, 0, 1)).astype(np.float32)

            xyz_to_cam = np.array(img.xyz_to_cam[:3], dtype=np.float32)
            if not np.any(xyz_to_cam):
                if "D65" in img.color_matrix:
                    xyz_to_cam = np.array(img.color_matrix["D65"], dtype=np.float32).reshape(3, 3)
                elif img.color_matrix:
                    first_key = list(img.color_matrix.keys())[0]
                    xyz_to_cam = np.array(img.color_matrix[first_key], dtype=np.float32).reshape(3, 3)
                else:
                    xyz_to_cam = np.eye(3, dtype=np.float32)

            rgb = apply_color_matrix(rgb, xyz_to_cam)
            rgb = apply_aces(rgb)
            rgb = apply_srgb_gamma(rgb)
            rgb = apply_orientation(rgb, img.orientation)

        out = np.clip(np.round(np.transpose(rgb, (1, 2, 0)) * 255.0), 0, 255).astype(np.uint8)
        file.parent.mkdir(exist_ok=True)
        Image.fromarray(out, 'RGB').save(file)
        print(f"  Saved: {file}")
