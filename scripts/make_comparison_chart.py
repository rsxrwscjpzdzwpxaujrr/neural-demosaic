"""
Crop several square patches from result.png and result_markesteijn.png,
then assemble them into a side-by-side comparison chart.

Usage:
    python scripts/make_comparison_chart.py [--output output/comparison.png]
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GAP = 16         # pixels between cells
LABEL_H = 32     # height of the label bar above each column

# Each entry: (cx_frac, cy_frac, crop_size_px)
CROPS = [
    (0.5, 0.37, 256),
    (0.27, 0.36, 256),
    (0.42, 0.38, 256),
    (0.33, 0.92, 256),
]

BG_COLOR = (20, 20, 20)
LABEL_BG = (40, 40, 40)
TEXT_COLOR = (220, 220, 220)
BORDER_COLOR = (60, 60, 60)


def crop_at(img: Image.Image, cx_frac: float, cy_frac: float, size: int) -> Image.Image:
    w, h = img.size
    cx = int(cx_frac * w)
    cy = int(cy_frac * h)
    half = size // 2
    left = max(0, min(cx - half, w - size))
    top = max(0, min(cy - half, h - size))
    return img.crop((left, top, left + size, top + size))


def make_chart(result_path: Path, markesteijn_path: Path, out_path: Path):
    neural = Image.open(result_path).convert("RGB")
    markesteijn = Image.open(markesteijn_path).convert("RGB")

    sizes = [size for _, _, size in CROPS]
    row_h = max(sizes)
    rows = 2  # neural | markesteijn

    label_w = 56
    col_xs = []
    x = label_w + GAP
    for size in sizes:
        col_xs.append(x)
        x += size + GAP

    chart_w = x
    chart_h = rows * row_h + (rows + 1) * GAP

    chart = Image.new("RGB", (chart_w, chart_h), BG_COLOR)
    draw = ImageDraw.Draw(chart)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except Exception:
        font = ImageFont.load_default()

    labels = ["Neural", "Markesteijn"]
    sources = [neural, markesteijn]

    for row_i, (label, src) in enumerate(zip(labels, sources)):
        y = GAP + row_i * (row_h + GAP)
        lbl_rect = [GAP // 2, y, GAP // 2 + label_w, y + row_h]
        draw.rectangle(lbl_rect, fill=LABEL_BG)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        txt_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(txt_img).text((0, 0), label, fill=TEXT_COLOR, font=font)
        txt_img = txt_img.rotate(90, expand=True)
        tx = GAP // 2 + (label_w - txt_img.width) // 2
        ty = y + (row_h - txt_img.height) // 2
        chart.paste(txt_img, (tx, ty), txt_img)

        for col_i, (cx, cy, size) in enumerate(CROPS):
            px = col_xs[col_i]
            py = y + (row_h - size) // 2
            patch = crop_at(src, cx, cy, size)
            chart.paste(patch, (px, py))
            draw.rectangle([px, py, px + size - 1, py + size - 1], outline=BORDER_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(out_path)
    print(f"Saved comparison chart → {out_path}  ({chart_w}×{chart_h})")


def main():
    parser = argparse.ArgumentParser(description="Build a crop comparison chart.")
    parser.add_argument("--result", default="output/result.png")
    parser.add_argument("--markesteijn", default="output/result_markesteijn.png")
    parser.add_argument("--output", default="output/comparison.png")
    args = parser.parse_args()

    make_chart(
        Path(args.result),
        Path(args.markesteijn),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
