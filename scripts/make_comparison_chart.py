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
    (0.5, 0.37, 320),
    (0.27, 0.36, 320),
    (0.42, 0.38, 320),
    (0.62, 0.76, 320)
]
CROP_SCALE = 2
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


def make_chart(pics: [(str, Path)], out_path: Path):
    sizes = [size for _, _, size in CROPS]
    row_h = max(sizes)
    rows = len(pics)

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

    font = ImageFont.load_default(26)

    for row_i, (label, pic_path) in enumerate(pics):
        src = Image.open(pic_path).convert("RGB")
        y = GAP + row_i * (row_h + GAP)
        lbl_rect = [GAP // 2, y, GAP // 2 + label_w, y + row_h]
        draw.rectangle(lbl_rect, fill=LABEL_BG)
        rect_w = lbl_rect[2] - lbl_rect[0]
        rect_h = lbl_rect[3] - lbl_rect[1]
        txt_img = Image.new("RGBA", (rect_h, rect_w), (0, 0, 0, 0))
        ImageDraw.Draw(txt_img).text(
            (rect_h / 2, rect_w / 2), label, fill=TEXT_COLOR, font=font, anchor="mm"
        )
        txt_img = txt_img.rotate(90, expand=True)
        chart.paste(txt_img, (lbl_rect[0], lbl_rect[1]), txt_img)

        for col_i, (cx, cy, size) in enumerate(CROPS):
            px = col_xs[col_i]
            py = y + (row_h - size) // 2
            patch = crop_at(src, cx, cy, size // CROP_SCALE)
            patch = patch.resize((size, size), resample=Image.Resampling.NEAREST)
            chart.paste(patch, (px, py))
            draw.rectangle([px, py, px + size - 1, py + size - 1], outline=BORDER_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(out_path)
    print(f"Saved comparison chart → {out_path}  ({chart_w}×{chart_h})")


def main():
    parser = argparse.ArgumentParser(description="Build a crop comparison chart.")
    parser.add_argument("--pics", type=Path, nargs="+")
    parser.add_argument("--labels", type=str, nargs="+")
    parser.add_argument("--output", type=str, default="assets/comparison.png")
    args = parser.parse_args()

    make_chart(list(zip(args.labels, args.pics)), Path(args.output))


if __name__ == "__main__":
    main()
