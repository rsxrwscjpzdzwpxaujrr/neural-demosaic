import os
import sys
import time
from pathlib import Path

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import lin_rgb_to_luv
from src.cfa import XTRANS_PATTERN, get_mosaic_arr
from src.models import PackedXTransNet
from src.demosaic_opencl import MarkesteijnOpenCLDemosaicer

print = tqdm.write

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
GT_DIR = Path("data/gt")
VAL_DIR = Path("data/val")
CKPT_PATH = Path("weights/")
LOG_PATH = Path("logs/")
PATCH = 0
BATCH = 0
ITERS = 0
LR = 0.0
LOG_EVERY = 50
VAL_EVERY = 1000
PLOT_EVERY = 500
SAMPLES_TO_PLOT = 5

EXPOSURE_RANGE = (-2.5, 2.5)

YY, XX, CFA_INDICES = None, None, None

class GTPatches(Dataset):
    def __init__(self, files):
        self.images = [np.load(f, mmap_mode="r") for f in files]

        areas = torch.tensor([img.shape[-2] * img.shape[-1] for img in self.images], dtype=torch.float32)

        self.weights = areas / areas.sum()

    def __len__(self):
        return ITERS * BATCH

    def __getitem__(self, _):
        img = self.images[torch.multinomial(self.weights, 1).item()]
        h, w, _ = img.shape
        y = torch.randint(h - PATCH + 1, ()).item()
        x = torch.randint(w - PATCH + 1, ()).item()
        patch = img[y:y + PATCH, x:x + PATCH]

        return torch.tensor(patch)


def gamma(x):
    return x.clamp_min(0).add(1e-3).pow(1 / 2.4)


def psnr(a, b):
    return -10 * torch.log10(F.mse_loss(a, b)).item()


# 0 db is just noticable difference, more means less noticable, inspired by CIE76
def percept_diff(a, b):
    a_luv = lin_rgb_to_luv(a)
    b_luv = lin_rgb_to_luv(b)

    jnd = 2.3

    return 20 * np.log10(jnd) + psnr(a_luv, b_luv)


def make_eval_pairs(images, size=1080):
    assert size % 6 == 0, "size must be divisible by 6"
    pairs = []
    for img in images:
        h, w, _ = img.shape
        h_new, w_new = min(size, h // 6 * 6), min(size, w // 6 * 6)
        y, x = (h - h_new) // 2, (w - w_new) // 2
        crop = np.ascontiguousarray(img[y:y + h_new, x:x + w_new], dtype=np.float32)
        gt = torch.from_numpy(crop).permute(2, 0, 1)[None]
        yy, xx, cfa_indices = get_mosaic_arr((h_new, w_new), XTRANS_PATTERN)
        mosaic = torch.from_numpy(crop[yy, xx, cfa_indices])[None, None]
        pairs.append((mosaic, gt))
    return pairs


@torch.no_grad()
def evaluate(fn, pairs):
    vals = []

    for mo, gt in pairs:
        mo_d = mo.to(DEVICE)
        gt_d = gt.to(DEVICE)

        res = fn(mo_d)

        white_psnr = psnr(res, gt_d)
        diff = percept_diff(res, gt_d)

        vals.append([white_psnr, diff])

    return tuple(np.mean(vals, axis=0))


def markesteijn_psnr(kernels, pairs):
    pattern = XTRANS_PATTERN.astype(np.uint8)
    vals = []
    for mo, gt in pairs:
        mosaic = np.pad(mo[0, 0].numpy(), 18, mode="symmetric")
        out = kernels.demosaic(raw_image=mosaic, xtrans_pattern=pattern, passes=3, crop=False)
        out = torch.from_numpy(np.ascontiguousarray(out[18:-18, 18:-18])).permute(2, 0, 1)[None]

        white_psnr = psnr(out, gt)
        diff = percept_diff(out, gt)

        vals.append([white_psnr, diff])
    return tuple(np.mean(vals, axis=0))


def pretty_psnrs(psnrs: tuple) -> str:
    return f"{psnrs[0]:.2f} ({', '.join(f'{x:.1f}' for x in psnrs[1:])})"

@torch.no_grad()
def preprocess(patch: torch.Tensor):
    patch = patch.permute(0, 3, 1, 2)

    if torch.rand(()) < 0.5:
        patch = patch.flip(dims=[2])
    if torch.rand(()) < 0.5:
        patch = patch.flip(dims=[3])
    if torch.rand(()) < 0.5:
        patch = patch.permute(0, 1, 3, 2)

    exposure = (torch.rand((patch.shape[0], 1, 1, 1), device=DEVICE) *
                (EXPOSURE_RANGE[1] - EXPOSURE_RANGE[0]) + EXPOSURE_RANGE[0])
    patch = patch * (2 ** exposure)

    patch = patch.to(dtype=torch.float32)

    mosaic = patch[:, CFA_INDICES, YY, XX][:, None]

    return mosaic, patch


def init_plt():
    dpi = plt.rcParams['figure.dpi']

    fig, (ax_pic, ax_loss) = plt.subplots(1, 2, figsize=((3 * 4 * PATCH) / dpi, (SAMPLES_TO_PLOT * PATCH) / dpi),
                                          width_ratios=(0.25, 0.75))
    ax_pic.set_position([0, 0, 0.25, 1])
    ax_pic.axis('off')

    ax_loss.grid(True, axis='x', color='lightgray', linewidth=1)
    ax_loss.grid(True, axis='y', color='lightblue', linewidth=1)
    ax_loss.autoscale(enable=True, axis='y')

    ax_loss.set_xlim(left=0, right=ITERS)

    line_loss, = ax_loss.plot([0], color='tab:blue')
    ax_loss.set_ylabel("Loss", color='blue')
    line_loss.set_xdata(range(0, ITERS))

    ax_loss.set_xlabel("Iteration")
    ax_loss.tick_params(axis='y', labelcolor='tab:blue')

    ax_psnr = ax_loss.twinx()
    ax_psnr.grid(True, axis='y', color='mistyrose', linewidth=1)
    ax_psnr.autoscale(enable=True, axis='y')

    line_psnr, = ax_psnr.plot([0], color='tab:red', marker='o')

    ax_psnr.set_ylabel("PSNR", color='red')
    ax_psnr.yaxis.set_major_formatter(ticker.EngFormatter(unit='dB'))
    ax_psnr.tick_params(axis='y', labelcolor='tab:red')

    plt.ion()
    plt.show()

    return fig, ax_pic, ax_loss, line_loss, ax_psnr, line_psnr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="packed", help="The name for weight file without format. "
                                                                   "Default 'packed'.")
    parser.add_argument("--width", type=int, default=32, help="Model width. Default 32.")
    parser.add_argument("--depth", type=int, default=8, help="Model depth. Default 8.")
    parser.add_argument("--iters", type=int, default=50000, help="Training iteration count. Default 50000.")
    parser.add_argument("--patch", type=int, default=96, help="Training patch size. Default 96.")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size. Default 32.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate. Default 1e-3.")
    parser.add_argument("--val-since", type=float, default=0.5, help="The amount of progress before we to start doing "
                                                                     "val. Default 0.5.")
    parser.add_argument("--faststart", action="store_true", help="Skip Markesteijn demosaic at start and limit train "
                                                                 "and val files by 10.")
    parser.add_argument("--plt", action="store_true", help="Show result pics during training.")
    args = parser.parse_args()

    global ITERS, PATCH, BATCH, LR, YY, XX, CFA_INDICES
    ITERS, PATCH, BATCH, LR = args.iters, args.patch, args.batch, args.lr

    yy, xx, cfa_indices = get_mosaic_arr((PATCH, PATCH), XTRANS_PATTERN)
    YY, XX, CFA_INDICES = (
        torch.tensor(         yy, dtype=torch.int32).to(DEVICE),
        torch.tensor(         xx, dtype=torch.int32).to(DEVICE),
        torch.tensor(cfa_indices, dtype=torch.int32).to(DEVICE),
    )

    train_files = sorted(GT_DIR.glob("*.npy"))
    val_files = sorted(VAL_DIR.glob("*.npy"))

    if args.faststart:
        train_files = train_files[:10]
        val_files = val_files[:10]

    print(f"{len(train_files)} train images")
    print(f"{len(val_files)} val images")

    loader = DataLoader(GTPatches(train_files), batch_size=BATCH, num_workers=4,
                        persistent_workers=True)
    model = PackedXTransNet(XTRANS_PATTERN, width=args.width, depth=args.depth).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)

    train_eval = make_eval_pairs([np.load(f, mmap_mode="r") for f in train_files])
    val_eval = make_eval_pairs([np.load(f, mmap_mode="r") for f in val_files])
    base_train = evaluate(model.baseline, train_eval)
    base_val = evaluate(model.baseline, val_eval)
    print(f"bilinear PSNR:    train {pretty_psnrs(base_train)}, val {pretty_psnrs(base_val)}")

    if not args.faststart:
        os.environ.setdefault("PYOPENCL_CTX", "0")
        kernels = MarkesteijnOpenCLDemosaicer(kernel_path="src/kernels/demosaic_markesteijn.cl")
        mark_train = markesteijn_psnr(kernels, train_eval)
        mark_val = markesteijn_psnr(kernels, val_eval)
        print(f"markesteijn PSNR: train {pretty_psnrs(mark_train)}, val {pretty_psnrs(mark_val)}")
    else:
        mark_train = mark_val = (float('-inf'), float('-inf'))

    CKPT_PATH.mkdir(exist_ok=True)
    ckpt_file = CKPT_PATH / f"{args.name}.pt"

    best_psnr = tuple((0.0, float('-inf')))
    running = 0.0
    t0 = time.time()

    if args.plt:
        fig, ax_pic, ax_loss, line_loss, ax_psnr, line_psnr = init_plt()

        step_loss_x = np.zeros(ITERS)
        step_loss_y = np.zeros(ITERS)
        step_loss_count = 0

        step_psnr_x = np.zeros(ITERS)
        step_psnr_y = np.zeros(ITERS)
        step_psnr_count = 0

    pbar = tqdm(enumerate(loader, 1), total=ITERS, miniters=10, mininterval=0.25, maxinterval=99999.9, smoothing=0.2)
    for step, gt in pbar:
        mosaic, gt = preprocess(gt.to(DEVICE))
        out = model(mosaic)

        loss = F.l1_loss(out, gt) + F.l1_loss(gamma(out), gamma(gt))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if step % LOG_EVERY == 0:
            print(f"step {step}/{ITERS}  loss {running / LOG_EVERY:.5f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {LOG_EVERY * BATCH / (time.time() - t0):.0f} img/s")
            running = 0.0
            t0 = time.time()

            if args.plt:
                step_loss_x[step_loss_count] = step
                step_loss_y[step_loss_count] = loss.item()

                step_loss_count += 1

                ax_loss.set_ylim(bottom=0, top=None)

                line_loss.set_xdata(step_loss_x[:step_loss_count])
                line_loss.set_ydata(step_loss_y[:step_loss_count])

        if (step / ITERS) >= args.val_since and step % VAL_EVERY == 0:
            model.eval()
            train_psnr = evaluate(model, train_eval)
            val_psnr = evaluate(model, val_eval)
            model.train()
            marker = ""
            if val_psnr[0] > best_psnr[0]:
                best_psnr = val_psnr
                torch.save(model.state_dict(), ckpt_file)
                pbar.set_postfix_str(f"Best PSNR: {best_psnr[0]:.2f}dB")
                marker = " *"
            print(
                f"PSNR train {pretty_psnrs(train_psnr)} (mark {pretty_psnrs(mark_train)}) | "
                f"val {pretty_psnrs(val_psnr)} (mark {pretty_psnrs(mark_val)}){marker}")

            if args.plt:
                step_psnr_x[step_psnr_count] = step
                step_psnr_y[step_psnr_count] = val_psnr[0]

                step_psnr_count += 1

                line_psnr.set_xdata(step_psnr_x[:step_psnr_count])
                line_psnr.set_ydata(step_psnr_y[:step_psnr_count])

                ax_psnr.relim()
                ax_psnr.autoscale_view()

        if args.plt:
            if step % PLOT_EVERY == 0:
                with torch.no_grad():
                    out_a = gt[0:SAMPLES_TO_PLOT, ...]
                    out_b = out[0:SAMPLES_TO_PLOT, ...]

                    pics = [gamma(out_a), gamma(out_b), (((out_a - out_b) * 10.0) + 0.5)]
                    pics = [pic.permute(0, 2, 3, 1).clamp(0, 1).reshape(-1, PATCH, 3) for pic in pics]

                    pic = torch.cat(pics, dim=1)

                    ax_pic.imshow(pic.cpu().numpy())

            fig.canvas.flush_events()

    if args.plt:
        LOG_PATH.mkdir(exist_ok=True)
        plt.savefig(LOG_PATH / f"{args.name}.png")

if __name__ == "__main__":
    main()
