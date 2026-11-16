import os
import sys
import time
from pathlib import Path

import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import apply_approximate_wb
from src.cfa import XTRANS_PATTERN, get_mosaic_arr
from src.models import PackedConstraintXTransNet
from src.demosaic_opencl import MarkesteijnOpenCLDemosaicer
from src.plot import Plot

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
GT_DIR = Path("data/gt")
VAL_DIR = Path("data/val")
CKPT_PATH = Path("weights/")
LOG_PATH = Path("logs/")
PATCH = 0
BATCH = 0
ITERS = 0
LR = 0.0
WARMUP_ITERS = 250
LOG_EVERY = 250
VAL_EVERY = 1000
EZ_DISCARD_THRESHOLD = 0.0

FILES_EVERY = 2500
FILE_CACHE = 8 * 1024 * 1024 * 1024

EXPOSURE_RANGE = (-2.0, 2.0)
ISO_GAIN_RANGE = (-3.0, 3.0)
NOISE = False

YY, XX, CFA_INDICES = None, None, None

LOG_TXT = None

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)


def our_write(cls='', *args, **kwargs):
    tqdm.write(cls, *args, **kwargs)

    if LOG_TXT is not None:
        __builtins__.print(cls, file=LOG_TXT, flush=True, *args, **kwargs)


class GTPatches(Dataset):
    def __init__(self, files):
        self.counters = torch.zeros(2, dtype=torch.long).share_memory_()  # [total, ez_discarded]
        self.global_idx = {f.stem: k for k, f in enumerate(files)}
        self.ez = torch.zeros(len(files), dtype=torch.uint8).share_memory_()

        self.load_files(files, first_run=True)

    @property
    def total_patches(self):
        return int(self.counters[0])

    @property
    def ez_discarded(self):
        return int(self.counters[1])

    def __len__(self):
        return ITERS * BATCH

    def load_files(self, files, first_run=False):
        if not first_run and len(files) == len(self.images):
            return

        self.images = []
        self.scores = []
        self.names = []
        self.gids = []

        size = 0

        for f in tqdm(random.sample(files, len(files)), unit='file', desc='Loading files', delay=2.0):
            file = np.load(f, mmap_mode="r")
            size += file.size * file.itemsize

            if size > FILE_CACHE:
                # print(f"reached size {size} with {len(self.images)} files")
                break

            score = float(f.stem.split('_')[0]) ** 2

            self.images.append(file)
            self.names.append(f.stem)
            self.gids.append(self.global_idx[f.stem])
            self.scores.append(score)

        self.scores = np.array(self.scores, dtype=np.float32)
        self.order = np.argsort(self.scores)

    def apply_patch_stats(self, stats):
        for gid, loss in stats.items():
            if loss < EZ_DISCARD_THRESHOLD:
                self.ez[gid] = 1

            if loss > EZ_DISCARD_THRESHOLD * 1.25:
                self.ez[gid] = 0

    def __getitem__(self, i):
        # img = self.images[torch.multinomial(self.weights, 1).item()]
        # position within the batch -> one stratum of [0, 1)
        k = ((BATCH - 1) - i) % BATCH

        for i in range(0, 2):
            u = (k + torch.rand(()).item()) / BATCH
            j = min(int(u * len(self.order)), len(self.order) - 1)
            gid = self.gids[self.order[j]]

            if not self.ez[gid]:
                if i > 0:
                    self.counters[1] += 1

                break

        img = self.images[self.order[j]]
        h, w, _ = img.shape
        y = torch.randint(h - PATCH + 1, ()).item()
        x = torch.randint(w - PATCH + 1, ()).item()
        patch = torch.tensor(img[y:y + PATCH, x:x + PATCH])

        self.counters[0] += 1

        return patch, gid


def gamma(x):
    return x.clamp_min(0).add(1e-3).pow(1 / 2.4)


def psnr(a, b):
    return -10 * torch.log10(F.mse_loss(a, b)).item()


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


def calc_vals(res, tg):
    res = apply_approximate_wb(res)
    tg = apply_approximate_wb(tg)

    psnr_val = psnr(res, tg)
    # diff = percept_diff(res, gt)
    ssim_val = ssim(gamma(res), gamma(tg)).item()

    return [psnr_val, ssim_val]


def pretty_score(psnrs: tuple) -> str:
    return f"PSNR: {psnrs[0]:.2f}dB, SSIM: {psnrs[1]:.4f}"


@torch.no_grad()
def evaluate(fn, pairs):
    vals = []

    for mo, gt in tqdm(pairs, unit='pair', desc='Evaluation', delay=2.0):
        mo = mo.to(DEVICE)
        gt = gt.to(DEVICE)

        res = fn(mo)

        vals.append(calc_vals(res, gt))

    return tuple(np.mean(vals, axis=0))


def markesteijn_psnr(kernels, pairs):
    pattern = XTRANS_PATTERN.astype(np.uint8)
    vals = []
    for mo, gt in tqdm(pairs, unit='pair', desc='Markesteijn', delay=2.0):
        mosaic = np.pad(mo[0, 0].numpy(), 18, mode="symmetric")
        res = kernels.demosaic(raw_image=mosaic, xtrans_pattern=pattern, passes=3, crop=False)
        res = torch.from_numpy(np.ascontiguousarray(res[18:-18, 18:-18])).permute(2, 0, 1)[None]

        vals.append(calc_vals(res, gt))
    return tuple(np.mean(vals, axis=0))


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

    if NOISE:
        k_gain = 2  # 2 e-/DN
        iso_gain = 2 ** (
            torch.rand((patch.shape[0], 1, 1, 1), device=DEVICE) *
            (ISO_GAIN_RANGE[1] - ISO_GAIN_RANGE[0]) + ISO_GAIN_RANGE[0]
        )
        maxval = 16383.0
        read_noise_amount_dn = 1.7

        electrons = (torch.clamp(patch, min=0) / iso_gain) * maxval * k_gain
        shot_noise = (torch.poisson(electrons)) * (iso_gain / (maxval * k_gain))
        read_noise = torch.randn_like(patch) * (read_noise_amount_dn / maxval) * iso_gain

        patch = patch + shot_noise + read_noise

    mosaic = patch[:, None, CFA_INDICES, YY, XX]

    return mosaic, patch


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
    parser.add_argument("--finetune", type=str, default=None, help="Specify a model if you want to finetune.")
    parser.add_argument("--discard-threshold", type=float, default=0.3, help="Discard patches with loss above value. "
                                                                             "Default 0.3.")
    parser.add_argument("--ez-discard-threshold", type=float, default=0.009, help="Discard patches with loss below "
                                                                                  "value. Default 0.009.")
    parser.add_argument("--noise", action="store_true", help="Enable realistic noise simulation.")
    args = parser.parse_args()

    global ITERS, PATCH, BATCH, LR, YY, XX, CFA_INDICES, LOG_TXT, EZ_DISCARD_THRESHOLD, NOISE
    ITERS, PATCH, BATCH, LR = args.iters, args.patch, args.batch, args.lr
    EZ_DISCARD_THRESHOLD, NOISE = args.ez_discard_threshold, args.noise

    LOG_PATH.mkdir(exist_ok=True)
    LOG_TXT = open(LOG_PATH / f"{args.name}.txt", "w", encoding="utf-8")

    print = our_write

    for key, value in vars(args).items():
        print(f"{key} {value}")

    print()

    yy, xx, cfa_indices = get_mosaic_arr((PATCH, PATCH), XTRANS_PATTERN)
    YY, XX, CFA_INDICES = (
        torch.tensor(         yy, dtype=torch.int32).to(DEVICE),
        torch.tensor(         xx, dtype=torch.int32).to(DEVICE),
        torch.tensor(cfa_indices, dtype=torch.int32).to(DEVICE),
    )

    non_compiled_model = PackedConstraintXTransNet(XTRANS_PATTERN, width=args.width, depth=args.depth).to(DEVICE)

    if args.finetune is not None:
        model_dict = torch.load(CKPT_PATH / Path(args.finetune + '.pt'), map_location=DEVICE)
        non_compiled_model.load_state_dict(model_dict)

    model = torch.compile(non_compiled_model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_ITERS),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS - WARMUP_ITERS)
        ], milestones=[WARMUP_ITERS]
    )

    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print()

    random.seed(0)

    train_files = sorted(GT_DIR.glob("*.npy"))
    val_files = sorted(VAL_DIR.glob("*.npy"))

    if args.faststart:
        train_files = train_files[:10]
        val_files = val_files[:10]

    print(f"{len(train_files)} train images")
    print(f"{len(val_files)} val images")
    print()

    train_pairs = make_eval_pairs([np.load(f, mmap_mode="r") for f in random.sample(train_files, min(256, len(train_files)))])
    val_pairs = make_eval_pairs([np.load(f, mmap_mode="r") for f in val_files])

    print('Evaluating base model...')
    base_train = evaluate(model.baseline, train_pairs)
    base_val = evaluate(model.baseline, val_pairs)
    print(f"Bilinear val {pretty_score(base_val)}, train {pretty_score(base_train)}")

    mark_val_score = None

    if not args.faststart:
        print('Evaluating Markesteijn...')
        os.environ.setdefault("PYOPENCL_CTX", "0")
        kernels = MarkesteijnOpenCLDemosaicer(kernel_path="src/kernels/demosaic_markesteijn.cl")
        mark_train_score = markesteijn_psnr(kernels, train_pairs)
        mark_val_score = markesteijn_psnr(kernels, val_pairs)
        print(f"Markesteijn val {pretty_score(mark_val_score)}, train {pretty_score(mark_train_score)}")
    else:
        mark_train_score = mark_val_score = (float('-inf'), float('-inf'))

    CKPT_PATH.mkdir(exist_ok=True)
    ckpt_file = CKPT_PATH / f"{args.name}.pt"

    best_val_score = 0.0, float('-inf')
    running = 0.0
    t0 = time.time()
    max_loss = 0.0
    hard_discarded = 0

    if args.plt:
        plot = Plot(
            ITERS, PATCH,
            LOG_PATH, args.name,
            args.val_since, VAL_EVERY,
            (base_val[0], base_train[0]),
            None if mark_val_score is None else mark_val_score[0]
        )

    loader = DataLoader(GTPatches(train_files), batch_size=BATCH, num_workers=4,
                        prefetch_factor=10, persistent_workers=True, shuffle=False)

    pbar = tqdm(enumerate(loader, 1), total=ITERS, miniters=10, mininterval=0.0, maxinterval=99999.9, smoothing=0.2, position=0)
    statusbar = tqdm(total=0, bar_format="{desc}", position=1)
    for step, (gt, gid) in pbar:
        mosaic, gt = preprocess(gt.to(DEVICE))
        out = model(mosaic)

        # loss = F.l1_loss(out, gt)
        scale = gt.mean(dim=(1, 2, 3)).detach().clamp(min=2 ** -7)
        loss = F.l1_loss(out, gt, reduction='none').mean(dim=(1, 2, 3)) / scale

        stats = dict(zip(gid, loss.detach().cpu().tolist()))

        hard_discard_mask = loss > args.discard_threshold
        hard_discarded += hard_discard_mask.sum().item()
        loss[hard_discard_mask] = 0.0

        loss = loss.mean()

        if loss.item() == 0.0:
            print("Zero loss, training failed")
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=1.0,
            norm_type=2.0,
        )
        opt.step()
        sched.step()
        running += loss.item()

        loader.dataset.apply_patch_stats(stats)

        if step % FILES_EVERY == 0 and step != ITERS:
            loader.dataset.load_files(train_files)

        if step % LOG_EVERY == 0 or step == ITERS:
            avg_loss = running / LOG_EVERY
            easy_patches = loader.dataset.ez_discarded / loader.dataset.total_patches
            loader.dataset.counters.zero_()
            print(f"step {step}/{ITERS}  loss {avg_loss:.5f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  "
                  f"discarded ez: {easy_patches * 100.0:.1f}%, "
                  f"hard: {hard_discarded / (LOG_EVERY * BATCH) * 100.0:.3f}%  "

                  f"{LOG_EVERY * BATCH / (time.time() - t0):.0f} img/s")
            hard_discarded = 0
            running = 0.0
            t0 = time.time()

            max_loss = max(max_loss, avg_loss)



        val_score = None

        if (step / ITERS) >= args.val_since and step % VAL_EVERY == 0 or (step == 1 and args.finetune is not None):
            model.eval()
            train_score = evaluate(model, train_pairs)
            val_score = evaluate(model, val_pairs)
            model.train()
            marker = ""
            if val_score[0] > best_val_score[0]:
                best_val_score = val_score
                torch.save(non_compiled_model.state_dict(), ckpt_file)
                marker = " *"

            statusbar.set_description_str(f"Last val {pretty_score(val_score)},\tBest val {pretty_score(val_score)}")
            print(f"val {pretty_score(val_score)} train {pretty_score(train_score)}{marker}")

        pic = None

        if plot is not None and step % Plot.SAMPLES_EVERY == 0:
            with torch.no_grad():
                out_a = apply_approximate_wb(gt[0:Plot.PIC_SAMPLES, ...])
                out_b = apply_approximate_wb(out[0:Plot.PIC_SAMPLES, ...])

                pics = [
                    gamma(out_a),
                    gamma(out_b),
                    (((out_a - out_b) * 10.0) + 0.5)
                ]
                pics = [pic.permute(0, 2, 3, 1).clamp(0, 1).reshape(-1, PATCH, 3) for pic in pics]

                pic = torch.cat(pics, dim=1).cpu().numpy()

        if plot is not None:
            plot.tick(step, loss.item(), None if val_score is None else (val_score[0], train_score[0]), pic)

    LOG_TXT.close()

    if plot is not None:
        plot.save_file()


if __name__ == "__main__":
    main()
