import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import random
import argparse
from termcolor import colored
from strip_ansi import strip_ansi
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import apply_approximate_wb, to_patches
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
EVAL_PATCH = 288
BATCH = 0
ITERS = 0
LR = 0.0
VAL_EVERY = 1000
EZ_DISCARD_THRESHOLD = 0.0

FILES_EVERY = 2500
FILE_CACHE = 16 * 1024 * 1024 * 1024

MAX_EVAL_FILES = 256
MAX_EVAL_PATCHES = 256

EXPOSURE_RANGE = (-2.0, 2.0)
ISO_GAIN_RANGE = (-3.0, 3.0)
NOISE = False

YY, XX, CFA_INDICES = None, None, None

LOG_TXT = None


def our_write(cls='', *args, **kwargs):
    tqdm.write(cls, *args, **kwargs)

    if LOG_TXT is not None:
        __builtins__.print(strip_ansi(cls), file=LOG_TXT, flush=True, *args, **kwargs)


class Logger:
    LOG_EVERY = 50

    def __init__(self, iters: int, batch: int, dataset):
        assert iters > Logger.LOG_EVERY, "Total iter count can't be higher than LOG_EVERY"

        self.iters = iters
        self.batch = batch
        self.dataset = dataset
        self.running = 0.0
        self.t0 = time.time()
        self.hard_discarded = 0

    def tick(self, step: int, loss: float, last_lr: float, hard_discarded: int):
        self.running += loss
        self.hard_discarded += hard_discarded

        if step % Logger.LOG_EVERY == 0 or step == self.iters:
            avg_loss = self.running / Logger.LOG_EVERY
            easy_patches = self.dataset.ez_discarded / self.dataset.total_patches
            self.dataset.counters.zero_()
            print(f"step {step}/{self.iters}  loss {avg_loss:.5f}  "
                  f"lr {last_lr:.2e}  "
                  f"discarded ez: {easy_patches * 100.0:.1f}%, "
                  f"hard: {self.hard_discarded / (Logger.LOG_EVERY * self.batch) * 100.0:.3f}%  "
                  f"{Logger.LOG_EVERY * self.batch / (time.time() - self.t0):.0f} img/s")
            self.hard_discarded = 0
            self.running = 0.0
            self.t0 = time.time()


class Evaluator:
    def __init__(self, train_mos, train_gts, val_mos, val_gts):
        self.train_mos = train_mos
        self.train_gts = train_gts
        self.val_mos = val_mos
        self.val_gts = val_gts

        self.best_val = [float('-inf'), float('-inf')]
        self.best_train = [float('-inf'), float('-inf')]

    def evaluate_base(self, model):
        print('Evaluating base model...')
        train = Evaluator._evaluate(model.baseline, self.train_mos, self.train_gts)
        val = Evaluator._evaluate(model.baseline, self.val_mos, self.val_gts)

        return train, val

    def evaluate_mark(self):
        print('Evaluating Markesteijn...')
        os.environ.setdefault("PYOPENCL_CTX", "0")
        kernels = MarkesteijnOpenCLDemosaicer(kernel_path="src/kernels/demosaic_markesteijn.cl")
        train = Evaluator._evaluate_markesteijn(kernels, self.train_mos, self.train_gts)
        val = Evaluator._evaluate_markesteijn(kernels, self.val_mos, self.val_gts)

        return train, val

    def evaluate_model(self, model):
        print('Evaluating model...')
        model.eval()
        train = Evaluator._evaluate(model, self.train_mos, self.train_gts)
        val = Evaluator._evaluate(model, self.val_mos, self.val_gts)
        model.train()

        return train, val

    def check_if_best(self, train, val):
        best = False

        if val[0] > self.best_val[0]:
            self.best_val[0] = val[0]
            best = True

        if val[1] > self.best_val[1]:
            self.best_val[1] = val[1]

        if train[0] > self.best_train[0]:
            self.best_train[0] = train[0]

        if train[1] > self.best_train[1]:
            self.best_train[1] = train[1]

        return best

    @torch.no_grad()
    def _calc_vals(res, tg):
        res = apply_approximate_wb(res)
        tg = apply_approximate_wb(tg)

        psnr_vals = Evaluator._psnr(res, tg)
        ssim_vals = ssim_fn(gamma(res), gamma(tg), data_range=1.0, reduction="none")

        return psnr_vals.mean().item(), ssim_vals.mean().item()

    @torch.no_grad()
    def _evaluate(fn, mosaics, gts):
        mosaics = mosaics.to(DEVICE)
        gts = gts.to(DEVICE)

        res = fn(mosaics)

        return Evaluator._calc_vals(res, gts)

    def _evaluate_markesteijn(kernels, mosaics, gts):
        pattern = XTRANS_PATTERN.astype(np.uint8)
        b = mosaics.shape[0]
        size = mosaics.shape[2]

        ress = torch.empty((b, 3, size, size))

        for i in tqdm(range(0, b), unit='pair', desc='Markesteijn', delay=2.0):
            mosaic = np.pad(mosaics[i, 0].numpy(), 18, mode="symmetric")
            res = kernels.demosaic(raw_image=mosaic, xtrans_pattern=pattern, passes=3, crop=False)
            res = torch.from_numpy(np.ascontiguousarray(res[18:-18, 18:-18])).permute(2, 0, 1)[None]

            ress[i] = res

        return Evaluator._calc_vals(ress, gts)

    def _psnr(a, b):
        return -10 * torch.log10(F.mse_loss(a, b))


@torch.no_grad()
def plot_pics(gt, out):
    out_a = apply_approximate_wb(gt[0:Plot.PIC_SAMPLES, ...])
    out_b = apply_approximate_wb(out[0:Plot.PIC_SAMPLES, ...])

    pics = [
        gamma(out_a),
        gamma(out_b),
        (((out_a - out_b) * 10.0) + 0.5)
    ]
    pics = [pic.permute(0, 2, 3, 1).clamp(0, 1).reshape(-1, PATCH, 3) for pic in pics]

    return torch.cat(pics, dim=1).cpu().numpy()


def bold_if(str, cond):
    return colored(str, color="light_green" if cond else None, attrs=["bold"] if cond else [])


def pretty_score(score: tuple, best: tuple = (float('-inf'), float('-inf'))) -> str:
    return f"PSNR: {bold_if(f"{score[0]:.2f}dB", score[0] == best[0])}, "\
           f"SSIM: {bold_if(f"{score[1]:.4f}", score[1] == best[1])}"


def make_eval_pairs(image_paths, size):
    def pick(k, max_amount):
        rng = np.random.default_rng(0)
        return rng.choice(k, min(max_amount, k), replace=False)

    images = (np.load(image_paths[i], mmap_mode="r") for i in pick(len(image_paths), MAX_EVAL_FILES))
    crops = np.concatenate([to_patches(img, size)[0] for img in images])
    crops = crops[pick(len(crops), MAX_EVAL_PATCHES)]

    yy, xx, cfa_indices = get_mosaic_arr((size, size), XTRANS_PATTERN)

    mosaics = torch.from_numpy(crops[:, yy, xx, cfa_indices]).unsqueeze(1).to(torch.float32)
    gts = torch.from_numpy(crops).permute(0, 3, 1, 2).to(torch.float32)

    return mosaics, gts


def gamma(x):
    return x.clamp_min(0).add(1e-3).pow(1 / 2.4)


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
    parser.add_argument("--warmup-iters", type=int, default=1000, help="Warmup iteration count. Default 1000.")
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
    parser.add_argument("--no-compile", action="store_true", help="Disable compiling the model.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate. Specify --name, --width, and --depth")
    args = parser.parse_args()

    assert args.warmup_iters < args.iters, "Warmup iter count should be lower than total iter count."

    global print
    global ITERS, PATCH, BATCH, LR, YY, XX, CFA_INDICES, LOG_TXT, EZ_DISCARD_THRESHOLD, NOISE
    ITERS, PATCH, BATCH, LR = args.iters, args.patch, args.batch, args.lr
    EZ_DISCARD_THRESHOLD, NOISE = args.ez_discard_threshold, args.noise

    start_time = datetime.now(timezone.utc)

    log_filename = f"{args.name}_{start_time.isoformat(timespec="seconds")}"

    LOG_PATH.mkdir(exist_ok=True)
    LOG_TXT = open(LOG_PATH / f"{log_filename}.txt", "w", encoding="utf-8")

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

    if args.eval_only:
        args.finetune = args.name

    non_compiled_model = PackedConstraintXTransNet(XTRANS_PATTERN, width=args.width, depth=args.depth).to(DEVICE)

    if args.finetune is not None:
        model_dict = torch.load(CKPT_PATH / Path(args.finetune + '.pt'), map_location=DEVICE)
        non_compiled_model.load_state_dict(model_dict)

    model = non_compiled_model if args.no_compile else torch.compile(non_compiled_model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_iters),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS - args.warmup_iters)
        ], milestones=[args.warmup_iters]
    )

    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print()

    train_files = []
    train_dirs = [GT_DIR]
    train_dirs.extend(sorted(GT_DIR.glob("*/")))

    for train_dir in train_dirs:
        files = sorted(train_dir.glob("*.npy"))
        print(f"{len(files)} train images from {train_dir.stem}")
        train_files.extend(files)

    val_files = sorted(VAL_DIR.glob("*.npy"))

    if args.faststart:
        train_files = train_files[:10]
        val_files = val_files[:10]

    print(f"{len(train_files)} train images total")
    print(f"{len(val_files)} val images")
    print()

    random.seed(0)

    evaluator = Evaluator(*make_eval_pairs(train_files, PATCH), *make_eval_pairs(val_files, EVAL_PATCH))

    base_train, base_val = evaluator.evaluate_base(model)
    print(f"Bilinear val {pretty_score(base_val)}, train {pretty_score(base_train)}")

    mark_val_score = None

    if not args.faststart:
        mark_train_score, mark_val_score = evaluator.evaluate_mark()
        print(f"Markesteijn val {pretty_score(mark_val_score)}, train {pretty_score(mark_train_score)}")
    else:
        mark_train_score = mark_val_score = (float('-inf'), float('-inf'))

    if args.eval_only:
        train_score, val_score = evaluator.evaluate_base(model)
        print(f"val {pretty_score(val_score)} train {pretty_score(train_score)}")

        LOG_TXT.close()
        return

    CKPT_PATH.mkdir(exist_ok=True)
    ckpt_file = CKPT_PATH / f"{args.name}.pt"

    if args.plt:
        plot = Plot(
            ITERS, PATCH,
            LOG_PATH, log_filename,
            args.val_since, VAL_EVERY,
            (base_val[0], base_train[0]),
            None if mark_val_score is None else mark_val_score[0]
        )

    random.seed(0)

    loader = DataLoader(GTPatches(train_files), batch_size=BATCH, num_workers=4,
                        prefetch_factor=10, persistent_workers=True, shuffle=False)

    logger = Logger(args.iters, args.batch, loader.dataset)

    pbar = tqdm(enumerate(loader, 1), total=ITERS, miniters=10, mininterval=0.0, maxinterval=99999.9, smoothing=0.2)
    for step, (gt, gid) in pbar:
        mosaic, gt = preprocess(gt.to(DEVICE))
        out = model(mosaic)

        # loss = F.l1_loss(out, gt)
        scale = gt.mean(dim=(1, 2, 3)).detach().clamp(min=2 ** -7)
        loss = F.l1_loss(out, gt, reduction='none').mean(dim=(1, 2, 3)) / scale

        stats = dict(zip(gid, loss.detach().cpu().tolist()))

        hard_discard_mask = loss > args.discard_threshold
        hard_discarded = hard_discard_mask.sum().item()
        loss[hard_discard_mask] = 0.0

        loss = loss.mean()
        loss_num = loss.item()

        if loss_num == 0.0:
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

        loader.dataset.apply_patch_stats(stats)

        if step % FILES_EVERY == 0 and step != ITERS:
            loader.dataset.load_files(train_files)

        logger.tick(step, loss_num, sched.get_last_lr()[0], hard_discarded)

        val_score = None

        if (step / ITERS) >= args.val_since and step % VAL_EVERY == 0 or (step == 1 and args.finetune is not None):
            train_score, val_score = evaluator.evaluate_model(model)

            best = evaluator.check_if_best(train_score, val_score)

            if best:
                torch.save(non_compiled_model.state_dict(), ckpt_file)

            print(f"val {pretty_score(val_score, evaluator.best_val)} "
                  f"train {pretty_score(train_score, evaluator.best_train)}{" *" if best else ""}")

        if plot is not None:
            pic = None

            if step % Plot.SAMPLES_EVERY == 0:
                pic = plot_pics(gt, out)

            plot.tick(step, loss_num, None if val_score is None else (val_score[0], train_score[0]), pic)

    since_start = datetime.now(timezone.utc) - start_time
    since_start = since_start - timedelta(microseconds=since_start.microseconds)

    print(f"Time since start: {since_start}")

    LOG_TXT.close()

    if plot is not None:
        plot.save_file()


if __name__ == "__main__":
    main()
