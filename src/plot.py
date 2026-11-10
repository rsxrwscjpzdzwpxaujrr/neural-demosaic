import time
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Plot:
    MAX_LOSS = 0.04
    SAMPLES_EVERY = 250
    PIC_SAMPLES = 4
    PIC_COLUMNS = 3
    ITERS_PER_PIXEL = 50
    TARGET_FLUSH_DT = 0.03

    def __init__(self, iters: int, patch: int, file_path: Path, filename, val_since: float, val_every: int, base_val, mark_val):
        self.iters = iters
        self.patch = patch
        self.file_path = file_path
        self.val_since = val_since
        self.val_every = val_every
        self.filename = filename
        self.last_flush_time = 0.0

        assert self.iters % Plot.ITERS_PER_PIXEL == 0, "Iteration amount should be divisible by ITERS_PER_PIXEL"

        plt.style.use('dark_background')

        dpi = 96.0
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi

        plt.ion()

        self.pic_width = (self.patch * Plot.PIC_COLUMNS)
        self.plot_width = self.pic_width * 5
        self.full_width = self.pic_width + self.plot_width

        self.fig, (self.ax_pic, self.ax_loss) = plt.subplots(
            1, 2,
            figsize=(self.full_width / dpi, (self.patch * Plot.PIC_SAMPLES) / dpi),
            width_ratios=(self.pic_width / self.full_width, 1 - self.pic_width / self.full_width),
            gridspec_kw={'wspace': 0, 'hspace': 0}
        )

        self.ax_pic.axis('off')
        self.ax_pic.set_position([0, 0, self.pic_width / self.full_width, 1])
        self.ax_pic_art = self.ax_pic.imshow(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8), aspect='auto')
        self.ax_loss.grid(True, axis='x', color='#ffffff30', linewidth=1)
        self.ax_loss.grid(True, axis='y', color='#0000ff70', linewidth=1)
        self.ax_loss.autoscale(enable=True, axis='y')

        self.height = int(self.ax_loss.get_position().height * Plot.PIC_SAMPLES * self.patch)

        self.loss_graph = np.zeros((self.height, self.iters // Plot.ITERS_PER_PIXEL), dtype=np.uint8)
        self.loss_graph[0] = 255
        self.ax_loss.set_xlim(left=0, right=self.iters)
        self.ax_loss.set_ylim(bottom=0, top=Plot.MAX_LOSS)
        self.ax_loss.set_ylabel("Loss", color='tab:blue')
        self.ax_loss_art = self.ax_loss.imshow(self.loss_graph, extent=[0, self.iters, 0, Plot.MAX_LOSS], aspect='auto', zorder=-1, alpha=1)
        self.ax_loss.format_xdata = lambda x: f"{x:.0f} it"

        self.ax_loss.set_xlabel("Iteration")
        self.ax_loss.tick_params(axis='y', labelcolor='tab:blue')

        self.ax_psnr = self.ax_loss.twinx()

        for ax in (self.ax_loss, self.ax_psnr):
            ax.callbacks.connect('xlim_changed', self.__clamp_x)

        self.ax_psnr.format_xdata = lambda x: f"{x:.0f} it"
        self.ax_psnr.grid(True, axis='y', color='#ff000040', linewidth=1)
        self.ax_psnr.autoscale(enable=True, axis='y')
        self.ax_psnr.format_ydata = lambda y: f"{y:.2f} dB"

        self.line_psnr, = self.ax_psnr.plot([], color='tab:red', marker='o')

        self.ax_psnr.set_ylabel("PSNR", color='tab:red')
        self.ax_psnr.yaxis.set_major_formatter(ticker.EngFormatter(unit='dB'))
        self.ax_psnr.tick_params(axis='y', labelcolor='tab:red')

        self.step_psnr_x = []
        self.step_psnr_y = []

        if mark_val is not None:
            self.ax_psnr.axhline(y=mark_val, color='#00ff00a0', linewidth=1, label="val mark PSNR")

        if base_val is not None:
            self.ax_psnr.axhline(y=base_val, color='#ffb000a0', linewidth=1, label="val bili PSNR")
            self.__append_psnr(0, base_val)

        self.ax_psnr.legend()

        plt.show()

        self.running_loss = 0.0
        self.max_loss = 0.0

    def __clamp_x(self, ax):
        lo, hi = ax.get_xlim()
        span = hi - lo
        if span > self.iters:
            new = (0, self.iters)
        elif lo < 0:
            new = (0, span)
        elif hi > self.iters:
            new = (self.iters - span, self.iters)
        else:
            return
        ax.set_xlim(new)

    def tick(self, step: int, loss: float, val_psnr: float | None, pic: np.ndarray | None):
        assert step > 0, "Steps should start from 1"

        self.running_loss += loss

        if loss > self.max_loss:
            self.max_loss = loss

        if step % Plot.ITERS_PER_PIXEL == 0:
            avg_loss = self.running_loss / Plot.ITERS_PER_PIXEL

            max_id = max(int(self.max_loss * (self.height / Plot.MAX_LOSS)), 0)
            avg_id = max(int(avg_loss * (self.height / Plot.MAX_LOSS)), 0)

            col = (step - 1) // Plot.ITERS_PER_PIXEL

            if max_id: self.loss_graph[-max_id:, col] = 128
            if avg_id: self.loss_graph[-avg_id:, col] = 255

            self.running_loss = 0.0
            self.max_loss = 0.0

        if val_psnr is not None:
            self.__append_psnr(step, val_psnr)

        if pic is not None:
            self.ax_loss_art.set_data(self.loss_graph)
            self.ax_pic_art.set_data(pic)
            self.fig.canvas.draw_idle()
            self.save_file()

        now = time.time()

        if now - self.last_flush_time > Plot.TARGET_FLUSH_DT:
            self.fig.canvas.flush_events()
            self.last_flush_time = now

    def __append_psnr(self, step: int, psnr: float):
        self.step_psnr_x.append(step)
        self.step_psnr_y.append(psnr)

        self.line_psnr.set_xdata(self.step_psnr_x)
        self.line_psnr.set_ydata(self.step_psnr_y)

        self.ax_psnr.relim()
        self.ax_psnr.autoscale_view()

        self.fig.canvas.draw_idle()

    def save_file(self):
        self.fig.savefig(self.file_path / f"{self.filename}.svg")
