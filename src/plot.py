import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Plot:
    DRAW_EVERY = 5
    MAX_LOSS = 0.0333
    HEIGHT = 512
    SAMPLES_EVERY = 250
    FLUSH_EVERY = 5
    PIC_SAMPLES = 5
    ITERS_PER_PIXEL = 50

    def __init__(self, iters: int, patch: int, file_path: Path, filename, val_since: float, val_every: int, base_val, mark_val):
        plt.style.use('dark_background')

        dpi = 96.0
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi

        self.iters = iters
        self.patch = patch
        self.file_path = file_path
        self.val_since = val_since
        self.val_every = val_every
        self.filename = filename

        plt.ion()

        self.fig, (self.ax_pic, self.ax_loss) = plt.subplots(1, 2, figsize=((3 * 4 * self.patch) / dpi, (Plot.PIC_SAMPLES * self.patch) / dpi),
                                            width_ratios=(0.25, 0.75))
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        self.ax_pic.set_position([0, 0, 0.25, 1])
        self.ax_pic.axis('off')
        self.ax_pic_art = self.ax_pic.imshow(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8), aspect='auto')

        self.ax_loss.grid(True, axis='x', color='#ffffff30', linewidth=1)
        self.ax_loss.grid(True, axis='y', color='#0000ff70', linewidth=1)
        self.ax_loss.autoscale(enable=True, axis='y')

        self.loss_graph = np.zeros((Plot.HEIGHT, self.iters // Plot.ITERS_PER_PIXEL), dtype=np.uint8)
        self.loss_graph[0] = 255
        self.ax_loss.set_xlim(left=0, right=self.iters)
        self.ax_loss.set_ylim(bottom=0, top=Plot.MAX_LOSS)
        self.ax_loss.set_ylabel("Loss", color='tab:blue')
        self.ax_loss_art = self.ax_loss.imshow(self.loss_graph, extent=[0, self.iters, 0, Plot.MAX_LOSS], aspect='auto', zorder=-1, alpha=1)

        self.ax_loss.set_xlabel("Iteration")
        self.ax_loss.tick_params(axis='y', labelcolor='tab:blue')

        self.ax_psnr = self.ax_loss.twinx()
        self.ax_psnr.grid(True, axis='y', color='#ff000040', linewidth=1)
        self.ax_psnr.autoscale(enable=True, axis='y')

        self.line_psnr, = self.ax_psnr.plot([], color='tab:red', marker='o')

        self.ax_psnr.set_ylabel("PSNR", color='tab:red')
        self.ax_psnr.yaxis.set_major_formatter(ticker.EngFormatter(unit='dB'))
        self.ax_psnr.tick_params(axis='y', labelcolor='tab:red')

        self.step_psnr_x = np.arange(int(self.iters * val_since), self.iters + 1, self.val_every)
        self.step_psnr_y = np.zeros(len(self.step_psnr_x))
        self.step_psnr_count = 0

        if base_val is not None:
            self.ax_psnr.axhline(y=base_val, color='#ffb000a0', linewidth=1, label="val bili PSNR")
            self.step_psnr_y[0] = base_val

            self.step_psnr_count += 1

        if mark_val is not None:
            self.ax_psnr.axhline(y=mark_val, color='#00ff00a0', linewidth=1, label="val mark PSNR")

        self.ax_psnr.legend()

        plt.show()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.running_loss = 0.0
        self.max_loss = 0.0

    def tick(self, step: int, loss: float, val_psnr: float | None, pic: np.ndarray | None):
        self.running_loss += loss

        if loss > self.max_loss:
            self.max_loss = loss

        if step % Plot.ITERS_PER_PIXEL == 0:
            avg_loss = self.running_loss / Plot.ITERS_PER_PIXEL

            max_id = max(self.max_loss * (Plot.HEIGHT / Plot.MAX_LOSS), 0)
            avg_id = max(avg_loss * (Plot.HEIGHT / Plot.MAX_LOSS), 0)

            col = (step - 1) // Plot.ITERS_PER_PIXEL

            if max_id: self.loss_graph[-int(max_id):, col] = 128
            if avg_id: self.loss_graph[-int(avg_id):, col] = 255

            self.running_loss = 0.0
            self.max_loss = 0.0

        if val_psnr is not None:
            self.step_psnr_x[self.step_psnr_count] = step
            self.step_psnr_y[self.step_psnr_count] = val_psnr

            self.step_psnr_count += 1

            self.line_psnr.set_xdata(self.step_psnr_x[:self.step_psnr_count])
            self.line_psnr.set_ydata(self.step_psnr_y[:self.step_psnr_count])

            self.ax_psnr.relim()
            self.ax_psnr.autoscale_view()

        if step % Plot.SAMPLES_EVERY == 0:
            self.ax_loss_art.set_data(self.loss_graph)
            self.ax_pic_art.set_data(pic)

            self.save_file()

        if step % Plot.DRAW_EVERY == 0:
            self.fig.canvas.flush_events()

    def save_file(self):
        self.fig.savefig(self.file_path / f"{self.filename}.svg")
