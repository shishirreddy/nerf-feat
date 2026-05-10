"""Lightweight training-curve logger with PNG export.

Usage
-----
    plotter = LossPlotter()

    # inside the training loop, at log_interval:
    plotter.update(iteration, color_loss=float(loss_a), silhouette_loss=float(loss_b))

    # at preview_interval, save accumulated curves:
    plotter.save(artifacts.previews / "loss_curves.png")
"""

from collections import defaultdict
from pathlib import Path

import numpy as np


class LossPlotter:
    """Accumulates scalar losses and saves a publication-ready PNG loss curve."""

    def __init__(self) -> None:
        self._steps: list[int] = []
        self._losses: dict[str, list[float]] = defaultdict(list)

    def update(self, step: int, **losses: float) -> None:
        """Record one set of loss values at *step*.

        Call this every time the training loop logs to the console so the
        plot and the log stay in sync.
        """
        self._steps.append(step)
        for name, value in losses.items():
            self._losses[name].append(float(value))

    def save(self, path: Path) -> None:
        """Render all tracked loss curves and write a PNG to *path*.

        Each loss gets its own subplot on a shared x-axis.  The raw curve is
        drawn in a muted colour; a smoothed trend line is overlaid in a
        contrasting colour once enough data points have accumulated.
        """
        import matplotlib
        matplotlib.use("Agg")  # headless — no display required
        import matplotlib.pyplot as plt

        names = list(self._losses)
        if not names or not self._steps:
            return

        n_plots = len(names)
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(10, 3 * n_plots),
            sharex=True,
            squeeze=False,
        )

        steps = np.array(self._steps, dtype=np.float32)
        for ax, name in zip(axes[:, 0], names):
            values = np.array(self._losses[name], dtype=np.float32)

            ax.plot(steps, values, linewidth=0.8, alpha=0.45, color="#5B9BD5", label="raw")

            # Smoothed trend — window = ~2 % of total steps, minimum 10.
            window = max(10, len(values) // 50)
            if len(values) >= window:
                kernel = np.ones(window) / window
                smoothed = np.convolve(values, kernel, mode="valid")
                smooth_steps = steps[window - 1:]
                ax.plot(smooth_steps, smoothed, linewidth=1.8, color="#E05A4E", label=f"smoothed (w={window})")
                ax.legend(fontsize=8, loc="upper right")

            ax.set_ylabel(name, fontsize=9)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
            ax.tick_params(labelsize=8)

        axes[-1, 0].set_xlabel("step", fontsize=9)
        fig.suptitle("Training loss curves", fontsize=11, y=1.01)
        fig.tight_layout()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=130, bbox_inches="tight")
        plt.close(fig)
