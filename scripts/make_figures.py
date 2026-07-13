"""Generate the repository's signature figures.

- figures/dose_ladder.gif: one field of view swept from dose 2000 down to
  dose 1 and back, with U-Net detections overlaid and per-frame F1.
- figures/preset_gallery.png: the four materials presets, clean and at
  dose 30.

Run from the repository root after training models/unet_seg.pt.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from atomfinder.detect import peaks_from_heatmap
from atomfinder.metrics import filter_margin, match_positions
from atomfinder.net import UNet, predict_heatmap
from atomfinder.refine import refine_com
from atomfinder.sim import preset_config, simulate_image

MARGIN, TOL = 8.0, 4.0
DOSES = [2000.0, 1000.0, 500.0, 250.0, 125.0, 60.0, 30.0, 16.0, 8.0, 4.0, 2.0, 1.0]


def dose_ladder(model: UNet) -> None:
    frames = []
    for dose in DOSES:
        result = simulate_image(
            preset_config("graphene", size=256, dose=dose, rotation_deg=8.0),
            np.random.default_rng(3),
        )
        peaks = peaks_from_heatmap(predict_heatmap(model, result.image))
        pred = filter_margin(refine_com(result.image, peaks), result.image.shape, MARGIN)
        true = filter_margin(result.positions, result.image.shape, MARGIN)
        res = match_positions(true, pred, TOL)
        frames.append((dose, result.image, pred, res.f1))

    fig, ax = plt.subplots(figsize=(4.6, 4.9))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)

    def draw(i: int):
        dose, image, pred, f1 = frames[i]
        ax.clear()
        ax.imshow(image, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.6)
        ax.scatter(pred[:, 1], pred[:, 0], s=14, marker="x", c="#ff5c39", lw=0.9)
        ax.set_title(f"graphene, dose {dose:g} counts/peak, U-Net F1 {f1:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save("figures/dose_ladder.gif", writer=PillowWriter(fps=2), dpi=88)
    plt.close(fig)
    print("saved figures/dose_ladder.gif")


def preset_gallery() -> None:
    presets = ["graphene", "mos2", "srtio3", "fcc110"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.6), layout="constrained")
    for col, preset in enumerate(presets):
        for row, dose in enumerate((2000.0, 30.0)):
            result = simulate_image(
                preset_config(preset, size=256, dose=dose, rotation_deg=10.0),
                np.random.default_rng(5),
            )
            ax = axes[row, col]
            ax.imshow(result.image, cmap="gray", interpolation="nearest")
            ax.set_title(f"{preset}, dose {dose:g}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Materials presets: clean (top) and dose-limited (bottom)")
    fig.savefig("figures/preset_gallery.png", dpi=140)
    plt.close(fig)
    print("saved figures/preset_gallery.png")


if __name__ == "__main__":
    unet = UNet()
    unet.load_state_dict(torch.load("models/unet_seg.pt", weights_only=True))
    dose_ladder(unet)
    preset_gallery()
