"""Benchmark both detectors across electron dose.

Simulates fresh images at each dose level (fixed seeds, several images
per dose), runs the LoG baseline and the trained U-Net, and reports mean
F1 and position RMSE per dose. Writes results/dose_sweep.json and the
dose-sweep figure.

The LoG baseline is reported twice: once with its fixed default
threshold, and once oracle-tuned, where at every dose the threshold is
chosen from a grid to maximise mean F1 using ground-truth access the
U-Net never gets. The gap between the two LoG rows measures how much the
classical detector depends on per-condition retuning; the U-Net runs
with one fixed threshold everywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from atomfinder.detect import detect_log, peaks_from_heatmap
from atomfinder.metrics import filter_margin, match_positions
from atomfinder.net import UNet, predict_heatmap
from atomfinder.refine import refine_com
from atomfinder.sim import SimConfig, simulate_image

DOSES = [1.0, 2.0, 4.0, 8.0, 30.0, 125.0, 500.0, 2000.0]
LOG_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
LOG_FIXED_THRESHOLD = 0.25
MARGIN = 8.0
TOLERANCE = 4.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/unet_atoms.pt"))
    parser.add_argument("--images-per-dose", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/dose_sweep.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/dose_sweep.png"))
    args = parser.parse_args()

    model = UNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))

    rows = []
    for dose in DOSES:
        samples = [
            simulate_image(
                SimConfig(size=256, dose=dose),
                np.random.default_rng(args.seed + 1000 * int(dose) + k),
            )
            for k in range(args.images_per_dose)
        ]
        truths = [filter_margin(s.positions, s.image.shape, MARGIN) for s in samples]

        def score(peaks: np.ndarray, sample, true) -> tuple[float, float]:
            pred = filter_margin(refine_com(sample.image, peaks), sample.image.shape, MARGIN)
            res = match_positions(true, pred, tolerance=TOLERANCE)
            return res.f1, res.rmse

        def mean_scores(peaks_per_sample: list[np.ndarray]) -> tuple[float, float]:
            f1s, rmses = [], []
            for peaks, sample, true in zip(peaks_per_sample, samples, truths):
                f1, rmse = score(peaks, sample, true)
                f1s.append(f1)
                if np.isfinite(rmse):
                    rmses.append(rmse)
            return float(np.mean(f1s)), float(np.mean(rmses)) if rmses else float("nan")

        log_at = {
            threshold: mean_scores(
                [
                    detect_log(s.image, sigma=s.config.probe_sigma, threshold_rel=threshold)
                    for s in samples
                ]
            )
            for threshold in LOG_THRESHOLDS
        }
        oracle_threshold = max(log_at, key=lambda t: log_at[t][0])
        unet_f1, unet_rmse = mean_scores(
            [peaks_from_heatmap(predict_heatmap(model, s.image)) for s in samples]
        )

        row = {
            "dose": dose,
            "log_fixed_f1": round(log_at[LOG_FIXED_THRESHOLD][0], 4),
            "log_fixed_rmse_px": round(log_at[LOG_FIXED_THRESHOLD][1], 4),
            "log_oracle_f1": round(log_at[oracle_threshold][0], 4),
            "log_oracle_rmse_px": round(log_at[oracle_threshold][1], 4),
            "log_oracle_threshold": oracle_threshold,
            "unet_f1": round(unet_f1, 4),
            "unet_rmse_px": round(unet_rmse, 4),
        }
        rows.append(row)
        print(
            f"dose {dose:6.0f}  LoG-fixed F1 {row['log_fixed_f1']:.3f}   "
            f"LoG-oracle F1 {row['log_oracle_f1']:.3f} (thr {oracle_threshold:.2f})   "
            f"U-Net F1 {row['unet_f1']:.3f} (RMSE {row['unet_rmse_px']:.2f})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images_per_dose": args.images_per_dose,
        "tolerance_px": TOLERANCE,
        "margin_px": MARGIN,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.out}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    doses = [r["dose"] for r in rows]
    series = (
        ("log_fixed", "#9bb0bd", "LoG, fixed threshold"),
        ("log_oracle", "#4dc3ff", "LoG, oracle-tuned per dose"),
        ("unet", "#ff5c39", "U-Net, fixed threshold"),
    )
    for method, color, label in series:
        ax1.plot(doses, [r[f"{method}_f1"] for r in rows], "o-", color=color, label=label)
        ax2.plot(doses, [r[f"{method}_rmse_px"] for r in rows], "o-", color=color, label=label)
    for ax, ylabel in ((ax1, "detection F1"), (ax2, "position RMSE (px)")):
        ax.set_xscale("log")
        ax.set_xlabel("dose (mean counts at a column peak)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.set_ylim(0.0, 1.02)
    fig.suptitle("Detector performance vs electron dose")
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
