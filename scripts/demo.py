"""End-to-end demo on the committed sample images.

For each committed sample (high, medium, low dose) this script runs both
detectors (LoG baseline and trained U-Net), refines detections to
sub-pixel accuracy, scores them against the known ground truth, writes
results/metrics.json and renders the detection overlay panel.
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
from atomfinder.io import load_sample
from atomfinder.metrics import DetectionResult, filter_margin, match_positions
from atomfinder.net import UNet, predict_heatmap
from atomfinder.refine import refine_com

MARGIN = 8.0
TOLERANCE = 4.0
SAMPLE_ORDER = ["high_dose", "mid_dose", "low_dose"]


def evaluate(
    image: np.ndarray, raw_peaks: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, DetectionResult]:
    """Refine raw detections and score them against the ground truth."""
    refined = refine_com(image, raw_peaks)
    pred = filter_margin(refined, image.shape, MARGIN)
    true = filter_margin(truth, image.shape, MARGIN)
    return pred, match_positions(true, pred, tolerance=TOLERANCE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sample"))
    parser.add_argument("--model", type=Path, default=Path("models/unet_atoms.pt"))
    parser.add_argument("--figure", type=Path, default=Path("figures/detection_overlay.png"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args()

    model = UNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    fig, axes = plt.subplots(len(SAMPLE_ORDER), 3, figsize=(12.5, 4.2 * len(SAMPLE_ORDER)))

    for row, name in enumerate(SAMPLE_ORDER):
        sample = load_sample(args.data / f"{name}.npz")
        image, truth = sample.image, sample.positions
        true_shown = filter_margin(truth, image.shape, MARGIN)

        detections = {
            "log": detect_log(image, sigma=sample.config.probe_sigma),
            "unet": peaks_from_heatmap(predict_heatmap(model, image)),
        }

        all_metrics[name] = {}
        panels = [("ground truth", None, None)]
        for method, raw in detections.items():
            pred, res = evaluate(image, raw, truth)
            all_metrics[name][method] = {
                "n_true": res.n_true,
                "n_pred": res.n_pred,
                "precision": round(res.precision, 4),
                "recall": round(res.recall, 4),
                "f1": round(res.f1, 4),
                "rmse_px": round(res.rmse, 4) if np.isfinite(res.rmse) else None,
            }
            panels.append((method, pred, res))
            print(
                f"{name:10s} {method:5s}  P {res.precision:.3f}  R {res.recall:.3f}  "
                f"F1 {res.f1:.3f}  RMSE {res.rmse:.2f} px"
            )

        for col, (label, pred, res) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", interpolation="nearest")
            ax.scatter(
                true_shown[:, 1],
                true_shown[:, 0],
                s=26,
                facecolors="none",
                edgecolors="#4dc3ff",
                linewidths=0.9,
                label="ground truth",
            )
            title = f"{name.replace('_', ' ')}: {label}"
            if pred is not None and res is not None:
                ax.scatter(
                    pred[:, 1], pred[:, 0], s=12, marker="x", c="#ff5c39", lw=0.9, label=label
                )
                title += f" (F1 {res.f1:.3f})"
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    print(f"saved {args.figure}")

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tolerance_px": TOLERANCE, "margin_px": MARGIN, "samples": all_metrics}
    args.metrics.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.metrics}")


if __name__ == "__main__":
    main()
