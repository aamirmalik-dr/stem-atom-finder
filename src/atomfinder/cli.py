"""Command-line interface.

Subcommands:
    atomfinder simulate   simulate a materials preset to .npz and/or .png
    atomfinder detect     run a detector on an .npz sample or a real image
    atomfinder train      train the U-Net detectors (and ablation variants)
    atomfinder benchmark  run a YAML benchmark config, save JSON + figure
    atomfinder samples    regenerate the committed sample images
    atomfinder demo       detect on every committed sample, save overlay
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

from atomfinder.benchmark import REFINERS, build_methods, run_config
from atomfinder.io import load_sample, save_sample
from atomfinder.metrics import margin_mask, match_positions
from atomfinder.plots import plot_payload
from atomfinder.real import crop_to_multiple, load_image
from atomfinder.sim import preset_config, simulate_image
from atomfinder.train import TrainSettings, train_unet

MARGIN = 8.0
TOLERANCE = 4.0

# (name, preset, dose, seed): the committed reference samples.
SAMPLES = (
    ("graphene_d50", "graphene", 50.0, 0),
    ("mos2_d50", "mos2", 50.0, 1),
    ("srtio3_d125", "srtio3", 125.0, 2),
    ("fcc110_d10", "fcc110", 10.0, 3),
)
DEFAULT_MODELS = {"unet": "models/unet_seg.pt", "unet_reg": "models/unet_reg.pt"}

ABLATION_VARIANTS: dict[str, dict] = {
    "full": {},
    "fixed_dose": {"randomize_dose": False},
    "fixed_geometry": {"randomize_geometry": False},
    "no_defects": {"include_defects": False},
    "no_scan_artifacts": {"include_scan_artifacts": False},
}


def _cmd_simulate(args: argparse.Namespace) -> None:
    config = preset_config(args.preset, size=args.size, dose=args.dose, rotation_deg=args.rotation)
    result = simulate_image(config, np.random.default_rng(args.seed))
    if args.out:
        save_sample(args.out, result)
        print(f"wrote {args.out}  ({len(result.positions)} columns)")
    if args.figure:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(result.image, cmap="gray", interpolation="nearest")
        ax.set_title(f"{args.preset}, dose {args.dose:g}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150)
        print(f"saved {args.figure}")
    if not args.out and not args.figure:
        print("nothing to do: pass --out and/or --figure")


def _load_input(path: Path, invert: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (image, truth or None) from an .npz sample or a real image."""
    if path.suffix == ".npz":
        sample = load_sample(path)
        return sample.image, sample.positions
    return crop_to_multiple(load_image(path, invert=invert)), None


def _cmd_detect(args: argparse.Namespace) -> None:
    image, truth = _load_input(Path(args.image), args.invert)
    methods = build_methods([args.method], {args.method: args.model})
    method = methods[args.method]
    threshold = args.threshold if args.threshold is not None else method.default_threshold
    peaks = method.detect(image, args.sigma, threshold)
    refine = REFINERS[args.refine]
    if refine is not None and len(peaks):
        peaks = refine(image, peaks)
    print(f"{len(peaks)} columns detected ({method.label}, threshold {threshold:g})")

    if truth is not None:
        pred = peaks[margin_mask(peaks, image.shape, MARGIN)] if len(peaks) else peaks
        true = truth[margin_mask(truth, image.shape, MARGIN)]
        res = match_positions(true, pred, TOLERANCE)
        print(
            f"vs ground truth: P {res.precision:.3f}  R {res.recall:.3f}  "
            f"F1 {res.f1:.3f}  RMSE {res.rmse:.2f} px"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.out, peaks, delimiter=",", header="row,col", comments="")
        print(f"wrote {args.out}")
    if args.figure:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(image, cmap="gray", interpolation="nearest")
        if truth is not None:
            ax.scatter(
                truth[:, 1],
                truth[:, 0],
                s=26,
                facecolors="none",
                edgecolors="#4dc3ff",
                linewidths=0.9,
                label="ground truth",
            )
        if len(peaks):
            ax.scatter(
                peaks[:, 1],
                peaks[:, 0],
                s=14,
                marker="x",
                c="#ff5c39",
                lw=0.9,
                label=method.label,
            )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150)
        print(f"saved {args.figure}")


def _train_one(settings: TrainSettings, out: Path, figure: Path | None) -> None:
    model, history = train_unet(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {out}  ({n_params} parameters, final loss {history[-1]:.4f})")
    if figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history, lw=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"training loss ({settings.target})")
        fig.tight_layout()
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=150)
        plt.close(fig)
        print(f"saved {figure}")


def _cmd_train(args: argparse.Namespace) -> None:
    if args.ablation:
        for variant, overrides in ABLATION_VARIANTS.items():
            print(f"-- ablation variant: {variant}")
            settings = TrainSettings(
                steps=args.steps, seed=args.seed, target="segmentation", **overrides
            )
            _train_one(settings, Path("models/ablation") / f"{variant}.pt", None)
        return
    settings = TrainSettings(steps=args.steps, seed=args.seed, target=args.target)
    default_out = "models/unet_seg.pt" if args.target == "segmentation" else "models/unet_reg.pt"
    out = Path(args.out or default_out)
    figure = Path(args.figure) if args.figure else None
    _train_one(settings, out, figure)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    for config in args.configs:
        payload = run_config(config)
        name = payload["config"].get("name", Path(config).stem)
        plot_payload(payload, Path("figures") / f"{name}.png")


def _cmd_samples(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, preset, dose, seed in SAMPLES:
        config = preset_config(preset, size=256, dose=dose, rotation_deg=12.0)
        result = simulate_image(config, np.random.default_rng(seed))
        save_sample(out / f"{name}.npz", result)
        print(f"wrote {out / (name + '.npz')}  ({len(result.positions)} columns)")


def _cmd_demo(args: argparse.Namespace) -> None:
    method_names = ["log", "unet"]
    methods = build_methods(method_names, DEFAULT_MODELS)
    files = [Path(args.data) / f"{name}.npz" for name, *_ in SAMPLES]

    all_metrics: dict[str, dict] = {}
    fig, axes = plt.subplots(len(files), 2, figsize=(9.0, 4.4 * len(files)))
    for row, path in enumerate(files):
        sample = load_sample(path)
        image, truth = sample.image, sample.positions
        true = truth[margin_mask(truth, image.shape, MARGIN)]
        all_metrics[path.stem] = {}

        panels: list[tuple[str, np.ndarray | None, object]] = []
        for name in method_names:
            method = methods[name]
            peaks = method.detect(image, sample.config.probe_sigma, method.default_threshold)
            if len(peaks):
                peaks = REFINERS["com"](image, peaks)
            pred = peaks[margin_mask(peaks, image.shape, MARGIN)] if len(peaks) else peaks
            res = match_positions(true, pred, TOLERANCE)
            all_metrics[path.stem][name] = {
                "n_true": res.n_true,
                "n_pred": res.n_pred,
                "precision": round(res.precision, 4),
                "recall": round(res.recall, 4),
                "f1": round(res.f1, 4),
                "rmse_px": round(res.rmse, 4) if np.isfinite(res.rmse) else None,
            }
            panels.append((method.label, pred, res))
            print(
                f"{path.stem:14s} {name:5s}  P {res.precision:.3f}  R {res.recall:.3f}  "
                f"F1 {res.f1:.3f}  RMSE {res.rmse:.2f} px"
            )

        for col, (label, pred, res) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", interpolation="nearest")
            ax.scatter(
                true[:, 1],
                true[:, 0],
                s=24,
                facecolors="none",
                edgecolors="#4dc3ff",
                linewidths=0.8,
            )
            title = f"{path.stem}: {label}"
            if pred is not None and res is not None:
                ax.scatter(pred[:, 1], pred[:, 0], s=12, marker="x", c="#ff5c39", lw=0.9)
                title += f" (F1 {res.f1:.3f})"
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=100)
    print(f"saved {figure}")

    metrics = Path(args.metrics)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tolerance_px": TOLERANCE, "margin_px": MARGIN, "samples": all_metrics}
    metrics.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {metrics}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the atomfinder console command."""
    parser = argparse.ArgumentParser(prog="atomfinder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate a materials preset")
    p.add_argument("--preset", default="graphene")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--dose", type=float, default=125.0)
    p.add_argument("--rotation", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help=".npz output with ground truth")
    p.add_argument("--figure", default=None, help=".png rendering")
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("detect", help="detect columns in an image")
    p.add_argument("image", help=".npz sample or real .png/.tif/.jpg image")
    p.add_argument("--method", default="log", help="log | localmax | ncc | unet | unet_reg")
    p.add_argument("--model", default="models/unet_seg.pt")
    p.add_argument("--sigma", type=float, default=2.6, help="column width in px")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--refine", default="com", choices=list(REFINERS))
    p.add_argument("--invert", action="store_true", help="invert real-image contrast")
    p.add_argument("--out", default=None, help="positions .csv")
    p.add_argument("--figure", default=None, help="overlay .png")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("train", help="train the learned detectors")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", default="segmentation", choices=["segmentation", "regression"])
    p.add_argument("--out", default=None)
    p.add_argument("--figure", default=None)
    p.add_argument("--ablation", action="store_true", help="train all ablation variants")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("benchmark", help="run YAML benchmark configs")
    p.add_argument("configs", nargs="+")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("samples", help="regenerate the committed samples")
    p.add_argument("--out", default="data/sample")
    p.set_defaults(func=_cmd_samples)

    p = sub.add_parser("demo", help="detect on every committed sample")
    p.add_argument("--data", default="data/sample")
    p.add_argument("--figure", default="figures/detection_overlay.png")
    p.add_argument("--metrics", default="results/metrics.json")
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
