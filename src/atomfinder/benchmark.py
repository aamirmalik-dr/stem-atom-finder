"""Config-driven benchmark harness.

Every benchmark is a YAML file (see configs/) with a `mode` and fixed
seeds, so each committed number regenerates bit-for-bit. Modes:

- "sweep": vary one SimConfig parameter, score every method at its fixed
  default threshold; methods listed under `oracle` are additionally
  scored with the per-condition best threshold from their grid (an upper
  bound that uses ground-truth access).
- "pr_curves": trace precision-recall by sweeping each method's
  threshold grid at a few fixed conditions.
- "operating_point": on a threshold x condition grid, compare the best
  single fixed threshold per method against per-condition oracle tuning.
  The gap is the fixed-threshold penalty, the cost of not being allowed
  to retune.
- "refinement": start from ground-truth positions rounded to whole
  pixels and measure position RMSE after each sub-pixel refiner, which
  isolates refinement accuracy from detection quality.
- "materials": run every method on each named materials preset and
  report F1 plus per-species recall (e.g. the near-invisible O columns
  of SrTiO3).

Results are written to results/<name>.json; figures to
figures/<name>.png.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from atomfinder.detect import detect_local_max, detect_log, detect_ncc, peaks_from_heatmap
from atomfinder.metrics import margin_mask, match_positions, per_species_recall
from atomfinder.net import UNet, predict_heatmap
from atomfinder.refine import refine_com, refine_gaussian
from atomfinder.sim import SimConfig, SimResult, simulate_image

REFINERS: dict[str, Callable | None] = {
    "none": None,
    "com": refine_com,
    "gauss": refine_gaussian,
}


@dataclass
class Method:
    """One detection method: a detector plus its threshold semantics.

    Attributes:
        name: Registry key, e.g. "unet".
        detect: Callable (image, probe_sigma, threshold) -> (N, 2) peaks.
        default_threshold: The fixed operating point used in sweeps.
        grid: Threshold grid for PR curves and oracle tuning.
        label: Human-readable name for figures.
    """

    name: str
    detect: Callable[[np.ndarray, float, float], np.ndarray]
    default_threshold: float
    grid: tuple[float, ...]
    label: str


def _load_unet(path: str | Path) -> UNet:
    model = UNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def build_methods(names: list[str], model_paths: dict[str, str] | None = None) -> dict[str, Method]:
    """Build the requested subset of the method registry.

    Args:
        names: Method names to build. Classical: "log", "localmax",
            "ncc". Learned: "unet" (segmentation head), "unet_reg"
            (regression head); these need entries in model_paths. A
            learned method may carry a variant tag, e.g.
            "unet@fixed_dose", which loads model_paths["unet@fixed_dose"]
            with the base method's semantics (used by the ablation).
        model_paths: Mapping of learned-method name to weight file.

    Returns:
        Mapping of name to Method.
    """
    model_paths = model_paths or {}
    rel_grid = tuple(np.round(np.arange(0.05, 0.55, 0.05), 2))
    heat_grid = tuple(np.round(np.arange(0.10, 0.75, 0.05), 2))
    methods: dict[str, Method] = {}
    for name in names:
        base = name.split("@")[0]
        if name == "log":
            methods[name] = Method(
                name,
                lambda img, ps, t: detect_log(img, sigma=ps, threshold_rel=t),
                0.25,
                rel_grid,
                "LoG",
            )
        elif name == "localmax":
            methods[name] = Method(
                name,
                lambda img, ps, t: detect_local_max(img, sigma=ps, threshold_rel=t),
                0.25,
                rel_grid,
                "Local max",
            )
        elif name == "ncc":
            methods[name] = Method(
                name,
                lambda img, ps, t: detect_ncc(img, sigma=ps, threshold=t),
                0.45,
                tuple(np.round(np.arange(0.20, 0.75, 0.05), 2)),
                "NCC template",
            )
        elif base in ("unet", "unet_reg"):
            if name not in model_paths:
                raise ValueError(f"method {name!r} needs a model path")
            model = _load_unet(model_paths[name])
            activation = "sigmoid" if base == "unet" else "linear"
            label = "U-Net (seg)" if base == "unet" else "U-Net (reg)"
            if "@" in name:
                label = f"U-Net [{name.split('@', 1)[1]}]"
            methods[name] = Method(
                name,
                lambda img, ps, t, m=model, a=activation: peaks_from_heatmap(
                    predict_heatmap(m, img, activation=a), threshold=t
                ),
                0.4,
                heat_grid,
                label,
            )
        else:
            raise ValueError(f"unknown method: {name!r}")
    return methods


def _simulate_set(
    base: dict, overrides: dict, n_images: int, seed: int, condition_index: int
) -> list[SimResult]:
    """Simulate a reproducible image set for one sweep condition."""
    samples = []
    for k in range(n_images):
        config = SimConfig(**{**base, **overrides})
        rng = np.random.default_rng(seed + 7919 * condition_index + k)
        samples.append(simulate_image(config, rng))
    return samples


def _score(
    samples: list[SimResult],
    peaks_per_sample: list[np.ndarray],
    refiner: str,
    tolerance: float,
    margin: float,
) -> dict[str, float]:
    """Refine and score detections over an image set; return mean metrics."""
    refine = REFINERS[refiner]
    f1s, precisions, recalls, rmses = [], [], [], []
    for sample, peaks in zip(samples, peaks_per_sample):
        shape = sample.image.shape
        if refine is not None and len(peaks):
            peaks = refine(sample.image, peaks)
        pred = peaks[margin_mask(peaks, shape, margin)] if len(peaks) else peaks
        true = sample.positions[margin_mask(sample.positions, shape, margin)]
        res = match_positions(true, pred, tolerance)
        f1s.append(res.f1)
        precisions.append(res.precision)
        recalls.append(res.recall)
        if np.isfinite(res.rmse):
            rmses.append(res.rmse)
    return {
        "f1": float(np.mean(f1s)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "rmse_px": float(np.mean(rmses)) if rmses else float("nan"),
    }


def _detect_all(method: Method, samples: list[SimResult], threshold: float) -> list[np.ndarray]:
    return [method.detect(s.image, s.config.probe_sigma, threshold) for s in samples]


def _oracle_best(
    method: Method,
    samples: list[SimResult],
    refiner: str,
    tolerance: float,
    margin: float,
) -> tuple[float, dict[str, float]]:
    """Return (best threshold, metrics at it) over the method's grid."""
    best_t, best = None, None
    for t in method.grid:
        scores = _score(samples, _detect_all(method, samples, t), refiner, tolerance, margin)
        if best is None or scores["f1"] > best["f1"]:
            best_t, best = t, scores
    return float(best_t), best


def _common(config: dict) -> tuple[dict, int, int, float, float, str, dict[str, Method]]:
    base = dict(config.get("base_config", {}))
    n_images = int(config.get("images_per_condition", 5))
    seed = int(config.get("seed", 10))
    tolerance = float(config.get("tolerance", 4.0))
    margin = float(config.get("margin", 8.0))
    refiner = str(config.get("refine", "com"))
    methods = build_methods(list(config.get("methods", [])), config.get("models"))
    return base, n_images, seed, tolerance, margin, refiner, methods


def run_sweep(config: dict) -> dict:
    """Run a one-parameter sweep; see module docstring."""
    base, n_images, seed, tolerance, margin, refiner, methods = _common(config)
    parameter = config["sweep"]["parameter"]
    values = config["sweep"]["values"]
    oracle_names = list(config.get("oracle", []))

    rows = []
    for i, value in enumerate(values):
        samples = _simulate_set(base, {parameter: value}, n_images, seed, i)
        row: dict = {parameter: value}
        for name, method in methods.items():
            peaks = _detect_all(method, samples, method.default_threshold)
            row[name] = _score(samples, peaks, refiner, tolerance, margin)
        for name in oracle_names:
            t, scores = _oracle_best(methods[name], samples, refiner, tolerance, margin)
            row[f"{name}_oracle"] = {**scores, "threshold": t}
        rows.append(row)
        summary = "  ".join(f"{k} F1 {v['f1']:.3f}" for k, v in row.items() if k != parameter)
        print(f"{parameter} {value:g}: {summary}")
    return {"mode": "sweep", "parameter": parameter, "rows": rows}


def run_pr_curves(config: dict) -> dict:
    """Trace precision-recall curves over each method's threshold grid."""
    base, n_images, seed, tolerance, margin, refiner, methods = _common(config)
    conditions = config["conditions"]  # list of SimConfig override dicts

    curves = []
    for i, overrides in enumerate(conditions):
        samples = _simulate_set(base, overrides, n_images, seed, i)
        entry: dict = {"condition": overrides, "methods": {}}
        for name, method in methods.items():
            points = []
            for t in method.grid:
                scores = _score(
                    samples, _detect_all(method, samples, t), refiner, tolerance, margin
                )
                points.append(
                    {
                        "threshold": float(t),
                        "precision": scores["precision"],
                        "recall": scores["recall"],
                        "f1": scores["f1"],
                    }
                )
            entry["methods"][name] = points
        curves.append(entry)
        print(f"pr condition {overrides}: done")
    return {"mode": "pr_curves", "curves": curves}


def run_operating_point(config: dict) -> dict:
    """Quantify the fixed-threshold penalty per method across conditions."""
    base, n_images, seed, tolerance, margin, refiner, methods = _common(config)
    parameter = config["sweep"]["parameter"]
    values = config["sweep"]["values"]

    sample_sets = [
        _simulate_set(base, {parameter: v}, n_images, seed, i) for i, v in enumerate(values)
    ]
    analysis: dict = {}
    for name, method in methods.items():
        # F1 grid: thresholds x conditions.
        grid = np.array(
            [
                [
                    _score(s, _detect_all(method, s, t), refiner, tolerance, margin)["f1"]
                    for s in sample_sets
                ]
                for t in method.grid
            ]
        )
        oracle_f1 = grid.max(axis=0)
        best_fixed_idx = int(grid.mean(axis=1).argmax())
        fixed_f1 = grid[best_fixed_idx]
        analysis[name] = {
            "best_fixed_threshold": float(method.grid[best_fixed_idx]),
            "fixed_mean_f1": float(fixed_f1.mean()),
            "fixed_worst_f1": float(fixed_f1.min()),
            "oracle_mean_f1": float(oracle_f1.mean()),
            "oracle_worst_f1": float(oracle_f1.min()),
            "mean_penalty": float((oracle_f1 - fixed_f1).mean()),
            "max_penalty": float((oracle_f1 - fixed_f1).max()),
            "fixed_f1_per_condition": [float(x) for x in fixed_f1],
            "oracle_f1_per_condition": [float(x) for x in oracle_f1],
        }
        print(
            f"{name}: fixed thr {analysis[name]['best_fixed_threshold']:.2f} "
            f"mean F1 {analysis[name]['fixed_mean_f1']:.3f} "
            f"(oracle {analysis[name]['oracle_mean_f1']:.3f}, "
            f"mean penalty {analysis[name]['mean_penalty']:.3f})"
        )
    return {
        "mode": "operating_point",
        "parameter": parameter,
        "values": values,
        "methods": analysis,
    }


def run_refinement(config: dict) -> dict:
    """Isolate sub-pixel refiner accuracy from detection quality."""
    base, n_images, seed, tolerance, margin, _, _ = _common(config)
    parameter = config["sweep"]["parameter"]
    values = config["sweep"]["values"]
    refiners = list(config.get("refiners", ["none", "com", "gauss"]))

    rows = []
    for i, value in enumerate(values):
        samples = _simulate_set(base, {parameter: value}, n_images, seed, i)
        row: dict = {parameter: value}
        for refiner in refiners:
            # Start every refiner from the ground truth rounded to whole
            # pixels, i.e. a perfect detector with integer-pixel output.
            peaks = [np.round(s.positions) for s in samples]
            row[refiner] = _score(samples, peaks, refiner, tolerance, margin)["rmse_px"]
        rows.append(row)
        print(f"{parameter} {value:g}: " + "  ".join(f"{r} {row[r]:.3f} px" for r in refiners))
    return {"mode": "refinement", "parameter": parameter, "refiners": refiners, "rows": rows}


def run_materials(config: dict) -> dict:
    """Score every method on each materials preset, with species recall."""
    from atomfinder.sim import preset_config

    _, n_images, seed, tolerance, margin, refiner, methods = _common(config)
    presets = list(config.get("presets", []))
    overrides = dict(config.get("base_config", {}))

    rows = []
    for i, preset in enumerate(presets):
        base = dataclasses.asdict(preset_config(preset, **overrides))
        samples = _simulate_set(base, {}, n_images, seed, i)
        row: dict = {"preset": preset}
        for name, method in methods.items():
            peaks = _detect_all(method, samples, method.default_threshold)
            scores = _score(samples, peaks, refiner, tolerance, margin)
            recalls: dict[str, list[float]] = {}
            refine = REFINERS[refiner]
            for sample, pk in zip(samples, peaks):
                if refine is not None and len(pk):
                    pk = refine(sample.image, pk)
                shape = sample.image.shape
                tmask = margin_mask(sample.positions, shape, margin)
                pred = pk[margin_mask(pk, shape, margin)] if len(pk) else pk
                for sp, rec in per_species_recall(
                    sample.positions[tmask],
                    sample.species[tmask],
                    sample.species_names,
                    pred,
                    tolerance,
                ).items():
                    recalls.setdefault(sp, []).append(rec)
            scores["species_recall"] = {sp: float(np.nanmean(v)) for sp, v in recalls.items()}
            row[name] = scores
        rows.append(row)
        print(f"{preset}: " + "  ".join(f"{n} F1 {row[n]['f1']:.3f}" for n in methods))
    return {"mode": "materials", "rows": rows}


MODES = {
    "sweep": run_sweep,
    "pr_curves": run_pr_curves,
    "operating_point": run_operating_point,
    "refinement": run_refinement,
    "materials": run_materials,
}


def run_config(path: str | Path, out_dir: str | Path = "results") -> dict:
    """Run one YAML benchmark config and write results/<name>.json.

    Args:
        path: YAML config file.
        out_dir: Directory for the JSON result.

    Returns:
        The result payload (also written to disk).
    """
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    mode = config.get("mode", "sweep")
    if mode not in MODES:
        raise ValueError(f"unknown benchmark mode: {mode!r}")
    print(f"== {path.stem} ({mode}) ==")
    payload = MODES[mode](config)
    payload["config"] = config
    out = Path(out_dir) / f"{config.get('name', path.stem)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {out}")
    return payload
