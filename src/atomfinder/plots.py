"""Figures for benchmark results.

Each function takes the JSON payload produced by atomfinder.benchmark
and writes one PNG. Colors are fixed per method so figures stay
comparable across benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {
    "log": "#4dc3ff",
    "log_oracle": "#1f77b4",
    "localmax": "#9bb0bd",
    "ncc": "#7bc96f",
    "unet": "#ff5c39",
    "unet_reg": "#c44e00",
}
METHOD_LABELS = {
    "log": "LoG",
    "log_oracle": "LoG (oracle-tuned)",
    "localmax": "Local max",
    "ncc": "NCC template",
    "unet": "U-Net (seg)",
    "unet_reg": "U-Net (reg)",
}


_FALLBACK = ("#7f3fbf", "#2ca089", "#d4a017", "#b03a5b", "#5470c6", "#3f7f3f")


def _style(name: str, index: int = 0) -> dict:
    base = name.replace("_oracle", "").split("@")[0]
    color = METHOD_COLORS.get(name) or METHOD_COLORS.get(base)
    if "@" in name or color is None:
        color = _FALLBACK[index % len(_FALLBACK)]
    label = METHOD_LABELS.get(name, name)
    if "@" in name:
        label = f"U-Net [{name.split('@', 1)[1]}]"
    return {
        "color": color,
        "label": label,
        "linestyle": "--" if name.endswith("_oracle") else "-",
        "marker": "o",
        "markersize": 4,
    }


def _log_x(parameter: str) -> bool:
    return parameter in ("dose",)


def plot_sweep(payload: dict, path: str | Path) -> None:
    """F1 and position RMSE versus the swept parameter, per method."""
    parameter = payload["parameter"]
    rows = payload["rows"]
    values = [r[parameter] for r in rows]
    names = [k for k in rows[0] if k != parameter]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, name in enumerate(names):
        ax1.plot(values, [r[name]["f1"] for r in rows], **_style(name, i))
        ax2.plot(values, [r[name]["rmse_px"] for r in rows], **_style(name, i))
    for ax, ylabel in ((ax1, "detection F1"), (ax2, "position RMSE (px)")):
        if _log_x(parameter):
            ax.set_xscale("log")
        ax.set_xlabel(parameter)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8)
    fig.suptitle(payload.get("config", {}).get("title", f"Sweep over {parameter}"))
    fig.tight_layout()
    _save(fig, path)


def plot_pr_curves(payload: dict, path: str | Path) -> None:
    """Precision-recall curves, one panel per condition."""
    curves = payload["curves"]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 4.2), squeeze=False)
    for ax, entry in zip(axes[0], curves):
        for name, points in entry["methods"].items():
            pts = sorted(points, key=lambda p: p["recall"])
            style = _style(name)
            ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], **style)
        label = ", ".join(f"{k}={v:g}" for k, v in entry["condition"].items())
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Precision-recall across each method's threshold grid")
    fig.tight_layout()
    _save(fig, path)


def plot_operating_point(payload: dict, path: str | Path) -> None:
    """Fixed-threshold versus oracle F1 per condition, plus mean penalty."""
    values = payload["values"]
    methods = payload["methods"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, entry in methods.items():
        style = _style(name)
        ax1.plot(values, entry["fixed_f1_per_condition"], **style)
        ax1.plot(
            values,
            entry["oracle_f1_per_condition"],
            color=style["color"],
            linestyle=":",
            alpha=0.7,
        )
    if _log_x(payload["parameter"]):
        ax1.set_xscale("log")
    ax1.set_xlabel(payload["parameter"])
    ax1.set_ylabel("detection F1")
    ax1.set_title("solid: best single fixed threshold, dotted: oracle per condition")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    names = list(methods)
    penalties = [methods[n]["mean_penalty"] for n in names]
    ax2.bar(
        range(len(names)),
        penalties,
        color=[_style(n)["color"] for n in names],
    )
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([METHOD_LABELS.get(n, n) for n in names], rotation=20, fontsize=8)
    ax2.set_ylabel("mean F1 penalty (oracle - fixed)")
    ax2.set_title("Cost of a single fixed threshold")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, path)


def plot_refinement(payload: dict, path: str | Path) -> None:
    """Position RMSE versus the swept parameter for each refiner."""
    parameter = payload["parameter"]
    rows = payload["rows"]
    values = [r[parameter] for r in rows]
    colors = {"none": "#9bb0bd", "com": "#4dc3ff", "gauss": "#ff5c39"}
    labels = {
        "none": "integer pixel (no refinement)",
        "com": "centre of mass",
        "gauss": "2D Gaussian fit",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for refiner in payload["refiners"]:
        ax.plot(
            values,
            [r[refiner] for r in rows],
            "o-",
            color=colors.get(refiner, "#666666"),
            label=labels.get(refiner, refiner),
        )
    if _log_x(parameter):
        ax.set_xscale("log")
    ax.set_xlabel(parameter)
    ax.set_ylabel("position RMSE (px)")
    ax.set_title("Sub-pixel refiner accuracy from integer-pixel starts")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_materials(payload: dict, path: str | Path) -> None:
    """Per-preset F1 bars plus per-species recall for the multi-species presets."""
    rows = payload["rows"]
    presets = [r["preset"] for r in rows]
    names = [k for k in rows[0] if k != "preset"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    width = 0.8 / len(names)
    x = np.arange(len(presets))
    for j, name in enumerate(names):
        ax1.bar(
            x + j * width,
            [r[name]["f1"] for r in rows],
            width,
            color=_style(name)["color"],
            label=METHOD_LABELS.get(name, name),
        )
    ax1.set_xticks(x + 0.4 - width / 2)
    ax1.set_xticklabels(presets)
    ax1.set_ylabel("detection F1")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("F1 per materials preset")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    bars, labels, colors = [], [], []
    for r in rows:
        for name in names:
            for sp, rec in r[name].get("species_recall", {}).items():
                if len(r[name]["species_recall"]) > 1:
                    bars.append(rec)
                    labels.append(f"{r['preset']}:{sp} ({METHOD_LABELS.get(name, name)})")
                    colors.append(_style(name)["color"])
    y = np.arange(len(bars))
    ax2.barh(y, bars, color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("recall")
    ax2.set_xlim(0, 1.05)
    ax2.set_title("Per-species recall (multi-species presets)")
    ax2.grid(alpha=0.3, axis="x")
    ax2.invert_yaxis()
    fig.tight_layout()
    _save(fig, path)


PLOTTERS = {
    "sweep": plot_sweep,
    "pr_curves": plot_pr_curves,
    "operating_point": plot_operating_point,
    "refinement": plot_refinement,
    "materials": plot_materials,
}


def plot_payload(payload: dict, path: str | Path) -> None:
    """Dispatch to the plotter for the payload's benchmark mode."""
    PLOTTERS[payload["mode"]](payload, path)


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")
