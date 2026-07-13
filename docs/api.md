# Python API

Everything below is importable from the top-level package. Coordinates
are float (row, col) pixels throughout. All snippets run as-is after
`pip install -e .`.

## Simulate a material

```python
import numpy as np
from atomfinder import preset_config, simulate_image

config = preset_config("srtio3", size=256, dose=125.0)
result = simulate_image(config, np.random.default_rng(0))

result.image          # (256, 256) float32, heaviest column peaks at ~1
result.positions      # (N, 2) ground-truth column centres
result.species_names  # ("Sr", "TiO", "O")
result.species        # (N,) index into species_names
result.weights        # (N,) relative column weights, heaviest = 1
```

Presets: `graphene`, `mos2`, `srtio3`, `fcc110`, plus generic
`hexagonal` and `square`. Every physical knob is a `SimConfig` field:

```python
from atomfinder import SimConfig

config = SimConfig(
    lattice="mos2",
    spacing=22.0,                  # cell parameter in px
    dose=8.0,                      # counts at the brightest column peak
    partial_vacancy_fraction=0.1,  # sulfur monovacancies
    drift_px=6.0,                  # slow sample drift over the frame
    jitter_sigma=0.4,              # fast per-row scan jitter
    background_variation=0.05,     # smooth contamination-like background
)
```

## Detect columns

```python
from atomfinder import detect_log, detect_local_max, detect_ncc

peaks = detect_log(result.image, sigma=2.6, threshold_rel=0.25)
peaks = detect_local_max(result.image, sigma=2.6, threshold_rel=0.25)
peaks = detect_ncc(result.image, sigma=2.6, threshold=0.45)
```

The learned detectors load a committed 30k-parameter U-Net:

```python
import torch
from atomfinder import UNet, predict_heatmap, peaks_from_heatmap

model = UNet()
model.load_state_dict(torch.load("models/unet_seg.pt", weights_only=True))
heatmap = predict_heatmap(model, result.image)          # sigmoid head
peaks = peaks_from_heatmap(heatmap, threshold=0.4)
```

`models/unet_reg.pt` is the heatmap-regression variant; pass
`activation="linear"` to `predict_heatmap` for it.

## Refine to sub-pixel accuracy

```python
from atomfinder import refine_com, refine_gaussian

fast = refine_com(result.image, peaks)        # iterative centre of mass
best = refine_gaussian(result.image, peaks)   # 2D Gaussian least squares
```

## Score against ground truth

```python
from atomfinder import filter_margin, match_positions, per_species_recall

pred = filter_margin(best, result.image.shape, margin=8.0)
true = filter_margin(result.positions, result.image.shape, margin=8.0)
res = match_positions(true, pred, tolerance=4.0)
res.precision, res.recall, res.f1, res.rmse   # Hungarian one-to-one matching

recalls = per_species_recall(
    true, result.species[:len(true)], result.species_names, pred
)
```

Note: `per_species_recall` expects the species array aligned with the
positions you pass; filter both with the same mask (see
`atomfinder.metrics.margin_mask`).

## Train

```python
from atomfinder import TrainSettings, train_unet

model, history = train_unet(TrainSettings(steps=300, target="segmentation"))
```

`TrainSettings` exposes the domain-randomisation switches used by the
ablation: `randomize_dose`, `randomize_geometry`, `include_defects`,
`include_scan_artifacts`.

## Real images

```python
from atomfinder import load_image, crop_to_multiple

image = load_image("data/real/al_001_haadf_wikimedia.png")
image = image[:448, :]            # crop the burned-in scale bar
image = crop_to_multiple(image)   # U-Net needs sides divisible by 4
```

Match the pixel sampling before detecting: the committed models expect
columns roughly 2 to 3 px wide (sigma), so downsample coarse images
first. The tutorial notebook walks through this on the committed real
image.

## Benchmarks

```python
from atomfinder.benchmark import run_config
from atomfinder.plots import plot_payload

payload = run_config("configs/dose_sweep.yaml")
plot_payload(payload, "figures/dose_sweep.png")
```

Or from the shell: `atomfinder benchmark configs/*.yaml`. Each config is
plain YAML with fixed seeds; see configs/ for the eight committed ones.
