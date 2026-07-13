# Model card: atomfinder U-Net detectors

## What these files are

| File | Head | Loss | Size |
|---|---|---|---|
| `unet_seg.pt` | sigmoid (segmentation) | BCE vs Gaussian-disk heatmap | 143 KB |
| `unet_reg.pt` | linear (regression) | MSE vs the same heatmap | 143 KB |
| `ablation/*.pt` | sigmoid | BCE, one domain-randomisation component removed each | 143 KB each |

## Architecture

Two-level U-Net: two encoder blocks (8 and 16 channels), a 32-channel
bottleneck, transpose-convolution upsampling with skip connections, and a
1x1 output head. 29,641 parameters. Input is a single-channel image
standardised to zero mean and unit variance, sides divisible by 4. Output
is a per-pixel column score; detections are local maxima of the score map
above a threshold (default 0.4), refined by centre of mass.

## Training regime

300 Adam steps at lr 2e-3, batch 8, seed 0, on 128 px images simulated
fresh at every step (2400 images total, none repeated), by this
repository's own HAADF simulator. Domain randomisation per image:

- material: hexagonal, square, graphene, MoS2, SrTiO3 [001], FCC Pt [110]
- lattice constant x0.85 to x1.2 of the preset default, random rotation
- probe sigma 2.2 to 3.0 px, background 0.04 to 0.12
- vacancies 0 to 8%, dopants 0 to 8% (generic lattices), S monovacancies
  0 to 15% (MoS2), static displacements 0.2 to 0.5 px
- scan jitter 0 to 0.6 px, slow drift 0 to 4 px, background variation
  0 to 0.05
- dose log-uniform from 2 to 2000 counts at the brightest column peak

Training takes a few minutes on a CPU. Reproduce with
`atomfinder train --steps 300` (add `--target regression` or
`--ablation`).

## Measured performance (this repository's benchmark, seeds fixed)

On the doped hexagonal dose sweep (`configs/dose_sweep.yaml`), the
segmentation U-Net holds F1 0.945 at dose 1 and 0.997 or better from dose 8
up, at its fixed default threshold. On the materials benchmark
(`configs/materials.yaml`, dose 125), it is the only method that detects
any SrTiO3 oxygen columns (recall 0.635 vs 0.000 for LoG and NCC), at the
cost of precision (0.805 vs 0.999). Full numbers: results/*.json and
RESULTS.md.

The ablation (`configs/ablation.yaml`) shows dose randomisation is the
load-bearing component: training at one fixed dose drops dose-1 F1 from
0.996 to 0.792. Geometry randomisation and defect exposure contribute
smaller but real margins (0.962 and 0.975 at dose 1); scan-artifact
randomisation is not exercised by this test (0.995).

## Intended use and limitations

Intended: detecting atomic-column positions in atomic-resolution
ADF/HAADF-style images whose sampling roughly matches the training regime
(columns 2 to 3 px wide, 10 to 25 px apart), and serving as a reproducible
learned baseline for detector comparisons.

Not intended: species classification, images far outside the sampling
regime without resampling, bright-field or phase-contrast images (invert
and validate first), or any quantitative claim on experimental data
without validation against a trusted reference.

Domain gap: the models were trained purely on simulation with Poisson
noise, Gaussian probe, and clean vacuum background. Real images add
contamination, amorphous regions, detector artifacts, and non-Poisson
noise. On the committed real Al [001] image the detections are
qualitatively sensible after resampling (see the tutorial notebook), but
there is no ground truth there and no accuracy claim is made. Fine-tuning
on a few labelled experimental patches is the expected path to production
use.
