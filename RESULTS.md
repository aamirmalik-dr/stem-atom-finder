# Results

All numbers below were produced by running the committed scripts in a fresh
Python 3.11 virtual environment with CPU-only torch 2.13.0. Nothing is
estimated or copied from elsewhere. Regenerate with the commands shown.

## Dose sweep (`python scripts/benchmark_dose.py`)

5 simulated 256 px images per dose, Hungarian matching at 4 px tolerance,
8 px border margin excluded. LoG appears twice: with its fixed default
threshold (0.25), and oracle-tuned, where the threshold is picked per dose
from {0.10 ... 0.40} to maximise mean F1 using ground truth. The U-Net uses
one fixed heatmap threshold (0.4) at every dose. Raw values in
`results/dose_sweep.json`.

| Dose | LoG fixed F1 | LoG oracle F1 | LoG oracle threshold | U-Net F1 | LoG oracle RMSE (px) | U-Net RMSE (px) |
|---|---|---|---|---|---|---|
| 1 | 0.871 | 0.980 | 0.15 | 0.997 | 0.87 | 0.82 |
| 2 | 0.921 | 0.997 | 0.10 | 0.998 | 0.63 | 0.59 |
| 4 | 0.980 | 0.997 | 0.10 | 0.997 | 0.45 | 0.43 |
| 8 | 0.997 | 0.998 | 0.10 | 0.998 | 0.33 | 0.32 |
| 30 | 0.999 | 0.999 | 0.10 | 0.999 | 0.20 | 0.20 |
| 125 | 1.000 | 1.000 | 0.10 | 1.000 | 0.13 | 0.13 |
| 500 | 1.000 | 1.000 | 0.10 | 1.000 | 0.09 | 0.09 |
| 2000 | 1.000 | 1.000 | 0.10 | 1.000 | 0.08 | 0.08 |

## Committed samples (`python scripts/demo.py`)

Detection on the three committed images in `data/sample/`, after
centre-of-mass refinement. Raw values in `results/metrics.json`.

| Sample | Dose | Method | Precision | Recall | F1 | RMSE (px) |
|---|---|---|---|---|---|---|
| high_dose | 500 | LoG | 1.000 | 1.000 | 1.000 | 0.09 |
| high_dose | 500 | U-Net | 1.000 | 1.000 | 1.000 | 0.09 |
| mid_dose | 30 | LoG | 1.000 | 1.000 | 1.000 | 0.20 |
| mid_dose | 30 | U-Net | 1.000 | 1.000 | 1.000 | 0.21 |
| low_dose | 2 | LoG | 0.997 | 0.943 | 0.969 | 0.64 |
| low_dose | 2 | U-Net | 0.997 | 1.000 | 0.998 | 0.60 |

## Training (`python scripts/train_unet.py`)

300 steps, batch 8, Adam at 2e-3, seed 0, fresh simulated 128 px images
every step with dose drawn log-uniform from 2 to 2000. BCE loss fell from
0.43 (steps 1-25 mean) to 0.18 (steps 276-300 mean); the committed
`figures/training_loss.png` is the actual curve. The committed
`models/unet_atoms.pt` (29,641 parameters, 143 KB) is the model these
results use.

## Reading the result

The oracle-tuned LoG row is the important control. Without it the benchmark
would suggest the U-Net detects columns far better at low dose (0.997 vs
0.871 at dose 1). With it, most of that gap is revealed as threshold
sensitivity rather than detection power: a retuned classical detector
reaches 0.980. The U-Net's genuine advantages in this simulation are a
small residual edge at the lowest dose and, more usefully, one fixed
operating point that works across the entire sweep.
