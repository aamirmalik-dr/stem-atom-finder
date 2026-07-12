# Data

Everything in this project is synthetic. There is no external dataset to
download and no license restriction on the committed files.

## Committed samples

`data/sample/` holds three simulated 256 px HAADF-STEM images produced by
the project's own simulator with fixed seeds:

| File | Dose (mean counts at a column peak) | Seed |
|---|---|---|
| `high_dose.npz` | 1000 | 0 |
| `mid_dose.npz` | 200 | 1 |
| `low_dose.npz` | 50 | 2 |

Each `.npz` contains the noisy image, the ground-truth column positions
(`positions`, float (row, col) pixels), a dopant flag per column
(`is_dopant`), the removed lattice sites (`vacancies`), and the full
simulation config as JSON.

## Regenerating

```
python scripts/generate_sample.py
```

rewrites the three files bit-for-bit, since the seeds are fixed in the
script. Training data is never stored at all: `scripts/train_unet.py`
simulates a fresh batch at every optimisation step.
