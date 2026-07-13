# Data

## Committed synthetic samples (data/sample/)

Four simulated 256 px HAADF-STEM images, one per materials preset, produced
by this repository's own simulator with fixed seeds. Fully synthetic, no
license restrictions.

| File | Preset | Dose (counts at brightest column peak) | Seed |
|---|---|---|---|
| `graphene_d50.npz` | graphene honeycomb | 50 | 0 |
| `mos2_d50.npz` | MoS2 monolayer (8% S monovacancies) | 50 | 1 |
| `srtio3_d125.npz` | SrTiO3 perovskite [001] | 125 | 2 |
| `fcc110_d10.npz` | FCC Pt [110] | 10 | 3 |

Each `.npz` contains the noisy image, ground-truth column positions
(`positions`, float (row, col) pixels), per-column species and relative
weights, a dopant flag, removed lattice sites (`vacancies`), and the full
simulation config as JSON. Regenerate bit-for-bit with:

```
atomfinder samples
```

Training data is never stored: the training loop simulates a fresh batch at
every optimisation step.

## Committed real image (data/real/)

`al_001_haadf_wikimedia.png` is an experimental high-resolution HAADF-STEM
micrograph of aluminium viewed along [001].

- Author: Matheustunes (own work), via Wikimedia Commons.
- Source: https://commons.wikimedia.org/wiki/File:Aluminium_Atomic_lattice.png
- License: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/
- The file is committed unmodified. This data license is separate from the
  repository's MIT code license and continues to apply to the image.

Practical notes: the image has a burned-in scale bar in the bottom rows
(crop to `[:448, :]` before detection) and coarser sampling than the
simulator defaults (columns roughly 40 px apart), so downsample about 3x
before using the committed models. The tutorial notebook shows the full
procedure. There is no ground truth for this image; results on it are
qualitative only.

Another clearly licensed test image (not committed, CC BY-SA 4.0, Hovden
Lab, University of Michigan): an atomic-resolution ADF image of 1T-TaS2 at
https://commons.wikimedia.org/wiki/File:Atomic_resolution_image_of_Tantalum_disulfide.png

## Bring your own image

Any PNG/TIFF/JPEG atomic-resolution image works:

```
atomfinder detect your_image.png --method unet --figure overlay.png
```

Use `--invert` for bright-field-like contrast, `--sigma` to match your
column width in pixels, and downsample or upsample first so columns are
roughly 5 to 8 px apart between centres at 2 to 3 px width.
