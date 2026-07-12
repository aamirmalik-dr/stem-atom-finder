"""Generate the committed sample images.

Writes three 256 px simulated HAADF images at high, medium and low dose
to data/sample/, each with its ground-truth column positions. All three
are fully synthetic and reproducible from the fixed seeds below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atomfinder.io import save_sample
from atomfinder.sim import SimConfig, simulate_image

SAMPLES = {
    "high_dose": (500.0, 0),
    "mid_dose": (30.0, 1),
    "low_dose": (2.0, 2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (dose, seed) in SAMPLES.items():
        config = SimConfig(size=256, rotation_deg=12.0, dose=dose)
        result = simulate_image(config, np.random.default_rng(seed))
        path = args.out / f"{name}.npz"
        save_sample(path, result)
        print(f"wrote {path}  ({len(result.positions)} columns, dose {dose:g})")


if __name__ == "__main__":
    main()
