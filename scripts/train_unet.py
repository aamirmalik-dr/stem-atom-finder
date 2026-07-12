"""Train the column-detection U-Net on simulated data and save the weights.

CPU-friendly: the default 300 steps of batch 8 takes a few minutes and
produces the committed models/unet_atoms.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from atomfinder.train import train_unet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models/unet_atoms.pt"))
    parser.add_argument("--figure", type=Path, default=Path("figures/training_loss.png"))
    args = parser.parse_args()

    model, history = train_unet(steps=args.steps, batch_size=args.batch_size, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {args.out}  ({n_params} parameters)")

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("BCE loss")
    ax.set_title("U-Net training loss (fresh simulated batch per step)")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
