"""Training loop for the column-detection U-Net.

Training data is generated on the fly by the simulator with randomised
lattice spacing, rotation, defect rates and dose, so the network never
sees the same image twice and learns to be robust across imaging
conditions instead of memorising one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from atomfinder.net import UNet, heatmap_target, normalize_image
from atomfinder.sim import SimConfig, simulate_image


def random_config(rng: np.random.Generator, size: int = 128) -> SimConfig:
    """Draw a randomised simulation config for domain-randomised training."""
    return SimConfig(
        size=size,
        lattice=str(rng.choice(["hexagonal", "square"])),
        spacing=float(rng.uniform(11.0, 18.0)),
        rotation_deg=None,
        dopant_fraction=float(rng.uniform(0.0, 0.10)),
        vacancy_fraction=float(rng.uniform(0.0, 0.10)),
        probe_sigma=float(rng.uniform(2.2, 3.0)),
        displacement_sigma=float(rng.uniform(0.2, 0.5)),
        jitter_sigma=float(rng.uniform(0.0, 0.6)),
        background=float(rng.uniform(0.04, 0.12)),
        dose=float(np.exp(rng.uniform(np.log(2.0), np.log(2000.0)))),
    )


def make_batch(
    rng: np.random.Generator, batch_size: int, size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a batch of images and their heatmap targets."""
    images, targets = [], []
    for _ in range(batch_size):
        result = simulate_image(random_config(rng, size), rng)
        images.append(normalize_image(result.image))
        targets.append(heatmap_target(result.image.shape, result.positions))
    x = torch.from_numpy(np.stack(images))[:, None]
    y = torch.from_numpy(np.stack(targets))[:, None]
    return x, y


def train_unet(
    steps: int = 300,
    batch_size: int = 8,
    lr: float = 2e-3,
    seed: int = 0,
    size: int = 128,
    log_every: int = 25,
) -> tuple[UNet, list[float]]:
    """Train the U-Net on freshly simulated data.

    Args:
        steps: Number of optimisation steps (one fresh batch each).
        batch_size: Images per batch.
        lr: Adam learning rate.
        seed: Seed for both simulation and weight initialisation.
        size: Training patch side length in pixels.
        log_every: Print running loss every this many steps.

    Returns:
        The trained model and the per-step loss history.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    history: list[float] = []
    for step in range(1, steps + 1):
        x, y = make_batch(rng, batch_size, size)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if log_every and step % log_every == 0:
            recent = float(np.mean(history[-log_every:]))
            print(f"step {step:4d}/{steps}  loss {recent:.4f}")
    return model, history
