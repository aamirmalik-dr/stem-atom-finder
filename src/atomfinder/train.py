"""Training loops for the learned detectors.

Training data is generated on the fly by the simulator with randomised
material, lattice spacing, rotation, defect rates, scan artifacts and
dose, so the network never sees the same image twice and learns to be
robust across imaging conditions instead of memorising one.

Two learned detectors share the same U-Net backbone:

- "segmentation": sigmoid head trained with binary cross-entropy against
  Gaussian disks at the column positions.
- "regression": linear head trained with mean squared error against the
  same Gaussian heatmap.

Each domain-randomisation component can be switched off individually,
which is what the ablation in the benchmark uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from atomfinder.net import UNet, heatmap_target, normalize_image
from atomfinder.sim import SimConfig, preset_config, simulate_image

TRAIN_LATTICES = ("hexagonal", "square", "graphene", "mos2", "srtio3", "fcc110")


@dataclass
class TrainSettings:
    """Training configuration.

    Attributes:
        steps: Optimisation steps (one fresh simulated batch each).
        batch_size: Images per batch.
        lr: Adam learning rate.
        seed: Seed for simulation and weight initialisation.
        size: Training patch side length in pixels.
        target: "segmentation" (BCE + sigmoid) or "regression" (MSE).
        randomize_dose: Draw dose log-uniform from [2, 2000]; if False,
            train at a fixed dose of 500.
        randomize_geometry: Draw material, spacing, rotation and probe
            size at random; if False, train on one fixed hexagonal
            lattice.
        include_defects: Randomise vacancy, dopant and partial-vacancy
            fractions; if False, simulate defect-free crystals.
        include_scan_artifacts: Randomise jitter and drift; if False,
            simulate a perfectly stable scan.
    """

    steps: int = 300
    batch_size: int = 8
    lr: float = 2e-3
    seed: int = 0
    size: int = 128
    target: str = "segmentation"
    randomize_dose: bool = True
    randomize_geometry: bool = True
    include_defects: bool = True
    include_scan_artifacts: bool = True


def random_config(
    rng: np.random.Generator, settings: TrainSettings | None = None, size: int | None = None
) -> SimConfig:
    """Draw one randomised simulation config for training.

    Args:
        rng: NumPy random generator.
        settings: Which domain-randomisation components are active.
        size: Patch size override; defaults to settings.size.

    Returns:
        A SimConfig sampled according to the active components.
    """
    settings = settings or TrainSettings()
    size = size or settings.size

    if settings.randomize_geometry:
        lattice = str(rng.choice(TRAIN_LATTICES))
        config = preset_config(lattice, size=size)
        config.spacing *= float(rng.uniform(0.85, 1.2))
        config.rotation_deg = None
        config.probe_sigma = float(rng.uniform(2.2, 3.0))
    else:
        config = preset_config("hexagonal", size=size, rotation_deg=12.0)

    if settings.include_defects:
        config.vacancy_fraction = float(rng.uniform(0.0, 0.08))
        if config.lattice in ("hexagonal", "square"):
            config.dopant_fraction = float(rng.uniform(0.0, 0.08))
        if config.lattice == "mos2":
            config.partial_vacancy_fraction = float(rng.uniform(0.0, 0.15))
        config.displacement_sigma = float(rng.uniform(0.2, 0.5))
    else:
        config.vacancy_fraction = 0.0
        config.dopant_fraction = 0.0
        config.partial_vacancy_fraction = 0.0
        config.displacement_sigma = 0.0

    if settings.include_scan_artifacts:
        config.jitter_sigma = float(rng.uniform(0.0, 0.6))
        config.drift_px = float(rng.uniform(0.0, 4.0))
        config.background_variation = float(rng.uniform(0.0, 0.05))
    else:
        config.jitter_sigma = 0.0
        config.drift_px = 0.0
        config.background_variation = 0.0

    config.background = float(rng.uniform(0.04, 0.12))
    config.dose = (
        float(np.exp(rng.uniform(np.log(2.0), np.log(2000.0))))
        if settings.randomize_dose
        else 500.0
    )
    return config


def make_batch(
    rng: np.random.Generator, settings: TrainSettings
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a batch of images and their heatmap targets."""
    images, targets = [], []
    for _ in range(settings.batch_size):
        result = simulate_image(random_config(rng, settings), rng)
        images.append(normalize_image(result.image))
        targets.append(heatmap_target(result.image.shape, result.positions))
    x = torch.from_numpy(np.stack(images))[:, None]
    y = torch.from_numpy(np.stack(targets))[:, None]
    return x, y


def train_unet(
    settings: TrainSettings | None = None, log_every: int = 25, **overrides
) -> tuple[UNet, list[float]]:
    """Train a U-Net detector on freshly simulated data.

    Args:
        settings: Training configuration; built from overrides if None.
        log_every: Print running loss every this many steps (0 = silent).
        **overrides: TrainSettings fields, convenience for
            train_unet(steps=300, target="regression").

    Returns:
        The trained model and the per-step loss history.
    """
    if settings is None:
        settings = TrainSettings(**overrides)
    if settings.target not in ("segmentation", "regression"):
        raise ValueError(f"unknown target style: {settings.target!r}")

    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    model = UNet()
    opt = torch.optim.Adam(model.parameters(), lr=settings.lr)
    loss_fn = nn.BCEWithLogitsLoss() if settings.target == "segmentation" else nn.MSELoss()

    model.train()
    history: list[float] = []
    for step in range(1, settings.steps + 1):
        x, y = make_batch(rng, settings)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if log_every and step % log_every == 0:
            recent = float(np.mean(history[-log_every:]))
            print(f"step {step:4d}/{settings.steps}  loss {recent:.4f}")
    return model, history
