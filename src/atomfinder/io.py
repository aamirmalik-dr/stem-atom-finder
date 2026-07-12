"""Saving and loading simulated samples as .npz files."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from atomfinder.sim import SimConfig, SimResult


def save_sample(path: str | Path, result: SimResult) -> None:
    """Write a SimResult to a compressed .npz file.

    Args:
        path: Destination file path.
        result: The simulated image and ground truth to store.
    """
    np.savez_compressed(
        path,
        image=result.image,
        positions=result.positions,
        is_dopant=result.is_dopant,
        vacancies=result.vacancies,
        config=json.dumps(dataclasses.asdict(result.config)),
    )


def load_sample(path: str | Path) -> SimResult:
    """Load a SimResult previously written by save_sample.

    Args:
        path: Path to the .npz file.

    Returns:
        The reconstructed SimResult.
    """
    with np.load(path, allow_pickle=False) as data:
        config = SimConfig(**json.loads(str(data["config"])))
        return SimResult(
            image=data["image"],
            positions=data["positions"],
            is_dopant=data["is_dopant"],
            vacancies=data["vacancies"],
            config=config,
        )
