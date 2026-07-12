"""Synthetic HAADF-STEM image simulator.

Generates atomic-resolution annular dark-field images of a 2D crystal with
known ground-truth atomic column positions. The model is deliberately simple
but physics-motivated:

- Column intensity scales as Z**1.7 (incoherent Z-contrast imaging).
- The finite probe is modelled as a Gaussian point-spread function.
- Vacancies remove columns, substitutional dopants change column intensity.
- Small random static displacements mimic relaxation and thermal disorder.
- Slow horizontal scan jitter mimics fly-back and drift error.
- Shot noise is Poisson, controlled by an electron dose parameter.

Coordinates are float (row, col) pixels throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

Z_EXPONENT = 1.7


@dataclass
class SimConfig:
    """Configuration for one simulated HAADF image.

    Attributes:
        size: Image side length in pixels (square image).
        lattice: "hexagonal" or "square".
        spacing: Nearest-neighbour column spacing in pixels.
        rotation_deg: Lattice rotation. If None, drawn uniformly at random.
        host_z: Atomic number of the host species.
        dopant_z: Atomic number of the substitutional dopant.
        dopant_fraction: Fraction of columns replaced by the dopant.
        vacancy_fraction: Fraction of columns removed entirely.
        probe_sigma: Gaussian probe standard deviation in pixels.
        displacement_sigma: Static random displacement of columns in pixels.
        jitter_sigma: Row-to-row scan jitter amplitude in pixels.
        background: Constant background as a fraction of host column peak.
        dose: Mean electron counts at a host column peak. Lower is noisier.
    """

    size: int = 256
    lattice: str = "hexagonal"
    spacing: float = 14.0
    rotation_deg: float | None = None
    host_z: int = 42
    dopant_z: int = 74
    dopant_fraction: float = 0.05
    vacancy_fraction: float = 0.05
    probe_sigma: float = 2.6
    displacement_sigma: float = 0.35
    jitter_sigma: float = 0.4
    background: float = 0.08
    dose: float = 500.0


@dataclass
class SimResult:
    """A simulated image with its ground truth.

    Attributes:
        image: Float32 image, normalised so the noise-free host peak is ~1.
        positions: (N, 2) float array of column centres as (row, col).
        is_dopant: (N,) bool array, True where the column is a dopant.
        vacancies: (M, 2) float array of removed lattice sites.
        config: The configuration that produced this image.
    """

    image: np.ndarray
    positions: np.ndarray
    is_dopant: np.ndarray
    vacancies: np.ndarray
    config: SimConfig = field(repr=False)


def _lattice_points(config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Return (N, 2) lattice points as (row, col) covering the image."""
    a = config.spacing
    if config.lattice == "hexagonal":
        v1 = np.array([0.0, a])
        v2 = np.array([a * np.sqrt(3.0) / 2.0, a / 2.0])
    elif config.lattice == "square":
        v1 = np.array([0.0, a])
        v2 = np.array([a, 0.0])
    else:
        raise ValueError(f"unknown lattice type: {config.lattice!r}")

    theta = (
        np.deg2rad(config.rotation_deg)
        if config.rotation_deg is not None
        else rng.uniform(0.0, np.pi / 3.0)
    )
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v1, v2 = rot @ v1, rot @ v2

    n = int(np.ceil(2.0 * config.size / a)) + 2
    ii, jj = np.meshgrid(np.arange(-n, n), np.arange(-n, n), indexing="ij")
    pts = ii.reshape(-1, 1) * v1 + jj.reshape(-1, 1) * v2
    pts += rng.uniform(0.0, a, size=2) + config.size / 2.0

    pad = 2.0 * config.probe_sigma
    inside = np.all((pts >= -pad) & (pts <= config.size - 1 + pad), axis=1)
    return pts[inside]


def _splat(shape: tuple[int, int], points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bilinearly deposit point weights onto a pixel grid."""
    img = np.zeros(shape, dtype=np.float64)
    r, c = points[:, 0], points[:, 1]
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = r - r0, c - c0
    for dr, dc, w in (
        (0, 0, (1 - fr) * (1 - fc)),
        (0, 1, (1 - fr) * fc),
        (1, 0, fr * (1 - fc)),
        (1, 1, fr * fc),
    ):
        rr, cc = r0 + dr, c0 + dc
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        np.add.at(img, (rr[ok], cc[ok]), weights[ok] * w[ok])
    return img


def simulate_image(config: SimConfig, rng: np.random.Generator | None = None) -> SimResult:
    """Simulate one HAADF-STEM image with ground-truth column positions.

    Args:
        config: Simulation parameters.
        rng: NumPy random generator. A fixed generator gives a fixed image.

    Returns:
        A SimResult holding the noisy image and the ground truth.
    """
    if rng is None:
        rng = np.random.default_rng()

    pts = _lattice_points(config, rng)
    n = len(pts)

    keep = rng.uniform(size=n) >= config.vacancy_fraction
    vacancies = pts[~keep]
    pts = pts[keep]

    is_dopant = rng.uniform(size=len(pts)) < config.dopant_fraction
    pts = pts + rng.normal(0.0, config.displacement_sigma, size=pts.shape)

    # Peak amplitude of a splatted-then-blurred unit weight is
    # 1 / (2 pi sigma^2); scale weights so a host column peaks at ~1.
    peak_norm = 2.0 * np.pi * config.probe_sigma**2
    z_ratio = (config.dopant_z / config.host_z) ** Z_EXPONENT
    weights = np.where(is_dopant, z_ratio, 1.0) * peak_norm

    clean = _splat((config.size, config.size), pts, weights)
    clean = gaussian_filter(clean, sigma=config.probe_sigma, mode="constant")
    clean += config.background

    # Slow scan jitter: shift each row horizontally, and shift the ground
    # truth of every column by the jitter of its own row.
    jitter = gaussian_filter(rng.normal(0.0, 1.0, size=config.size), sigma=6.0)
    scale = jitter.std()
    jitter = jitter * (config.jitter_sigma / scale) if scale > 0 else jitter * 0.0
    cols = np.arange(config.size, dtype=np.float64)
    jittered = np.empty_like(clean)
    for row in range(config.size):
        jittered[row] = np.interp(cols - jitter[row], cols, clean[row])
    row_idx = np.clip(np.round(pts[:, 0]).astype(int), 0, config.size - 1)
    pts = pts.copy()
    pts[:, 1] += jitter[row_idx]

    noisy = rng.poisson(np.clip(jittered, 0.0, None) * config.dose) / config.dose

    inside = np.all((pts >= 0.0) & (pts <= config.size - 1.0), axis=1)
    return SimResult(
        image=noisy.astype(np.float32),
        positions=pts[inside],
        is_dopant=is_dopant[inside],
        vacancies=vacancies,
        config=config,
    )
