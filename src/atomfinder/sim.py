"""Synthetic HAADF-STEM image simulator.

Generates atomic-resolution annular dark-field images of 2D-projected
crystals with exact ground-truth column positions. The imaging model is
deliberately simple but physics-motivated:

- A column's scattering weight is the sum of Z**1.7 over the atoms it
  contains (incoherent Z-contrast imaging). Images are normalised so the
  heaviest column species peaks at ~1 before noise.
- The finite probe is a Gaussian point-spread function.
- Vacancies remove whole columns; partial vacancies remove one atom from
  a two-atom column (e.g. a sulfur monovacancy in MoS2), halving its
  weight while the column stays in the ground truth.
- Substitutional dopants replace a column's weight with the dopant's.
- Small random static displacements mimic relaxation and thermal disorder.
- Two scan artifacts: fast per-row horizontal jitter (fly-back error) and
  slow sample drift, a displacement that accumulates as the scan
  progresses down the frame. Both are applied as per-row offsets and the
  ground truth is shifted identically, to first order in the row
  coordinate (exact for constant drift, and accurate to O(drift^2/size)
  otherwise).
- The background is a documented two-term model: a constant pedestal
  (`background`, dark current plus diffuse scattering) plus an optional
  smooth low-frequency field (`background_variation`, mimicking thickness
  variation or surface contamination).
- Shot noise is Poisson, controlled by a single dose parameter: the mean
  electron counts at the peak of the heaviest column.

Coordinates are float (row, col) pixels throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

Z_EXPONENT = 1.7


def column_weight(*atomic_numbers: int) -> float:
    """Return the HAADF scattering weight of a column of the given atoms."""
    return float(sum(z**Z_EXPONENT for z in atomic_numbers))


@dataclass(frozen=True)
class ColumnSite:
    """One atomic column site in the 2D projected unit cell.

    Attributes:
        frac: Fractional (u, v) coordinates in the cell spanned by v1, v2.
        species: Column label, e.g. "Sr" or "S2".
        weight: HAADF scattering weight, sum of Z**1.7 over the column.
        reducible: True if the column holds two atoms and can lose one to
            a partial vacancy (weight halves, column stays).
    """

    frac: tuple[float, float]
    species: str
    weight: float
    reducible: bool = False


@dataclass(frozen=True)
class LatticeSpec:
    """A 2D projected crystal: cell vectors (units of the cell parameter
    `a`) plus a basis of column sites.

    Attributes:
        name: Preset name.
        v1: First cell vector as (row, col), in units of `a`.
        v2: Second cell vector as (row, col), in units of `a`.
        basis: Column sites in the cell.
        description: What the projection is and what `spacing` means.
    """

    name: str
    v1: tuple[float, float]
    v2: tuple[float, float]
    basis: tuple[ColumnSite, ...]
    description: str


_SQRT3 = float(np.sqrt(3.0))

PRESETS: dict[str, LatticeSpec] = {
    "graphene": LatticeSpec(
        name="graphene",
        v1=(0.0, 1.0),
        v2=(_SQRT3 / 2.0, 0.5),
        basis=(
            ColumnSite((0.0, 0.0), "C", column_weight(6)),
            ColumnSite((1.0 / 3.0, 1.0 / 3.0), "C", column_weight(6)),
        ),
        description=(
            "Graphene honeycomb. spacing = hexagonal cell parameter a "
            "(2.46 A); C-C bond = a/sqrt(3)."
        ),
    ),
    "mos2": LatticeSpec(
        name="mos2",
        v1=(0.0, 1.0),
        v2=(_SQRT3 / 2.0, 0.5),
        basis=(
            ColumnSite((0.0, 0.0), "Mo", column_weight(42)),
            ColumnSite((1.0 / 3.0, 1.0 / 3.0), "S2", column_weight(16, 16), reducible=True),
        ),
        description=(
            "MoS2 monolayer, plan view. spacing = cell parameter a (3.16 A). "
            "Mo and S2 columns alternate on the honeycomb; partial vacancies "
            "model sulfur monovacancies (S2 -> S, half weight)."
        ),
    ),
    "srtio3": LatticeSpec(
        name="srtio3",
        v1=(0.0, 1.0),
        v2=(1.0, 0.0),
        basis=(
            ColumnSite((0.0, 0.0), "Sr", column_weight(38)),
            ColumnSite((0.5, 0.5), "TiO", column_weight(22, 8)),
            ColumnSite((0.0, 0.5), "O", column_weight(8)),
            ColumnSite((0.5, 0.0), "O", column_weight(8)),
        ),
        description=(
            "SrTiO3-type perovskite viewed along [001]. spacing = cubic cell "
            "parameter a (3.905 A). Sr corner columns, Ti+O centre columns, "
            "and pure-O edge columns that are nearly invisible in HAADF."
        ),
    ),
    "fcc110": LatticeSpec(
        name="fcc110",
        v1=(0.0, 1.0 / np.sqrt(2.0)),
        v2=(1.0, 0.0),
        basis=(
            ColumnSite((0.0, 0.0), "Pt", column_weight(78)),
            ColumnSite((0.5, 0.5), "Pt", column_weight(78)),
        ),
        description=(
            "FCC metal (Pt) viewed along [110]. spacing = cubic cell "
            "parameter a (3.92 A); projected cell a/sqrt(2) x a, centred."
        ),
    ),
}


@dataclass
class SimConfig:
    """Configuration for one simulated HAADF image.

    Attributes:
        size: Image side length in pixels (square image).
        lattice: "hexagonal", "square", or a preset name from PRESETS
            ("graphene", "mos2", "srtio3", "fcc110").
        spacing: Cell parameter `a` in pixels. For the generic hexagonal
            and square lattices this equals the nearest-neighbour column
            spacing; for presets see the preset description.
        rotation_deg: Lattice rotation. If None, drawn uniformly at random.
        host_z: Atomic number of the host species (generic lattices only).
        dopant_z: Atomic number of the substitutional dopant.
        dopant_fraction: Fraction of columns replaced by the dopant.
        vacancy_fraction: Fraction of columns removed entirely.
        partial_vacancy_fraction: Fraction of reducible (two-atom) columns
            that lose one atom. Only presets with reducible sites (mos2)
            are affected.
        probe_sigma: Gaussian probe standard deviation in pixels.
        displacement_sigma: Static random column displacement in pixels.
        jitter_sigma: Fast per-row horizontal scan jitter in pixels.
        drift_px: Total slow sample drift accumulated over the frame, in
            pixels. Direction is drift_angle_deg, or random if None.
        drift_angle_deg: Drift direction in degrees (0 = +col, 90 = +row).
        background: Constant background as a fraction of the heaviest
            column's peak.
        background_variation: Amplitude of an additional smooth
            low-frequency background field, same units as `background`.
            Zero disables it.
        dose: Mean electron counts at the heaviest column's peak.
    """

    size: int = 256
    lattice: str = "hexagonal"
    spacing: float = 14.0
    rotation_deg: float | None = None
    host_z: int = 42
    dopant_z: int = 74
    dopant_fraction: float = 0.0
    vacancy_fraction: float = 0.05
    partial_vacancy_fraction: float = 0.0
    probe_sigma: float = 2.6
    displacement_sigma: float = 0.35
    jitter_sigma: float = 0.4
    drift_px: float = 0.0
    drift_angle_deg: float | None = None
    background: float = 0.08
    background_variation: float = 0.0
    dose: float = 500.0


@dataclass
class SimResult:
    """A simulated image with its ground truth.

    Attributes:
        image: Float32 image, normalised so the heaviest noise-free column
            peaks at ~1.
        positions: (N, 2) float array of column centres as (row, col).
        species: (N,) int array indexing into species_names.
        species_names: Distinct column labels, e.g. ("Sr", "TiO", "O").
        weights: (N,) float array of relative column weights (heaviest = 1),
            after dopant substitution and partial vacancies.
        is_dopant: (N,) bool array, True where the column is a dopant.
        vacancies: (M, 2) float array of fully removed lattice sites.
        config: The configuration that produced this image.
    """

    image: np.ndarray
    positions: np.ndarray
    species: np.ndarray
    species_names: tuple[str, ...]
    weights: np.ndarray
    is_dopant: np.ndarray
    vacancies: np.ndarray
    config: SimConfig = field(repr=False)


def preset_config(name: str, **overrides) -> SimConfig:
    """Return a SimConfig with sensible defaults for a named preset.

    Spacing is chosen so nearest-neighbour columns sit 12-14 px apart at
    the default probe size. Any SimConfig field can be overridden.

    Args:
        name: Preset name from PRESETS, or "hexagonal" / "square".
        **overrides: SimConfig fields to override.

    Returns:
        A ready-to-use SimConfig.
    """
    defaults: dict[str, dict] = {
        "graphene": {"spacing": 21.0, "vacancy_fraction": 0.02},
        "mos2": {"spacing": 22.0, "vacancy_fraction": 0.0, "partial_vacancy_fraction": 0.08},
        "srtio3": {"spacing": 18.0, "vacancy_fraction": 0.0},
        "fcc110": {"spacing": 20.0, "vacancy_fraction": 0.02},
        "hexagonal": {"spacing": 14.0},
        "square": {"spacing": 14.0},
    }
    if name not in defaults:
        raise ValueError(f"unknown lattice or preset: {name!r}")
    config = SimConfig(lattice=name, **defaults[name])
    return replace(config, **overrides) if overrides else config


def _resolve_spec(config: SimConfig) -> LatticeSpec:
    """Return the LatticeSpec for a config, building generic ones on the fly."""
    if config.lattice in PRESETS:
        return PRESETS[config.lattice]
    weight = column_weight(config.host_z)
    if config.lattice == "hexagonal":
        return LatticeSpec(
            name="hexagonal",
            v1=(0.0, 1.0),
            v2=(_SQRT3 / 2.0, 0.5),
            basis=(ColumnSite((0.0, 0.0), "host", weight),),
            description="Generic hexagonal lattice; spacing = NN distance.",
        )
    if config.lattice == "square":
        return LatticeSpec(
            name="square",
            v1=(0.0, 1.0),
            v2=(1.0, 0.0),
            basis=(ColumnSite((0.0, 0.0), "host", weight),),
            description="Generic square lattice; spacing = NN distance.",
        )
    raise ValueError(f"unknown lattice type: {config.lattice!r}")


def _lattice_points(
    config: SimConfig, spec: LatticeSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, site_index) covering the image with a pad margin."""
    a = config.spacing
    v1 = np.array(spec.v1) * a
    v2 = np.array(spec.v2) * a

    theta = (
        np.deg2rad(config.rotation_deg)
        if config.rotation_deg is not None
        else rng.uniform(0.0, np.pi / 3.0)
    )
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v1, v2 = rot @ v1, rot @ v2
    offset = rng.uniform(0.0, a, size=2) + config.size / 2.0

    n = int(np.ceil(2.0 * config.size / a)) + 2
    ii, jj = np.meshgrid(np.arange(-n, n), np.arange(-n, n), indexing="ij")
    cells = ii.reshape(-1, 1) * v1 + jj.reshape(-1, 1) * v2

    points, site_index = [], []
    for k, site in enumerate(spec.basis):
        pts = cells + site.frac[0] * v1 + site.frac[1] * v2 + offset
        points.append(pts)
        site_index.append(np.full(len(pts), k))
    pts = np.concatenate(points)
    idx = np.concatenate(site_index)

    pad = 2.0 * config.probe_sigma
    inside = np.all((pts >= -pad) & (pts <= config.size - 1 + pad), axis=1)
    return pts[inside], idx[inside]


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


def _row_offsets(config: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return per-output-row (dx, dy) offsets from jitter plus slow drift."""
    size = config.size
    dx = np.zeros(size)
    dy = np.zeros(size)

    if config.jitter_sigma > 0:
        jitter = gaussian_filter(rng.normal(0.0, 1.0, size=size), sigma=6.0)
        scale = jitter.std()
        if scale > 0:
            dx += jitter * (config.jitter_sigma / scale)

    if config.drift_px > 0:
        angle = (
            np.deg2rad(config.drift_angle_deg)
            if config.drift_angle_deg is not None
            else rng.uniform(0.0, 2.0 * np.pi)
        )
        progress = np.arange(size) / max(size - 1, 1)
        dx += config.drift_px * np.cos(angle) * progress
        dy += config.drift_px * np.sin(angle) * progress

    return dx, dy


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

    spec = _resolve_spec(config)
    pts, site_idx = _lattice_points(config, spec, rng)

    keep = rng.uniform(size=len(pts)) >= config.vacancy_fraction
    vacancies = pts[~keep]
    pts, site_idx = pts[keep], site_idx[keep]

    weights = np.array([spec.basis[k].weight for k in site_idx])
    species_names = tuple(dict.fromkeys(site.species for site in spec.basis))
    name_to_id = {name: i for i, name in enumerate(species_names)}
    species = np.array([name_to_id[spec.basis[k].species] for k in site_idx])

    if config.partial_vacancy_fraction > 0:
        reducible = np.array([spec.basis[k].reducible for k in site_idx])
        reduced = reducible & (rng.uniform(size=len(pts)) < config.partial_vacancy_fraction)
        weights = np.where(reduced, weights / 2.0, weights)

    is_dopant = rng.uniform(size=len(pts)) < config.dopant_fraction
    weights = np.where(is_dopant, column_weight(config.dopant_z), weights)
    weights = weights / weights.max() if len(weights) else weights

    pts = pts + rng.normal(0.0, config.displacement_sigma, size=pts.shape)

    # Peak amplitude of a splatted-then-blurred unit weight is
    # 1 / (2 pi sigma^2); rescale so a weight-1 column peaks at ~1.
    peak_norm = 2.0 * np.pi * config.probe_sigma**2
    clean = _splat((config.size, config.size), pts, weights * peak_norm)
    clean = gaussian_filter(clean, sigma=config.probe_sigma, mode="constant")

    clean += config.background
    if config.background_variation > 0:
        fluct = gaussian_filter(
            rng.normal(0.0, 1.0, size=(config.size, config.size)), sigma=config.size / 8.0
        )
        scale = fluct.std()
        if scale > 0:
            clean += np.abs(fluct) * (config.background_variation / scale)

    # Scan artifacts: warp the image with per-row offsets and shift the
    # ground truth of each column by the offsets of its own row.
    dx, dy = _row_offsets(config, rng)
    rows, cols = np.meshgrid(
        np.arange(config.size, dtype=np.float64),
        np.arange(config.size, dtype=np.float64),
        indexing="ij",
    )
    warped = map_coordinates(
        clean, [rows - dy[:, None], cols - dx[:, None]], order=1, mode="nearest"
    )
    row_idx = np.clip(np.round(pts[:, 0]).astype(int), 0, config.size - 1)
    pts = pts.copy()
    pts[:, 0] += dy[row_idx]
    pts[:, 1] += dx[row_idx]

    noisy = rng.poisson(np.clip(warped, 0.0, None) * config.dose) / config.dose

    inside = np.all((pts >= 0.0) & (pts <= config.size - 1.0), axis=1)
    return SimResult(
        image=noisy.astype(np.float32),
        positions=pts[inside],
        species=species[inside],
        species_names=species_names,
        weights=weights[inside],
        is_dopant=is_dopant[inside],
        vacancies=vacancies,
        config=config,
    )
