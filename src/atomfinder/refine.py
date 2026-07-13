"""Sub-pixel position refinement.

Detectors return integer pixel positions. For quantitative work (strain,
polarisation, bond lengths) columns must be located to a fraction of a
pixel. Two standard refiners are provided:

- Iterative local centre of mass after background subtraction: fast,
  robust, slightly biased when a neighbour's tail leaks into the window.
- 2D Gaussian least-squares fit (the maximum-likelihood estimator under
  Gaussian noise): slower, and the field standard for quantitative
  column positions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def refine_com(
    image: np.ndarray,
    positions: np.ndarray,
    window: int = 5,
    iterations: int = 3,
) -> np.ndarray:
    """Refine detections to sub-pixel accuracy with iterative centre of mass.

    Args:
        image: 2D grayscale image.
        positions: (N, 2) initial positions as (row, col).
        window: Half-width of the local window in pixels.
        iterations: Number of recentring iterations.

    Returns:
        (N, 2) float array of refined positions. Detections whose window
        would leave the image are returned unrefined.
    """
    image = image.astype(np.float64)
    h, w = image.shape
    refined = positions.astype(np.float64).copy()
    ax = np.arange(-window, window + 1, dtype=np.float64)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")

    for i in range(len(refined)):
        r, c = refined[i]
        for _ in range(iterations):
            ri, ci = int(round(r)), int(round(c))
            if not (window <= ri < h - window and window <= ci < w - window):
                break
            patch = image[ri - window : ri + window + 1, ci - window : ci + window + 1]
            weights = patch - patch.min()
            total = weights.sum()
            if total <= 0:
                break
            r = ri + float((weights * yy).sum() / total)
            c = ci + float((weights * xx).sum() / total)
        refined[i] = (r, c)
    return refined


def _fit_one_gaussian(
    patch: np.ndarray, yy: np.ndarray, xx: np.ndarray, sigma0: float
) -> tuple[float, float] | None:
    """Fit A * exp(-r^2 / 2 s^2) + b to a patch; return (dr, dc) or None."""
    b0 = float(patch.min())
    a0 = float(patch.max() - b0)
    if a0 <= 0:
        return None

    def residual(p: np.ndarray) -> np.ndarray:
        amp, r0, c0, s, b = p
        model = amp * np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2 * s**2)) + b
        return (model - patch).ravel()

    half = float(yy.max())
    try:
        fit = least_squares(
            residual,
            x0=np.array([a0, 0.0, 0.0, sigma0, b0]),
            bounds=(
                [0.0, -half, -half, 0.3 * sigma0, -np.inf],
                [np.inf, half, half, 3.0 * sigma0, np.inf],
            ),
            method="trf",
            max_nfev=60,
        )
    except ValueError:
        return None
    if not fit.success and fit.status <= 0:
        return None
    return float(fit.x[1]), float(fit.x[2])


def refine_gaussian(
    image: np.ndarray,
    positions: np.ndarray,
    window: int = 5,
    sigma0: float = 2.6,
) -> np.ndarray:
    """Refine detections by least-squares 2D Gaussian fitting.

    Fits amplitude, centre, isotropic width and constant offset to the
    local window around each detection. This is the maximum-likelihood
    position estimate under additive Gaussian noise, and the standard
    sub-pixel method for quantitative STEM.

    Args:
        image: 2D grayscale image.
        positions: (N, 2) initial positions as (row, col).
        window: Half-width of the fit window in pixels.
        sigma0: Initial guess for the column width in pixels.

    Returns:
        (N, 2) float array of refined positions. Detections whose window
        would leave the image, or whose fit fails, are returned unrefined.
    """
    image = image.astype(np.float64)
    h, w = image.shape
    refined = positions.astype(np.float64).copy()
    ax = np.arange(-window, window + 1, dtype=np.float64)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")

    for i in range(len(refined)):
        ri, ci = int(round(refined[i, 0])), int(round(refined[i, 1]))
        if not (window <= ri < h - window and window <= ci < w - window):
            continue
        patch = image[ri - window : ri + window + 1, ci - window : ci + window + 1]
        offset = _fit_one_gaussian(patch, yy, xx, sigma0)
        if offset is not None:
            refined[i] = (ri + offset[0], ci + offset[1])
    return refined
