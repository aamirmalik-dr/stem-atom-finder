"""Sub-pixel position refinement.

Detectors return integer pixel positions. For quantitative work (strain,
polarisation, bond lengths) columns must be located to a fraction of a
pixel, so each detection is refined by an iterative local centre of mass
after background subtraction.
"""

from __future__ import annotations

import numpy as np


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
