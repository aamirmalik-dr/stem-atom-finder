"""Classical atomic column detection.

The baseline detector is multi-scale Laplacian of Gaussian (LoG) blob
detection: the LoG response of a Gaussian blob peaks when the filter scale
matches the blob scale, so local maxima of the scale-normalised response
locate columns without any training.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter


def _local_maxima(response: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Return (N, 2) integer (row, col) local maxima above a threshold."""
    size = 2 * min_distance + 1
    footprint_max = maximum_filter(response, size=size, mode="constant", cval=-np.inf)
    peaks = (response == footprint_max) & (response > threshold)
    return np.argwhere(peaks).astype(np.float64)


def detect_log(
    image: np.ndarray,
    sigma: float = 2.6,
    threshold_rel: float = 0.25,
    min_distance: int = 4,
    n_scales: int = 5,
) -> np.ndarray:
    """Detect columns with multi-scale Laplacian-of-Gaussian filtering.

    Args:
        image: 2D grayscale image.
        sigma: Expected column width in pixels; scales bracket this value.
        threshold_rel: Peak threshold relative to the maximum LoG response.
        min_distance: Minimum separation between detections in pixels.
        n_scales: Number of scales spanning [0.75 * sigma, 1.5 * sigma].

    Returns:
        (N, 2) float array of detected positions as (row, col).
    """
    image = image.astype(np.float64)
    scales = np.linspace(0.75 * sigma, 1.5 * sigma, n_scales)
    responses = np.stack([-(s**2) * gaussian_laplace(image, s) for s in scales])
    best = responses.max(axis=0)
    threshold = threshold_rel * best.max()
    return _local_maxima(best, threshold, min_distance)


def peaks_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.4,
    min_distance: int = 4,
) -> np.ndarray:
    """Extract (row, col) peak positions from a model-predicted heatmap.

    Args:
        heatmap: 2D array of per-pixel column scores in [0, 1].
        threshold: Absolute score threshold for a peak.
        min_distance: Minimum separation between peaks in pixels.

    Returns:
        (N, 2) float array of peak positions as (row, col).
    """
    return _local_maxima(heatmap.astype(np.float64), threshold, min_distance)
