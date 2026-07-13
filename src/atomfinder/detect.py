"""Classical atomic column detection.

Four detector families microscopists actually reach for:

- Multi-scale Laplacian of Gaussian (LoG) blob detection: the
  scale-normalised LoG response peaks when the filter scale matches the
  blob scale, so local maxima locate columns without training.
- Smoothed local-maximum peak finding: Gaussian-smooth the image, then
  take local maxima above an intensity threshold. The simplest standard
  approach, usually followed by sub-pixel refinement.
- Normalised cross-correlation (NCC) template matching against a
  Gaussian column template, thresholded on the correlation coefficient.
- Model-based peak extraction from a CNN heatmap (peaks_from_heatmap),
  shared by the learned detectors in atomfinder.net.

All detectors return integer-pixel candidates; sub-pixel refinement lives
in atomfinder.refine.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter, uniform_filter
from scipy.signal import fftconvolve


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


def detect_local_max(
    image: np.ndarray,
    sigma: float = 2.6,
    threshold_rel: float = 0.25,
    min_distance: int = 4,
) -> np.ndarray:
    """Detect columns as local maxima of the Gaussian-smoothed image.

    Args:
        image: 2D grayscale image.
        sigma: Smoothing width in pixels, matched to the column width.
        threshold_rel: Intensity threshold as a fraction of the smoothed
            image's dynamic range above its minimum.
        min_distance: Minimum separation between detections in pixels.

    Returns:
        (N, 2) float array of detected positions as (row, col).
    """
    smoothed = gaussian_filter(image.astype(np.float64), sigma)
    lo, hi = smoothed.min(), smoothed.max()
    threshold = lo + threshold_rel * (hi - lo)
    return _local_maxima(smoothed, threshold, min_distance)


def gaussian_template(sigma: float, radius: int | None = None) -> np.ndarray:
    """Return a zero-mean, unit-norm Gaussian column template.

    Args:
        sigma: Template standard deviation in pixels.
        radius: Half-width; defaults to ceil(3 * sigma).

    Returns:
        A (2R+1, 2R+1) float array with zero mean and unit L2 norm.
    """
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    template = np.exp(-(yy**2 + xx**2) / (2 * sigma**2))
    template -= template.mean()
    return template / np.linalg.norm(template)


def detect_ncc(
    image: np.ndarray,
    sigma: float = 2.6,
    threshold: float = 0.45,
    min_distance: int = 4,
) -> np.ndarray:
    """Detect columns by normalised cross-correlation with a Gaussian template.

    The correlation coefficient is computed at every pixel against a
    zero-mean unit-norm template, so the score is contrast-invariant and
    lies in [-1, 1].

    Args:
        image: 2D grayscale image.
        sigma: Template width in pixels, matched to the column width.
        threshold: Minimum correlation coefficient for a detection.
        min_distance: Minimum separation between detections in pixels.

    Returns:
        (N, 2) float array of detected positions as (row, col).
    """
    image = image.astype(np.float64)
    # A 2.3-sigma radius keeps the invalid border band (one radius wide)
    # inside the 8 px evaluation margin used throughout the benchmark,
    # while truncating only the faint tail of the template.
    template = gaussian_template(sigma, radius=int(np.ceil(2.3 * sigma)))
    n = template.size
    window = template.shape[0]

    # Numerator: since the template is zero-mean, subtracting the local
    # patch mean changes nothing, so plain correlation suffices.
    numerator = fftconvolve(image, template[::-1, ::-1], mode="same")

    # Denominator: local patch L2 norm about its own mean.
    local_mean = uniform_filter(image, size=window, mode="reflect")
    local_sq = uniform_filter(image**2, size=window, mode="reflect")
    patch_var = np.clip(local_sq - local_mean**2, 0.0, None)
    denominator = np.sqrt(patch_var * n)

    ncc = np.where(denominator > 1e-12, numerator / np.maximum(denominator, 1e-12), 0.0)

    # Within one template radius of the border the template only partially
    # overlaps the image, so the coefficient is not meaningful there.
    radius = window // 2
    ncc[:radius, :] = -np.inf
    ncc[-radius:, :] = -np.inf
    ncc[:, :radius] = -np.inf
    ncc[:, -radius:] = -np.inf
    return _local_maxima(ncc, threshold, min_distance)


def peaks_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.4,
    min_distance: int = 4,
) -> np.ndarray:
    """Extract (row, col) peak positions from a model-predicted heatmap.

    Args:
        heatmap: 2D array of per-pixel column scores.
        threshold: Absolute score threshold for a peak.
        min_distance: Minimum separation between peaks in pixels.

    Returns:
        (N, 2) float array of peak positions as (row, col).
    """
    return _local_maxima(heatmap.astype(np.float64), threshold, min_distance)
