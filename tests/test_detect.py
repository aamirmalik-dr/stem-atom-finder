"""Detector tests."""

import numpy as np
import pytest

from atomfinder.detect import (
    detect_local_max,
    detect_log,
    detect_ncc,
    gaussian_template,
    peaks_from_heatmap,
)
from atomfinder.metrics import filter_margin, match_positions
from atomfinder.sim import SimConfig, simulate_image

CLEAN = SimConfig(size=256, dose=5000.0, vacancy_fraction=0.0, dopant_fraction=0.0)


def _score(detector, **kwargs):
    result = simulate_image(CLEAN, np.random.default_rng(0))
    pred = filter_margin(detector(result.image, **kwargs), (256, 256), 8.0)
    true = filter_margin(result.positions, (256, 256), 8.0)
    return match_positions(true, pred, tolerance=4.0)


@pytest.mark.parametrize(
    "detector,kwargs",
    [
        (detect_log, {"sigma": 2.6}),
        (detect_local_max, {"sigma": 2.6}),
        (detect_ncc, {"sigma": 2.6}),
    ],
)
def test_classical_detectors_on_clean_image(detector, kwargs):
    res = _score(detector, **kwargs)
    assert res.recall > 0.95
    assert res.precision > 0.95


def test_gaussian_template_properties():
    template = gaussian_template(2.5)
    assert abs(template.mean()) < 1e-12
    assert abs(np.linalg.norm(template) - 1.0) < 1e-12
    assert template.shape[0] == template.shape[1]


def test_ncc_is_contrast_invariant():
    result = simulate_image(CLEAN, np.random.default_rng(1))
    a = detect_ncc(result.image, sigma=2.6)
    b = detect_ncc(result.image * 50.0 + 3.0, sigma=2.6)
    np.testing.assert_array_equal(a, b)


def test_peaks_from_heatmap_finds_isolated_bumps():
    heatmap = np.zeros((64, 64))
    centres = [(20, 20), (20, 40), (45, 30)]
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    for r, c in centres:
        heatmap += np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 2.0**2))
    peaks = peaks_from_heatmap(heatmap, threshold=0.4, min_distance=4)
    assert len(peaks) == 3
    res = match_positions(np.array(centres, dtype=float), peaks, tolerance=1.5)
    assert res.n_matched == 3


def test_peaks_empty_heatmap():
    peaks = peaks_from_heatmap(np.zeros((32, 32)), threshold=0.4)
    assert len(peaks) == 0
