"""Detector tests."""

import numpy as np

from atomfinder.detect import detect_log, peaks_from_heatmap
from atomfinder.metrics import filter_margin, match_positions
from atomfinder.sim import SimConfig, simulate_image


def test_log_finds_columns_on_clean_image():
    config = SimConfig(size=256, dose=5000.0, vacancy_fraction=0.0, dopant_fraction=0.0)
    result = simulate_image(config, np.random.default_rng(0))
    pred = filter_margin(detect_log(result.image, sigma=config.probe_sigma), (256, 256), 8.0)
    true = filter_margin(result.positions, (256, 256), 8.0)
    res = match_positions(true, pred, tolerance=4.0)
    assert res.recall > 0.95
    assert res.precision > 0.95


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
