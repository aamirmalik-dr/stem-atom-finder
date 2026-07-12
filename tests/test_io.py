"""Sample save/load round-trip tests."""

import numpy as np

from atomfinder.io import load_sample, save_sample
from atomfinder.sim import SimConfig, simulate_image


def test_roundtrip(tmp_path):
    result = simulate_image(SimConfig(size=64), np.random.default_rng(0))
    path = tmp_path / "sample.npz"
    save_sample(path, result)
    loaded = load_sample(path)
    np.testing.assert_array_equal(loaded.image, result.image)
    np.testing.assert_array_equal(loaded.positions, result.positions)
    np.testing.assert_array_equal(loaded.is_dopant, result.is_dopant)
    np.testing.assert_array_equal(loaded.vacancies, result.vacancies)
    assert loaded.config == result.config
