"""Sample save/load round-trip tests."""

import numpy as np

from atomfinder.io import load_sample, save_sample
from atomfinder.sim import SimConfig, preset_config, simulate_image


def test_roundtrip_generic(tmp_path):
    result = simulate_image(SimConfig(size=64), np.random.default_rng(0))
    path = tmp_path / "sample.npz"
    save_sample(path, result)
    loaded = load_sample(path)
    np.testing.assert_array_equal(loaded.image, result.image)
    np.testing.assert_array_equal(loaded.positions, result.positions)
    np.testing.assert_array_equal(loaded.species, result.species)
    np.testing.assert_array_equal(loaded.weights, result.weights)
    np.testing.assert_array_equal(loaded.is_dopant, result.is_dopant)
    np.testing.assert_array_equal(loaded.vacancies, result.vacancies)
    assert loaded.config == result.config


def test_roundtrip_preset_species_names(tmp_path):
    result = simulate_image(preset_config("srtio3", size=96), np.random.default_rng(1))
    path = tmp_path / "sample.npz"
    save_sample(path, result)
    loaded = load_sample(path)
    assert loaded.species_names == result.species_names
    assert loaded.config.lattice == "srtio3"
