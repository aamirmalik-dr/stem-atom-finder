"""Simulator tests."""

import numpy as np
import pytest

from atomfinder.sim import SimConfig, simulate_image


def test_deterministic_with_seed():
    config = SimConfig(size=128)
    a = simulate_image(config, np.random.default_rng(3))
    b = simulate_image(config, np.random.default_rng(3))
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_array_equal(a.positions, b.positions)


def test_shapes_and_dtype():
    result = simulate_image(SimConfig(size=96), np.random.default_rng(0))
    assert result.image.shape == (96, 96)
    assert result.image.dtype == np.float32
    assert result.positions.shape[1] == 2
    assert len(result.is_dopant) == len(result.positions)


def test_column_count_matches_lattice_density():
    config = SimConfig(size=256, spacing=14.0, vacancy_fraction=0.0, lattice="square")
    result = simulate_image(config, np.random.default_rng(1))
    expected = 256.0**2 / 14.0**2
    assert 0.8 * expected <= len(result.positions) <= 1.2 * expected


def test_vacancies_remove_columns():
    rng_a, rng_b = np.random.default_rng(2), np.random.default_rng(2)
    full = simulate_image(SimConfig(size=256, vacancy_fraction=0.0), rng_a)
    sparse = simulate_image(SimConfig(size=256, vacancy_fraction=0.3), rng_b)
    assert len(sparse.positions) < len(full.positions)
    assert len(sparse.vacancies) > 0


def test_dopants_are_brighter():
    config = SimConfig(size=256, dopant_fraction=0.15, dose=1e6, jitter_sigma=0.0)
    result = simulate_image(config, np.random.default_rng(4))
    assert result.is_dopant.any() and (~result.is_dopant).any()
    idx = np.round(result.positions).astype(int)
    peak = result.image[idx[:, 0], idx[:, 1]]
    assert peak[result.is_dopant].mean() > 1.3 * peak[~result.is_dopant].mean()


def test_higher_dose_is_less_noisy():
    low = simulate_image(SimConfig(size=128, dose=30.0), np.random.default_rng(5))
    high = simulate_image(SimConfig(size=128, dose=3000.0), np.random.default_rng(5))
    # Background regions should be far noisier at low dose.
    assert low.image.std() > high.image.std()


def test_unknown_lattice_raises():
    with pytest.raises(ValueError):
        simulate_image(SimConfig(lattice="kagome"), np.random.default_rng(0))
