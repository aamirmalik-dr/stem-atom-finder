"""Simulator tests."""

import numpy as np
import pytest

from atomfinder.sim import PRESETS, SimConfig, column_weight, preset_config, simulate_image


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
    assert len(result.species) == len(result.positions)
    assert len(result.weights) == len(result.positions)


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
    config = SimConfig(
        size=256, dopant_fraction=0.15, dose=1e6, jitter_sigma=0.0, vacancy_fraction=0.0
    )
    result = simulate_image(config, np.random.default_rng(4))
    assert result.is_dopant.any() and (~result.is_dopant).any()
    idx = np.round(result.positions).astype(int)
    peak = result.image[idx[:, 0], idx[:, 1]]
    assert peak[result.is_dopant].mean() > 1.3 * peak[~result.is_dopant].mean()


def test_higher_dose_is_less_noisy():
    low = simulate_image(SimConfig(size=128, dose=30.0), np.random.default_rng(5))
    high = simulate_image(SimConfig(size=128, dose=3000.0), np.random.default_rng(5))
    assert low.image.std() > high.image.std()


def test_unknown_lattice_raises():
    with pytest.raises(ValueError):
        simulate_image(SimConfig(lattice="kagome"), np.random.default_rng(0))


def test_column_weight():
    assert column_weight(6) == pytest.approx(6**1.7)
    assert column_weight(16, 16) == pytest.approx(2 * 16**1.7)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets_build(preset):
    config = preset_config(preset, size=192, dose=500.0)
    result = simulate_image(config, np.random.default_rng(0))
    assert len(result.positions) > 20
    assert result.image.max() > 0.5
    expected = {site.species for site in PRESETS[preset].basis}
    assert set(result.species_names) == expected


def test_preset_config_rejects_unknown():
    with pytest.raises(ValueError):
        preset_config("wurtzite")


def test_mos2_sulfur_columns_are_dimmer():
    config = preset_config("mos2", size=256, dose=1e6, jitter_sigma=0.0, rotation_deg=5.0)
    result = simulate_image(config, np.random.default_rng(6))
    idx = np.round(result.positions).astype(int)
    peak = result.image[idx[:, 0], idx[:, 1]]
    names = result.species_names
    mo = peak[result.species == names.index("Mo")]
    s2 = peak[result.species == names.index("S2")]
    assert 0.2 < s2.mean() / mo.mean() < 0.6


def test_partial_vacancy_halves_weight():
    config = preset_config("mos2", size=256, partial_vacancy_fraction=1.0, dopant_fraction=0.0)
    result = simulate_image(config, np.random.default_rng(7))
    names = result.species_names
    s2 = result.weights[result.species == names.index("S2")]
    # Every S2 column reduced: weight ratio vs Mo should be half the pristine ratio.
    mo = result.weights[result.species == names.index("Mo")]
    pristine_ratio = column_weight(16, 16) / column_weight(42)
    assert np.allclose(s2.mean() / mo.mean(), pristine_ratio / 2.0, rtol=1e-6)


def test_srtio3_has_faint_oxygen_columns():
    config = preset_config("srtio3", size=256)
    result = simulate_image(config, np.random.default_rng(8))
    names = result.species_names
    o_weight = result.weights[result.species == names.index("O")]
    sr_weight = result.weights[result.species == names.index("Sr")]
    assert o_weight.mean() < 0.1 * sr_weight.mean()


def test_drift_shifts_ground_truth_consistently():
    kwargs = dict(
        size=128,
        jitter_sigma=0.0,
        displacement_sigma=0.0,
        vacancy_fraction=0.0,
        rotation_deg=10.0,
        dose=1e6,
    )
    still = simulate_image(SimConfig(**kwargs), np.random.default_rng(9))
    drifted = simulate_image(
        SimConfig(drift_px=8.0, drift_angle_deg=0.0, **kwargs), np.random.default_rng(9)
    )
    # Same rng stream, so the lattices match. Drift at angle 0 shifts each
    # column by drift_px * row / (size - 1) in +col, rows unchanged.
    checked = 0
    for row, col in still.positions:
        expected_col = col + 8.0 * round(row) / 127.0
        if not 0.0 <= expected_col <= 127.0:
            continue
        same_row = drifted.positions[np.abs(drifted.positions[:, 0] - row) < 1e-9]
        assert len(same_row) > 0
        nearest = same_row[np.argmin(np.abs(same_row[:, 1] - expected_col))]
        assert abs(nearest[1] - expected_col) < 1e-6
        checked += 1
    assert checked > 20


def test_background_variation_raises_background_level():
    kwargs = dict(size=128, dose=1e6, vacancy_fraction=0.0)
    flat = simulate_image(SimConfig(background_variation=0.0, **kwargs), np.random.default_rng(11))
    varied = simulate_image(
        SimConfig(background_variation=0.2, **kwargs), np.random.default_rng(11)
    )
    assert varied.image.mean() > flat.image.mean()
