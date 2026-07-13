"""Network and training-loop tests."""

import numpy as np
import pytest
import torch

from atomfinder.net import UNet, heatmap_target, normalize_image, predict_heatmap
from atomfinder.train import TrainSettings, make_batch, random_config, train_unet


def test_forward_shape():
    model = UNet()
    x = torch.zeros(2, 1, 64, 64)
    assert model(x).shape == (2, 1, 64, 64)


def test_model_is_small():
    n_params = sum(p.numel() for p in UNet().parameters())
    assert n_params < 100_000


def test_heatmap_target_peaks_at_positions():
    target = heatmap_target((32, 32), np.array([[16.0, 16.0]]))
    assert np.isclose(target[16, 16], 1.0, atol=1e-6)
    assert target.max() <= 1.0
    assert target[0, 0] == 0.0


def test_normalize_image():
    rng = np.random.default_rng(0)
    z = normalize_image(rng.normal(5.0, 3.0, size=(64, 64)))
    assert abs(z.mean()) < 1e-5
    assert abs(z.std() - 1.0) < 1e-5


def test_predict_heatmap_sigmoid_range():
    model = UNet()
    heatmap = predict_heatmap(model, np.random.default_rng(1).normal(size=(64, 64)))
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_predict_heatmap_linear_nonnegative():
    model = UNet()
    heatmap = predict_heatmap(
        model, np.random.default_rng(2).normal(size=(64, 64)), activation="linear"
    )
    assert heatmap.min() >= 0.0


def test_predict_heatmap_rejects_unknown_activation():
    with pytest.raises(ValueError):
        predict_heatmap(UNet(), np.zeros((64, 64)), activation="softmax")


def test_make_batch_shapes():
    settings = TrainSettings(batch_size=2, size=64)
    x, y = make_batch(np.random.default_rng(0), settings)
    assert x.shape == (2, 1, 64, 64)
    assert y.shape == (2, 1, 64, 64)
    assert float(y.max()) <= 1.0


def test_random_config_respects_ablation_flags():
    rng = np.random.default_rng(0)
    fixed_dose = random_config(rng, TrainSettings(randomize_dose=False))
    assert fixed_dose.dose == 500.0
    no_artifacts = random_config(rng, TrainSettings(include_scan_artifacts=False))
    assert no_artifacts.jitter_sigma == 0.0 and no_artifacts.drift_px == 0.0
    no_defects = random_config(rng, TrainSettings(include_defects=False))
    assert no_defects.vacancy_fraction == 0.0 and no_defects.dopant_fraction == 0.0
    fixed_geom = random_config(rng, TrainSettings(randomize_geometry=False))
    assert fixed_geom.lattice == "hexagonal" and fixed_geom.rotation_deg == 12.0


def test_train_smoke_both_targets():
    for target in ("segmentation", "regression"):
        model, history = train_unet(
            TrainSettings(steps=2, batch_size=2, size=64, target=target), log_every=0
        )
        assert len(history) == 2
        assert all(np.isfinite(loss) for loss in history)
        assert isinstance(model, UNet)


def test_train_rejects_unknown_target():
    with pytest.raises(ValueError):
        train_unet(TrainSettings(steps=1, target="classification"), log_every=0)
