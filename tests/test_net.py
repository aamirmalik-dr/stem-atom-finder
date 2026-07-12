"""Network and training-loop tests."""

import numpy as np
import torch

from atomfinder.net import UNet, heatmap_target, normalize_image, predict_heatmap
from atomfinder.train import make_batch, train_unet


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


def test_predict_heatmap_range():
    model = UNet()
    heatmap = predict_heatmap(model, np.random.default_rng(1).normal(size=(64, 64)))
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_make_batch_shapes():
    x, y = make_batch(np.random.default_rng(0), batch_size=2, size=64)
    assert x.shape == (2, 1, 64, 64)
    assert y.shape == (2, 1, 64, 64)
    assert float(y.max()) <= 1.0


def test_train_smoke():
    model, history = train_unet(steps=2, batch_size=2, size=64, log_every=0)
    assert len(history) == 2
    assert all(np.isfinite(loss) for loss in history)
    assert isinstance(model, UNet)
