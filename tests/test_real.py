"""Real-image loader tests."""

import numpy as np
import pytest
from PIL import Image

from atomfinder.real import crop_to_multiple, load_image


def _write_png(path, array):
    Image.fromarray((array * 255).astype(np.uint8)).save(path)


def test_load_image_normalises_to_unit_range(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "img.png"
    _write_png(path, rng.uniform(size=(40, 60)))
    image = load_image(path)
    assert image.shape == (40, 60)
    assert image.dtype == np.float32
    assert image.min() == 0.0 and image.max() == 1.0


def test_load_image_invert(tmp_path):
    path = tmp_path / "img.png"
    _write_png(path, np.linspace(0, 1, 100).reshape(10, 10))
    normal = load_image(path)
    inverted = load_image(path, invert=True)
    np.testing.assert_allclose(inverted, 1.0 - normal, atol=1e-6)


def test_crop_to_multiple():
    image = np.zeros((37, 42))
    cropped = crop_to_multiple(image, 4)
    assert cropped.shape == (36, 40)


def test_crop_too_small_raises():
    with pytest.raises(ValueError):
        crop_to_multiple(np.zeros((3, 3)), 4)
