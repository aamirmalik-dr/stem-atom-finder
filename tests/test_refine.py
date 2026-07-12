"""Sub-pixel refinement tests."""

import numpy as np

from atomfinder.refine import refine_com


def _gaussian_image(centre: tuple[float, float], size: int = 48, sigma: float = 2.5) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.exp(-((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) / (2 * sigma**2))


def test_recovers_subpixel_centre():
    centre = (24.37, 23.62)
    image = _gaussian_image(centre)
    refined = refine_com(image, np.array([[24.0, 24.0]]))
    assert np.linalg.norm(refined[0] - np.array(centre)) < 0.15


def test_improves_on_integer_start():
    centre = (20.45, 25.55)
    image = _gaussian_image(centre)
    start = np.array([[20.0, 26.0]])
    refined = refine_com(image, start)
    err_before = np.linalg.norm(start[0] - np.array(centre))
    err_after = np.linalg.norm(refined[0] - np.array(centre))
    assert err_after < err_before


def test_border_detection_left_unrefined():
    image = _gaussian_image((2.0, 2.0))
    start = np.array([[2.0, 2.0]])
    refined = refine_com(image, start, window=5)
    np.testing.assert_array_equal(refined, start)
