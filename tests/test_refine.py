"""Sub-pixel refinement tests."""

import numpy as np

from atomfinder.refine import refine_com, refine_gaussian


def _gaussian_image(
    centre: tuple[float, float], size: int = 48, sigma: float = 2.5, offset: float = 0.0
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.exp(-((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) / (2 * sigma**2)) + offset


def test_com_recovers_subpixel_centre():
    centre = (24.37, 23.62)
    refined = refine_com(_gaussian_image(centre), np.array([[24.0, 24.0]]))
    assert np.linalg.norm(refined[0] - np.array(centre)) < 0.15


def test_gaussian_fit_recovers_subpixel_centre():
    centre = (24.37, 23.62)
    refined = refine_gaussian(_gaussian_image(centre), np.array([[24.0, 24.0]]))
    assert np.linalg.norm(refined[0] - np.array(centre)) < 0.02


def test_gaussian_fit_handles_constant_offset():
    centre = (20.45, 25.55)
    image = _gaussian_image(centre, offset=0.7)
    refined = refine_gaussian(image, np.array([[20.0, 26.0]]))
    assert np.linalg.norm(refined[0] - np.array(centre)) < 0.02


def test_com_improves_on_integer_start():
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
    np.testing.assert_array_equal(refine_com(image, start, window=5), start)
    np.testing.assert_array_equal(refine_gaussian(image, start, window=5), start)


def test_flat_patch_left_unrefined_by_gaussian_fit():
    image = np.full((32, 32), 0.5)
    start = np.array([[16.0, 16.0]])
    np.testing.assert_array_equal(refine_gaussian(image, start), start)
