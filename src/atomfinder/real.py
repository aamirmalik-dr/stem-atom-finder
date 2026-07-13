"""Loading real experimental images for detection.

The detectors in this package operate on any 2D grayscale array, so
running them on an experimental atomic-resolution image only needs a
loader and light normalisation. Expect a domain gap relative to the
synthetic training data (contamination, amorphous background, detector
noise that is not purely Poisson); see the model card for details.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, invert: bool = False) -> np.ndarray:
    """Load an image file as a normalised 2D grayscale float array.

    Args:
        path: PNG, TIFF, or JPEG file. Colour images are converted to
            grayscale by luminance.
        invert: Invert contrast, for bright-field-like images where
            columns are dark.

    Returns:
        Float32 array scaled to [0, 1].
    """
    with Image.open(path) as img:
        arr = np.asarray(img.convert("F"), dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return 1.0 - arr if invert else arr


def crop_to_multiple(image: np.ndarray, multiple: int = 4) -> np.ndarray:
    """Centre-crop an image so both sides are multiples of `multiple`.

    The U-Net pools twice, so its input sides must be multiples of 4.
    Classical detectors do not need this.

    Args:
        image: 2D array.
        multiple: Required divisor of both side lengths.

    Returns:
        The centre-cropped view.
    """
    h, w = image.shape
    nh, nw = (h // multiple) * multiple, (w // multiple) * multiple
    if nh == 0 or nw == 0:
        raise ValueError(f"image {image.shape} too small to crop to multiple of {multiple}")
    top, left = (h - nh) // 2, (w - nw) // 2
    return image[top : top + nh, left : left + nw]
