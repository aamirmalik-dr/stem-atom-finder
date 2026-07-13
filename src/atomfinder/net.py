"""Compact U-Net that maps a STEM image to an atomic column heatmap.

The network is intentionally small (two downsampling stages, 8 base
channels, about 30k parameters) so it trains in minutes on a CPU and the
weight file stays tiny. The target it learns is a Gaussian disk of fixed
width centred on every ground-truth column.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Two-level U-Net producing a single-channel heatmap in [0, 1]."""

    def __init__(self, base: int = 8):
        super().__init__()
        self.enc1 = _block(1, base)
        self.enc2 = _block(base, base * 2)
        self.bottom = _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel column logits for a (B, 1, H, W) input."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottom(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def heatmap_target(shape: tuple[int, int], positions: np.ndarray, sigma: float = 1.8) -> np.ndarray:
    """Render the training target: Gaussian disks at column positions.

    Args:
        shape: (H, W) of the target.
        positions: (N, 2) column centres as (row, col).
        sigma: Disk standard deviation in pixels.

    Returns:
        Float32 array in [0, 1], one Gaussian bump per column.
    """
    target = np.zeros(shape, dtype=np.float32)
    rad = int(np.ceil(3 * sigma))
    ax = np.arange(-rad, rad + 1)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    for r, c in positions:
        ri, ci = int(round(r)), int(round(c))
        bump = np.exp(-((yy + ri - r) ** 2 + (xx + ci - c) ** 2) / (2 * sigma**2))
        r0, r1 = max(ri - rad, 0), min(ri + rad + 1, shape[0])
        c0, c1 = max(ci - rad, 0), min(ci + rad + 1, shape[1])
        br0, bc0 = r0 - (ri - rad), c0 - (ci - rad)
        window = target[r0:r1, c0:c1]
        np.maximum(window, bump[br0 : br0 + r1 - r0, bc0 : bc0 + c1 - c0], out=window)
    return target


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Standardise an image to zero mean and unit variance."""
    image = image.astype(np.float32)
    std = image.std()
    return (image - image.mean()) / (std if std > 0 else 1.0)


def predict_heatmap(model: nn.Module, image: np.ndarray, activation: str = "sigmoid") -> np.ndarray:
    """Run the network on a single image and return its heatmap.

    Args:
        model: A trained UNet.
        image: 2D image whose sides are multiples of 4 (two poolings).
        activation: "sigmoid" for the segmentation head, "linear" for the
            heatmap-regression head (raw output, clipped at zero).

    Returns:
        2D float32 heatmap of per-pixel column score.
    """
    if activation not in ("sigmoid", "linear"):
        raise ValueError(f"unknown activation: {activation!r}")
    model.eval()
    x = torch.from_numpy(normalize_image(image))[None, None]
    with torch.no_grad():
        logits = model(x)[0, 0]
    if activation == "sigmoid":
        return torch.sigmoid(logits).numpy()
    return np.clip(logits.numpy(), 0.0, None)
