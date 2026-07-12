"""Detection metrics with optimal one-to-one matching.

Predicted and ground-truth positions are matched with the Hungarian
algorithm on the pairwise distance matrix; a match counts as a true
positive only if it lies within a tolerance radius. This avoids the
double-counting that greedy nearest-neighbour matching allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class DetectionResult:
    """Matched detection metrics.

    Attributes:
        n_true: Number of ground-truth columns.
        n_pred: Number of predicted columns.
        n_matched: True positives (matched within tolerance).
        precision: n_matched / n_pred.
        recall: n_matched / n_true.
        f1: Harmonic mean of precision and recall.
        rmse: Root-mean-square position error of matched pairs, in pixels.
        matches: (K, 2) integer array of (true_index, pred_index) pairs.
    """

    n_true: int
    n_pred: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    rmse: float
    matches: np.ndarray


def filter_margin(points: np.ndarray, shape: tuple[int, int], margin: float) -> np.ndarray:
    """Drop points within `margin` pixels of the image border."""
    if len(points) == 0:
        return points
    h, w = shape
    ok = (
        (points[:, 0] >= margin)
        & (points[:, 0] <= h - 1 - margin)
        & (points[:, 1] >= margin)
        & (points[:, 1] <= w - 1 - margin)
    )
    return points[ok]


def match_positions(
    true_positions: np.ndarray,
    pred_positions: np.ndarray,
    tolerance: float = 4.0,
) -> DetectionResult:
    """Match predictions to ground truth and compute detection metrics.

    Args:
        true_positions: (N, 2) ground-truth positions as (row, col).
        pred_positions: (M, 2) predicted positions as (row, col).
        tolerance: Maximum centre distance in pixels for a true positive.

    Returns:
        A DetectionResult with counts, precision, recall, F1 and RMSE.
    """
    n_true, n_pred = len(true_positions), len(pred_positions)
    if n_true == 0 or n_pred == 0:
        return DetectionResult(
            n_true, n_pred, 0, 0.0, 0.0, 0.0, float("nan"), np.empty((0, 2), dtype=int)
        )

    dist = cdist(true_positions, pred_positions)
    # Penalise out-of-tolerance pairs so the assignment prefers real matches.
    cost = np.where(dist <= tolerance, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    ok = dist[rows, cols] <= tolerance
    matches = np.stack([rows[ok], cols[ok]], axis=1)

    n_matched = len(matches)
    precision = n_matched / n_pred
    recall = n_matched / n_true
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    rmse = (
        float(np.sqrt(np.mean(dist[matches[:, 0], matches[:, 1]] ** 2)))
        if n_matched
        else float("nan")
    )
    return DetectionResult(n_true, n_pred, n_matched, precision, recall, f1, rmse, matches)
