"""Matching-metric tests."""

import numpy as np

from atomfinder.metrics import filter_margin, match_positions


def test_perfect_match():
    pts = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 5.0]])
    res = match_positions(pts, pts.copy(), tolerance=2.0)
    assert res.n_matched == 3
    assert res.precision == 1.0 and res.recall == 1.0 and res.f1 == 1.0
    assert res.rmse == 0.0


def test_extra_prediction_lowers_precision():
    true = np.array([[10.0, 10.0]])
    pred = np.array([[10.0, 10.0], [50.0, 50.0]])
    res = match_positions(true, pred, tolerance=2.0)
    assert res.n_matched == 1
    assert res.precision == 0.5 and res.recall == 1.0


def test_missed_column_lowers_recall():
    true = np.array([[10.0, 10.0], [50.0, 50.0]])
    pred = np.array([[10.0, 10.0]])
    res = match_positions(true, pred, tolerance=2.0)
    assert res.precision == 1.0 and res.recall == 0.5


def test_out_of_tolerance_is_not_a_match():
    res = match_positions(np.array([[10.0, 10.0]]), np.array([[10.0, 16.0]]), tolerance=4.0)
    assert res.n_matched == 0 and res.f1 == 0.0


def test_no_double_counting_one_to_one():
    true = np.array([[10.0, 10.0]])
    pred = np.array([[10.0, 11.0], [11.0, 10.0]])
    res = match_positions(true, pred, tolerance=4.0)
    assert res.n_matched == 1
    assert res.precision == 0.5


def test_assignment_is_optimal_not_greedy():
    # Greedy matching would pair the first prediction with the nearer truth
    # and strand the second; optimal assignment matches both.
    true = np.array([[0.0, 0.0], [0.0, 3.0]])
    pred = np.array([[0.0, 1.4], [0.0, 0.1]])
    res = match_positions(true, pred, tolerance=2.0)
    assert res.n_matched == 2


def test_empty_inputs():
    empty = np.empty((0, 2))
    pts = np.array([[1.0, 1.0]])
    assert match_positions(empty, pts).n_matched == 0
    assert match_positions(pts, empty).recall == 0.0


def test_rmse_reflects_position_error():
    res = match_positions(np.array([[10.0, 10.0]]), np.array([[10.0, 11.0]]), tolerance=4.0)
    assert np.isclose(res.rmse, 1.0)


def test_filter_margin():
    pts = np.array([[2.0, 50.0], [50.0, 50.0], [50.0, 98.0]])
    kept = filter_margin(pts, (100, 100), margin=8.0)
    assert len(kept) == 1
    np.testing.assert_array_equal(kept[0], [50.0, 50.0])
