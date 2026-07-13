"""Benchmark harness tests (small, fast configurations)."""

import numpy as np
import pytest
import torch
import yaml

from atomfinder.benchmark import build_methods, run_config
from atomfinder.net import UNet

BASE = {"size": 96, "lattice": "hexagonal", "spacing": 14.0}


def _write_config(tmp_path, payload):
    path = tmp_path / f"{payload['name']}.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _tiny_model(tmp_path):
    path = tmp_path / "unet.pt"
    torch.save(UNet().state_dict(), path)
    return str(path)


def test_build_methods_rejects_unknown():
    with pytest.raises(ValueError):
        build_methods(["dog"])


def test_build_methods_requires_model_path():
    with pytest.raises(ValueError):
        build_methods(["unet"])


def test_sweep_mode(tmp_path):
    config = {
        "name": "t_sweep",
        "mode": "sweep",
        "sweep": {"parameter": "dose", "values": [500.0]},
        "base_config": BASE,
        "images_per_condition": 1,
        "seed": 1,
        "methods": ["localmax"],
        "oracle": ["localmax"],
    }
    payload = run_config(_write_config(tmp_path, config), out_dir=tmp_path)
    row = payload["rows"][0]
    assert row["dose"] == 500.0
    assert 0.0 <= row["localmax"]["f1"] <= 1.0
    assert row["localmax_oracle"]["f1"] >= row["localmax"]["f1"] - 1e-9
    assert (tmp_path / "t_sweep.json").exists()


def test_sweep_with_untrained_unet(tmp_path):
    config = {
        "name": "t_unet",
        "mode": "sweep",
        "sweep": {"parameter": "dose", "values": [500.0]},
        "base_config": BASE,
        "images_per_condition": 1,
        "seed": 1,
        "methods": ["unet"],
        "models": {"unet": _tiny_model(tmp_path)},
    }
    payload = run_config(_write_config(tmp_path, config), out_dir=tmp_path)
    assert "unet" in payload["rows"][0]


def test_operating_point_mode(tmp_path):
    config = {
        "name": "t_op",
        "mode": "operating_point",
        "sweep": {"parameter": "dose", "values": [125.0, 500.0]},
        "base_config": BASE,
        "images_per_condition": 1,
        "seed": 2,
        "methods": ["localmax"],
    }
    payload = run_config(_write_config(tmp_path, config), out_dir=tmp_path)
    entry = payload["methods"]["localmax"]
    assert entry["oracle_mean_f1"] >= entry["fixed_mean_f1"] - 1e-9
    assert entry["mean_penalty"] >= 0.0
    assert len(entry["fixed_f1_per_condition"]) == 2


def test_refinement_mode(tmp_path):
    config = {
        "name": "t_ref",
        "mode": "refinement",
        "sweep": {"parameter": "dose", "values": [2000.0]},
        "base_config": BASE,
        "images_per_condition": 1,
        "seed": 3,
        "refiners": ["none", "com", "gauss"],
    }
    payload = run_config(_write_config(tmp_path, config), out_dir=tmp_path)
    row = payload["rows"][0]
    # Rounding truth to integers gives ~0.4 px RMSE; refiners must beat it.
    assert row["none"] > row["com"]
    assert row["none"] > row["gauss"]


def test_materials_mode(tmp_path):
    config = {
        "name": "t_mat",
        "mode": "materials",
        "presets": ["srtio3"],
        "base_config": {"size": 128, "dose": 500.0},
        "images_per_condition": 1,
        "seed": 4,
        "methods": ["log"],
    }
    payload = run_config(_write_config(tmp_path, config), out_dir=tmp_path)
    row = payload["rows"][0]
    recall = row["log"]["species_recall"]
    assert set(recall) == {"Sr", "TiO", "O"}
    # O columns are nearly invisible in HAADF; Sr columns are easy.
    assert recall["Sr"] > recall["O"]


def test_pr_curves_mode(tmp_path):
    config = {
        "name": "t_pr",
        "mode": "pr_curves",
        "conditions": [{"dose": 500.0}],
        "base_config": BASE,
        "images_per_condition": 1,
        "seed": 5,
        "methods": ["localmax"],
    }
    payload = run_config(_write_config(tmp_path, config), out_dir=tmp_path)
    points = payload["curves"][0]["methods"]["localmax"]
    assert len(points) > 3
    assert all(0.0 <= p["precision"] <= 1.0 for p in points)


def test_unknown_mode_raises(tmp_path):
    path = _write_config(tmp_path, {"name": "t_bad", "mode": "banana"})
    with pytest.raises(ValueError):
        run_config(path, out_dir=tmp_path)


def test_sweep_is_deterministic(tmp_path):
    config = {
        "name": "t_det",
        "mode": "sweep",
        "sweep": {"parameter": "dose", "values": [125.0]},
        "base_config": BASE,
        "images_per_condition": 2,
        "seed": 6,
        "methods": ["log"],
    }
    path = _write_config(tmp_path, config)
    a = run_config(path, out_dir=tmp_path)
    b = run_config(path, out_dir=tmp_path)
    assert np.isclose(a["rows"][0]["log"]["f1"], b["rows"][0]["log"]["f1"])
    assert a["rows"][0]["log"]["rmse_px"] == b["rows"][0]["log"]["rmse_px"]
