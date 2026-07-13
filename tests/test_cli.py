"""CLI smoke tests."""

import numpy as np
from PIL import Image

from atomfinder.cli import main


def test_simulate_writes_sample(tmp_path):
    out = tmp_path / "img.npz"
    main(
        [
            "simulate",
            "--preset",
            "graphene",
            "--size",
            "128",
            "--dose",
            "500",
            "--seed",
            "0",
            "--out",
            str(out),
        ]
    )
    assert out.exists()


def test_detect_on_npz_sample(tmp_path, capsys):
    sample = tmp_path / "img.npz"
    main(["simulate", "--preset", "fcc110", "--size", "128", "--out", str(sample)])
    out = tmp_path / "peaks.csv"
    main(["detect", str(sample), "--method", "localmax", "--out", str(out)])
    captured = capsys.readouterr().out
    assert "columns detected" in captured
    assert "vs ground truth" in captured
    assert out.exists()
    peaks = np.loadtxt(out, delimiter=",", skiprows=1)
    assert peaks.shape[1] == 2


def test_detect_on_real_png(tmp_path, capsys):
    rng = np.random.default_rng(0)
    png = tmp_path / "real.png"
    Image.fromarray((rng.uniform(size=(96, 96)) * 255).astype(np.uint8)).save(png)
    main(["detect", str(png), "--method", "log", "--threshold", "0.9"])
    assert "columns detected" in capsys.readouterr().out
