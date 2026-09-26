# Changelog

## Unreleased

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`.
- Citation file (`CITATION.cff`) and this changelog added.
- Continuous integration added: ruff, black, and pytest on CPU through GitHub Actions, with the badge in the README.
- Related repositories section in the README linking the six sibling electron-microscopy repositories.
- Test guarding the package `__version__` against the installed distribution metadata.

## 0.2.0 (2026-07-13)

- Multi-species HAADF-STEM simulator with materials presets (graphene, MoS2, SrTiO3, FCC Pt), vacancies, dopants, static displacements, scan jitter, drift, a two-term background, and Poisson noise, with ground truth tracked through every distortion.
- Five detectors: multi-scale Laplacian of Gaussian, smoothed local maxima, normalised cross-correlation, and two compact U-Nets (segmentation and regression heads) trained with domain randomisation; two sub-pixel refiners (centre of mass, 2D Gaussian fit).
- Hungarian-matched precision, recall, F1, localisation RMSE, and per-species recall; a YAML benchmark harness with sweep, precision-recall, operating-point, refinement, and per-material modes; the `atomfinder` CLI and a real-image path with one attributed CC BY-SA image.
- Committed trained weights, ablation weights, preset samples, results JSON, figures including the dose-ladder GIF, model card, API docs, and executed tutorial.
- Maintenance after publication: ruff and black pinned to exact versions, README expanded and restructured, detection overlay re-exported at higher resolution.

## 0.1.0 (2026-07-13)

- First version: single-species simulator, Laplacian of Gaussian and U-Net detectors, matched-detection metrics, committed samples and a trained model.
