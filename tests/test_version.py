"""Guard that the package version and the installed distribution metadata agree."""

from importlib.metadata import version

import atomfinder


def test_version_matches_distribution_metadata() -> None:
    assert atomfinder.__version__ == version("atomfinder")
