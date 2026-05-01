"""Smoke test: verify the package is importable and version matches."""
from surveyforge import __version__


def test_package_importable():
    assert __version__ == "0.1.0"
