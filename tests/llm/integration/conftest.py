"""Shared fixtures for live-API integration tests.

Tests under this directory are gated by the `integration` pytest marker (see
pyproject.toml `addopts = "-m 'not integration'"`); run them explicitly with
`pytest -m integration`.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def skip_if_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip if SJTU key absent; strip SOCKS proxy vars (httpx + missing socksio)."""
    if not os.environ.get("SJTU_MODELS_API_KEY"):
        pytest.skip("SJTU_MODELS_API_KEY not set")
    for proxy_var in ("ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(proxy_var, raising=False)
