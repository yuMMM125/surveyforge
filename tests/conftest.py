"""Shared pytest fixtures."""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load .env if present (no-op in CI)."""
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """Set fake API keys for unit tests.

    Also strips system proxy env vars so that httpx (used by ChatOpenAI) does
    not try to set up a SOCKS transport when the dev machine has ALL_PROXY set.
    """
    keys = {
        "SJTU_MODELS_API_KEY": "fake-sjtu",
        "LANGFUSE_PUBLIC_KEY": "fake-pub",
        "LANGFUSE_SECRET_KEY": "fake-sec",
        "LANGFUSE_HOST": "https://example.test",
    }
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    # Clear proxy vars — unit tests make no real network calls, and a SOCKS
    # proxy requires the optional `socksio` package which is not in dev deps.
    for proxy_var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(proxy_var, raising=False)
    yield keys


def has_real_key(name: str) -> bool:
    val = os.environ.get(name, "")
    return bool(val) and not val.startswith("fake-")
