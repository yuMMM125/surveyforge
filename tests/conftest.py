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
    """Set fake API keys for unit tests."""
    keys = {
        "SJTU_MODELS_API_KEY": "fake-sjtu",
        "LANGFUSE_PUBLIC_KEY": "fake-pub",
        "LANGFUSE_SECRET_KEY": "fake-sec",
        "LANGFUSE_HOST": "https://example.test",
    }
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    yield keys


def has_real_key(name: str) -> bool:
    val = os.environ.get(name, "")
    return bool(val) and not val.startswith("fake-")
