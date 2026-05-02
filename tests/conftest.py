"""Shared pytest fixtures.

Postgres fixtures (`postgres_url` / `initialized_pool` / `conn`) are at the
project root rather than under `tests/runtime/` so that `tests/tools/*` (and
later `tests/agents/*`) can share the same testcontainers Postgres + the
force-rollback transaction isolation, without copy-pasting.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from testcontainers.postgres import PostgresContainer

from litweave.runtime.db import init_db


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
        "MODELS_API_KEY": "fake-models",
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


# ---- testcontainers Postgres fixtures (moved up from tests/runtime/conftest.py
# so tests/tools/, tests/agents/, etc. can share without duplication) ----


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine") as pg:
        # `driver=None` returns plain `postgresql://...` (what psycopg3 wants).
        # The `driver="psycopg"` form returns SQLAlchemy's `postgresql+psycopg://...`
        # URL scheme, which `psycopg.connect` / `ConnectionPool` reject.
        yield pg.get_connection_url(driver=None)


@pytest.fixture(scope="session")
def initialized_pool(postgres_url: str) -> Iterator[ConnectionPool]:
    pool = ConnectionPool(postgres_url, min_size=1, max_size=2, open=True)
    with pool.connection() as conn, conn.transaction():
        init_db(conn)
    yield pool
    pool.close()


@pytest.fixture
def conn(initialized_pool: ConnectionPool) -> Iterator[psycopg.Connection]:
    """Per-test connection: changes roll back at end of test."""
    with initialized_pool.connection() as c, c.transaction(force_rollback=True):
        yield c


@pytest.fixture
def patch_agent_transaction(
    monkeypatch: pytest.MonkeyPatch, conn: psycopg.Connection,
) -> Callable[[str], None]:
    """Returns a function that patches `<module_path>.transaction` to yield the
    test's force-rollback `conn`.

    Lifted from `tests/agents/conftest.py` so root-level tests
    (`test_graph_smoke.py`, `test_cli.py`) can use it — pytest conftest
    discovery walks UP from the test file, so a fixture in `tests/agents/`
    is invisible to tests at `tests/`.

    Usage:
        def test_x(patch_agent_transaction, conn):
            patch_agent_transaction("litweave.agents.researcher_wide")
            # node call now shares conn's transaction
    """

    @contextmanager
    def _yield_conn() -> Iterator[psycopg.Connection]:
        yield conn

    def _patch(module_path: str) -> None:
        monkeypatch.setattr(f"{module_path}.transaction", _yield_conn)

    return _patch
