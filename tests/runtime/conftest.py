"""Shared fixtures for tests/runtime/.

Spins up Postgres via testcontainers for the test session, applies the schema
once, then yields a per-test connection wrapped in a force-rollback transaction
so tests are isolated without per-test schema reset.
"""
from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg_pool import ConnectionPool
from testcontainers.postgres import PostgresContainer

from surveyforge.runtime.db import init_db


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
