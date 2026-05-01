"""PostgreSQL connection management for SurveyForge runtime.

Per Architecture Decision (2026-05-01): PostgreSQL is the runtime backend; no
SQLite fallback. Connection string from `SURVEYFORGE_DATABASE_URL` env var.
Local dev uses the Postgres service in `docker-compose.yml`. Tests use
`testcontainers` to spin up an ephemeral instance per session.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

ENV_DATABASE_URL = "SURVEYFORGE_DATABASE_URL"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Return process-wide connection pool, opened lazily on first call."""
    global _pool
    if _pool is None:
        url = os.environ.get(ENV_DATABASE_URL)
        if not url:
            raise RuntimeError(
                f"{ENV_DATABASE_URL} is not set. For local dev: "
                f"`docker compose up -d postgres`, then "
                f"`export {ENV_DATABASE_URL}=postgresql://surveyforge:surveyforge@localhost:5432/surveyforge`"
            )
        _pool = ConnectionPool(url, min_size=1, max_size=10, open=True)
    return _pool


def reset_pool() -> None:
    """Close the cached pool. Used by tests that swap DATABASE_URL between sessions."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_db(conn: psycopg.Connection) -> None:
    """Apply schema.sql idempotently. Caller controls transaction scope."""
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)


@contextmanager
def transaction() -> Iterator[psycopg.Connection]:
    """Yield a pooled connection inside a transaction (commit on success, rollback on error)."""
    pool = get_pool()
    with pool.connection() as conn, conn.transaction():
        yield conn
