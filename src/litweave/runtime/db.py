"""PostgreSQL connection management for SurveyForge runtime.

Per Architecture Decision (2026-05-01): PostgreSQL is the runtime backend; no
SQLite fallback. Connection string from `LITWEAVE_DATABASE_URL` env var.
Local dev uses the Postgres service in `docker-compose.yml`. Tests use
`testcontainers` to spin up an ephemeral instance per session.
"""
from __future__ import annotations

import atexit
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

ENV_DATABASE_URL = "LITWEAVE_DATABASE_URL"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"

# NOTE (Task 6): the global `_pool` is mutated by `get_pool()` and `reset_pool()`
# without synchronization. Today's single-threaded execution makes this safe;
# when LangGraph's threaded/async runner lands (Task 6), wrap both functions in
# a `threading.Lock` to avoid races on first-call construction and concurrent
# reset.
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
                f"`export {ENV_DATABASE_URL}=postgresql://litweave:litweave@localhost:5432/litweave`"
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
    """Apply schema.sql idempotently within the caller's transaction.

    The connection MUST NOT be in autocommit mode — `schema.sql` issues
    multiple DDL statements, and without an enclosing transaction a
    mid-script failure leaves the schema half-applied. Wrap the call site
    in `db.transaction()` (or open `conn.transaction()` manually) before
    invoking this function.
    """
    if conn.autocommit:
        raise RuntimeError(
            "init_db requires a non-autocommit connection so multi-statement "
            "DDL applies atomically. Wrap the call in `db.transaction()`."
        )
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)


@contextmanager
def transaction() -> Iterator[psycopg.Connection]:
    """Yield a pooled connection inside a transaction (commit on success, rollback on error)."""
    pool = get_pool()
    with pool.connection() as conn, conn.transaction():
        yield conn


# Register cleanup at module import so long-running processes (CLI, server)
# don't leak a `ConnectionPool` worker thread on interpreter exit. Tests don't
# rely on this — the conftest fixture closes its own pool explicitly.
atexit.register(reset_pool)
