"""Shared fixtures for tests/agents/.

Agent node closures use `surveyforge.runtime.db.transaction()` to get a
pooled connection for `RunManager.update_stage(...)`. Unit tests want to
share the test's force-rollback `conn` fixture so creates + node-side
reads are visible within the same transaction. We monkeypatch the
`transaction` symbol imported into agent modules to yield the test's conn.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
import pytest


@pytest.fixture
def patch_planner_transaction(
    monkeypatch: pytest.MonkeyPatch, conn: psycopg.Connection
) -> psycopg.Connection:
    """Make `surveyforge.agents.planner.transaction()` yield the test conn.

    Lets the planner node's `RunManager.update_stage` write inside the
    test's force-rollback transaction (so creates + reads stay coherent).
    """

    @contextmanager
    def _fake_transaction() -> Iterator[psycopg.Connection]:
        yield conn

    monkeypatch.setattr(
        "surveyforge.agents.planner.transaction", _fake_transaction
    )
    return conn
