"""Shared fixtures for tests/agents/.

The `patch_agent_transaction` factory monkeypatches `transaction()` inside a
specific agent module so DB writes from the node share the test's force-rollback
`conn` fixture (rather than opening a new pool connection that bypasses
isolation). Each agent test file imports this fixture and parametrizes with
its own module path.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import psycopg
import pytest


@pytest.fixture
def patch_agent_transaction(
    monkeypatch: pytest.MonkeyPatch, conn: psycopg.Connection,
) -> Callable[[str], None]:
    """Returns a function that patches `<module_path>.transaction` to yield the
    test's force-rollback `conn`.

    Usage:
        def test_x(patch_agent_transaction, conn):
            patch_agent_transaction("surveyforge.agents.researcher_wide")
            # ... node call now shares conn's transaction
    """

    @contextmanager
    def _yield_conn() -> Iterator[psycopg.Connection]:
        yield conn

    def _patch(module_path: str) -> None:
        monkeypatch.setattr(f"{module_path}.transaction", _yield_conn)

    return _patch


# Backward-compat alias for the existing planner_unit tests.
@pytest.fixture
def patch_planner_transaction(
    patch_agent_transaction: Callable[[str], None],
) -> None:
    patch_agent_transaction("surveyforge.agents.planner")
