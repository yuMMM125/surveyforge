"""Shared fixtures for tests/agents/.

`patch_agent_transaction` itself now lives in `tests/conftest.py` so root-level
tests (e.g., `tests/test_graph_smoke.py`) can use it. Tests in this directory
inherit it automatically via pytest's conftest hierarchy.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture
def patch_planner_transaction(
    patch_agent_transaction: Callable[[str], None],
) -> None:
    """Backward-compat alias for the existing planner_unit tests."""
    patch_agent_transaction("surveyforge.agents.planner")
