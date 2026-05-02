"""Live Planner integration test using the configured model gateway.

Marked `pytest.mark.integration`; default `pytest` skips this file.
Run with `pytest tests/agents/integration -m integration`.
Requires `MODELS_API_KEY` in env (loaded via tests/conftest.py::load_env).
"""
from __future__ import annotations

import os
import time

import pytest
from langchain_core.runnables import RunnableConfig
from psycopg_pool import ConnectionPool

from litweave.agents.planner import make_planner_node
from litweave.llm.providers import ProviderName
from litweave.llm.roles import AgentRole
from litweave.llm.router import RoleBinding
from litweave.prompts.loader import PromptRegistry
from litweave.runtime.runs import RunManager
from litweave.state import make_initial_state

# Common placeholder patterns that would slip past `os.environ.get(...)` and
# trigger a real provider call. Skip the test when MODELS_API_KEY matches any of
# these; saves wasted requests and avoids leaking diagnostic info to the gateway
# audit log.
_PLACEHOLDER_KEY_PREFIXES = ("fake-", "fake_", "PASTE_YOUR_", "your-key", "test-", "dummy-", "placeholder")


def _get_model_api_key() -> str:
    return os.environ.get("MODELS_API_KEY") or os.environ.get("SJTU_MODELS_API_KEY", "")


def _is_placeholder_key(key: str) -> bool:
    """True if `key` looks like a placeholder/example value, not a real API key."""
    return any(key.startswith(prefix) for prefix in _PLACEHOLDER_KEY_PREFIXES)


@pytest.mark.integration
def test_planner_live_produces_outline_for_rlhf_survey(
    postgres_url: str,
    initialized_pool: ConnectionPool,  # ensures schema is applied before planner uses transaction()
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = _get_model_api_key()
    if not api_key:
        pytest.skip("MODELS_API_KEY not set")
    if _is_placeholder_key(api_key):
        pytest.skip(
            f"MODELS_API_KEY appears to be a placeholder ({api_key[:12]}...); "
            "skipping to avoid wasted live calls"
        )

    # Wire the testcontainer URL into the production DB code path so that the
    # run created here and the planner node's stage update go through the same
    # Postgres instance.
    monkeypatch.setenv("LITWEAVE_DATABASE_URL", postgres_url)
    from litweave.runtime.db import reset_pool, transaction
    reset_pool()

    try:
        # Use RateLimitedRouter so the live test follows the production router path.
        from litweave.llm.rate_limit import RateLimitConfig, RateLimitedRouter
        router = RateLimitedRouter(
            bindings={
                AgentRole.PLANNER: RoleBinding(
                    provider=ProviderName.GLM,
                    model="glm-5.1",
                    temperature=0.0,
                ),
            },
            config=RateLimitConfig(),
        )
        registry = PromptRegistry()
        node = make_planner_node(router, registry)

        with transaction() as conn:
            rm = RunManager(conn)
            run = rm.create(
                topic="Survey of RLHF progress",
                idempotency_key=f"live-{time.time_ns()}",
            )

        state = make_initial_state(topic="Survey of RLHF progress")
        config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

        result = node(state, config)

        assert len(result["outline"]) >= 3, f"expected >=3 sections, got {len(result['outline'])}"
        for section in result["outline"]:
            assert section["section_id"], "section_id must be non-empty"
            assert section["title"], "title must be non-empty"
            assert len(section["research_questions"]) >= 1
            assert len(section["must_find_evidence"]) >= 1
    finally:
        # Always flush so subsequent tests re-read env-var after monkeypatch teardown.
        reset_pool()
