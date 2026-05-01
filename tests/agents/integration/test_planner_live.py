"""Live Planner integration test — real GLM call.

Marked `pytest.mark.integration`; default `pytest` skips this file.
Run with `pytest tests/agents/integration -m integration`.
Requires `SJTU_MODELS_API_KEY` in env (loaded via tests/conftest.py::load_env).
"""
from __future__ import annotations

import os
import time

import pytest
from langchain_core.runnables import RunnableConfig
from psycopg_pool import ConnectionPool

from surveyforge.agents.planner import make_planner_node
from surveyforge.llm.providers import ProviderName
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import RoleBinding
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.runs import RunManager
from surveyforge.state import make_initial_state


@pytest.mark.integration
def test_planner_live_produces_outline_for_rlhf_survey(
    postgres_url: str,
    initialized_pool: ConnectionPool,  # ensures schema is applied before planner uses transaction()
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not os.environ.get("SJTU_MODELS_API_KEY"):
        pytest.skip("SJTU_MODELS_API_KEY not set")

    # Wire the testcontainer URL into the production DB code path so that
    # the run created here AND the planner node's RunManager.update_stage call
    # both go through the same Postgres instance. Without this, transaction()
    # would read SURVEYFORGE_DATABASE_URL (env-var, possibly unset or pointing
    # at a different DB) and fail the run lookup. Depending on `initialized_pool`
    # (session-scoped) guarantees schema.sql has been applied to this container.
    monkeypatch.setenv("SURVEYFORGE_DATABASE_URL", postgres_url)
    from surveyforge.runtime.db import reset_pool, transaction
    reset_pool()  # flush any pool cached from previous tests with a different URL

    # Use RateLimitedRouter (production-grade) — see Task 3 polish: planner
    # node accepts RouterProtocol; this verifies the protocol path end-to-end.
    from surveyforge.llm.rate_limit import RateLimitConfig, RateLimitedRouter
    router = RateLimitedRouter(
        bindings={
            AgentRole.PLANNER: RoleBinding(
                provider=ProviderName.GLM, model="glm-5.1", temperature=0.0,
            ),
        },
        config=RateLimitConfig(),  # all-defaults is fine for a single-call live test
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

    # Cleanup: flush pool so subsequent tests re-read env-var (which monkeypatch
    # will restore to its pre-test value at fixture teardown).
    reset_pool()
