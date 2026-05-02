"""Live Researcher-Wide integration test — real DeepSeek + real arxiv_search.

Marked `pytest.mark.integration`. Uses the same env-var-injection +
reset_pool pattern as test_planner_live (Task 3 polish), with the same
placeholder-key rejection.
"""
from __future__ import annotations

import os
import time

import pytest
from langchain_core.runnables import RunnableConfig
from psycopg_pool import ConnectionPool

from surveyforge.agents.researcher_wide import make_researcher_wide_node
from surveyforge.llm.providers import ProviderName
from surveyforge.llm.rate_limit import RateLimitConfig, RateLimitedRouter
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import RoleBinding
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.budget import BudgetManager
from surveyforge.runtime.runs import RunManager
from surveyforge.schemas.planner import PlannerSection
from surveyforge.state import make_initial_state

_PLACEHOLDER_KEY_PREFIXES = (
    "fake-", "fake_", "PASTE_YOUR_", "your-key", "test-", "dummy-", "placeholder",
)


def _is_placeholder_key(key: str) -> bool:
    return any(key.startswith(p) for p in _PLACEHOLDER_KEY_PREFIXES)


@pytest.mark.integration
def test_researcher_wide_live_finds_papers_for_long_context_topic(
    postgres_url: str,
    initialized_pool: ConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = os.environ.get("MODELS_API_KEY") or os.environ.get("SJTU_MODELS_API_KEY", "")
    if not api_key:
        pytest.skip("MODELS_API_KEY not set")
    if _is_placeholder_key(api_key):
        pytest.skip(f"MODELS_API_KEY appears to be a placeholder ({api_key[:12]}...)")

    monkeypatch.setenv("SURVEYFORGE_DATABASE_URL", postgres_url)
    from surveyforge.runtime.db import reset_pool, transaction
    reset_pool()

    try:
        router = RateLimitedRouter(
            bindings={
                AgentRole.RESEARCHER_WIDE: RoleBinding(
                    provider=ProviderName.DEEPSEEK, model="deepseek-chat", temperature=0.0,
                ),
            },
            config=RateLimitConfig(),
        )
        registry = PromptRegistry()
        budget_manager = BudgetManager()
        node = make_researcher_wide_node(router, registry, budget_manager)

        with transaction() as conn:
            run = RunManager(conn).create(
                topic="long-context LLM benchmarks",
                idempotency_key=f"live-wide-{time.time_ns()}",
            )

        state = make_initial_state(topic="long-context LLM benchmarks")
        state["outline"] = [
            PlannerSection(
                section_id="S1",
                title="Long-context benchmark methodologies",
                research_questions=[
                    "What benchmarks evaluate context length scaling?",
                    "How do they measure recall at long context?",
                ],
                must_find_evidence=[
                    "needle-in-haystack benchmark or RULER or LongBench"
                ],
            ).model_dump()
        ]
        config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

        result = node(state, config)

        assert "S1" in result["section_notes"], "S1 should have section_notes after wide run"
        # Live LLM behavior: at minimum, some paper should be queued for deep read
        # (relax the assertion if DeepSeek is too conservative; spec target >=1)
        assert len(result["deep_read_queue"]) >= 1, (
            f"expected >=1 paper in deep_read_queue, got {result['deep_read_queue']}"
        )
    finally:
        reset_pool()
