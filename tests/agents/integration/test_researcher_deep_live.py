"""Live Researcher-Deep integration test — real MiniMax + real s2_lookup.

Marked `pytest.mark.integration`. Same env-var-injection + reset_pool pattern
as test_planner_live / test_researcher_wide_live (Task 3 / Task 4 polish).
"""
from __future__ import annotations

import os
import time

import pytest
from langchain_core.runnables import RunnableConfig
from psycopg_pool import ConnectionPool

from surveyforge.agents.researcher_deep import make_researcher_deep_node
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
def test_researcher_deep_live_extracts_evidence_for_rlhf_paper(
    postgres_url: str,
    initialized_pool: ConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
):
    api_key = os.environ.get("SJTU_MODELS_API_KEY", "")
    if not api_key:
        pytest.skip("SJTU_MODELS_API_KEY not set")
    if _is_placeholder_key(api_key):
        pytest.skip(f"SJTU_MODELS_API_KEY appears to be a placeholder ({api_key[:12]}...)")

    monkeypatch.setenv("SURVEYFORGE_DATABASE_URL", postgres_url)
    from surveyforge.runtime.db import reset_pool, transaction
    reset_pool()

    try:
        router = RateLimitedRouter(
            bindings={
                AgentRole.RESEARCHER_DEEP: RoleBinding(
                    provider=ProviderName.MINIMAX, model="minimax", temperature=0.0,
                ),
            },
            config=RateLimitConfig(),
        )
        registry = PromptRegistry()
        budget_manager = BudgetManager()
        node = make_researcher_deep_node(router, registry, budget_manager)

        with transaction() as conn:
            run = RunManager(conn).create(
                topic="Survey of RLHF",
                idempotency_key=f"live-deep-{time.time_ns()}",
            )

        section = PlannerSection(
            section_id="S1",
            title="Background",
            research_questions=["What is RLHF?", "Why does it matter for AI alignment?"],
            must_find_evidence=["Original RLHF paper"],
        )
        state = make_initial_state(topic="Survey of RLHF")
        state["outline"] = [section.model_dump()]
        state["section_notes"] = {
            "S1": [{
                "paper_id": "arxiv:1706.03741",
                "title": "Original RLHF paper",
                "source": "arxiv",
                "why_relevant": "Original RLHF formulation",
                "handoff_to_deep": True,
            }]
        }
        state["deep_read_queue"] = ["arxiv:1706.03741"]
        config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

        node(state, config)

        # Verify ≥1 EvidenceCard in evidence_items table
        with transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM evidence_items WHERE run_id = %s",
                (run.run_id,),
            )
            count = cur.fetchone()[0]
        assert count >= 1, f"expected >=1 EvidenceCard, got {count}"
    finally:
        reset_pool()
