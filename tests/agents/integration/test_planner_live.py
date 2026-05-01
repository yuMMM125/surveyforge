"""Live Planner integration test — real GLM call.

Marked `pytest.mark.integration`; default `pytest` skips this file.
Run with `pytest tests/agents/integration -m integration`.
Requires `SJTU_MODELS_API_KEY` in env (loaded via tests/conftest.py::load_env).
"""
from __future__ import annotations

import os
import time

import psycopg
import pytest
from langchain_core.runnables import RunnableConfig

from surveyforge.agents.planner import make_planner_node
from surveyforge.llm.providers import ProviderName
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import LLMRouter, RoleBinding
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.runs import RunManager
from surveyforge.state import make_initial_state


@pytest.mark.integration
def test_planner_live_produces_outline_for_rlhf_survey(conn: psycopg.Connection) -> None:
    if not os.environ.get("SJTU_MODELS_API_KEY"):
        pytest.skip("SJTU_MODELS_API_KEY not set")

    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(
            provider=ProviderName.GLM, model="glm-5.1", temperature=0.0,
        ),
    })
    registry = PromptRegistry()
    node = make_planner_node(router, registry)

    rm = RunManager(conn)
    run = rm.create(topic="Survey of RLHF progress", idempotency_key=f"live-{time.time_ns()}")
    state = make_initial_state(topic="Survey of RLHF progress")
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

    result = node(state, config)

    assert len(result["outline"]) >= 3, f"expected >=3 sections, got {len(result['outline'])}"
    for section in result["outline"]:
        assert section["section_id"], "section_id must be non-empty"
        assert section["title"], "title must be non-empty"
        assert len(section["research_questions"]) >= 1
        assert len(section["must_find_evidence"]) >= 1
