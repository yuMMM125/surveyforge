"""Planner node unit tests — mocked LLMRouter + structured_call.

Live LLM testing lives in tests/agents/integration/test_planner_live.py.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from langchain_core.runnables import RunnableConfig

from surveyforge.agents.planner import make_planner_node
from surveyforge.llm.providers import ProviderName
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import LLMRouter, RoleBinding
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.runs import RunManager, RunStatus
from surveyforge.state import make_initial_state

CANNED_PLANNER_OUTPUT = {
    "topic": "Survey of RLHF",
    "sections": [
        {
            "section_id": "S1",
            "title": "Background",
            "research_questions": ["What is RLHF?", "Why does it matter?"],
            "must_find_evidence": ["Original RLHF paper"],
        },
        {
            "section_id": "S2",
            "title": "Methods",
            "research_questions": ["What methods exist?", "How do they compare?"],
            "must_find_evidence": ["PPO-based RLHF"],
        },
        {
            "section_id": "S3",
            "title": "Open challenges",
            "research_questions": ["What are limitations?", "What's next?"],
            "must_find_evidence": ["Reward hacking findings"],
        },
    ],
    "rationale": "Three sections cover background, methods, and open challenges.",
}


def _make_run(conn: psycopg.Connection) -> str:
    rm = RunManager(conn)
    return rm.create(topic="test", idempotency_key=f"key-{id(conn)}").run_id


@pytest.fixture
def planner_node(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Planner node bound to mocked router + real PromptRegistry."""
    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(
            provider=ProviderName.GLM, model="glm-5.1", temperature=0.0,
        ),
    })
    # Mock get_llm to return a MagicMock ChatOpenAI
    mock_llm = MagicMock()
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=mock_llm))
    # Mock structured_call to return our canned output
    monkeypatch.setattr(
        "surveyforge.agents.planner.structured_call",
        MagicMock(return_value=CANNED_PLANNER_OUTPUT),
    )
    registry = PromptRegistry()
    return make_planner_node(router, registry)


def test_planner_node_populates_outline_in_state(
    conn: psycopg.Connection,
    patch_planner_transaction: psycopg.Connection,
    planner_node,  # type: ignore[no-untyped-def]
) -> None:
    run_id = _make_run(conn)
    state = make_initial_state(topic="Survey of RLHF")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = planner_node(state, config)
    assert len(result["outline"]) == 3
    assert result["outline"][0]["section_id"] == "S1"
    assert result["outline"][0]["title"] == "Background"


def test_planner_node_preserves_other_state_fields(
    conn: psycopg.Connection,
    patch_planner_transaction: psycopg.Connection,
    planner_node,  # type: ignore[no-untyped-def]
) -> None:
    run_id = _make_run(conn)
    state = make_initial_state(topic="Survey of RLHF")
    state["deep_read_queue"] = ["arxiv:2401.12345"]  # pre-existing field
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = planner_node(state, config)
    assert result["deep_read_queue"] == ["arxiv:2401.12345"]  # untouched
    assert result["topic"] == "Survey of RLHF"


def test_planner_node_missing_topic_raises(
    conn: psycopg.Connection,
    patch_planner_transaction: psycopg.Connection,
    planner_node,  # type: ignore[no-untyped-def]
) -> None:
    run_id = _make_run(conn)
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    with pytest.raises(KeyError, match="topic"):
        planner_node({}, config)  # type: ignore[arg-type]


def test_planner_node_calls_run_manager_update_stage(
    conn: psycopg.Connection,
    patch_planner_transaction: psycopg.Connection,
    planner_node,  # type: ignore[no-untyped-def]
) -> None:
    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    planner_node(state, config)
    # Verify runs.current_stage == "planning" after node runs
    rm = RunManager(conn)
    refreshed = rm.get(run_id)
    assert refreshed.current_stage == "planning"
    assert refreshed.status == RunStatus.RUNNING


def test_planner_node_invokes_structured_call_with_planner_output_schema(
    conn: psycopg.Connection,
    patch_planner_transaction: psycopg.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify structured_call gets the right schema + config."""
    captured: dict[str, Any] = {}

    def fake_structured_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["schema"] = kwargs.get("schema") or args[2]
        captured["config"] = kwargs.get("config")
        captured["supports_fc"] = kwargs.get("supports_fc")
        return CANNED_PLANNER_OUTPUT

    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(
            provider=ProviderName.GLM, model="glm-5.1", temperature=0.0,
        ),
    })
    mock_llm = MagicMock()
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=mock_llm))
    monkeypatch.setattr(
        "surveyforge.agents.planner.structured_call", fake_structured_call,
    )
    node = make_planner_node(router, PromptRegistry())

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Schema is PlannerOutput's JSON Schema
    assert captured["schema"]["title"] == "PlannerOutput"
    # Config carries Langfuse correlation metadata
    assert captured["config"]["metadata"]["run_id"] == run_id
    assert captured["config"]["metadata"]["stage"] == "planning"
    assert captured["config"]["metadata"]["agent_role"] == "planner"
    assert "prompt_version" in captured["config"]["metadata"]


def test_planner_node_factory_returns_callable() -> None:
    """make_planner_node returns a Callable bound to the router + registry."""
    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    })
    registry = PromptRegistry()
    node = make_planner_node(router, registry)
    assert callable(node)
