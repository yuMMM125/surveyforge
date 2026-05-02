"""with_run_metadata returns Langfuse-compatible callback config dict."""
from __future__ import annotations

from litweave.llm.roles import AgentRole
from litweave.runtime.observability import with_run_metadata


def test_basic_fields_present():
    cfg = with_run_metadata(
        run_id="run_abc", stage="planning", agent_role=AgentRole.PLANNER,
    )
    assert cfg["metadata"] == {
        "run_id": "run_abc",
        "stage": "planning",
        "agent_role": "planner",
    }
    assert "planning" in cfg["tags"]
    assert "planner" in cfg["tags"]


def test_extras_merge_into_metadata():
    cfg = with_run_metadata(
        run_id="run_abc",
        stage="research_wide",
        agent_role=AgentRole.RESEARCHER_WIDE,
        prompt_version="0.1.0",
        section_id="S1",
    )
    assert cfg["metadata"]["prompt_version"] == "0.1.0"
    assert cfg["metadata"]["section_id"] == "S1"
    # Required fields still present
    assert cfg["metadata"]["run_id"] == "run_abc"


def test_extras_cannot_overwrite_required_fields():
    """Caller passing run_id=... in extras must not silently override the explicit one."""
    cfg = with_run_metadata(
        run_id="run_abc",
        stage="planning",
        agent_role=AgentRole.PLANNER,
        run_id_override_attempt="should-land-as-this-key-only",
    )
    assert cfg["metadata"]["run_id"] == "run_abc"
    assert cfg["metadata"]["run_id_override_attempt"] == "should-land-as-this-key-only"
