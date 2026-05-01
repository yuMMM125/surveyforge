"""Langfuse callback metadata helper per spec § 2.7.7."""
from __future__ import annotations

from typing import Any

from surveyforge.llm.roles import AgentRole


def with_run_metadata(
    run_id: str,
    stage: str,
    agent_role: AgentRole,
    **extra: Any,
) -> dict[str, Any]:
    """Build a Langfuse callback config dict with required correlation fields.

    Spec § 2.7.7 requires every model_call trace to carry run_id / stage /
    agent_role so post-hoc analysis can correlate spans within a run. Extra
    kwargs (e.g., prompt_version, paper_id) merge into metadata; if an extra
    key collides with a required field, the explicit positional argument wins.
    """
    metadata: dict[str, Any] = {
        **extra,
        "run_id": run_id,
        "stage": stage,
        "agent_role": agent_role.value,
    }
    return {"metadata": metadata, "tags": [stage, agent_role.value]}
