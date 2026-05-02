"""Planner LangGraph node — produces survey outline from topic.

Spec § 2.6.3: Planner has empty `allowed_tools` (no tool calls); pure
LLM-side decomposition of topic → 3-7 sections, each with 2-4 research
questions and 1-3 must-find-evidence items. `structured_call` validates
against `PlannerOutput.model_json_schema()`; if the LLM returns invalid
output the corrective-retry path inside `structured_call` re-prompts.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from litweave.llm.roles import AgentRole
from litweave.llm.router import RouterProtocol
from litweave.llm.structured_output import structured_call
from litweave.prompts.loader import PromptRegistry
from litweave.runtime.db import transaction
from litweave.runtime.observability import with_run_metadata
from litweave.runtime.runs import RunManager
from litweave.schemas.planner import PlannerOutput
from litweave.state import SurveyState

PlannerNode = Callable[[SurveyState, RunnableConfig], SurveyState]


def make_planner_node(
    router: RouterProtocol,
    registry: PromptRegistry,
) -> PlannerNode:
    """Build a Planner node bound to a specific LLMRouter + PromptRegistry.

    Returned callable signature: `(state, config) -> state` (LangGraph idiomatic).
    `config["configurable"]["thread_id"]` MUST be set by the caller — that's
    the `run_id` used for `RunManager.update_stage` + Langfuse correlation.
    """
    template = registry.load(AgentRole.PLANNER)
    binding = router.binding(AgentRole.PLANNER)

    def planner_node(state: SurveyState, config: RunnableConfig) -> SurveyState:
        topic = state["topic"]  # KeyError if missing — caller's contract violation
        run_id = config["configurable"]["thread_id"]

        # 1. Stage transition: PENDING → RUNNING (current_stage = "planning")
        with transaction() as conn:
            RunManager(conn).update_stage(run_id, "planning")

        # 2. Render prompt + build Langfuse metadata
        user_message = template.format(topic=topic)
        callback_config = with_run_metadata(
            run_id=run_id,
            stage="planning",
            agent_role=AgentRole.PLANNER,
            prompt_version=template.version,
        )

        # 3. Call LLM via structured_call (validates against PlannerOutput JSON Schema)
        llm = router.get_llm(AgentRole.PLANNER)
        result_dict: dict[str, Any] = structured_call(
            llm,
            [HumanMessage(content=user_message)],
            schema=PlannerOutput.model_json_schema(),
            tool_name="planner_output",
            max_retries=2,
            supports_fc=binding.fc_enabled(),
            config=callback_config,  # type: ignore[arg-type]
        )

        # 4. Pydantic re-validation (defense in depth + typed access)
        output = PlannerOutput.model_validate(result_dict)

        # 5. State mutation: outline is the produced sections
        new_state: SurveyState = {**state, "outline": [s.model_dump() for s in output.sections]}
        return new_state

    return planner_node
