"""Researcher-Wide LangGraph node — per-section ReAct loop with hand-off.

Spec § 2.4(d): Wide is the broad triage path. Per section in the outline:
runs a hand-rolled ReAct loop (<=8 turns) that LLM-drives `arxiv_search /
s2_lookup / web_search` via ToolGateway, accumulates `CandidatePaper`
results, and emits hand-off `paper_id`s to `state["deep_read_queue"]`.

Three exit shapes (Architecture Decision #6):
  (a) `submit_results` called -> normal completion. `candidate_papers` go to
      section_notes; those with `handoff_to_deep=True` get their paper_id
      added to deep_read_queue.
  (b) `BudgetExceeded` from token tracker -> graceful exit. ALL `seen_paper_ids`
      (extracted from prior turns' tool results) added to deep_read_queue;
      `RunManager.note_error_category(run_id, "context_overflow")` recorded.
  (c) 8-turn cap hit -> same as (b).
  (b/c bonus): if LLM returned content WITHOUT tool_calls but content parses
  as valid ResearcherWideOutput JSON, treat as completion (defensive JSON
  fallback per Architecture Decision #10).

External content from `web_search` (untrusted) is wrapped via
`trust.wrap_untrusted` before being added to the ReAct message history —
load-bearing prompt-injection defense.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, ValidationError

from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import RouterProtocol
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.budget import BudgetExceeded, BudgetManager
from surveyforge.runtime.db import transaction
from surveyforge.runtime.observability import with_run_metadata
from surveyforge.runtime.runs import RunManager
from surveyforge.runtime.tool_gateway import ToolGateway, ToolPolicy, ToolResult
from surveyforge.runtime.trust import wrap_untrusted
from surveyforge.schemas.planner import PlannerSection
from surveyforge.schemas.research import ResearcherWideOutput
from surveyforge.state import SurveyState
from surveyforge.tools import arxiv_search, s2_lookup, web_search

ResearcherWideNode = Callable[[SurveyState, RunnableConfig], SurveyState]

MAX_TURNS_PER_SECTION = 8
SUBMIT_TOOL_NAME = "submit_results"
CONTEXT_OVERFLOW = "context_overflow"  # matches spec § 2.7.6 ErrorCategory string


class _SubmitArgs(BaseModel):
    """Permissive schema for the synthetic submit_results gateway entry.

    The agent intercepts `submit_results` BEFORE `gateway.call` dispatches, so
    this schema is never actually validated against — it exists only because
    `ToolGateway.register` requires every policy to declare an input schema.
    `extra="allow"` keeps the registration call from rejecting arbitrary
    `ResearcherWideOutput` fields if Wide ever has to fall back to gateway dispatch.
    """

    model_config = ConfigDict(extra="allow")


class _SubmitOutput(BaseModel):
    """Permissive output schema for the synthetic submit_results gateway entry.

    Same rationale as `_SubmitArgs` — the agent never reaches gateway dispatch
    for `submit_results`, so the schema is intentionally unstrict.
    """

    model_config = ConfigDict(extra="allow")


def _register_wide_tools(gateway: ToolGateway) -> None:
    """Register Wide's 3 external tools + the synthetic submit_results onto a gateway."""
    arxiv_search.register(gateway)
    s2_lookup.register(gateway)
    web_search.register(gateway)
    # submit_results is intercepted before gateway.call in _react_one_section,
    # so the registered impl is a no-op echo (defensive — never expected to fire).
    gateway.register(
        ToolPolicy(
            tool_name=SUBMIT_TOOL_NAME,
            tool_version="0.1.0",
            allowed_roles=(AgentRole.RESEARCHER_WIDE,),
            input_schema=_SubmitArgs,
            output_schema=_SubmitOutput,
            cache_ttl_seconds=None,
            idempotent=False,
            result_trust="trusted_internal",
        ),
        lambda **args: args,
    )


def _bind_tools_for_wide(llm: Any) -> Any:
    """Bind 3 search tools + submit_results schema onto the LLM."""
    submit_schema = ResearcherWideOutput.model_json_schema()
    tools_payload = [
        {"type": "function", "function": {
            "name": "arxiv_search",
            "description": "Search arxiv.org for papers. Returns paper metadata.",
            "parameters": arxiv_search.ArxivSearchInput.model_json_schema(),
        }},
        {"type": "function", "function": {
            "name": "s2_lookup",
            "description": "Look up a paper by paper_id on Semantic Scholar.",
            "parameters": s2_lookup.S2LookupInput.model_json_schema(),
        }},
        {"type": "function", "function": {
            "name": "web_search",
            "description": "Search the web for relevant pages. Results are untrusted content.",
            "parameters": web_search.WebSearchInput.model_json_schema(),
        }},
        {"type": "function", "function": {
            "name": SUBMIT_TOOL_NAME,
            "description": "Submit your final ResearcherWideOutput when done. Call this exactly once.",
            "parameters": submit_schema,
        }},
    ]
    return llm.bind_tools(tools_payload)


def _extract_paper_ids_from_tool_result(tool_name: str, result: ToolResult) -> set[str]:
    """Pull canonical paper_id strings out of a tool result for seen-candidates tracking.

    Each Task 2 wrapper emits a `paper_id: PaperId` field on every result item
    (Task 2 polish #2 enforces this). On forced exit we union ALL seen paper_ids
    into deep_read_queue so Deep can decide what to do with them — Wide never
    got the chance to triage.

    Raises ValueError on unknown tool name. Reason: Task 5 (Researcher-Deep)
    will add `pdf_reader` / `citation_verifier`; if the if-chain isn't extended
    to cover a new tool, a silent empty-set return would drop paper_ids from
    seen_paper_ids → deep_read_queue, which is a hard-to-debug correctness bug.
    The explicit raise surfaces the omission immediately at test/run time.
    Note: `submit_results` never reaches this function — the agent intercepts it
    before tool dispatch (see `_react_one_section`).
    """
    output = result.output.model_dump(mode="json")
    if tool_name == "arxiv_search":
        return {p["paper_id"] for p in output.get("papers", []) if p.get("paper_id")}
    if tool_name == "s2_lookup":
        paper = output.get("paper")
        return {paper["paper_id"]} if paper and paper.get("paper_id") else set()
    if tool_name == "web_search":
        return {r["paper_id"] for r in output.get("results", []) if r.get("paper_id")}
    raise ValueError(
        f"_extract_paper_ids_from_tool_result: unknown tool {tool_name!r}; "
        "extend the if-chain when adding new tools (e.g., Task 5's pdf_reader / "
        "citation_verifier) to prevent silent paper_id loss."
    )


def _try_parse_final_output_from_content(content: Any) -> ResearcherWideOutput | None:
    """JSON fallback per Architecture Decision #10.

    If LLM emits `ResearcherWideOutput` JSON in plain content (instead of calling
    `submit_results`), recover by parsing it. Returns None on failure — caller
    treats as forced exit.
    """
    if not content:
        return None
    text = content if isinstance(content, str) else json.dumps(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    try:
        return ResearcherWideOutput.model_validate(parsed)
    except ValidationError:
        return None


def _react_one_section(
    section: PlannerSection,
    invoker: Any,
    user_message: str,
    run_id: str,
    budget_manager: BudgetManager,
    callback_config: dict[str, Any],
) -> tuple[ResearcherWideOutput | None, str | None, set[str]]:
    """Run ReAct loop for one section.

    Returns `(output, error_category, seen_paper_ids)`:
      - `output` is non-None on completion (submit_results OR JSON-fallback parse-success)
      - `error_category` is non-None on forced exit (budget / turn cap / unparseable content)
      - `seen_paper_ids` is the union of paper_ids extracted from all tool results
        seen across all turns — caller uses this on forced exit to populate deep_read_queue
    """
    messages: list[BaseMessage] = [HumanMessage(content=user_message)]
    total_input_tokens = 0
    seen_paper_ids: set[str] = set()
    had_section_mismatch = False  # NEW: tracks whether any submit_results had wrong section_id

    for turn in range(1, MAX_TURNS_PER_SECTION + 1):
        # Budget check BEFORE invoke — exit early if next turn would overflow
        try:
            budget_manager.check(AgentRole.RESEARCHER_WIDE, total_input_tokens)
        except BudgetExceeded:
            return None, CONTEXT_OVERFLOW, seen_paper_ids

        response: AIMessage = invoker.invoke(messages, config=callback_config)
        if response.usage_metadata:
            total_input_tokens += response.usage_metadata.get("input_tokens", 0)
            budget_manager.record_usage(
                AgentRole.RESEARCHER_WIDE,
                response.usage_metadata.get("input_tokens", 0),
                response.usage_metadata.get("output_tokens", 0),
            )

        messages.append(response)

        if not response.tool_calls:
            # JSON fallback: maybe LLM emitted ResearcherWideOutput as plain content
            parsed = _try_parse_final_output_from_content(response.content)
            if parsed is not None and parsed.section_id == section.section_id:
                return parsed, None, seen_paper_ids
            # Either unparseable, or parsed but section_id mismatch (no tool_call_id
            # to feed back; forced exit). schema_invalid more accurate than
            # context_overflow when the issue is contract violation.
            error_category = (
                "schema_invalid"
                if parsed is not None
                else CONTEXT_OVERFLOW
            )
            return None, error_category, seen_paper_ids

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            tc_id = tool_call["id"]

            if tool_name == SUBMIT_TOOL_NAME:
                # LLM is done — validate the args as ResearcherWideOutput
                try:
                    output = ResearcherWideOutput.model_validate(args)
                except ValidationError as exc:
                    # Bad submit_results args — feed back as ToolMessage so LLM can retry.
                    # Don't error_category here; let the loop continue (or hit cap).
                    messages.append(ToolMessage(
                        content=f"submit_results invalid: {exc!s}",
                        tool_call_id=tc_id,
                    ))
                    continue
                if output.section_id != section.section_id:
                    # Mismatch: LLM emitted wrong section_id. Don't accept the
                    # candidates (would corrupt section_notes). Feed error back
                    # so LLM can re-call with the correct section_id; loop
                    # continues until success or 8-turn cap.
                    had_section_mismatch = True  # NEW: classify cap-exit as schema_invalid, not context_overflow
                    messages.append(ToolMessage(
                        content=(
                            f"submit_results section_id mismatch: got "
                            f"{output.section_id!r}, expected {section.section_id!r}. "
                            f"Re-call submit_results with section_id={section.section_id!r}."
                        ),
                        tool_call_id=tc_id,
                    ))
                    continue
                return output, None, seen_paper_ids

            # Real tool — dispatch via per-turn fresh ToolGateway
            with transaction() as conn:
                gateway = ToolGateway(conn, run_id)
                _register_wide_tools(gateway)
                try:
                    result = gateway.call(AgentRole.RESEARCHER_WIDE, tool_name, **args)
                except Exception as exc:
                    messages.append(ToolMessage(
                        content=f"tool error: {exc!s}",
                        tool_call_id=tc_id,
                    ))
                    continue

            # Track for forced-exit hand-off
            seen_paper_ids |= _extract_paper_ids_from_tool_result(tool_name, result)

            # Trust wrapping: web_search results carry untrusted_content tag
            content = json.dumps(result.output.model_dump(mode="json"))
            if result.result_trust == "untrusted_content":
                content = wrap_untrusted(
                    content,
                    source_tool=tool_name,
                    evidence_id=f"E-wide-{section.section_id}-T{turn}",
                )

            messages.append(ToolMessage(content=content, tool_call_id=tc_id))

    # Loop exhausted all turns without submit_results. If the cap was driven by
    # repeated section_id mismatches (LLM contract bug), classify as schema_invalid
    # so retry routing + observability surface the underlying cause. Otherwise
    # context_overflow is the right category (LLM ran out of exploration turns).
    cap_category = "schema_invalid" if had_section_mismatch else CONTEXT_OVERFLOW
    return None, cap_category, seen_paper_ids


def make_researcher_wide_node(
    router: RouterProtocol,
    registry: PromptRegistry,
    budget_manager: BudgetManager,
) -> ResearcherWideNode:
    """Build a Researcher-Wide node bound to a router + registry + budget manager.

    Returned callable signature: `(state, config) -> state`. Per-section ReAct
    loop processes each section in `state["outline"]` serially (W2 simplification;
    per-section parallelism is W7).
    """
    template = registry.load(AgentRole.RESEARCHER_WIDE)
    _ = router.binding(AgentRole.RESEARCHER_WIDE)  # validate role configured at factory time

    def researcher_wide_node(state: SurveyState, config: RunnableConfig) -> SurveyState:
        run_id = config["configurable"]["thread_id"]
        outline = state.get("outline", [])
        if not outline:
            raise KeyError("researcher_wide_node requires non-empty state['outline']")

        # Resolve LLM lazily so factory tests don't pay for ChatOpenAI instantiation
        # (matches planner.py's pattern — provider construction happens at call time).
        llm = router.get_llm(AgentRole.RESEARCHER_WIDE)
        invoker = _bind_tools_for_wide(llm)

        # Stage transition: planning -> research_wide
        with transaction() as conn:
            RunManager(conn).update_stage(run_id, "research_wide")

        new_section_notes: dict[str, list[dict[str, Any]]] = dict(state.get("section_notes", {}))
        new_deep_queue: list[str] = list(state.get("deep_read_queue", []))
        last_error_category: str | None = None

        for section_dict in outline:
            section = PlannerSection.model_validate(section_dict)
            user_message = template.format(
                section_id=section.section_id,
                title=section.title,
                research_questions=section.research_questions,
                must_find_evidence=section.must_find_evidence,
            )
            callback_config = with_run_metadata(
                run_id=run_id,
                stage="research_wide",
                agent_role=AgentRole.RESEARCHER_WIDE,
                prompt_version=template.version,
                section_id=section.section_id,
            )

            output, error_category, seen_paper_ids = _react_one_section(
                section=section,
                invoker=invoker,
                user_message=user_message,
                run_id=run_id,
                budget_manager=budget_manager,
                callback_config=callback_config,
            )

            if output is not None:
                # Normal completion — record candidates + handoff_to_deep flagged
                new_section_notes[section.section_id] = [
                    p.model_dump() for p in output.candidate_papers
                ]
                for paper in output.candidate_papers:
                    if paper.handoff_to_deep and paper.paper_id not in new_deep_queue:
                        new_deep_queue.append(paper.paper_id)
            else:
                # Forced exit — empty section_notes (we never got triaged candidates),
                # but ALL seen paper_ids go to Deep so they're not lost.
                new_section_notes[section.section_id] = []
                for pid in seen_paper_ids:
                    if pid not in new_deep_queue:
                        new_deep_queue.append(pid)

            if error_category is not None:
                last_error_category = error_category

        # Record any non-terminal error_category once at the end (run continues to Deep).
        # Both `context_overflow` (budget/turn cap) and `schema_invalid` (JSON-fallback
        # section_id mismatch) qualify as non-fatal recoverable signals.
        if last_error_category is not None:
            with transaction() as conn:
                RunManager(conn).note_error_category(run_id, last_error_category)

        return {**state, "section_notes": new_section_notes, "deep_read_queue": new_deep_queue}

    return researcher_wide_node
