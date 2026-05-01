"""SurveyState: LangGraph TypedDict per spec § 2.3.

W2 populates only `topic / outline / section_notes / deep_read_queue`. The
remaining fields are declared so the schema is stable from day 1, but they
default empty/None and are populated by W3+ tasks (Synthesizer, Writer,
Critic, Judge). `run_id` flows through `RunnableConfig["configurable"]["thread_id"]`,
NOT through SurveyState — that decouples state schema from runtime concerns
and matches LangGraph's checkpointer contract (thread_id == run_id, spec § 2.7.2).
"""
from __future__ import annotations

from typing import Any, TypedDict


class SurveyState(TypedDict):
    """Working memory shared across graph nodes within a single run.

    See spec § 2.5 for memory layering (working / checkpoint / evidence /
    long-term semantic) — SurveyState is ONLY working memory; evidence flows
    via `evidence_store` reads/writes (id references, not raw blobs in state).

    `outline` carries dumped `PlannerSection` dicts (via `model_dump()`) rather
    than the Pydantic model itself: LangGraph's checkpointer requires
    JSON-serializable state, and downstream nodes index via `section["title"]`.
    """

    # W2 — populated by Tasks 3-5
    topic: str
    outline: list[dict[str, Any]]               # PlannerSection dicts; produced by Planner (Task 3)
    section_notes: dict[str, list[dict[str, Any]]]  # section_id → notes; Wide/Deep (Tasks 4-5)
    deep_read_queue: list[str]                  # paper_ids hand-off; Wide → Deep (Tasks 4-5)

    # W3+ — declared but defaulted; populated by later tasks
    structured_extracts: dict[str, Any]         # Synthesizer (W3)
    section_drafts: dict[str, str]              # Writer (W4)
    section_critiques: dict[str, dict[str, Any]]  # Critic-section (W5)
    final_critique: dict[str, Any] | None       # Critic-final (W5)
    citations: list[dict[str, Any]]             # accumulated across nodes
    retry_counts: dict[str, int]                # per-stage retry counter
    final_survey: str | None                    # Writer final assembly (W4)


def make_initial_state(topic: str) -> SurveyState:
    """Construct an initial SurveyState for a new run.

    Defaults all post-W2 fields to empty/None. Caller is responsible for
    threading `run_id` via the RunnableConfig at graph invoke time.
    """
    return SurveyState(
        topic=topic,
        outline=[],
        section_notes={},
        deep_read_queue=[],
        structured_extracts={},
        section_drafts={},
        section_critiques={},
        final_critique=None,
        citations=[],
        retry_counts={},
        final_survey=None,
    )
