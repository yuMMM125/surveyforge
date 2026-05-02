"""W2 stub Synthesizer — bullet-list evidence dedup.

Real W3 Synthesizer will output structured `SynthesizerOutput` Pydantic with
comparison_matrix, taxonomy clustering, etc. The W2 stub returns a dict with
the same TOP-LEVEL shape (`section_id` / `papers_cited` / `claims`) so the
Writer's interface is forward-compatible — only the content depth differs.

Reads `evidence_items` table for run_id (NOT from state — Bundle 1c separated
evidence storage from working memory; state stays small + JSON-serializable).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from surveyforge.runtime.db import transaction
from surveyforge.runtime.evidence import EvidenceStore
from surveyforge.runtime.runs import RunManager
from surveyforge.state import SurveyState

SynthesizeStubNode = Callable[[SurveyState, RunnableConfig], SurveyState]


def make_synthesize_stub_node() -> SynthesizeStubNode:
    """Build the W2 stub Synthesizer node.

    For each section in `state["outline"]`: read its `evidence_items` rows,
    dedup paper_ids (preserving first-seen order), populate
    `state["structured_extracts"][section_id]` with a forward-compat dict.
    """
    def synthesize_node(state: SurveyState, config: RunnableConfig) -> SurveyState:
        run_id = config["configurable"]["thread_id"]
        outline = state.get("outline", [])

        # Stage transition: research_deep → synthesize
        with transaction() as conn:
            RunManager(conn).update_stage(run_id, "synthesize")

        new_extracts: dict[str, Any] = dict(state.get("structured_extracts", {}))

        with transaction() as conn:
            store = EvidenceStore(conn)
            for section_dict in outline:
                section_id = section_dict["section_id"]
                items = store.list_by_section(run_id, section_id)
                seen_paper_ids: set[str] = set()
                papers_cited: list[str] = []
                claims: list[dict[str, Any]] = []
                for item in items:
                    if item.paper_id not in seen_paper_ids:
                        seen_paper_ids.add(item.paper_id)
                        papers_cited.append(item.paper_id)
                    claims.append({
                        "evidence_id": item.evidence_id,
                        "paper_id": item.paper_id,
                        "claim": item.claim,
                        "confidence": item.confidence,
                    })
                new_extracts[section_id] = {
                    "section_id": section_id,
                    "papers_cited": papers_cited,
                    "claims": claims,
                }

        return {**state, "structured_extracts": new_extracts}

    return synthesize_node
