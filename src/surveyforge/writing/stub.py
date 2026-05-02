"""W2 stub Writer — markdown bullet draft with inline `[E-...]` citations.

Real W4 Writer will produce flowing prose with proper transitions, academic-
style citations, multi-paragraph synthesis. The W2 stub assembles a minimal
markdown section draft so the e2e pipeline emits SOMETHING readable —
sufficient to verify plumbing works end-to-end.

Reads `state["structured_extracts"]` (from Synthesize stub); writes to
`state["section_drafts"]`.
"""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig

from surveyforge.runtime.db import transaction
from surveyforge.runtime.runs import RunManager
from surveyforge.state import SurveyState

WriteStubNode = Callable[[SurveyState, RunnableConfig], SurveyState]


def make_write_stub_node() -> WriteStubNode:
    """Build the W2 stub Writer node.

    Reads `state["structured_extracts"]`, produces markdown per section as
    `## Title\\n\\n- claim [E-...]\\n` lines into `state["section_drafts"]`.
    """
    def write_node(state: SurveyState, config: RunnableConfig) -> SurveyState:
        run_id = config["configurable"]["thread_id"]
        outline = state.get("outline", [])
        extracts = state.get("structured_extracts", {})

        with transaction() as conn:
            RunManager(conn).update_stage(run_id, "write")

        new_drafts: dict[str, str] = dict(state.get("section_drafts", {}))
        for section_dict in outline:
            section_id = section_dict["section_id"]
            title = section_dict.get("title", section_id)
            section_extract = extracts.get(section_id, {})
            claims = section_extract.get("claims", [])

            lines: list[str] = [f"## {title}", ""]
            if not claims:
                lines.append("_No evidence available for this section in W2._")
            else:
                for claim_obj in claims:
                    evidence_id = claim_obj["evidence_id"]
                    claim_text = claim_obj["claim"]
                    lines.append(f"- {claim_text} [{evidence_id}]")
            new_drafts[section_id] = "\n".join(lines)

        return {**state, "section_drafts": new_drafts}

    return write_node
