"""Stub Writer — markdown draft with inline `[E-...]` citations.

A future real Writer will produce flowing prose with proper transitions,
academic-style citations, multi-paragraph synthesis. The stub assembles a
minimal markdown section draft so the e2e pipeline emits SOMETHING
readable — sufficient to verify plumbing works end-to-end.

Reads `state["structured_extracts"]` (from Synthesize); writes to
`state["section_drafts"]`. Renders all 8 SynthesizerOutput fields when
present (claims, comparison_matrix, taxonomy, cross_paper_synthesis,
coverage_gaps), with backwards-compat for empty/missing fields.
"""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig

from litweave.runtime.db import transaction
from litweave.runtime.runs import RunManager
from litweave.state import SurveyState

WriteStubNode = Callable[[SurveyState, RunnableConfig], SurveyState]


def make_write_stub_node() -> WriteStubNode:
    """Build the stub Writer node.

    Renders each SynthesizerOutput section as markdown:
      - `## title` heading
      - `- claim [E-...]` bullets for `claims`
      - `### Cross-paper synthesis` italic paragraphs for `cross_paper_synthesis`
      - `### Comparison matrix` markdown table for `comparison_matrix`
      - `### Categories` nested list for `taxonomy.categories`
      - `### Coverage gaps` warning footer for `coverage_gaps`
    Empty/missing fields are silently skipped.
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
            matrix = section_extract.get("comparison_matrix", {})
            taxonomy = section_extract.get("taxonomy", {})
            synthesis_claims = section_extract.get("cross_paper_synthesis", [])
            coverage_gaps = section_extract.get("coverage_gaps", [])

            lines: list[str] = [f"## {title}", ""]

            # Claims (backwards-compat with the pre-real-Synth path)
            if claims:
                for claim_obj in claims:
                    evidence_id = claim_obj["evidence_id"]
                    claim_text = claim_obj["claim"]
                    lines.append(f"- {claim_text} [{evidence_id}]")
            else:
                lines.append("_No evidence available for this section._")

            # Cross-paper synthesis (Synth-emitted, not Writer-generated prose)
            if synthesis_claims:
                lines.extend(["", "### Cross-paper synthesis", ""])
                for sc in synthesis_claims:
                    evidence_marks = " ".join(
                        f"[{eid}]" for eid in sc.get("evidence_ids", [])
                    )
                    lines.append(f"_{sc.get('claim_text', '')}_ {evidence_marks}")
                    lines.append("")

            # Comparison matrix (markdown table)
            rows = matrix.get("rows", [])
            dims = matrix.get("dimensions", [])
            if rows and dims:
                lines.extend(
                    [
                        "",
                        "### Comparison matrix",
                        "",
                        "| paper_id | " + " | ".join(dims) + " |",
                    ]
                )
                lines.append("|" + " --- |" * (len(dims) + 1))
                for row in rows:
                    cells = row.get("cells", {})
                    cell_values = [cells.get(d, {}).get("value", "") for d in dims]
                    lines.append(
                        f"| {row.get('paper_id', '')} | "
                        + " | ".join(cell_values)
                        + " |"
                    )

            # Taxonomy (nested list)
            categories = taxonomy.get("categories", [])
            if categories:
                lines.extend(["", "### Categories", ""])
                for cat in categories:
                    paper_ids_str = ", ".join(cat.get("paper_ids", []))
                    lines.append(
                        f"- **{cat.get('name', '')}**: "
                        f"{cat.get('description', '')} ({paper_ids_str})"
                    )

            # Coverage gaps (warning footer)
            if coverage_gaps:
                lines.extend(["", "### Coverage gaps", ""])
                for gap in coverage_gaps:
                    lines.append(
                        f"- {gap.get('must_find_evidence_item', '')} "
                        f"({gap.get('reason', '')}): "
                        f"{gap.get('description', '')}"
                    )

            new_drafts[section_id] = "\n".join(lines)

        return {**state, "section_drafts": new_drafts}

    return write_node
