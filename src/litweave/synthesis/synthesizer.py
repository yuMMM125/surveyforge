"""Real Synthesizer node — schema-guided cross-paper extraction.

Replaces the stub (`stub.py`) for production use; stub remains in repo
for fast non-LLM tests (test_graph_smoke.py uses it via mock).

Per spec § 1.3 P3, Synthesizer is the single core differentiator from
STORM: cross-paper structured extraction with evidence-traceable
comparison matrix and taxonomy.

Design (Plan #3 ADs):
  - AD #1: single structured_call per section
  - AD #2: ONE schema-repair retry (max 2 LLM calls; structured_call max_retries=0)
  - AD #3: host-side grounding validation (paper_id / evidence_id subset + evidence-paper consistency)
  - AD #4: direct EvidenceStore read (no ToolGateway tool wrapper)
  - AD #5: 96K input budget enforcement (top_k rerank fallback; call-then-catch on BudgetExceeded)
  - AD #6: greedy whole-paper top_k by (avg_confidence DESC, claim_count DESC)
  - AD #7: coverage_gaps populated for missing_evidence + budget_truncation
  - AD #8: empty section → empty SynthesizerOutput (structurally present)
  - AD #11: factory closure pattern matching the existing agent factories
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from litweave.llm.roles import AgentRole
from litweave.llm.router import RouterProtocol
from litweave.llm.structured_output import StructuredCallError, structured_call
from litweave.prompts.loader import PromptRegistry, PromptTemplate
from litweave.runtime.budget import BudgetExceeded, BudgetManager
from litweave.runtime.db import transaction
from litweave.runtime.errors import classify_exception
from litweave.runtime.evidence import EvidenceItem, EvidenceStore
from litweave.runtime.observability import with_run_metadata
from litweave.runtime.runs import RunManager
from litweave.schemas.planner import PlannerSection
from litweave.schemas.synthesis import (
    ComparisonMatrix,
    CoverageGap,
    SectionTaxonomy,
    SynthesizerOutput,
)
from litweave.state import SurveyState

SynthesizerNode = Callable[[SurveyState, RunnableConfig], SurveyState]

CONTEXT_OVERFLOW = "context_overflow"
SCHEMA_INVALID = "schema_invalid"

# Rough token estimate: 4 chars per token (consistent with Wide/Deep heuristic).
# A real tokenizer can swap in here later if drift is observed.
_CHARS_PER_TOKEN = 4

# Literal glue text emitted by `_build_user_message` between the base
# template and the per-paper evidence blocks. Hoisted as module-level
# constants so `_packed_total_tokens` can charge their token overhead
# without duplicating string literals (and so a future style change here
# is reflected in budget accounting automatically).
_EVIDENCE_SECTION_HEADER = "\n\n## EvidenceItems for this section\n\n"
_BLOCK_JOINER = "\n\n"


def _estimate_tokens(text: str) -> int:
    """Rough English heuristic. Same as Wide/Deep."""
    return len(text) // _CHARS_PER_TOKEN


def _format_evidence_items_for_prompt(items: list[EvidenceItem]) -> str:
    """Render EvidenceItems into a stable prompt format. One block per item."""
    blocks = []
    for item in items:
        blocks.append(
            f"### {item.evidence_id}\n"
            f"paper_id: {item.paper_id}\n"
            f"confidence: {item.confidence:.2f}\n"
            f"claim: {item.claim}\n"
            f"source_span: {item.source_span or '(none)'}"
        )
    return _BLOCK_JOINER.join(blocks)


def _packed_total_tokens(items: list[EvidenceItem], base_tokens: int) -> int:
    """Canonical token accounting: base + EvidenceItems section header +
    per-paper-block sum + inter-block joiners.

    Matches the actual prompt string `_build_user_message` produces, so the
    up-front budget check, `_top_k_rerank` packing, and the AD #15
    deterministic test all share one accounting and agree on the budget
    boundary. Counting only `base + Σ per_paper_blocks` (without header /
    joiners) under-estimates real prompt size; including the overhead here
    makes the helper a pessimistic upper bound on real prompt size — safer.

    Empty items → just `base_tokens` (no EvidenceItems section emitted).
    """
    by_paper: dict[str, list[EvidenceItem]] = {}
    for item in items:
        by_paper.setdefault(item.paper_id, []).append(item)
    if not by_paper:
        return base_tokens
    header_tokens = _estimate_tokens(_EVIDENCE_SECTION_HEADER)
    per_paper_tokens = sum(
        _estimate_tokens(_format_evidence_items_for_prompt(paper_items))
        for paper_items in by_paper.values()
    )
    joiner_tokens = (len(by_paper) - 1) * _estimate_tokens(_BLOCK_JOINER)
    return base_tokens + header_tokens + per_paper_tokens + joiner_tokens


def _top_k_rerank(
    items: list[EvidenceItem],
    base_prompt_tokens: int,
    budget_tokens: int,
) -> tuple[list[EvidenceItem], list[str]]:
    """Greedy whole-paper top-K packing.

    Sort papers by (avg_confidence DESC, claim_count DESC); accept whole-paper
    if `_packed_total_tokens(selected + paper_items)` is within budget;
    reject and record paper_id otherwise.

    Per-candidate-set acceptance consults `_packed_total_tokens` so packing
    accounts for header + inter-block joiners on the same scale as the
    up-front budget projection — single source of truth, no slop.

    Whole-paper atomicity preserves paper_facts coherence — partial-paper
    truncation would let the LLM produce contradictory paper_facts entries
    citing only some of a paper's evidence.
    """
    by_paper: dict[str, list[EvidenceItem]] = {}
    for item in items:
        by_paper.setdefault(item.paper_id, []).append(item)

    def _paper_score(paper_items: list[EvidenceItem]) -> tuple[float, int]:
        avg_conf = sum(i.confidence for i in paper_items) / len(paper_items)
        return (avg_conf, len(paper_items))

    sorted_papers = sorted(
        by_paper.items(),
        key=lambda kv: _paper_score(kv[1]),
        reverse=True,
    )

    selected: list[EvidenceItem] = []
    dropped: list[str] = []
    for paper_id, paper_items in sorted_papers:
        candidate = selected + list(paper_items)
        if _packed_total_tokens(candidate, base_prompt_tokens) <= budget_tokens:
            selected = candidate
        else:
            dropped.append(paper_id)
    return selected, dropped


def _build_user_message(
    section: PlannerSection,
    items: list[EvidenceItem],
    template: PromptTemplate,
) -> str:
    """Format the prompt with section context + evidence_items.

    `template` is a `PromptTemplate` (the loader's frozen dataclass), not a
    raw `str` — call `template.format(**kwargs)` which delegates to
    `template.body.format(**kwargs)` per `prompts/loader.py:93-94`.
    """
    evidence_text = _format_evidence_items_for_prompt(items)
    return template.format(
        section_id=section.section_id,
        section_title=section.title,
        must_find_evidence=section.must_find_evidence,
    ) + _EVIDENCE_SECTION_HEADER + evidence_text


class GroundingError(ValueError):
    """Output structurally valid (Pydantic) but content not grounded in input
    evidence. Triggers schema-repair retry alongside ValidationError. Per AD #3.
    """


def _validate_grounding(
    output: SynthesizerOutput,
    input_paper_ids: set[str],
    input_evidence_ids: set[str],
    evidence_id_to_paper_id: dict[str, str],
    expected_section_id: str,
) -> None:
    """Host-side grounding validation (AD #3). Six layers (0-5):

      Layer 0 — section_id match: output.section_id MUST equal
                expected_section_id (the section currently being processed).
                Mismatch typically means the LLM lost track of which section
                it was synthesizing for (multi-section context confusion).
      Layer 1 — paper_id subset: every paper_id referenced anywhere in output
                MUST be in input_paper_ids. Includes claims[*].paper_id.
      Layer 2 — evidence_id subset: every evidence_id referenced anywhere
                MUST be in input_evidence_ids.
      Layer 3 — evidence-paper consistency, both directions:
                (3a) paper_facts[i].evidence_ids → items where paper_id == paper_facts[i].paper_id
                (3b) comparison_matrix.rows[i].cells[d].evidence_ids → items where paper_id == rows[i].paper_id
                (3c) cross_paper_synthesis[i].evidence_ids → items where paper_id ∈ synthesis_claim.paper_ids
                (3d) cross_paper_synthesis[i].paper_ids ⊆ {paper-of(eid) for eid in evidence_ids}
                     — every supporting paper must have ≥ 1 covering evidence_id; schema
                     `min_length=2` on evidence_ids is count-only, not coverage.
                (3e) claims[i].evidence_id → item where paper_id == claims[i].paper_id
                     — each backwards-compat claim must cite an evidence whose source
                     paper matches the claim's declared paper_id.
                (taxonomy.rationale_evidence_ids deliberately relaxed: rationale
                 evidence may legitimately come from any input paper.)
      Layer 4 — taxonomy: any TaxonomyCategory with non-empty paper_ids MUST
                have non-empty rationale_evidence_ids. Schema allows both fields
                to default to empty list independently; this layer enforces the
                "non-empty categories must be backed by rationale" contract.
      Layer 5 — non-empty-input coverage: when the section has input evidence
                (input_paper_ids non-empty), the output cannot be structurally
                empty.
                (5a) set(papers_cited) MUST equal input_paper_ids
                (5b) set(paper_facts[*].paper_id) MUST equal input_paper_ids
                     — one entry per cited paper, even if individual fields are
                     empty strings for unknown method/dataset/metric/result
                (5c) set(comparison_matrix.rows[*].paper_id) MUST equal input_paper_ids
                     — one row per cited paper, even with empty cells dict if no
                     dimension has supporting evidence
                (Empty-section path returns _empty_output_for_section directly
                without calling the LLM, so Layer 5 is never invoked there.)

    Raises `GroundingError` on first violation. Caller catches alongside
    `ValidationError` and feeds the error message back to the LLM in the
    schema-repair retry loop.
    """
    # Layer 0: section_id match
    if output.section_id != expected_section_id:
        raise GroundingError(
            f"output.section_id={output.section_id!r} does not match the "
            f"section currently being processed ({expected_section_id!r}). "
            f"Each Synthesizer call processes one section at a time; the "
            f"output's section_id MUST echo the input section_id verbatim."
        )

    # Layer 1: paper_id subset (claims[*].paper_id included)
    output_paper_ids: set[str] = set()
    output_paper_ids.update(output.papers_cited)
    output_paper_ids.update(c.paper_id for c in output.claims)
    output_paper_ids.update(pf.paper_id for pf in output.paper_facts)
    output_paper_ids.update(row.paper_id for row in output.comparison_matrix.rows)
    for cat in output.taxonomy.categories:
        output_paper_ids.update(cat.paper_ids)
    for sc in output.cross_paper_synthesis:
        output_paper_ids.update(sc.paper_ids)
    rogue_papers = output_paper_ids - input_paper_ids
    if rogue_papers:
        raise GroundingError(
            f"Output references paper_ids not in input evidence: "
            f"{sorted(rogue_papers)[:5]}. All paper_ids MUST be from input "
            f"evidence_items (input has: {sorted(input_paper_ids)[:5]}"
            f"{'...' if len(input_paper_ids) > 5 else ''})."
        )

    # Layer 2: evidence_id subset
    output_evidence_ids: set[str] = set()
    output_evidence_ids.update(c.evidence_id for c in output.claims)
    for pf in output.paper_facts:
        output_evidence_ids.update(pf.evidence_ids)
    for row in output.comparison_matrix.rows:
        for cell in row.cells.values():
            output_evidence_ids.update(cell.evidence_ids)
    for cat in output.taxonomy.categories:
        output_evidence_ids.update(cat.rationale_evidence_ids)
    for sc in output.cross_paper_synthesis:
        output_evidence_ids.update(sc.evidence_ids)
    rogue_evidences = output_evidence_ids - input_evidence_ids
    if rogue_evidences:
        raise GroundingError(
            f"Output references evidence_ids not in input: "
            f"{sorted(rogue_evidences)[:5]}. All evidence_ids MUST be from "
            f"input evidence_items."
        )

    # Layer 3a: paper_facts evidence-paper consistency
    for pf in output.paper_facts:
        for eid in pf.evidence_ids:
            actual_paper = evidence_id_to_paper_id.get(eid)
            if actual_paper != pf.paper_id:
                raise GroundingError(
                    f"paper_facts entry for {pf.paper_id!r} cites evidence_id "
                    f"{eid!r} which actually belongs to paper {actual_paper!r}. "
                    f"Each paper_facts entry's evidence_ids MUST come from the "
                    f"same paper_id."
                )

    # Layer 3b: matrix cell evidence-paper consistency
    for row in output.comparison_matrix.rows:
        for dim, cell in row.cells.items():
            for eid in cell.evidence_ids:
                actual_paper = evidence_id_to_paper_id.get(eid)
                if actual_paper != row.paper_id:
                    raise GroundingError(
                        f"comparison_matrix row for {row.paper_id!r}, dimension "
                        f"{dim!r}, cites evidence_id {eid!r} which belongs to "
                        f"paper {actual_paper!r}, not {row.paper_id!r}."
                    )

    # Layer 3c: cross_paper_synthesis evidence ⊆ supporting papers
    # Layer 3d: every supporting paper has ≥ 1 evidence_id covering it
    #
    # Both directions matter: 3c rejects "evidence from a paper not listed",
    # 3d rejects "paper listed but no evidence for it" (e.g., supporting
    # paper_ids=[A, B], evidence_ids=[E1] where E1 → A only). Schema-level
    # `min_length=2` on evidence_ids only ensures count ≥ 2, NOT coverage.
    for sc in output.cross_paper_synthesis:
        sc_paper_set = set(sc.paper_ids)
        evidence_papers: set[str] = set()
        for eid in sc.evidence_ids:
            actual_paper = evidence_id_to_paper_id.get(eid)
            if actual_paper not in sc_paper_set:
                raise GroundingError(
                    f"cross_paper_synthesis claim {sc.claim_text[:50]!r} cites "
                    f"evidence_id {eid!r} from paper {actual_paper!r} which is "
                    f"NOT in supporting paper_ids {sorted(sc_paper_set)}."
                )
            if actual_paper is not None:
                evidence_papers.add(actual_paper)
        uncovered = sc_paper_set - evidence_papers
        if uncovered:
            raise GroundingError(
                f"cross_paper_synthesis claim {sc.claim_text[:50]!r} lists "
                f"supporting paper_ids {sorted(uncovered)} without any covering "
                f"evidence_id. Each supporting paper MUST have ≥ 1 evidence_id "
                f"in evidence_ids (schema's min_length=2 is count-only, not "
                f"coverage)."
            )

    # Layer 3e: claims evidence-paper consistency
    for claim in output.claims:
        actual_paper = evidence_id_to_paper_id.get(claim.evidence_id)
        if actual_paper != claim.paper_id:
            raise GroundingError(
                f"claims entry cites evidence_id {claim.evidence_id!r} as "
                f"belonging to paper {claim.paper_id!r}, but that evidence "
                f"actually came from paper {actual_paper!r}. Each claim's "
                f"evidence_id MUST match its declared paper_id."
            )

    # Layer 4: taxonomy categories with paper_ids must have rationale_evidence_ids
    for cat in output.taxonomy.categories:
        if cat.paper_ids and not cat.rationale_evidence_ids:
            raise GroundingError(
                f"taxonomy category {cat.name!r} has paper_ids "
                f"{sorted(cat.paper_ids)} but empty rationale_evidence_ids. "
                f"Any non-empty TaxonomyCategory MUST have ≥ 1 rationale_evidence_id "
                f"explaining WHY those papers cluster together."
            )

    # Layer 5: non-empty input → output cannot be structurally empty
    if input_paper_ids:
        output_papers_cited = set(output.papers_cited)
        if output_papers_cited != input_paper_ids:
            raise GroundingError(
                f"papers_cited {sorted(output_papers_cited)} does not match "
                f"input paper set {sorted(input_paper_ids)}. Hard rule 5: "
                f"papers_cited MUST equal the dedup'd union of input "
                f"evidence_items' paper_ids — do NOT add or drop papers."
            )
        pf_paper_ids = {pf.paper_id for pf in output.paper_facts}
        missing_pf = input_paper_ids - pf_paper_ids
        if missing_pf:
            raise GroundingError(
                f"paper_facts is missing entries for input paper_ids "
                f"{sorted(missing_pf)}. There MUST be one paper_facts entry "
                f"per cited paper (use empty string '' for unknown method / "
                f"dataset / metric / result fields)."
            )
        matrix_paper_ids = {row.paper_id for row in output.comparison_matrix.rows}
        missing_matrix = input_paper_ids - matrix_paper_ids
        if missing_matrix:
            raise GroundingError(
                f"comparison_matrix is missing rows for input paper_ids "
                f"{sorted(missing_matrix)}. There MUST be one MatrixRow per "
                f"cited paper (use an empty `cells` dict if no dimension has "
                f"supporting evidence for that paper)."
            )


def _empty_output_for_section(section: PlannerSection) -> SynthesizerOutput:
    """Per AD #8: 0-evidence section → structurally present empty output +
    coverage_gaps for every must_find_evidence item."""
    return SynthesizerOutput(
        section_id=section.section_id,
        papers_cited=[],
        claims=[],
        paper_facts=[],
        comparison_matrix=ComparisonMatrix(),
        taxonomy=SectionTaxonomy(),
        cross_paper_synthesis=[],
        coverage_gaps=[
            CoverageGap(
                must_find_evidence_item=mfe,
                reason="missing_evidence",
                description="Section has 0 EvidenceItems from Researcher-Deep.",
            )
            for mfe in section.must_find_evidence
        ],
    )


def make_synthesizer_node(
    router: RouterProtocol,
    registry: PromptRegistry,
    budget_manager: BudgetManager,
) -> SynthesizerNode:
    """Build the real Synthesizer node.

    Factory closure pattern matching the existing agent factories: deps
    captured at graph init time; node body is pure (state, config) → state.
    """
    template = registry.load(AgentRole.SYNTHESIZER)
    _ = router.binding(AgentRole.SYNTHESIZER)  # validate role configured

    def synthesizer_node(state: SurveyState, config: RunnableConfig) -> SurveyState:
        run_id = config["configurable"]["thread_id"]
        outline = state.get("outline", [])

        # Stage transition: research_deep → synthesize
        with transaction() as conn:
            RunManager(conn).update_stage(run_id, "synthesize")

        new_extracts: dict[str, Any] = dict(state.get("structured_extracts", {}))
        last_error_category: str | None = None

        for section_dict in outline:
            section = PlannerSection.model_validate(section_dict)

            # Read evidence directly from EvidenceStore (AD #4 — repository
            # boundary, not tool gateway). Rationale: evidence read is internal
            # trusted-DB query with no audit value worth a tool_calls row.
            with transaction() as conn:
                store = EvidenceStore(conn)
                items = store.list_by_section(run_id, section.section_id)

            # Empty section: emit structurally-present empty output (AD #8)
            if not items:
                new_extracts[section.section_id] = _empty_output_for_section(
                    section
                ).model_dump()
                continue

            # Budget enforcement (AD #5 + #6): call-then-catch on BudgetExceeded;
            # the exception itself carries `max_input_tokens`, so we don't peek
            # into BudgetManager private state. Whole-paper atomicity in top_k.
            base_template_text = template.format(
                section_id=section.section_id,
                section_title=section.title,
                must_find_evidence=section.must_find_evidence,
            )
            base_tokens = _estimate_tokens(base_template_text)

            # AD #5: count tokens via `_packed_total_tokens` — the canonical
            # helper that also drives `_top_k_rerank` packing. The helper
            # charges header + joiner overhead so it equals (modulo integer-
            # division flooring) `_estimate_tokens(_build_user_message(...))`.
            # Using the helper directly keeps the up-front budget check, rerank
            # packing, and the AD #15 deterministic test on a single accounting
            # AND avoids the string allocation on the no-overflow path.
            full_tokens = _packed_total_tokens(items, base_tokens)

            dropped_paper_ids: list[str] = []
            try:
                budget_manager.check(AgentRole.SYNTHESIZER, full_tokens)
                user_message = _build_user_message(section, items, template)
            except BudgetExceeded as exc:
                # Use exception's reported budget (cleanest API surface).
                items, dropped_paper_ids = _top_k_rerank(
                    items, base_tokens, exc.max_input_tokens
                )
                if not items:
                    # Top_k emptied pool — degrade to empty output for section,
                    # record context_overflow on the run.
                    last_error_category = CONTEXT_OVERFLOW
                    new_extracts[section.section_id] = _empty_output_for_section(
                        section
                    ).model_dump()
                    continue
                user_message = _build_user_message(section, items, template)
                # Recheck after truncation, same packed-total accounting.
                try:
                    budget_manager.check(
                        AgentRole.SYNTHESIZER,
                        _packed_total_tokens(items, base_tokens),
                    )
                except BudgetExceeded:
                    last_error_category = CONTEXT_OVERFLOW
                    new_extracts[section.section_id] = _empty_output_for_section(
                        section
                    ).model_dump()
                    continue

            # Build host-side grounding sets for AD #3 validation
            input_paper_ids = {item.paper_id for item in items}
            input_evidence_ids = {item.evidence_id for item in items}
            evidence_id_to_paper_id = {item.evidence_id: item.paper_id for item in items}

            llm = router.get_llm(AgentRole.SYNTHESIZER)
            binding = router.binding(AgentRole.SYNTHESIZER)
            callback_config = with_run_metadata(
                run_id=run_id,
                stage="synthesize",
                agent_role=AgentRole.SYNTHESIZER,
                prompt_version=template.version,
                section_id=section.section_id,
            )

            messages: list[Any] = [HumanMessage(content=user_message)]

            # AD #2 + AD #3: max 2 LLM calls (`max_retries=0` on each
            # structured_call to disable its built-in retry; outer loop
            # provides the schema-repair retry semantics). Both Pydantic
            # ValidationError AND host-side GroundingError trigger the
            # repair retry; the second failure records SCHEMA_INVALID.
            output: SynthesizerOutput | None = None
            for attempt in (1, 2):
                try:
                    result_dict = structured_call(
                        llm,
                        messages,
                        schema=SynthesizerOutput.model_json_schema(),
                        tool_name="synthesizer_output",
                        max_retries=0,  # CRITICAL — see AD #2; default 2 → up to 6 LLM calls
                        supports_fc=binding.fc_enabled(),
                        config=callback_config,  # type: ignore[arg-type]
                    )
                    candidate = SynthesizerOutput.model_validate(result_dict)
                    _validate_grounding(
                        candidate,
                        input_paper_ids,
                        input_evidence_ids,
                        evidence_id_to_paper_id,
                        section.section_id,
                    )
                    output = candidate
                    break  # both shape + grounding passed
                except (StructuredCallError, ValidationError, GroundingError) as exc:
                    if attempt == 1:
                        # Schema-repair retry: feed exception text back to model.
                        # The model is told to fix ONLY the structural / grounding
                        # issue, not change semantic content.
                        if isinstance(exc, GroundingError):
                            repair_msg = (
                                "Your previous output passed JSON schema but FAILED "
                                "host-side grounding validation:\n\n"
                                f"{exc}\n\n"
                                "Fix ONLY the grounding issue: ensure every paper_id "
                                "comes from input evidence, every evidence_id is from "
                                "input, and evidence-paper consistency holds. Do NOT "
                                "change semantic content beyond fixing references."
                            )
                        else:
                            repair_msg = (
                                "Your previous output failed Pydantic / JSON schema "
                                f"validation:\n\n{exc!r}\n\n"
                                "Fix ONLY the structural issue (field/type/constraint). "
                                "Do NOT change semantic content."
                            )
                        messages.append(HumanMessage(content=repair_msg))
                        continue
                    # Second failure: record + continue to next section
                    last_error_category = SCHEMA_INVALID
                    output = None
                    break
                except Exception as exc:
                    # Transport / provider errors: classify; unclassified propagate.
                    classified = classify_exception(exc)
                    if classified is None:
                        raise
                    last_error_category = classified.value
                    output = None
                    break

            if output is None:
                continue  # error already recorded

            # Augment coverage_gaps with budget_truncation entries if any
            # paper was dropped during top_k rerank (AD #7).
            if dropped_paper_ids:
                augmented_gaps = [
                    *output.coverage_gaps,
                    CoverageGap(
                        must_find_evidence_item="(budget truncation)",
                        reason="budget_truncation",
                        description=f"Dropped papers due to input budget: {dropped_paper_ids}",
                    ),
                ]
                output = output.model_copy(update={"coverage_gaps": augmented_gaps})

            new_extracts[section.section_id] = output.model_dump()

        # Record any non-terminal error_category once at the end
        if last_error_category is not None:
            with transaction() as conn:
                RunManager(conn).note_error_category(run_id, last_error_category)

        return {**state, "structured_extracts": new_extracts}

    return synthesizer_node
