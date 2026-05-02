"""Researcher-Deep LangGraph node — produces EvidenceCards from triaged papers.

Spec § 2.4(d) Deep path. Single `structured_call` per section (NOT a ReAct
loop — Deep takes pre-triaged papers and produces structured EvidenceCard
output in one shot). Pre-fetches abstracts via `s2_lookup` (Bundle 1b cache
hits free re-fetches Wide already did). EvidenceCards persisted via the
real `evidence_store_write` impl (replaces Bundle 1b placeholder).

W2 limitations (Architecture Decisions):
  - `web:` papers skipped (no web re-fetch path; Decision #3) — removed from
    `deep_read_queue` to avoid cycling forever; W3 will need a separate path
  - No real `pdf_reader` — abstracts only; W3 adds full text
  - Wide forced-exit papers come in as CandidatePaper-shaped stubs (Decision #5)
    so `section_notes` cross-reference picks them up uniformly with real candidates

`deep_read_queue` semantics: papers that successfully made it through a section's
`structured_call` (regardless of whether the LLM emitted evidence cards or marked
them insufficient) are removed from the queue. Papers from sections that hit
transient failures (provider_429/5xx, BudgetExceeded, schema_invalid, abstract
fetch errors) STAY in the queue so an upstream retry / orchestrator can re-enter.
Web: papers also leave the queue (intentional W2 skip).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, ValidationError

from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import RouterProtocol
from surveyforge.llm.structured_output import StructuredCallError, structured_call
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.budget import BudgetExceeded, BudgetManager
from surveyforge.runtime.db import transaction
from surveyforge.runtime.errors import classify_exception
from surveyforge.runtime.evidence import EvidenceItem, EvidenceStore
from surveyforge.runtime.observability import with_run_metadata
from surveyforge.runtime.runs import RunManager
from surveyforge.runtime.tool_gateway import ToolGateway, ToolPolicy
from surveyforge.schemas.planner import PlannerSection
from surveyforge.schemas.research import ResearcherDeepOutput
from surveyforge.state import SurveyState
from surveyforge.tools import s2_lookup

ResearcherDeepNode = Callable[[SurveyState, RunnableConfig], SurveyState]

CONTEXT_OVERFLOW = "context_overflow"
SCHEMA_INVALID = "schema_invalid"


class _EvidenceWriteOutput(BaseModel):
    """Real `evidence_store_write` impl output schema (replaces Bundle 1b _OpaqueOutput)."""
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    saved: bool


def _make_evidence_write_impl(conn: psycopg.Connection) -> Callable[..., dict[str, Any]]:
    """Build the real `evidence_store_write` impl bound to a specific connection.

    The impl takes EvidenceItem fields as kwargs (validated by input_schema),
    constructs an EvidenceItem (Pydantic re-validation = persistence-boundary
    safety per Bundle 1c), and saves via EvidenceStore. Returns evidence_id +
    saved=True so callers can confirm persistence.
    """
    store = EvidenceStore(conn)

    def write_impl(**args: Any) -> dict[str, Any]:
        item = EvidenceItem(**args)
        store.save(item)
        return {"evidence_id": item.evidence_id, "saved": True}

    return write_impl


def _register_deep_tools(gateway: ToolGateway, conn: psycopg.Connection) -> None:
    """Register Deep's tools onto a per-call gateway.

    `s2_lookup`: for abstract pre-fetch. `evidence_store_write` (real impl,
    replaces Bundle 1b placeholder): for EvidenceCard persistence.
    """
    s2_lookup.register(gateway)
    evidence_write_policy = ToolPolicy(
        tool_name="evidence_store_write",
        tool_version="0.1.0",
        allowed_roles=(AgentRole.RESEARCHER_DEEP,),
        input_schema=EvidenceItem,
        output_schema=_EvidenceWriteOutput,
        idempotent=False,
        cache_ttl_seconds=None,
        result_trust="trusted_internal",
    )
    gateway.register(evidence_write_policy, _make_evidence_write_impl(conn))


def _fetch_abstract(gateway: ToolGateway, paper_id: str) -> str | None:
    """Look up paper abstract via s2_lookup.

    Returns None for legitimate "no abstract available" cases:
      - `web:` prefix (W2 doesn't fetch web)
      - s2 lookup returned `paper=None` (404 — handled inside the wrapper)
      - paper exists but `abstract` field is empty/null

    Does NOT catch exceptions. The caller (node body) wraps invocations in a
    try/except + classify_exception so transport errors (httpx.HTTPStatusError
    429/5xx) become `provider_429`/`provider_5xx` (retry-eligible), and
    programmer errors (`ToolRoleDenied`, `ValidationError`) propagate unmasked.
    Pretending these are "no abstract" would silently drop retryable papers
    AND mask config bugs.
    """
    if paper_id.startswith("web:"):
        return None
    result = gateway.call(AgentRole.RESEARCHER_DEEP, "s2_lookup", paper_id=paper_id)
    # `result.output` is typed as the generic `BaseModel`; narrow to the concrete
    # `S2LookupOutput` so attribute access is mypy-strict-clean.
    output = s2_lookup.S2LookupOutput.model_validate(result.output.model_dump())
    paper = output.paper
    if paper is None or not paper.abstract:
        return None
    return paper.abstract


def make_researcher_deep_node(
    router: RouterProtocol,
    registry: PromptRegistry,
    budget_manager: BudgetManager,
) -> ResearcherDeepNode:
    """Build a Researcher-Deep node bound to a router + registry + budget manager."""
    template = registry.load(AgentRole.RESEARCHER_DEEP)
    _ = router.binding(AgentRole.RESEARCHER_DEEP)  # validate role configured

    def researcher_deep_node(state: SurveyState, config: RunnableConfig) -> SurveyState:
        run_id = config["configurable"]["thread_id"]
        deep_read_queue = state.get("deep_read_queue", [])
        section_notes = state.get("section_notes", {})
        outline = state.get("outline", [])

        # Stage transition: research_wide → research_deep
        with transaction() as conn:
            RunManager(conn).update_stage(run_id, "research_deep")

        # Build section_id → PlannerSection map
        sections_by_id: dict[str, PlannerSection] = {}
        for s_dict in outline:
            parsed = PlannerSection.model_validate(s_dict)
            sections_by_id[parsed.section_id] = parsed

        # Build section_id → list[paper_id] map by cross-referencing section_notes.
        # With Decision #5, Wide always writes section_notes entries (real candidates
        # OR forced-exit stubs), so every paper Wide saw has a section.
        deep_queue_set = set(deep_read_queue)
        papers_by_section: dict[str, list[str]] = {}
        for section_id, papers in section_notes.items():
            for paper in papers:
                pid = paper.get("paper_id")
                if pid in deep_queue_set:
                    papers_by_section.setdefault(section_id, []).append(pid)

        last_error_category: str | None = None
        # Track which papers reached a "section's structured_call completed
        # successfully" state. Anything left in `deep_read_queue` after the
        # outer loop = unprocessed (transient failure / no abstract).
        processed: set[str] = set()

        for section_id, paper_ids in papers_by_section.items():
            section = sections_by_id.get(section_id)
            if section is None:
                continue

            # Pre-fetch abstracts. _fetch_abstract no longer swallows exceptions:
            # transport errors (provider_429/5xx) propagate so the section gets
            # marked retry-eligible (papers stay in deep_read_queue). web: papers
            # legitimately return None.
            abstracts: dict[str, str] = {}
            try:
                with transaction() as conn:
                    fetch_gateway = ToolGateway(conn, run_id)
                    _register_deep_tools(fetch_gateway, conn)
                    for pid in paper_ids:
                        abstract = _fetch_abstract(fetch_gateway, pid)
                        if abstract is not None:
                            abstracts[pid] = abstract
            except Exception as exc:
                # Classify transport-level errors; programmer bugs (e.g.,
                # ToolRoleDenied) propagate unmasked.
                classified = classify_exception(exc)
                if classified is None:
                    raise
                last_error_category = classified.value
                continue  # papers stay in deep_read_queue (retryable)

            if not abstracts:
                # All section's papers had no abstract (404 / empty abstract /
                # all web: papers). Not a transient infrastructure issue, but
                # papers stay in queue too — caller may want to retry with a
                # different mechanism (W3 pdf_reader). web: papers get cleaned
                # at end as a special case (intentional W2 skip).
                continue

            # Build prompt — include paper abstracts in must_find_evidence context
            paper_text = "\n\n".join(
                f"### Paper {pid}\n{abstract}" for pid, abstract in abstracts.items()
            )
            user_message = template.format(
                section_id=section_id,
                paper_ids=list(abstracts.keys()),
                must_find_evidence=section.must_find_evidence,
            ) + f"\n\n### Available paper abstracts:\n\n{paper_text}"

            # Budget check (rough estimate: 4 chars per token)
            try:
                budget_manager.check(AgentRole.RESEARCHER_DEEP, len(user_message) // 4)
            except BudgetExceeded:
                last_error_category = CONTEXT_OVERFLOW
                continue

            # LLM call
            llm = router.get_llm(AgentRole.RESEARCHER_DEEP)
            binding = router.binding(AgentRole.RESEARCHER_DEEP)
            callback_config = with_run_metadata(
                run_id=run_id,
                stage="research_deep",
                agent_role=AgentRole.RESEARCHER_DEEP,
                prompt_version=template.version,
                section_id=section_id,
            )

            # Narrow exception classification (Decision #8): contract violations
            # → schema_invalid; transport-level errors → classify_exception;
            # unclassified → propagate (no silent mask).
            try:
                result_dict = structured_call(
                    llm,
                    [HumanMessage(content=user_message)],
                    schema=ResearcherDeepOutput.model_json_schema(),
                    tool_name="researcher_deep_output",
                    supports_fc=binding.fc_enabled(),
                    config=callback_config,  # type: ignore[arg-type]
                )
                output = ResearcherDeepOutput.model_validate(result_dict)
            except (StructuredCallError, ValidationError):
                # LLM output failed JSON Schema or Pydantic validation — recoverable
                last_error_category = SCHEMA_INVALID
                continue
            except Exception as exc:
                # Transport / provider errors (httpx.HTTPStatusError 429/5xx etc.)
                # use classify_exception; unclassified exceptions PROPAGATE.
                classified = classify_exception(exc)
                if classified is None:
                    raise  # don't silently swallow unknown errors
                last_error_category = classified.value
                continue

            # Receiver-side section_id validation (output level)
            if output.section_id != section_id:
                last_error_category = SCHEMA_INVALID
                continue

            # Section completed successfully (LLM produced validated output).
            # Mark all input papers as processed — whether the LLM emitted
            # evidence cards for them or marked them insufficient, we have the
            # LLM's answer and the paper doesn't need a retry. Papers without
            # abstracts (e.g., web: in this section) were never in `abstracts.keys()`
            # and stay in the queue (web: gets stripped at the bottom).
            processed.update(abstracts.keys())

            # Persist EvidenceCards via real evidence_store_write
            input_paper_ids = set(abstracts.keys())  # for per-card paper_id validation
            section_evidence_count = 0
            with transaction() as conn:
                write_gateway = ToolGateway(conn, run_id)
                _register_deep_tools(write_gateway, conn)
                for card in output.evidence_cards:
                    # Per-card validation (Decision #7b + #7c):
                    # - card.section_id must match current section
                    # - card.paper_id must be one we actually fed in (defends
                    #   against LLM hallucination / cross-section pollution)
                    if card.section_id != section_id:
                        continue
                    if card.paper_id not in input_paper_ids:
                        continue
                    evidence_id = f"E-{run_id}-{section_id}-{section_evidence_count}"
                    section_evidence_count += 1
                    # No try/except for ValidationError here:
                    # `ResearcherDeepOutput.model_validate(result_dict)` upstream
                    # already validates every EvidenceCard field (PaperId prefix,
                    # confidence ∈ [0,1], etc.) — that's the structured-output
                    # gate per spec § 2.6. By the time we reach this site, all
                    # cards in `output.evidence_cards` are Pydantic-valid.
                    # `EvidenceItem(**args)` inside the gateway's write_impl
                    # adds run_id + created_by + source_locator (host-supplied,
                    # not LLM-supplied) and re-validates — by construction those
                    # are well-formed too. Any exception here = infrastructure
                    # or config bug (DB lost, ToolRoleDenied, UniqueViolation),
                    # so we PROPAGATE rather than silent-drop the audit trail.
                    write_gateway.call(
                        AgentRole.RESEARCHER_DEEP,
                        "evidence_store_write",
                        evidence_id=evidence_id,
                        run_id=run_id,
                        section_id=section_id,
                        paper_id=card.paper_id,
                        claim=card.claim,
                        source_span=card.source_span,
                        source_locator=None,  # W2: no structured locator
                        confidence=card.confidence,
                        created_by=AgentRole.RESEARCHER_DEEP.value,
                    )

        # Record any non-terminal error (run continues to Synthesizer / Writer)
        if last_error_category is not None:
            with transaction() as conn:
                RunManager(conn).note_error_category(run_id, last_error_category)

        # Build the new deep_read_queue:
        #   keep papers that were NOT processed AND NOT web: (web: is intentional
        #   W2 skip; W3 will use a separate mechanism for those).
        remaining_queue = [
            pid for pid in deep_read_queue
            if pid not in processed and not pid.startswith("web:")
        ]

        return {**state, "deep_read_queue": remaining_queue}

    return researcher_deep_node
