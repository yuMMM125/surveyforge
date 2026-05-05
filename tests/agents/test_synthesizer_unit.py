"""Synthesizer unit tests — schema, top_k rerank, schema-repair retry,
budget overflow, empty section, missing evidence."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from litweave.llm.roles import AgentRole
from litweave.prompts.loader import PromptRegistry
from litweave.runtime.budget import (
    BUDGET_PER_ROLE,
    BudgetManager,
    BudgetSpec,
    OverflowFallback,
)
from litweave.runtime.evidence import EvidenceItem, EvidenceStore
from litweave.runtime.runs import RunManager
from litweave.schemas.planner import PlannerSection
from litweave.schemas.synthesis import (
    ComparisonMatrix,
    MatrixCell,
    MatrixRow,
    PaperFacts,
    SynthesizerOutput,
)
from litweave.state import make_initial_state
from litweave.synthesis.synthesizer import (
    _estimate_tokens,
    _packed_total_tokens,
    _top_k_rerank,
    make_synthesizer_node,
)

# ---- schema validation ----

def test_synthesizer_output_schema_minimal_valid():
    out = SynthesizerOutput(section_id="S1")
    assert out.papers_cited == []
    assert out.comparison_matrix.dimensions == ["method", "dataset", "metric", "result"]
    assert out.taxonomy.categories == []


def test_matrix_cell_requires_evidence_ids():
    """Empty evidence_ids must fail Pydantic min_length=1 (Open Decision #2)."""
    with pytest.raises(ValidationError):
        MatrixCell(value="some method", evidence_ids=[])


def test_synthesis_claim_requires_two_papers_and_evidence():
    """SynthesisClaim requires ≥ 2 papers + ≥ 2 evidence (Open Decision #1 + AD #3 grounding)."""
    from litweave.schemas.synthesis import SynthesisClaim
    # Single paper → fail
    with pytest.raises(ValidationError):
        SynthesisClaim(
            claim_text="some claim", paper_ids=["arxiv:123"], evidence_ids=["E-1", "E-2"]
        )
    # Empty evidence_ids → fail
    with pytest.raises(ValidationError):
        SynthesisClaim(
            claim_text="some claim",
            paper_ids=["arxiv:123", "arxiv:456"],
            evidence_ids=[],
        )


# ---- top_k rerank ----

def _make_evidence(evidence_id: str, paper_id: str, claim: str, confidence: float) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        run_id="run_test",
        paper_id=paper_id,
        section_id="S1",
        claim=claim,
        source_span=None,
        source_locator=None,
        confidence=confidence,
        created_by=AgentRole.RESEARCHER_DEEP,
    )


def test_top_k_rerank_orders_by_confidence_then_count():
    items = [
        _make_evidence("E-1", "arxiv:lo1", "lo claim", 0.3),
        _make_evidence("E-2", "arxiv:hi1", "hi claim 1", 0.9),
        _make_evidence("E-3", "arxiv:hi1", "hi claim 2", 0.8),
        _make_evidence("E-4", "arxiv:mid1", "mid claim", 0.6),
    ]
    # Generous budget: keep all
    selected, dropped = _top_k_rerank(items, base_prompt_tokens=0, budget_tokens=100_000)
    assert dropped == []
    # Insertion order: hi1 first (highest avg conf), mid1 second, lo1 last
    paper_order = []
    for it in selected:
        if it.paper_id not in paper_order:
            paper_order.append(it.paper_id)
    assert paper_order == ["arxiv:hi1", "arxiv:mid1", "arxiv:lo1"]


def test_top_k_rerank_drops_lowest_when_budget_tight():
    items = [
        _make_evidence("E-1", "arxiv:p1", "claim with about thirty characters of body text per paper", 0.9),
        _make_evidence("E-2", "arxiv:p2", "claim with about thirty characters of body text per paper", 0.5),
        _make_evidence("E-3", "arxiv:p3", "claim with about thirty characters of body text per paper", 0.3),
    ]
    # Each paper ~30 tokens. Budget 80 → keep p1 (highest conf), drop p2, p3.
    selected, dropped = _top_k_rerank(items, base_prompt_tokens=20, budget_tokens=80)
    selected_papers = {it.paper_id for it in selected}
    assert "arxiv:p1" in selected_papers
    assert "arxiv:p2" in dropped or "arxiv:p3" in dropped


def test_top_k_rerank_whole_paper_atomicity():
    """If a paper has multiple evidence_items, reject all-or-nothing — never partial."""
    items = [
        _make_evidence("E-1", "arxiv:big", "very very very long claim text " * 20, 0.9),
        _make_evidence("E-2", "arxiv:big", "very very very long claim text " * 20, 0.9),
    ]
    selected, dropped = _top_k_rerank(items, base_prompt_tokens=0, budget_tokens=50)
    # Either both kept (impossible if budget too small) or both dropped
    selected_paper_ids = {it.paper_id for it in selected}
    if selected_paper_ids:
        assert len(selected) == 2  # both items present
    else:
        assert dropped == ["arxiv:big"]


# ---- token estimate ----

def test_estimate_tokens_4_chars_per_token():
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("abcdefgh") == 2
    assert _estimate_tokens("") == 0


# ---- end-to-end with mocked structured_call ----

def _make_mock_router() -> MagicMock:
    router = MagicMock()
    router.binding.return_value = MagicMock(fc_enabled=lambda: True)
    router.get_llm.return_value = MagicMock()
    return router


def test_synthesizer_empty_section_emits_structurally_present_output(
    monkeypatch, conn, patch_agent_transaction
):
    """0 evidence_items for a section → all fields present, coverage_gaps for every must_find_evidence."""
    patch_agent_transaction("litweave.synthesis.synthesizer")
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k1")
    router = _make_mock_router()
    registry = PromptRegistry()
    budget = BudgetManager()
    node = make_synthesizer_node(router, registry, budget)

    section = PlannerSection(
        section_id="S1",
        title="Background",
        research_questions=["q1", "q2"],
        must_find_evidence=["evidence A", "evidence B"],
    )
    state = make_initial_state(topic="t")
    state["outline"] = [section.model_dump()]
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

    result = node(state, config)

    extract = result["structured_extracts"]["S1"]
    assert extract["section_id"] == "S1"
    assert extract["papers_cited"] == []
    assert extract["claims"] == []
    assert extract["paper_facts"] == []
    assert extract["comparison_matrix"]["rows"] == []
    assert extract["taxonomy"]["categories"] == []
    assert extract["cross_paper_synthesis"] == []
    assert len(extract["coverage_gaps"]) == 2
    reasons = {g["reason"] for g in extract["coverage_gaps"]}
    assert reasons == {"missing_evidence"}


def test_synthesizer_schema_repair_retry_succeeds_on_second_attempt(
    monkeypatch, conn, patch_agent_transaction
):
    """First structured_call returns invalid output; second call (with repair msg) returns valid."""
    patch_agent_transaction("litweave.synthesis.synthesizer")
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k2")

    # Pre-populate evidence_items via EvidenceStore
    store = EvidenceStore(conn)
    item = EvidenceItem(
        evidence_id="E-test-1",
        run_id=run.run_id,
        paper_id="arxiv:1",
        section_id="S1",
        claim="claim 1",
        source_span=None,
        source_locator=None,
        confidence=0.9,
        created_by=AgentRole.RESEARCHER_DEEP,
    )
    store.save(item)

    invalid_first = {"section_id": "S1", "papers_cited": ["not_a_paper_id_format"]}
    # ↑ paper_id "not_a_paper_id_format" lacks prefix → ValidationError

    valid_second = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:1"],
        claims=[{
            "evidence_id": "E-test-1", "paper_id": "arxiv:1",
            "claim": "claim 1", "confidence": 0.9,
        }],
        paper_facts=[PaperFacts(
            paper_id="arxiv:1", method="", dataset="", metric="", result="",
            evidence_ids=["E-test-1"],
        )],
        comparison_matrix=ComparisonMatrix(rows=[
            MatrixRow(paper_id="arxiv:1", cells={}),
        ]),
    ).model_dump()

    call_count = {"n": 0}
    def fake_structured_call(*args, **kwargs):
        call_count["n"] += 1
        return invalid_first if call_count["n"] == 1 else valid_second
    monkeypatch.setattr(
        "litweave.synthesis.synthesizer.structured_call",
        fake_structured_call,
    )

    router = _make_mock_router()
    registry = PromptRegistry()
    budget = BudgetManager()
    node = make_synthesizer_node(router, registry, budget)

    section = PlannerSection(
        section_id="S1",
        title="t",
        research_questions=["q1", "q2"],
        must_find_evidence=["a"],
    )
    state = make_initial_state(topic="t")
    state["outline"] = [section.model_dump()]
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

    result = node(state, config)

    assert call_count["n"] == 2
    assert result["structured_extracts"]["S1"]["section_id"] == "S1"


def test_synthesizer_schema_invalid_after_two_attempts_marks_section(
    monkeypatch, conn, patch_agent_transaction
):
    """Both structured_call attempts return invalid → record schema_invalid + section skipped."""
    patch_agent_transaction("litweave.synthesis.synthesizer")
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k3")

    store = EvidenceStore(conn)
    item = EvidenceItem(
        evidence_id="E-test-2",
        run_id=run.run_id,
        paper_id="arxiv:1",
        section_id="S1",
        claim="claim",
        source_span=None,
        source_locator=None,
        confidence=0.9,
        created_by=AgentRole.RESEARCHER_DEEP,
    )
    store.save(item)

    invalid = {"section_id": "S1", "papers_cited": ["bad_format"]}
    call_count = {"n": 0}
    def always_invalid(*args, **kwargs):
        call_count["n"] += 1
        return invalid
    monkeypatch.setattr(
        "litweave.synthesis.synthesizer.structured_call",
        always_invalid,
    )

    router = _make_mock_router()
    registry = PromptRegistry()
    budget = BudgetManager()
    node = make_synthesizer_node(router, registry, budget)

    section = PlannerSection(
        section_id="S1",
        title="t",
        research_questions=["q1", "q2"],
        must_find_evidence=["a"],
    )
    state = make_initial_state(topic="t")
    state["outline"] = [section.model_dump()]
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

    result = node(state, config)

    assert call_count["n"] == 2  # exactly 2 attempts, no third
    # Section did not produce extract on schema_invalid path:
    assert "S1" not in result["structured_extracts"]
    # runs.error_category should be schema_invalid
    rm2 = RunManager(conn)
    run_obj = rm2.get(run.run_id)
    assert run_obj.error_category == "schema_invalid"


def test_synthesizer_passes_max_retries_zero_to_structured_call(
    monkeypatch, conn, patch_agent_transaction
):
    """AD #2 hard cap: every structured_call invocation MUST use max_retries=0.
    Without this guard, structured_call's default max_retries=2 would
    silently 3x the LLM budget (2 outer attempts x 3 inner = 6 calls per
    section). This test inspects the kwargs of each structured_call invocation."""
    patch_agent_transaction("litweave.synthesis.synthesizer")
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k3a")
    store = EvidenceStore(conn)
    store.save(EvidenceItem(
        evidence_id="E-mr0", run_id=run.run_id, paper_id="arxiv:1",
        section_id="S1", claim="c", source_span=None, source_locator=None,
        confidence=0.9, created_by=AgentRole.RESEARCHER_DEEP,
    ))

    captured_kwargs: list[dict] = []
    valid_output = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:1"],
        claims=[{
            "evidence_id": "E-mr0", "paper_id": "arxiv:1",
            "claim": "c", "confidence": 0.9,
        }],
        paper_facts=[PaperFacts(
            paper_id="arxiv:1", method="", dataset="", metric="", result="",
            evidence_ids=["E-mr0"],
        )],
        comparison_matrix=ComparisonMatrix(rows=[
            MatrixRow(paper_id="arxiv:1", cells={}),
        ]),
    ).model_dump()

    def capturing_structured_call(*args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return valid_output

    monkeypatch.setattr(
        "litweave.synthesis.synthesizer.structured_call",
        capturing_structured_call,
    )

    router = _make_mock_router()
    registry = PromptRegistry()
    budget = BudgetManager()
    node = make_synthesizer_node(router, registry, budget)
    section = PlannerSection(
        section_id="S1", title="t",
        research_questions=["q1", "q2"], must_find_evidence=["a"],
    )
    state = make_initial_state(topic="t")
    state["outline"] = [section.model_dump()]
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}
    node(state, config)

    assert captured_kwargs, "structured_call must have been called at least once"
    for kw in captured_kwargs:
        assert kw.get("max_retries") == 0, (
            f"AD #2 violated: structured_call called with max_retries={kw.get('max_retries')!r}, "
            f"must be 0 to bound LLM calls per section to 2 (outer repair retry only)."
        )


def test_synthesizer_grounding_failure_triggers_repair_retry(
    monkeypatch, conn, patch_agent_transaction
):
    """AD #3: output passes Pydantic shape but fails grounding (paper_id not in
    input). Repair retry should fire; if second attempt also ungrounded, mark
    schema_invalid."""
    patch_agent_transaction("litweave.synthesis.synthesizer")
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k3b")
    store = EvidenceStore(conn)
    store.save(EvidenceItem(
        evidence_id="E-gr-1", run_id=run.run_id, paper_id="arxiv:input",
        section_id="S1", claim="c", source_span=None, source_locator=None,
        confidence=0.9, created_by=AgentRole.RESEARCHER_DEEP,
    ))

    # First call: structurally valid but cites a paper_id NOT in input
    ungrounded_first = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:fabricated"],  # not in input!
    ).model_dump()
    # Second call: grounded correctly with full coverage (Layer 5 satisfied)
    grounded_second = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:input"],
        claims=[{
            "evidence_id": "E-gr-1", "paper_id": "arxiv:input",
            "claim": "c", "confidence": 0.9,
        }],
        paper_facts=[PaperFacts(
            paper_id="arxiv:input", method="", dataset="", metric="", result="",
            evidence_ids=["E-gr-1"],
        )],
        comparison_matrix=ComparisonMatrix(rows=[
            MatrixRow(paper_id="arxiv:input", cells={}),
        ]),
    ).model_dump()

    call_count = {"n": 0}
    def two_responses(*args, **kwargs):
        call_count["n"] += 1
        return ungrounded_first if call_count["n"] == 1 else grounded_second

    monkeypatch.setattr(
        "litweave.synthesis.synthesizer.structured_call",
        two_responses,
    )

    router = _make_mock_router()
    registry = PromptRegistry()
    budget = BudgetManager()
    node = make_synthesizer_node(router, registry, budget)
    section = PlannerSection(
        section_id="S1", title="t",
        research_questions=["q1", "q2"], must_find_evidence=["a"],
    )
    state = make_initial_state(topic="t")
    state["outline"] = [section.model_dump()]
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}
    result = node(state, config)

    assert call_count["n"] == 2, "grounding error should trigger exactly 1 repair retry"
    assert result["structured_extracts"]["S1"]["papers_cited"] == ["arxiv:input"]


def test_synthesizer_grounding_validate_layer3_evidence_paper_consistency():
    """Unit test of `_validate_grounding` Layer 3a: paper_facts evidence_ids
    must match the paper_facts.paper_id."""
    from litweave.synthesis.synthesizer import GroundingError, _validate_grounding

    out = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:p1", "arxiv:p2"],
        paper_facts=[
            PaperFacts(
                paper_id="arxiv:p1",
                method="m", dataset="d", metric="metr", result="r",
                evidence_ids=["E-2"],  # WRONG — E-2 belongs to p2 per the map below
            ),
        ],
    )
    input_paper_ids = {"arxiv:p1", "arxiv:p2"}
    input_evidence_ids = {"E-1", "E-2"}
    evidence_map = {"E-1": "arxiv:p1", "E-2": "arxiv:p2"}

    with pytest.raises(GroundingError, match=r"paper_facts.*evidence_id"):
        _validate_grounding(out, input_paper_ids, input_evidence_ids, evidence_map, "S1")


def test_synthesizer_grounding_validate_layer3d_cross_paper_synthesis_paper_coverage():
    """Unit test of `_validate_grounding` Layer 3d: every supporting paper in
    cross_paper_synthesis must have ≥ 1 evidence_id covering it.

    Schema-level `min_length=2` on `evidence_ids` is count-only — having 2
    evidence items both from one paper while listing 2 supporting papers
    structurally validates but is semantically broken: the second paper
    is "supporting" with zero evidence backing it. Layer 3d catches this.
    """
    from litweave.schemas.synthesis import SynthesisClaim
    from litweave.synthesis.synthesizer import GroundingError, _validate_grounding

    out = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:p1", "arxiv:p2"],
        cross_paper_synthesis=[
            SynthesisClaim(
                claim_text="cross-paper claim spanning p1 and p2",
                paper_ids=["arxiv:p1", "arxiv:p2"],
                # Both evidence items are from p1; p2 listed as supporting
                # but has no covering evidence_id.
                evidence_ids=["E-1", "E-3"],
            ),
        ],
    )
    input_paper_ids = {"arxiv:p1", "arxiv:p2"}
    input_evidence_ids = {"E-1", "E-2", "E-3"}
    evidence_map = {"E-1": "arxiv:p1", "E-2": "arxiv:p2", "E-3": "arxiv:p1"}

    with pytest.raises(GroundingError, match="without any covering"):
        _validate_grounding(out, input_paper_ids, input_evidence_ids, evidence_map, "S1")


def test_synthesizer_budget_overflow_drives_top_k_rerank_and_records_gap(
    monkeypatch, conn, patch_agent_transaction
):
    """Deterministic overflow trigger via `_packed_total_tokens` (per AD #15).

    Build N evidence_items with predictable text. Compute the exact projected
    token count using `_packed_total_tokens(items, base_tokens)` — the same
    canonical helper both production budget check and `_top_k_rerank` use.
    Set the budget to (full - 1) so top_k is GUARANTEED to drop >= 1 paper
    regardless of estimator tweaks, header changes, or joiner format changes.
    Asserts:
      (i) BudgetExceeded is caught (no propagation)
      (ii) top_k drops at least 1 paper (greedy whole-paper, lowest-confidence first)
      (iii) coverage_gaps has a budget_truncation entry
      (iv) papers_cited count < input paper count
    """
    patch_agent_transaction("litweave.synthesis.synthesizer")

    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k4")
    store = EvidenceStore(conn)
    # 5 papers with 500-char claims; deterministic content (no randomness).
    big_claim = "x" * 500
    for i in range(5):
        store.save(EvidenceItem(
            evidence_id=f"E-{i}",
            run_id=run.run_id,
            paper_id=f"arxiv:{i}",
            section_id="S1",
            claim=big_claim,
            source_span=None,
            source_locator=None,
            confidence=0.9 - i * 0.1,  # p0 highest, p4 lowest
            created_by=AgentRole.RESEARCHER_DEEP,
        ))

    section = PlannerSection(
        section_id="S1",
        title="t",
        research_questions=["q1", "q2"],
        must_find_evidence=["a"],
    )

    # AD #15: drive the threshold with `_packed_total_tokens` — the canonical
    # helper that both the up-front budget check AND `_top_k_rerank` consult.
    # Pinning the test to this helper guarantees top_k's per-candidate-set
    # accept/reject calls land exactly on the `target_budget` boundary, so
    # `target_budget = full_tokens_exact - 1` deterministically forces ≥ 1
    # drop regardless of integer-division flooring slop or future header /
    # joiner literal changes (the helper charges those, top_k consults the
    # same helper).
    items_for_estimate = store.list_by_section(run.run_id, "S1")
    registry = PromptRegistry()
    template = registry.load(AgentRole.SYNTHESIZER)
    base_template_text = template.format(
        section_id=section.section_id,
        section_title=section.title,
        must_find_evidence=section.must_find_evidence,
    )
    base_tokens = _estimate_tokens(base_template_text)
    full_tokens_exact = _packed_total_tokens(items_for_estimate, base_tokens)
    target_budget = full_tokens_exact - 1  # GUARANTEED top_k drops ≥ 1 paper
    assert target_budget > 0, "Test setup: total prompt tokens must be > 1"

    monkeypatch.setitem(
        BUDGET_PER_ROLE,
        AgentRole.SYNTHESIZER,
        BudgetSpec(
            max_input_tokens=target_budget,
            reserved_output_tokens=500,
            fallback=OverflowFallback.TOP_K_RERANK,
        ),
    )

    # Predict which papers survive top_k packing — the same algorithm
    # production will run inside the node, so the predicted surviving set
    # equals the `input_paper_ids` production passes to `_validate_grounding`.
    # This lets us mock `structured_call` to return a fully-grounded
    # SynthesizerOutput covering exactly that set, exercising the REAL
    # combined "budget truncation + receiver grounding" path.
    predicted_selected, predicted_dropped = _top_k_rerank(
        items_for_estimate, base_tokens, target_budget,
    )
    assert len(predicted_dropped) >= 1, (
        "test setup invariant: target_budget = full - 1 should force top_k "
        f"to drop >= 1 paper; got dropped={predicted_dropped}"
    )

    selected_by_paper: dict[str, list[EvidenceItem]] = {}
    for it in predicted_selected:
        selected_by_paper.setdefault(it.paper_id, []).append(it)
    surviving_paper_ids = sorted(selected_by_paper.keys())

    grounded_output = SynthesizerOutput(
        section_id="S1",
        papers_cited=surviving_paper_ids,
        claims=[
            {
                "evidence_id": it.evidence_id, "paper_id": it.paper_id,
                "claim": "c", "confidence": it.confidence,
            }
            for it in predicted_selected
        ],
        paper_facts=[
            PaperFacts(
                paper_id=pid, method="", dataset="", metric="", result="",
                evidence_ids=[it.evidence_id for it in selected_by_paper[pid]],
            )
            for pid in surviving_paper_ids
        ],
        comparison_matrix=ComparisonMatrix(rows=[
            MatrixRow(paper_id=pid, cells={}) for pid in surviving_paper_ids
        ]),
    ).model_dump()

    monkeypatch.setattr(
        "litweave.synthesis.synthesizer.structured_call",
        lambda *a, **kw: grounded_output,
    )
    # NOTE: `_validate_grounding` deliberately NOT monkeypatched — this test
    # covers the REAL combined path of (a) budget overflow → top_k rerank →
    # (b) host-side grounding against the truncated input set →
    # (c) coverage_gaps augmentation with the budget_truncation entry.

    router = _make_mock_router()
    budget = BudgetManager()
    node = make_synthesizer_node(router, registry, budget)
    state = make_initial_state(topic="t")
    state["outline"] = [section.model_dump()]
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

    result = node(state, config)

    extract = result["structured_extracts"]["S1"]
    # (i) BudgetExceeded was caught — node returned cleanly with an extract
    # (ii) top_k dropped at least 1 paper
    assert len(extract["papers_cited"]) < 5, (
        f"top_k should have dropped >=1 paper; got {len(extract['papers_cited'])} "
        f"of 5 input papers"
    )
    # (iii) coverage_gaps has at least one budget_truncation entry
    truncation_gaps = [
        g for g in extract["coverage_gaps"] if g["reason"] == "budget_truncation"
    ]
    assert len(truncation_gaps) >= 1, (
        f"expected budget_truncation gap; got {extract['coverage_gaps']}"
    )
    # (iv) Grounding was actually invoked on the rerank-truncated input set:
    # papers_cited equals the predicted surviving paper set. If Layer 5a
    # (papers_cited == input_paper_ids) had been silently bypassed, this
    # equality could have drifted undetected — e.g., output.papers_cited
    # mismatching the post-rerank input.
    assert set(extract["papers_cited"]) == set(surviving_paper_ids), (
        f"papers_cited {sorted(extract['papers_cited'])} should equal the "
        f"predicted top_k surviving set {surviving_paper_ids} — Layer 5a "
        f"is supposed to enforce this on the real grounding path"
    )


def test_synthesizer_section_id_mismatch_triggers_repair_retry(
    monkeypatch, conn, patch_agent_transaction
):
    """Layer 0: output.section_id MUST equal the section currently being
    processed. First call returns wrong section_id (LLM lost track of which
    section it's synthesizing); GroundingError fires; repair retry returns
    the correct section_id and the call succeeds."""
    patch_agent_transaction("litweave.synthesis.synthesizer")
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k_sid")
    store = EvidenceStore(conn)
    store.save(EvidenceItem(
        evidence_id="E-sid-1", run_id=run.run_id, paper_id="arxiv:1",
        section_id="S1", claim="c", source_span=None, source_locator=None,
        confidence=0.9, created_by=AgentRole.RESEARCHER_DEEP,
    ))

    def _full_output(section_id: str) -> dict:
        return SynthesizerOutput(
            section_id=section_id,
            papers_cited=["arxiv:1"],
            claims=[{
                "evidence_id": "E-sid-1", "paper_id": "arxiv:1",
                "claim": "c", "confidence": 0.9,
            }],
            paper_facts=[PaperFacts(
                paper_id="arxiv:1", method="", dataset="", metric="",
                result="", evidence_ids=["E-sid-1"],
            )],
            comparison_matrix=ComparisonMatrix(rows=[
                MatrixRow(paper_id="arxiv:1", cells={}),
            ]),
        ).model_dump()

    wrong_then_right = [_full_output("S99"), _full_output("S1")]
    call_count = {"n": 0}

    def two_responses(*args, **kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        return wrong_then_right[i]

    monkeypatch.setattr(
        "litweave.synthesis.synthesizer.structured_call",
        two_responses,
    )

    router = _make_mock_router()
    registry = PromptRegistry()
    budget = BudgetManager()
    node = make_synthesizer_node(router, registry, budget)
    section = PlannerSection(
        section_id="S1", title="t",
        research_questions=["q1", "q2"], must_find_evidence=["a"],
    )
    state = make_initial_state(topic="t")
    state["outline"] = [section.model_dump()]
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}
    result = node(state, config)

    assert call_count["n"] == 2, "section_id mismatch should trigger 1 repair retry"
    assert result["structured_extracts"]["S1"]["section_id"] == "S1"


def test_synthesizer_grounding_validate_claims_paper_evidence_mismatch():
    """Layer 3e: claims[i].evidence_id MUST come from an item whose paper_id
    matches claims[i].paper_id. Mis-attributed claim raises GroundingError
    even if the evidence_id and paper_id are individually in the input set."""
    from litweave.synthesis.synthesizer import GroundingError, _validate_grounding

    out = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:p1", "arxiv:p2"],
        claims=[{
            "evidence_id": "E-2", "paper_id": "arxiv:p1",  # E-2 actually belongs to p2
            "claim": "wrongly-attributed claim", "confidence": 0.9,
        }],
        paper_facts=[
            PaperFacts(paper_id="arxiv:p1", method="", dataset="", metric="",
                       result="", evidence_ids=["E-1"]),
            PaperFacts(paper_id="arxiv:p2", method="", dataset="", metric="",
                       result="", evidence_ids=["E-2"]),
        ],
        comparison_matrix=ComparisonMatrix(rows=[
            MatrixRow(paper_id="arxiv:p1", cells={}),
            MatrixRow(paper_id="arxiv:p2", cells={}),
        ]),
    )
    input_paper_ids = {"arxiv:p1", "arxiv:p2"}
    input_evidence_ids = {"E-1", "E-2"}
    evidence_map = {"E-1": "arxiv:p1", "E-2": "arxiv:p2"}

    with pytest.raises(GroundingError, match=r"claims.*evidence_id"):
        _validate_grounding(out, input_paper_ids, input_evidence_ids, evidence_map, "S1")


def test_synthesizer_non_empty_input_with_empty_output_raises_grounding_error():
    """Layer 5: when the section has input evidence (input_paper_ids non-empty),
    the model cannot return a structurally empty SynthesizerOutput. At minimum,
    set(papers_cited) MUST equal input_paper_ids; missing paper_facts entries
    or matrix.rows entries also raise."""
    from litweave.synthesis.synthesizer import GroundingError, _validate_grounding

    empty_out = SynthesizerOutput(section_id="S1")
    input_paper_ids = {"arxiv:p1", "arxiv:p2"}
    input_evidence_ids = {"E-1", "E-2"}
    evidence_map = {"E-1": "arxiv:p1", "E-2": "arxiv:p2"}

    with pytest.raises(GroundingError, match=r"papers_cited.*does not match"):
        _validate_grounding(empty_out, input_paper_ids, input_evidence_ids, evidence_map, "S1")


def test_synthesizer_taxonomy_paper_ids_without_rationale_raises_grounding_error():
    """Layer 4: a TaxonomyCategory with non-empty paper_ids MUST have non-empty
    rationale_evidence_ids. Schema allows both fields to default to empty list
    independently; this layer enforces the "non-empty categories must be
    rationalized" contract."""
    from litweave.schemas.synthesis import SectionTaxonomy, TaxonomyCategory
    from litweave.synthesis.synthesizer import GroundingError, _validate_grounding

    out = SynthesizerOutput(
        section_id="S1",
        papers_cited=["arxiv:p1"],
        paper_facts=[PaperFacts(
            paper_id="arxiv:p1", method="", dataset="", metric="", result="",
            evidence_ids=["E-1"],
        )],
        comparison_matrix=ComparisonMatrix(rows=[
            MatrixRow(paper_id="arxiv:p1", cells={}),
        ]),
        taxonomy=SectionTaxonomy(categories=[
            TaxonomyCategory(
                name="some category",
                description="a category that lists a paper without rationale",
                paper_ids=["arxiv:p1"],
                rationale_evidence_ids=[],  # EMPTY but paper_ids non-empty
            ),
        ]),
    )
    input_paper_ids = {"arxiv:p1"}
    input_evidence_ids = {"E-1"}
    evidence_map = {"E-1": "arxiv:p1"}

    with pytest.raises(GroundingError, match=r"rationale_evidence_ids"):
        _validate_grounding(out, input_paper_ids, input_evidence_ids, evidence_map, "S1")
