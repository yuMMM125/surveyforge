"""Researcher-Deep unit tests — single LLM call per section, mocked LLM output.

Live integration in tests/agents/integration/test_researcher_deep_live.py.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from langchain_core.runnables import RunnableConfig

from litweave.agents.researcher_deep import make_researcher_deep_node
from litweave.llm.providers import ProviderName
from litweave.llm.roles import AgentRole
from litweave.llm.router import LLMRouter, RoleBinding
from litweave.prompts.loader import PromptRegistry
from litweave.runtime.budget import BudgetManager
from litweave.runtime.runs import RunManager, RunStatus
from litweave.schemas.planner import PlannerSection
from litweave.state import make_initial_state

# ---- helpers ----

def _make_run(conn: psycopg.Connection) -> str:
    rm = RunManager(conn)
    return rm.create(topic="test", idempotency_key=f"key-{time.perf_counter_ns()}").run_id


def _outline() -> list[dict[str, Any]]:
    return [
        PlannerSection(
            section_id="S1",
            title="Background",
            research_questions=["What is RLHF?", "Why does it matter?"],
            must_find_evidence=["Original RLHF paper"],
        ).model_dump()
    ]


def _state_with_handoff(*paper_ids: str) -> dict[str, Any]:
    """Build a SurveyState as if Wide ran and produced these papers for S1."""
    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    state["section_notes"] = {
        "S1": [
            {"paper_id": pid, "title": f"Paper {pid}", "source": "arxiv",
             "why_relevant": "x", "handoff_to_deep": True}
            for pid in paper_ids
        ]
    }
    state["deep_read_queue"] = list(paper_ids)
    return state


@pytest.fixture
def deep_node_factory(monkeypatch: pytest.MonkeyPatch):
    """Factory: returns (make_callable, structured_call_capture, abstract_capture)."""
    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax", temperature=0.0,
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))

    structured_call_calls: list[dict[str, Any]] = []
    abstract_fetches: dict[str, str | None] = {}

    # Outputs queue: list of dicts to be returned in order via side_effect.
    # Single-section tests pass `[output_dict]`; multi-section tests pass
    # `[output_s1, output_s2, ...]`. structured_call() consumes one per call.
    _outputs: list[Any] = []  # Any = dict for success, Exception instance for raise

    def fake_structured_call(*args, **kwargs):
        structured_call_calls.append(kwargs)
        if not _outputs:
            raise RuntimeError("test bug: no scripted output for this structured_call invocation")
        next_output = _outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        return next_output

    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        fake_structured_call,
    )

    # Mock abstract fetch — caller sets abstract_fetches[paper_id] = "abstract" or None.
    # Mirror real `_fetch_abstract`: web: prefix returns None (W2 doesn't fetch web).
    # Without this mirror, web: papers would get a fake abstract and end up in
    # `abstracts.keys()` (= input_paper_ids), breaking the new strict coverage
    # check (`paper_ids_processed` from canned output wouldn't include them).
    def fake_fetch_abstract(gateway, paper_id):
        if paper_id in abstract_fetches:
            return abstract_fetches[paper_id]
        if paper_id.startswith("web:"):
            return None
        return "default abstract for testing"

    monkeypatch.setattr(
        "litweave.agents.researcher_deep._fetch_abstract",
        fake_fetch_abstract,
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()

    def make(scripted_outputs: dict[str, Any] | list[Any]):
        """Args:
            scripted_outputs: single dict (= single-section output) OR list of items
                where each item is either a dict (success output) or an Exception
                instance to be raised when structured_call fires.
        """
        nonlocal _outputs
        if isinstance(scripted_outputs, dict):
            _outputs = [scripted_outputs]
        else:
            _outputs = list(scripted_outputs)
        return (
            make_researcher_deep_node(router, registry, budget_manager),
            structured_call_calls,
            abstract_fetches,
        )

    return make


def _canned_deep_output(
    section_id: str = "S1",
    paper_ids: list[str] | None = None,
    n_cards: int = 1,
) -> dict[str, Any]:
    """Build a ResearcherDeepOutput dict with N evidence cards for the given section."""
    pids = paper_ids or ["arxiv:1706.03741"]
    return {
        "section_id": section_id,
        "paper_ids_processed": pids,
        "evidence_cards": [
            {
                "evidence_id": f"E-test-{section_id}-{i}",  # impl will override
                "paper_id": pids[i % len(pids)],
                "section_id": section_id,
                "claim": f"Claim {i}",
                "source_span": f"quote {i}",
                "confidence": 0.9,
            }
            for i in range(n_cards)
        ],
        "insufficient_evidence_paper_ids": [],
    }


# ---- happy path ----

def test_deep_node_persists_evidence_cards(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    patch_agent_transaction("litweave.agents.researcher_deep")
    node, _calls, _ = deep_node_factory(_canned_deep_output(n_cards=2))

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    node(state, config)

    # Verify evidence_items table has 2 rows for this run
    with conn.cursor() as cur:
        cur.execute(
            "SELECT evidence_id, section_id, paper_id, confidence FROM evidence_items "
            "WHERE run_id = %s ORDER BY evidence_id",
            (run_id,),
        )
        rows = cur.fetchall()
    assert len(rows) == 2
    for row in rows:
        evidence_id, section_id, paper_id, _confidence = row
        assert evidence_id.startswith(f"E-{run_id}-S1-")
        assert section_id == "S1"
        assert paper_id == "arxiv:1706.03741"


def test_deep_node_removes_processed_papers_from_queue(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """Successfully-processed papers (section's structured_call returned valid output)
    leave the queue. Other papers stay for upstream retry (Issue 1 fix)."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    node, _, _ = deep_node_factory(_canned_deep_output())
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)
    # Successfully processed → queue empty
    assert result["deep_read_queue"] == []


def test_deep_node_keeps_papers_in_queue_on_section_failure(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """When structured_call raises a transient error (provider_429), the section's
    papers MUST stay in deep_read_queue so upstream retry / orchestrator can
    re-enter (Issue 1 fix — previously cleared unconditionally)."""
    import httpx
    patch_agent_transaction("litweave.agents.researcher_deep")
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(429, request=request)
    rate_limit_error = httpx.HTTPStatusError("rate limited", request=request, response=response)
    node, _, _ = deep_node_factory([rate_limit_error])

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:transient.1")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # Paper stays in queue (retry-eligible)
    assert "arxiv:transient.1" in result["deep_read_queue"]
    # error_category recorded as provider_429
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "provider_429"


def test_deep_node_strips_web_papers_from_queue(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """web: papers leave the queue even though they're not "processed" — W2
    explicit skip (Issue 1 / Decision #3). Without this they'd cycle forever
    in any retry-loop wrapper."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    node, _, _ = deep_node_factory(_canned_deep_output())
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741", "web:abc123")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)
    # Both gone: arxiv via processed, web via explicit W2 strip
    assert result["deep_read_queue"] == []


# ---- web paper skip ----

def test_deep_node_skips_web_papers_in_w2(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """W2 has no pdf_reader / web re-fetch; web papers go to insufficient (here: just skipped)."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    node, _calls, _ = deep_node_factory(_canned_deep_output())
    run_id = _make_run(conn)
    # Mix of arxiv + web papers
    state = _state_with_handoff("arxiv:1706.03741", "web:abc123")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Verify ONLY arxiv paper produced evidence (web skipped)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paper_id FROM evidence_items WHERE run_id = %s",
            (run_id,),
        )
        paper_ids = {row[0] for row in cur.fetchall()}
    assert "arxiv:1706.03741" in paper_ids
    assert "web:abc123" not in paper_ids


# ---- Wide forced-exit stubs flow into Deep correctly ----

def test_deep_node_processes_wide_forced_exit_stubs(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """After Task 4 polish #4, Wide writes CandidatePaper-shaped stubs to
    section_notes on forced exit. Deep's cross-reference must pick them up
    just like real CandidatePaper entries (the shape is identical apart from
    the `_forced_exit_stub: True` marker which Deep ignores)."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    node, _, _ = deep_node_factory(_canned_deep_output(paper_ids=["arxiv:stub.1"]))
    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    # Wide forced-exit shape: stubs in section_notes, paper_ids in deep_read_queue
    state["section_notes"] = {
        "S1": [{
            "paper_id": "arxiv:stub.1",
            "title": "",
            "source": "arxiv",
            "why_relevant": "<forced-exit stub: not triaged by Wide>",
            "handoff_to_deep": True,
            "_forced_exit_stub": True,
        }]
    }
    state["deep_read_queue"] = ["arxiv:stub.1"]
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Stub paper got processed end-to-end: evidence_items row exists
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paper_id FROM evidence_items WHERE run_id = %s",
            (run_id,),
        )
        paper_ids = {row[0] for row in cur.fetchall()}
    assert "arxiv:stub.1" in paper_ids


# ---- section_id mismatch ----

def test_deep_node_output_section_id_mismatch_rejects_whole_output(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """If structured_call returns output.section_id != current section, reject all
    cards + record schema_invalid (Task 4 polish #2 lesson reused at Deep layer)."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    # Output claims "S99" but we process "S1"
    bad_output = _canned_deep_output(section_id="S99", n_cards=2)
    node, _, _ = deep_node_factory(bad_output)
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # No EvidenceCards persisted (whole output rejected)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM evidence_items WHERE run_id = %s", (run_id,))
        assert cur.fetchone() == (0,)
    # error_category recorded as schema_invalid
    rm = RunManager(conn)
    refreshed = rm.get(run_id)
    assert refreshed.error_category == "schema_invalid"
    assert refreshed.status == RunStatus.RUNNING  # non-terminal


def test_deep_node_per_card_section_id_mismatch_drops_card_only(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """A single bad card's section_id doesn't reject the whole output;
    only that card is dropped, others persist."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    output = _canned_deep_output(section_id="S1", n_cards=2)
    output["evidence_cards"][1]["section_id"] = "S99"  # wrong on second card only
    node, _, _ = deep_node_factory(output)
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Only the matching card persists
    with conn.cursor() as cur:
        cur.execute(
            "SELECT claim FROM evidence_items WHERE run_id = %s",
            (run_id,),
        )
        claims = [row[0] for row in cur.fetchall()]
    assert claims == ["Claim 0"]


# ---- per-card paper_id NOT in input drops card ----

def test_deep_node_rejects_output_with_hallucinated_card_paper_id(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """LLM may hallucinate a paper_id or reference a different section's paper.
    With the Codex P1 subset check on `card_paper_ids.issubset(input_paper_ids)`,
    a hallucinated card paper_id rejects the WHOLE output as schema_invalid
    (stricter than the previous per-card drop). Architecture Decision #7c +
    Codex P1 subset hardening."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    output = _canned_deep_output(section_id="S1", paper_ids=["arxiv:1706.03741"], n_cards=2)
    # Second card claims a paper_id NOT in section input
    output["evidence_cards"][1]["paper_id"] = "arxiv:hallucinated.1"
    node, _, _ = deep_node_factory(output)
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")  # input = arxiv:1706.03741 only
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # WHOLE output rejected — zero evidence persisted
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paper_id FROM evidence_items WHERE run_id = %s",
            (run_id,),
        )
        paper_ids = {row[0] for row in cur.fetchall()}
    assert paper_ids == set()
    assert "arxiv:hallucinated.1" not in paper_ids
    # error_category recorded as schema_invalid; paper stays in queue for retry
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "schema_invalid"
    assert "arxiv:1706.03741" in result["deep_read_queue"]


def test_deep_node_paper_ids_processed_must_cover_input_papers(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """If LLM returns valid output but `paper_ids_processed` is a STRICT SUBSET
    of input papers (some papers silently unreported), reject the whole output
    as schema_invalid + ALL section papers stay in deep_read_queue. Without
    this check, unreported papers would silently disappear (Codex P1)."""
    patch_agent_transaction("litweave.agents.researcher_deep")

    # 2 input papers, but LLM only reports 1 in paper_ids_processed
    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    state["section_notes"] = {
        "S1": [
            {"paper_id": "arxiv:p1", "title": "P1", "source": "arxiv",
             "why_relevant": "x", "handoff_to_deep": True},
            {"paper_id": "arxiv:p2", "title": "P2", "source": "arxiv",
             "why_relevant": "x", "handoff_to_deep": True},
        ]
    }
    state["deep_read_queue"] = ["arxiv:p1", "arxiv:p2"]

    bad_output = {
        "section_id": "S1",
        "paper_ids_processed": ["arxiv:p1"],  # ONLY 1 of 2 — coverage violation
        "evidence_cards": [{
            "evidence_id": "ignored",
            "paper_id": "arxiv:p1",
            "section_id": "S1",
            "claim": "Claim",
            "source_span": "quote",
            "confidence": 0.9,
        }],
        "insufficient_evidence_paper_ids": [],
    }
    node, _, _ = deep_node_factory(bad_output)

    run_id = _make_run(conn)
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # No evidence persisted (whole output rejected at coverage gate)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM evidence_items WHERE run_id = %s", (run_id,))
        assert cur.fetchone() == (0,)

    # error_category = schema_invalid
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "schema_invalid"

    # BOTH papers stay in queue (the unreported one + the reported one — they're
    # all retry-eligible because the section's output got rejected)
    assert "arxiv:p1" in result["deep_read_queue"]
    assert "arxiv:p2" in result["deep_read_queue"]


# ---- multi-section (real test, NOT placeholder) ----

def test_deep_node_processes_multiple_sections(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """Two sections in section_notes → structured_call called twice (one per
    section); each section's evidence persists independently. Verifies the
    side_effect fixture pattern + per-section isolation."""
    patch_agent_transaction("litweave.agents.researcher_deep")

    # Two sections, each with one paper
    state = make_initial_state(topic="x")
    state["outline"] = [
        PlannerSection(
            section_id="S1", title="A",
            research_questions=["q1.1", "q1.2"], must_find_evidence=["c1"],
        ).model_dump(),
        PlannerSection(
            section_id="S2", title="B",
            research_questions=["q2.1", "q2.2"], must_find_evidence=["c2"],
        ).model_dump(),
    ]
    state["section_notes"] = {
        "S1": [{"paper_id": "arxiv:s1.1", "title": "P1", "source": "arxiv",
                "why_relevant": "x", "handoff_to_deep": True}],
        "S2": [{"paper_id": "arxiv:s2.1", "title": "P2", "source": "arxiv",
                "why_relevant": "x", "handoff_to_deep": True}],
    }
    state["deep_read_queue"] = ["arxiv:s1.1", "arxiv:s2.1"]

    # Different output per section (via side_effect list)
    s1_out = _canned_deep_output(section_id="S1", paper_ids=["arxiv:s1.1"], n_cards=1)
    s1_out["evidence_cards"][0]["claim"] = "S1 specific claim"
    s2_out = _canned_deep_output(section_id="S2", paper_ids=["arxiv:s2.1"], n_cards=1)
    s2_out["evidence_cards"][0]["claim"] = "S2 specific claim"

    node, structured_calls, _ = deep_node_factory([s1_out, s2_out])

    run_id = _make_run(conn)
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # structured_call invoked exactly twice (one per section)
    assert len(structured_calls) == 2

    # evidence_items has one row per section, with the right paper + claim
    with conn.cursor() as cur:
        cur.execute(
            "SELECT section_id, paper_id, claim FROM evidence_items "
            "WHERE run_id = %s ORDER BY section_id",
            (run_id,),
        )
        rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0] == ("S1", "arxiv:s1.1", "S1 specific claim")
    assert rows[1] == ("S2", "arxiv:s2.1", "S2 specific claim")


# ---- exception classification ----

def test_deep_node_provider_429_classified_correctly(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """httpx.HTTPStatusError 429 from structured_call must be classified as
    `provider_429` (via classify_exception), not silently swallowed as schema_invalid."""
    import httpx
    patch_agent_transaction("litweave.agents.researcher_deep")
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(429, request=request)
    rate_limit_error = httpx.HTTPStatusError("rate limited", request=request, response=response)
    node, _, _ = deep_node_factory([rate_limit_error])

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    rm = RunManager(conn)
    refreshed = rm.get(run_id)
    assert refreshed.error_category == "provider_429"  # NOT schema_invalid
    assert refreshed.status == RunStatus.RUNNING


def test_deep_node_unclassified_exception_propagates(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """An unclassified exception (e.g., bare RuntimeError) must propagate
    rather than be silently swallowed as schema_invalid. This prevents
    masking real bugs in the implementation."""
    patch_agent_transaction("litweave.agents.researcher_deep")
    node, _, _ = deep_node_factory([RuntimeError("totally unexpected")])

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    with pytest.raises(RuntimeError, match="totally unexpected"):
        node(state, config)


# ---- abstract fetch error classification ----

def test_deep_node_abstract_fetch_provider_error_classified(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction, monkeypatch,
):
    """Issue 2 fix: _fetch_abstract no longer swallows exceptions.
    httpx.HTTPStatusError 5xx during s2_lookup → classify_exception →
    `provider_5xx` recorded; section's papers stay in deep_read_queue
    (retry-eligible)."""
    import httpx
    patch_agent_transaction("litweave.agents.researcher_deep")

    request = httpx.Request("GET", "https://api.semanticscholar.org/...")
    response = httpx.Response(503, request=request)
    fetch_error = httpx.HTTPStatusError("S2 down", request=request, response=response)

    # Override _fetch_abstract to raise the transport error (simulating a real
    # s2 outage caught by the node's try/classify wrap, NOT swallowed inside
    # _fetch_abstract).
    def raising_fetch(gateway, paper_id):
        raise fetch_error
    monkeypatch.setattr(
        "litweave.agents.researcher_deep._fetch_abstract",
        raising_fetch,
    )

    node, _, _ = deep_node_factory(_canned_deep_output())
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:transient.1")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "provider_5xx"
    # Paper stays in queue (transient — retryable)
    assert "arxiv:transient.1" in result["deep_read_queue"]


def test_deep_node_abstract_fetch_unclassified_exception_propagates(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction, monkeypatch,
):
    """ToolRoleDenied or programmer-bug exceptions during fetch propagate
    rather than being masked as 'no abstract' (Issue 2: prevent silent
    config-bug mask)."""
    from litweave.runtime.tool_gateway import ToolRoleDenied
    patch_agent_transaction("litweave.agents.researcher_deep")

    def raising_fetch(gateway, paper_id):
        raise ToolRoleDenied("config bug — Deep should be allowed for s2_lookup")
    monkeypatch.setattr(
        "litweave.agents.researcher_deep._fetch_abstract",
        raising_fetch,
    )

    node, _, _ = deep_node_factory(_canned_deep_output())
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    with pytest.raises(ToolRoleDenied):
        node(state, config)


# ---- evidence write failure handling ----

def test_deep_node_invalid_evidence_card_rejected_at_output_validation(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """Bad EvidenceCard fields (e.g., confidence out of [0, 1] range) fail at
    the upstream `ResearcherDeepOutput.model_validate` gate — the WHOLE output
    is rejected, NOT just the bad card. Single source of truth for shape: the
    structured-output gate. No partial persistence (would imply a per-card
    validation pass that we deliberately don't run; spec § 2.6 says structured
    output is one atomic contract per call).

    Asserts: 0 evidence_items rows, error_category=schema_invalid, paper STAYS
    in deep_read_queue (retry-eligible — LLM may produce valid output next try).
    """
    patch_agent_transaction("litweave.agents.researcher_deep")
    output = _canned_deep_output(section_id="S1", paper_ids=["arxiv:1706.03741"], n_cards=2)
    # Card 1 has confidence=1.7 (out of [0, 1] range). ResearcherDeepOutput's
    # nested EvidenceCard validation rejects the WHOLE output — Card 0 doesn't
    # land either, even though its fields are individually valid.
    output["evidence_cards"][1]["confidence"] = 1.7
    node, _, _ = deep_node_factory(output)

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # ZERO evidence persisted (whole output rejected at upstream gate)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM evidence_items WHERE run_id = %s", (run_id,))
        assert cur.fetchone() == (0,)
    # error_category recorded
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "schema_invalid"
    # Paper stays in queue (retry may yield valid LLM output)
    assert "arxiv:1706.03741" in result["deep_read_queue"]


def test_deep_node_evidence_write_infrastructure_error_propagates(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction, monkeypatch,
):
    """Issue 3 fix: non-ValidationError exceptions during evidence_store_write
    (DB lost / ToolRoleDenied / UniqueViolation) PROPAGATE — silent drop
    would corrupt the audit trail. Only LLM-side validation errors are caught."""
    from litweave.runtime.tool_gateway import ToolRoleDenied
    patch_agent_transaction("litweave.agents.researcher_deep")

    # Patch _register_deep_tools so its registered evidence_store_write impl raises
    # ToolRoleDenied — simulates a config-bug at write time.
    import litweave.agents.researcher_deep as rd_mod
    original_register = rd_mod._register_deep_tools

    def faulty_register(gateway, conn):
        original_register(gateway, conn)
        # Override evidence_store_write impl to raise
        gateway._impls["evidence_store_write"] = lambda **args: (_ for _ in ()).throw(
            ToolRoleDenied("simulated infra failure")
        )
    monkeypatch.setattr(rd_mod, "_register_deep_tools", faulty_register)

    node, _, _ = deep_node_factory(_canned_deep_output())
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    with pytest.raises(ToolRoleDenied):
        node(state, config)


# ---- arxiv_lookup fallback (Task 7 polish 5: SS demoted to optional) ----

def test_deep_node_fetches_via_arxiv_when_s2_429_on_arxiv_paper(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """When s2_lookup returns 429 on an arxiv:* paper, Deep falls back to
    arxiv_lookup and proceeds end-to-end. Both tool_calls audit rows persist:
    s2_lookup with provider_429 (the failed primary), arxiv_lookup with
    success (the fallback). Evidence is then written normally.
    """
    import httpx

    from litweave.agents.researcher_deep import make_researcher_deep_node
    from litweave.tools import arxiv_lookup, s2_lookup

    patch_agent_transaction("litweave.agents.researcher_deep")

    for var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)

    # Skip s2_lookup retry sleeps (deterministic test, no need to wait the
    # full 1+2+4 backoff sequence)
    monkeypatch.setattr("litweave.tools.s2_lookup.time.sleep", lambda s: None)

    # s2 returns 429 four times (1 initial + 3 retries) → exhausted, raises
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:1706.03741").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
        ]
    )
    # arxiv fallback returns the abstract
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(200, content="""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03741v1</id>
    <title>Original RLHF paper</title>
    <summary>We present a method for training reward models from human preferences.</summary>
  </entry>
</feed>
""")
    )

    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(return_value={
            "section_id": "S1",
            "paper_ids_processed": ["arxiv:1706.03741"],
            "evidence_cards": [{
                "evidence_id": "ignored-overridden",
                "paper_id": "arxiv:1706.03741",
                "section_id": "S1",
                "claim": "Reward modeling from human preferences",
                "source_span": "We present a method...",
                "confidence": 0.95,
            }],
            "insufficient_evidence_paper_ids": [],
        }),
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # tool_calls has both s2_lookup (provider_429) and arxiv_lookup (success)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tool_name, error_category FROM tool_calls "
            "WHERE run_id = %s ORDER BY created_at",
            (run_id,),
        )
        rows = cur.fetchall()
    tool_names_with_status = [(r[0], r[1]) for r in rows]
    assert ("s2_lookup", "provider_429") in tool_names_with_status, (
        f"expected s2_lookup with provider_429; got {tool_names_with_status}"
    )
    assert any(
        name == "arxiv_lookup" and err is None
        for name, err in tool_names_with_status
    ), f"expected arxiv_lookup success row; got {tool_names_with_status}"

    # Evidence persisted via the fallback's abstract
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paper_id, section_id FROM evidence_items WHERE run_id = %s",
            (run_id,),
        )
        evidence_rows = cur.fetchall()
    assert len(evidence_rows) == 1
    assert evidence_rows[0] == ("arxiv:1706.03741", "S1")


def test_deep_node_no_fallback_for_s2_papers_when_s2_429(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """s2:* paper that 429s on s2_lookup MUST NOT trigger arxiv_lookup
    fallback (arxiv API doesn't index s2 ids). Section gets skipped; paper
    stays in deep_read_queue for upstream retry.
    """
    import httpx

    from litweave.agents.researcher_deep import make_researcher_deep_node
    from litweave.tools import s2_lookup

    patch_agent_transaction("litweave.agents.researcher_deep")

    for var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr("litweave.tools.s2_lookup.time.sleep", lambda s: None)

    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/abc123").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
        ]
    )

    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(return_value=_canned_deep_output()),
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_with_handoff("s2:abc123")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # tool_calls has s2_lookup (provider_429) but NO arxiv_lookup (paper_id
    # not arxiv:* prefix → fallback skipped)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tool_name FROM tool_calls WHERE run_id = %s",
            (run_id,),
        )
        tool_names = {r[0] for r in cur.fetchall()}
    assert "s2_lookup" in tool_names
    assert "arxiv_lookup" not in tool_names, (
        f"s2:* papers must not trigger arxiv fallback; got {tool_names}"
    )

    # Paper stays in queue (transient — retry-eligible per existing semantics)
    assert "s2:abc123" in result["deep_read_queue"]
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "provider_429"


def test_deep_node_no_fallback_for_non_transient_s2_errors(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """Non-transient s2 errors (e.g., 403 — auth/policy) must NOT trigger
    arxiv_lookup fallback. arxiv won't fix a 403; the error propagates per
    existing classify_exception semantics (or raw if unclassified).
    """
    import httpx

    from litweave.agents.researcher_deep import make_researcher_deep_node
    from litweave.tools import s2_lookup

    patch_agent_transaction("litweave.agents.researcher_deep")

    for var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)

    # 403 on s2 — not retried by s2_lookup (only 429 is), not fallbackable.
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:1706.03741").mock(
        return_value=httpx.Response(403, text="forbidden")
    )

    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(return_value=_canned_deep_output()),
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    # 403 has no classify_exception mapping (not 429, not 5xx, not schema).
    # Per existing _fetch_abstract semantics in researcher_deep.py, the node
    # catches the exception INSIDE the transaction, then classify_exception
    # returns None for 403, so the node re-raises (no silent mask). Verify
    # this happens AND no arxiv_lookup row was inserted.
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        node(state, config)
    assert exc_info.value.response.status_code == 403

    with conn.cursor() as cur:
        cur.execute(
            "SELECT tool_name FROM tool_calls WHERE run_id = %s",
            (run_id,),
        )
        tool_names = {r[0] for r in cur.fetchall()}
    assert "s2_lookup" in tool_names
    assert "arxiv_lookup" not in tool_names, (
        f"non-transient s2 errors must not trigger arxiv fallback; "
        f"got {tool_names}"
    )


# ---- Task 7 polish 6: 3-paper end-to-end fallback coverage ----
#
# Polish 5 unit tests (above) cover individual fallback decisions on a single
# paper. These three tests exercise the full Deep node end-to-end with
# multiple papers exercising the s2-429 → arxiv-fallback path, validating the
# hypothesis that today's bounded-smoke "1 + 1 = 2 calls only" mystery is an
# external provider effect (LLM 429 / network) rather than a Deep code bug.


def _arxiv_atom_for(arxiv_id: str, abstract: str, title: str = "Test paper") -> str:
    """Build a minimal arxiv atom feed for a given id + abstract."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
    <title>{title}</title>
    <summary>{abstract}</summary>
  </entry>
</feed>
"""


def _state_3_paper_handoff() -> dict[str, Any]:
    """Build a SurveyState with 3 arxiv papers in section S1 (S1 outline + section_notes)."""
    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    paper_ids = ["arxiv:1234.5678", "arxiv:2345.6789", "arxiv:3456.7890"]
    state["section_notes"] = {
        "S1": [
            {"paper_id": pid, "title": f"Paper {pid}", "source": "arxiv",
             "why_relevant": "x", "handoff_to_deep": True}
            for pid in paper_ids
        ]
    }
    state["deep_read_queue"] = list(paper_ids)
    return state


def _setup_3_paper_s2_429_arxiv_success(monkeypatch, respx_mock):
    """Common HTTP-mock setup for Tests 1/2/3.

    s2: 4x 429 per paper (1 initial + 3 retries -> exhausted -> raise).
    arxiv: 200 with valid atom feed per paper, indexed by id_list query param.
    Returns (s2_routes, arxiv_routes) dicts keyed by bare arxiv id.
    """
    import httpx

    from litweave.tools import arxiv_lookup, s2_lookup

    # Strip system proxy env vars — without this, httpx.Client construction
    # inside s2_lookup tries SOCKS5 transport (socksio not in dev deps).
    for var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)

    # Skip retry sleeps - 1+2+4 backoff x 3 papers = 21s otherwise.
    monkeypatch.setattr("litweave.tools.s2_lookup.time.sleep", lambda s: None)
    monkeypatch.setattr("litweave.tools.arxiv_lookup.time.sleep", lambda s: None)

    bare_ids = ["1234.5678", "2345.6789", "3456.7890"]
    abstracts = {
        "1234.5678": "Paper 1 abstract.",
        "2345.6789": "Paper 2 abstract.",
        "3456.7890": "Paper 3 abstract.",
    }

    s2_routes: dict[str, Any] = {}
    arxiv_routes: dict[str, Any] = {}
    for bare in bare_ids:
        # s2: 4 sequential 429 responses (1 initial attempt + 3 retries)
        s2_routes[bare] = respx_mock.get(
            f"{s2_lookup.S2_API_BASE}/paper/arXiv:{bare}",
            name=f"s2_{bare}",
        ).mock(side_effect=[httpx.Response(429)] * 4)
        # arxiv: 200 success, distinguished per-paper via id_list query param
        arxiv_routes[bare] = respx_mock.get(
            arxiv_lookup.ARXIV_API_BASE,
            params={"id_list": bare},
            name=f"arxiv_{bare}",
        ).mock(
            return_value=httpx.Response(
                200, content=_arxiv_atom_for(bare, abstracts[bare])
            )
        )
    return s2_routes, arxiv_routes


def test_deep_node_3_papers_s2_429_arxiv_fallback_persists_evidence_happy_path(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """3 arxiv papers, s2 429s every paper, arxiv fallback succeeds with abstracts,
    LLM returns valid output covering all 3 → evidence_items has 3 rows; no error.

    This is the precise end-to-end shape today's bounded smoke run was supposed
    to produce. If this test passes, the fallback orchestrator + per-paper fetch
    loop + evidence_store_write integration all work correctly across multiple
    papers in a single section.
    """
    patch_agent_transaction("litweave.agents.researcher_deep")
    _setup_3_paper_s2_429_arxiv_success(monkeypatch, respx_mock)

    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(return_value={
            "section_id": "S1",
            "paper_ids_processed": [
                "arxiv:1234.5678", "arxiv:2345.6789", "arxiv:3456.7890",
            ],
            "insufficient_evidence_paper_ids": [],
            # Note: evidence_id values are ignored — Deep overrides with
            # f"E-{run_id}-{section_id}-{i}". They're required by the
            # EvidenceCard schema (validated upstream of Deep's overwrite).
            "evidence_cards": [
                {"evidence_id": "ignored-1", "section_id": "S1",
                 "paper_id": "arxiv:1234.5678", "claim": "claim 1",
                 "source_span": "Paper 1 abstract.", "confidence": 0.9},
                {"evidence_id": "ignored-2", "section_id": "S1",
                 "paper_id": "arxiv:2345.6789", "claim": "claim 2",
                 "source_span": "Paper 2 abstract.", "confidence": 0.8},
                {"evidence_id": "ignored-3", "section_id": "S1",
                 "paper_id": "arxiv:3456.7890", "claim": "claim 3",
                 "source_span": "Paper 3 abstract.", "confidence": 0.85},
            ],
        }),
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_3_paper_handoff()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # All 3 papers processed → queue empty
    assert result["deep_read_queue"] == [], (
        f"expected all 3 papers processed; remaining queue: {result['deep_read_queue']}"
    )

    # 3 evidence_items rows persisted
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM evidence_items WHERE run_id = %s", (run_id,),
        )
        ev_count = cur.fetchone()[0]
    assert ev_count == 3, f"expected 3 evidence_items rows, got {ev_count}"

    # 3 evidence_store_write tool_calls rows
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM tool_calls "
            "WHERE run_id = %s AND tool_name = 'evidence_store_write'",
            (run_id,),
        )
        write_count = cur.fetchone()[0]
    assert write_count == 3, (
        f"expected 3 evidence_store_write tool_calls rows, got {write_count}"
    )

    # No error_category (clean run)
    rm = RunManager(conn)
    assert rm.get(run_id).error_category is None


def test_deep_node_llm_provider_429_after_fallback_success_records_error_no_evidence(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """s2 429 + arxiv fallback success (fetch phase OK), but structured_call raises
    HTTPStatusError(429). Verify Deep handles LLM-side 429 cleanly:
    no evidence written, papers stay in queue, runs.error_category=provider_429.

    This is one of the candidate explanations for today's bounded-smoke
    "0 evidence" outcome — LLM gateway throttled the structured_call.
    """
    import httpx

    patch_agent_transaction("litweave.agents.researcher_deep")
    _setup_3_paper_s2_429_arxiv_success(monkeypatch, respx_mock)

    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))

    llm_429 = httpx.HTTPStatusError(
        "Rate limited",
        request=httpx.Request("POST", "https://models.sjtu.edu.cn/api/v1/chat/completions"),
        response=httpx.Response(429),
    )
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(side_effect=llm_429),
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_3_paper_handoff()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # Zero evidence persisted (LLM call failed before output validation)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM evidence_items WHERE run_id = %s", (run_id,),
        )
        ev_count = cur.fetchone()[0]
    assert ev_count == 0, f"expected 0 evidence_items, got {ev_count}"

    # Zero evidence_store_write rows (LLM 429 → never reached write loop)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM tool_calls "
            "WHERE run_id = %s AND tool_name = 'evidence_store_write'",
            (run_id,),
        )
        write_count = cur.fetchone()[0]
    assert write_count == 0, (
        f"expected 0 evidence_store_write rows on LLM 429, got {write_count}"
    )

    # Papers stay queued — at least one (in practice all 3 since the only section
    # failed at LLM step)
    assert len(result["deep_read_queue"]) >= 1, (
        f"expected ≥ 1 paper still queued after LLM 429; got: {result['deep_read_queue']}"
    )

    # error_category latched as provider_429
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "provider_429"


def test_deep_node_fetch_loop_iterates_each_paper_no_cache_dedup(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """Defensive test for today's bounded-smoke mystery (s2_lookup x 1 +
    arxiv_lookup x 1 = only 2 calls total despite multi-paper queue).

    Verify the per-section pre-fetch loop calls s2_lookup AND arxiv_lookup
    EXACTLY ONCE per paper — no early break, no in-section cache dedup.
    structured_call is mocked to return empty evidence_cards (still valid output)
    so the test focuses purely on fetch-loop iteration count.

    If this test passes, the fetch loop is correct and today's external mystery
    has another explanation (LLM 429, real provider state, etc). If it fails,
    we've found a real Deep bug.
    """
    patch_agent_transaction("litweave.agents.researcher_deep")
    s2_routes, arxiv_routes = _setup_3_paper_s2_429_arxiv_success(monkeypatch, respx_mock)

    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))

    # Empty evidence_cards is valid output as long as paper_ids_processed covers
    # all input papers (insufficient_evidence_paper_ids carries the explanation).
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(return_value={
            "section_id": "S1",
            "paper_ids_processed": [
                "arxiv:1234.5678", "arxiv:2345.6789", "arxiv:3456.7890",
            ],
            "insufficient_evidence_paper_ids": [
                "arxiv:1234.5678", "arxiv:2345.6789", "arxiv:3456.7890",
            ],
            "evidence_cards": [],
        }),
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_3_paper_handoff()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Per-route: each s2 paper hit 4x (1 initial + 3 retries) = 12 total s2 HTTP
    # requests; each arxiv route hit exactly 1x = 3 total arxiv HTTP requests.
    for bare, route in s2_routes.items():
        assert route.call_count == 4, (
            f"s2 route for {bare} expected 4 hits (1 + 3 retries), got {route.call_count}"
        )
    for bare, route in arxiv_routes.items():
        assert route.call_count == 1, (
            f"arxiv route for {bare} expected 1 hit, got {route.call_count}"
        )

    # tool_calls counts (one row per ToolGateway.call invocation regardless of
    # how many HTTP retries happened inside the impl): 3 s2_lookup + 3 arxiv_lookup
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM tool_calls "
            "WHERE run_id = %s AND tool_name = 's2_lookup'",
            (run_id,),
        )
        s2_count = cur.fetchone()[0]
    assert s2_count == 3, (
        f"expected exactly 3 s2_lookup tool_calls rows (one per paper), got {s2_count}"
    )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM tool_calls "
            "WHERE run_id = %s AND tool_name = 'arxiv_lookup'",
            (run_id,),
        )
        arxiv_count = cur.fetchone()[0]
    assert arxiv_count == 3, (
        f"expected exactly 3 arxiv_lookup tool_calls rows (fallback once per paper), "
        f"got {arxiv_count}"
    )

    # Defensive: every s2_lookup row has cache_hit=False (failed calls aren't
    # cached, but if a programmer-bug stored 429 errors as success-cached, this
    # would catch it).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT cache_hit FROM tool_calls "
            "WHERE run_id = %s AND tool_name = 's2_lookup'",
            (run_id,),
        )
        cache_flags = [row[0] for row in cur.fetchall()]
    assert all(flag is False for flag in cache_flags), (
        f"all s2_lookup rows should be cache_hit=False; got: {cache_flags}"
    )


# ---- factory shape ----

def test_make_researcher_deep_node_returns_callable():
    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)
    assert callable(node)


def test_make_researcher_deep_node_accepts_rate_limited_router():
    from litweave.llm.rate_limit import RateLimitConfig, RateLimitedRouter
    router = RateLimitedRouter(
        bindings={
            AgentRole.RESEARCHER_DEEP: RoleBinding(
                provider=ProviderName.MINIMAX, model="minimax",
            ),
        },
        config=RateLimitConfig(),
    )
    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)
    assert callable(node)


# ---- real ToolGateway evidence-write integration ----

def test_deep_node_real_gateway_writes_evidence_items(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """Use REAL `_register_deep_tools` (no monkeypatch) + respx-mocked s2 API.
    Verify evidence_items table receives a row via the REAL evidence_store_write
    impl (replaces Bundle 1b placeholder)."""
    import httpx

    from litweave.agents.researcher_deep import make_researcher_deep_node
    from litweave.tools import s2_lookup

    # Strip system proxy env vars (ALL_PROXY etc.) — without this, httpx.Client
    # construction inside s2_lookup tries to set up a SOCKS5 transport (which
    # requires the optional `socksio` package not in dev deps). Mirrors the
    # autouse fixture in tests/tools/conftest.py for the one agent test that
    # actually exercises real httpx clients (others mock _fetch_abstract).
    for proxy_var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                      "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(proxy_var, raising=False)

    patch_agent_transaction("litweave.agents.researcher_deep")

    # Mock s2 API — Deep's pre-fetch uses s2_lookup
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:1706.03741").mock(
        return_value=httpx.Response(200, json={
            "paperId": "abc123",
            "externalIds": {"ArXiv": "1706.03741"},
            "title": "Original RLHF paper",
            "abstract": "We present a method for training reward models from human preferences...",
            "authors": [{"name": "Author"}],
            "year": 2017,
            "venue": "NeurIPS",
            "citationCount": 1000,
        })
    )

    # Real router + mock LLM (avoid live ChatOpenAI instantiation)
    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(return_value={
            "section_id": "S1",
            "paper_ids_processed": ["arxiv:1706.03741"],
            "evidence_cards": [{
                "evidence_id": "ignored-overridden",
                "paper_id": "arxiv:1706.03741",
                "section_id": "S1",
                "claim": "Reward modeling from human preferences",
                "source_span": "We present a method...",
                "confidence": 0.95,
            }],
            "insufficient_evidence_paper_ids": [],
        }),
    )

    # NOTE: do NOT monkeypatch _register_deep_tools — let the real one run
    # (this is the load-bearing real-gateway test)

    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Verify evidence_items row inserted via REAL EvidenceStore
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paper_id, section_id, claim, confidence, created_by "
            "FROM evidence_items WHERE run_id = %s",
            (run_id,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    paper_id, section_id, claim, confidence, created_by = rows[0]
    assert paper_id == "arxiv:1706.03741"
    assert section_id == "S1"
    assert "Reward modeling" in claim
    assert confidence == pytest.approx(0.95)
    assert created_by == "researcher_deep"


def test_deep_node_real_gateway_preserves_tool_calls_audit_on_503(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, respx_mock,
):
    """When s2_lookup raises httpx 503, the failed tool_calls audit row MUST
    persist (not get rolled back when the exception propagates). Codex P1:
    the original code did `try/except` OUTSIDE `with transaction()`, which
    triggered rollback of the audit insert. Fix: catch INSIDE the transaction.

    Uses real ToolGateway + real `_register_deep_tools` + respx-mocked HTTP.
    Asserts: runs.error_category=provider_5xx AND tool_calls row exists with
    tool_name=s2_lookup + error_category=provider_5xx."""
    import httpx

    from litweave.agents.researcher_deep import make_researcher_deep_node
    from litweave.tools import s2_lookup

    patch_agent_transaction("litweave.agents.researcher_deep")

    # Strip proxy env (see other real-gateway tests)
    for var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)

    # Mock s2 API to return 503 for the lookup
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:transient.1").mock(
        return_value=httpx.Response(503, json={"error": "service unavailable"})
    )

    router = LLMRouter({
        AgentRole.RESEARCHER_DEEP: RoleBinding(
            provider=ProviderName.MINIMAX, model="minimax",
        ),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "litweave.agents.researcher_deep.structured_call",
        MagicMock(return_value=_canned_deep_output()),
    )

    # Real _register_deep_tools — do NOT monkeypatch
    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_deep_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:transient.1")
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = node(state, config)

    # runs.error_category recorded
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "provider_5xx"

    # Paper stays in queue (transient — retryable)
    assert "arxiv:transient.1" in result["deep_read_queue"]

    # Tool_calls audit row persisted (the load-bearing assertion: Codex P1
    # caught that this row was being rolled back when exception propagated
    # out of the transaction)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tool_name, error_category, agent_role FROM tool_calls "
            "WHERE run_id = %s ORDER BY created_at",
            (run_id,),
        )
        rows = cur.fetchall()
    s2_rows = [r for r in rows if r[0] == "s2_lookup"]
    assert len(s2_rows) == 1
    assert s2_rows[0][1] == "provider_5xx"  # audit captured the classification
    assert s2_rows[0][2] == "researcher_deep"
