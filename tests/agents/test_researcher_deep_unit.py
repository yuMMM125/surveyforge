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

from surveyforge.agents.researcher_deep import make_researcher_deep_node
from surveyforge.llm.providers import ProviderName
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import LLMRouter, RoleBinding
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.budget import BudgetManager
from surveyforge.runtime.runs import RunManager, RunStatus
from surveyforge.schemas.planner import PlannerSection
from surveyforge.state import make_initial_state

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
        "surveyforge.agents.researcher_deep.structured_call",
        fake_structured_call,
    )

    # Mock abstract fetch — caller sets abstract_fetches[paper_id] = "abstract" or None
    def fake_fetch_abstract(gateway, paper_id):
        return abstract_fetches.get(paper_id, "default abstract for testing")

    monkeypatch.setattr(
        "surveyforge.agents.researcher_deep._fetch_abstract",
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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

def test_deep_node_drops_card_with_paper_id_not_in_section_input(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """LLM may hallucinate a paper_id or reference a different section's paper.
    Deep must drop those cards (defends against cross-section pollution).
    Architecture Decision #7c."""
    patch_agent_transaction("surveyforge.agents.researcher_deep")
    output = _canned_deep_output(section_id="S1", paper_ids=["arxiv:1706.03741"], n_cards=2)
    # Second card claims a paper_id NOT in section input
    output["evidence_cards"][1]["paper_id"] = "arxiv:hallucinated.1"
    node, _, _ = deep_node_factory(output)
    run_id = _make_run(conn)
    state = _state_with_handoff("arxiv:1706.03741")  # input = arxiv:1706.03741 only
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT paper_id FROM evidence_items WHERE run_id = %s",
            (run_id,),
        )
        paper_ids = {row[0] for row in cur.fetchall()}
    assert paper_ids == {"arxiv:1706.03741"}  # only the input paper persisted
    assert "arxiv:hallucinated.1" not in paper_ids


# ---- multi-section (real test, NOT placeholder) ----

def test_deep_node_processes_multiple_sections(
    conn: psycopg.Connection, deep_node_factory, patch_agent_transaction
):
    """Two sections in section_notes → structured_call called twice (one per
    section); each section's evidence persists independently. Verifies the
    side_effect fixture pattern + per-section isolation."""
    patch_agent_transaction("surveyforge.agents.researcher_deep")

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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")

    request = httpx.Request("GET", "https://api.semanticscholar.org/...")
    response = httpx.Response(503, request=request)
    fetch_error = httpx.HTTPStatusError("S2 down", request=request, response=response)

    # Override _fetch_abstract to raise the transport error (simulating a real
    # s2 outage caught by the node's try/classify wrap, NOT swallowed inside
    # _fetch_abstract).
    def raising_fetch(gateway, paper_id):
        raise fetch_error
    monkeypatch.setattr(
        "surveyforge.agents.researcher_deep._fetch_abstract",
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
    from surveyforge.runtime.tool_gateway import ToolRoleDenied
    patch_agent_transaction("surveyforge.agents.researcher_deep")

    def raising_fetch(gateway, paper_id):
        raise ToolRoleDenied("config bug — Deep should be allowed for s2_lookup")
    monkeypatch.setattr(
        "surveyforge.agents.researcher_deep._fetch_abstract",
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
    patch_agent_transaction("surveyforge.agents.researcher_deep")
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
    from surveyforge.runtime.tool_gateway import ToolRoleDenied
    patch_agent_transaction("surveyforge.agents.researcher_deep")

    # Patch _register_deep_tools so its registered evidence_store_write impl raises
    # ToolRoleDenied — simulates a config-bug at write time.
    import surveyforge.agents.researcher_deep as rd_mod
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
    from surveyforge.llm.rate_limit import RateLimitConfig, RateLimitedRouter
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

    from surveyforge.agents.researcher_deep import make_researcher_deep_node
    from surveyforge.tools import s2_lookup

    # Strip system proxy env vars (ALL_PROXY etc.) — without this, httpx.Client
    # construction inside s2_lookup tries to set up a SOCKS5 transport (which
    # requires the optional `socksio` package not in dev deps). Mirrors the
    # autouse fixture in tests/tools/conftest.py for the one agent test that
    # actually exercises real httpx clients (others mock _fetch_abstract).
    for proxy_var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                      "all_proxy", "http_proxy", "https_proxy"):
        monkeypatch.delenv(proxy_var, raising=False)

    patch_agent_transaction("surveyforge.agents.researcher_deep")

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
        "surveyforge.agents.researcher_deep.structured_call",
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
