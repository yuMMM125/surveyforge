"""Researcher-Wide ReAct loop unit tests — mocked LLM scripts each turn.

Live integration in tests/agents/integration/test_researcher_wide_live.py.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from surveyforge.agents.researcher_wide import make_researcher_wide_node
from surveyforge.llm.providers import ProviderName
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import LLMRouter, RoleBinding
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.budget import BudgetManager, BudgetSpec, OverflowFallback
from surveyforge.runtime.runs import RunManager, RunStatus
from surveyforge.schemas.planner import PlannerSection
from surveyforge.state import make_initial_state

# ---- helpers: scripted LLM responses ----

def _ai_with_tool_call(name: str, args: dict[str, Any], tc_id: str = "tc_1") -> AIMessage:
    """Build an AIMessage with a single tool_call + minimal usage_metadata."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tc_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
    )


def _ai_submit(
    candidate_papers: list[dict[str, Any]],
    notes: str = "ok",
    *,
    section_id: str = "S1",
) -> AIMessage:
    """Build an AIMessage where the LLM calls submit_results to end the ReAct loop.

    `section_id` defaults to "S1" for tests using the single-section _outline_one_section()
    helper. Multi-section tests MUST pass the correct section_id per call to avoid
    silent contract violations (the production node validates output.section_id ==
    current PlannerSection.section_id and rejects mismatches)."""
    return _ai_with_tool_call(
        "submit_results",
        {
            "section_id": section_id,
            "query": "test query",
            "candidate_papers": candidate_papers,
            "notes": notes,
        },
        tc_id=f"tc_submit_{section_id}",
    )


def _make_run(conn: psycopg.Connection) -> str:
    rm = RunManager(conn)
    return rm.create(topic="test", idempotency_key=f"key-{id(conn)}").run_id


def _outline_one_section() -> list[dict[str, Any]]:
    return [
        PlannerSection(
            section_id="S1",
            title="Methods",
            research_questions=["What methods exist?", "How do they compare?"],
            must_find_evidence=["Original RLHF paper"],
        ).model_dump()
    ]


_DEFAULT_TOOL_OUTPUT: dict[str, dict[str, Any]] = {
    "arxiv_search": {"papers": []},
    "s2_lookup": {"paper": None},
    "web_search": {"results": []},
}


@pytest.fixture
def wide_node_factory(monkeypatch: pytest.MonkeyPatch):
    """Factory fixture for ResearcherWide node with mocked LLM + fake gateway.

    Returned `make` callable signature:
        make(scripted, *, fake_tool_output=None) -> (node, gateway_calls)

    Args:
        scripted: list of AIMessage to feed sequentially as `invoker.invoke` side_effect
        fake_tool_output: optional `dict[tool_name, output_dict]` per-tool override.
            Tools not in the dict fall back to `_DEFAULT_TOOL_OUTPUT` empty stubs.
            Used by forced-exit tests that need the fake gateway to return papers
            so we can assert seen_paper_ids -> deep_read_queue.

    Returns: (node_callable, gateway_calls_list_for_assertion)
    """
    router = LLMRouter({
        AgentRole.RESEARCHER_WIDE: RoleBinding(
            provider=ProviderName.DEEPSEEK, model="deepseek-chat", temperature=0.0,
        ),
    })

    # Mock LLM with bind_tools returning a chained mock invoker
    invoker = MagicMock()
    bound_llm = MagicMock()
    bound_llm.bind_tools.return_value = invoker
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=bound_llm))

    gateway_calls: list[dict[str, Any]] = []

    # `_outputs` is a per-fixture-instance closure; updated by each `make(...)` call
    # so the same fake gateway can serve different tool-output shapes across tests.
    _outputs: dict[str, dict[str, Any]] = {}

    def fake_register_wide_tools(gateway: Any) -> None:
        from unittest.mock import MagicMock as _MM

        def _capture_call(role: AgentRole, tool_name: str, **args: Any) -> Any:
            gateway_calls.append({"tool_name": tool_name, "args": args})
            output_dict = _outputs.get(tool_name, _DEFAULT_TOOL_OUTPUT.get(tool_name, {}))
            mock_result = _MM()
            mock_result.output.model_dump.return_value = output_dict
            mock_result.result_trust = (
                "untrusted_content" if tool_name == "web_search" else "trusted_internal"
            )
            return mock_result

        gateway.call = _capture_call

    monkeypatch.setattr(
        "surveyforge.agents.researcher_wide._register_wide_tools",
        fake_register_wide_tools,
    )

    registry = PromptRegistry()
    budget_manager = BudgetManager()

    def make(
        scripted: list[AIMessage],
        *,
        fake_tool_output: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        invoker.invoke.side_effect = list(scripted)
        _outputs.clear()
        if fake_tool_output:
            _outputs.update(fake_tool_output)
        return make_researcher_wide_node(router, registry, budget_manager), gateway_calls

    return make


# ---- normal completion path ----

def test_wide_node_submit_results_populates_section_notes(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    candidate_papers = [
        {"paper_id": "arxiv:2401.12345", "title": "Paper 1", "source": "arxiv",
         "why_relevant": "directly answers RQ1", "handoff_to_deep": False},
        {"paper_id": "arxiv:2402.99999", "title": "Paper 2", "source": "arxiv",
         "why_relevant": "ablation results", "handoff_to_deep": True},
    ]
    scripted = [_ai_submit(candidate_papers)]
    node, _ = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="RLHF survey")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    assert "S1" in result["section_notes"]
    assert len(result["section_notes"]["S1"]) == 2
    # handoff_to_deep=True papers are queued for Deep
    assert "arxiv:2402.99999" in result["deep_read_queue"]
    # handoff_to_deep=False paper is NOT queued
    assert "arxiv:2401.12345" not in result["deep_read_queue"]


def test_wide_node_dispatches_tool_call_then_submits(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """Two-turn ReAct: turn 1 calls arxiv_search, turn 2 calls submit_results."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    scripted = [
        _ai_with_tool_call("arxiv_search", {"query": "RLHF", "max_results": 5}, tc_id="tc_1"),
        _ai_submit([
            {"paper_id": "arxiv:2401.12345", "title": "P", "source": "arxiv",
             "why_relevant": "x", "handoff_to_deep": True},
        ]),
    ]
    node, gateway_calls = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    assert len(gateway_calls) == 1
    assert gateway_calls[0]["tool_name"] == "arxiv_search"
    assert gateway_calls[0]["args"]["query"] == "RLHF"
    assert "arxiv:2401.12345" in result["deep_read_queue"]


# ---- 8-turn hard cap ----

def test_wide_node_8_turn_cap_preserves_seen_paper_ids(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """LLM calls arxiv_search 8 times in a row without submit_results -> cap triggers.
    Per DoD: stubs for ALL seen paper_ids must appear in section_notes (Deep's
    cross-reference path), but deep_read_queue is bounded by
    MAX_HANDOFF_TO_DEEP_PER_SECTION to keep Deep within S2 rate limits.
    error_category=context_overflow must be recorded (non-terminal — run continues)."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    from surveyforge.agents.researcher_wide import MAX_HANDOFF_TO_DEEP_PER_SECTION
    scripted = [_ai_with_tool_call("arxiv_search", {"query": f"q{i}"}, tc_id=f"tc_{i}")
                for i in range(8)]
    # Each of the 8 arxiv_search calls returns one unique paper
    node, gateway_calls = wide_node_factory(
        scripted,
        fake_tool_output={
            "arxiv_search": {"papers": [
                {"paper_id": f"arxiv:2401.0000{i}", "arxiv_id": f"2401.0000{i}", "title": f"P{i}",
                 "authors": [], "abstract": "", "published": "2024-01-01T00:00:00Z",
                 "categories": [], "pdf_url": None}
                for i in range(8)
            ]},
        },
    )

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    assert len(gateway_calls) == 8  # all 8 turns dispatched
    # Forced-exit shape: section_notes contains stub entries (one per seen paper)
    # so Deep's cross-reference picks them up — NOT empty list. (Task 5 / Decision #5)
    stubs = result["section_notes"]["S1"]
    assert len(stubs) == 8  # one stub per seen paper (full set preserved for diagnostics)
    assert all(s["_forced_exit_stub"] is True for s in stubs)
    assert {s["paper_id"] for s in stubs} == {f"arxiv:2401.0000{i}" for i in range(8)}
    # deep_read_queue is CAPPED at MAX_HANDOFF_TO_DEEP_PER_SECTION (NOT all 8)
    # to keep Deep within S2 rate limits (Task 7 polish, 2026-05-02).
    assert len(result["deep_read_queue"]) == MAX_HANDOFF_TO_DEEP_PER_SECTION
    # error_category recorded, status unchanged
    rm = RunManager(conn)
    refreshed = rm.get(run_id)
    assert refreshed.error_category == "context_overflow"
    assert refreshed.status == RunStatus.RUNNING


# ---- budget overflow ----

def test_wide_node_budget_exceeded_preserves_seen_paper_ids(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch,
    wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """When BudgetManager.check raises, loop exits AND all seen paper_ids
    flow into deep_read_queue (per DoD: forced exit must preserve candidates)."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    # Override Wide's budget to a tiny number so turn 2's check trips it
    # (use setitem, NOT setattr — BUDGET_PER_ROLE is a dict, not an object)
    from surveyforge.runtime import budget as budget_mod
    monkeypatch.setitem(
        budget_mod.BUDGET_PER_ROLE,
        AgentRole.RESEARCHER_WIDE,
        BudgetSpec(max_input_tokens=100, reserved_output_tokens=50,
                   fallback=OverflowFallback.SNIPPET_ONLY),
    )

    # Turn 1: arxiv_search returns 2 papers (both go into seen_paper_ids).
    # Turn 2: budget check raises before invoke; no submit_results.
    scripted = [
        _ai_with_tool_call("arxiv_search", {"query": "x"}, tc_id="tc_1"),
        _ai_submit([]),  # would run if budget allowed (but won't)
    ]
    node, gateway_calls = wide_node_factory(
        scripted,
        fake_tool_output={
            "arxiv_search": {"papers": [
                {"paper_id": "arxiv:2401.11111", "arxiv_id": "2401.11111", "title": "P1",
                 "authors": [], "abstract": "", "published": "2024-01-01T00:00:00Z",
                 "categories": [], "pdf_url": None},
                {"paper_id": "arxiv:2401.22222", "arxiv_id": "2401.22222", "title": "P2",
                 "authors": [], "abstract": "", "published": "2024-01-02T00:00:00Z",
                 "categories": [], "pdf_url": None},
            ]},
        },
    )

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    # Only 1 tool dispatch — turn 2 budget check raises BudgetExceeded
    assert len(gateway_calls) == 1
    # BOTH paper_ids from turn-1 results land in deep_read_queue (forced exit)
    assert "arxiv:2401.11111" in result["deep_read_queue"]
    assert "arxiv:2401.22222" in result["deep_read_queue"]
    # Forced-exit shape: section_notes contains stub entries (one per seen paper)
    # so Deep's cross-reference picks them up — NOT empty list. (Task 5 / Decision #5)
    stubs = result["section_notes"]["S1"]
    assert len(stubs) == 2
    assert all(s["_forced_exit_stub"] is True for s in stubs)
    assert {s["paper_id"] for s in stubs} == {"arxiv:2401.11111", "arxiv:2401.22222"}


def test_wide_node_overflow_calls_note_error_category(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch,
    wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """Forced exit (budget OR turn cap) must call RunManager.note_error_category
    so observability captures the event without failing the run."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    from surveyforge.runtime import budget as budget_mod
    monkeypatch.setitem(
        budget_mod.BUDGET_PER_ROLE,
        AgentRole.RESEARCHER_WIDE,
        BudgetSpec(max_input_tokens=100, reserved_output_tokens=50,
                   fallback=OverflowFallback.SNIPPET_ONLY),
    )
    scripted = [_ai_with_tool_call("arxiv_search", {"query": "x"}, tc_id="tc_1")]
    node, _ = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Verify runs.error_category was set without changing status
    rm = RunManager(conn)
    refreshed = rm.get(run_id)
    assert refreshed.error_category == "context_overflow"
    # status should still be RUNNING (not FAILED — note_error_category is non-terminal)
    assert refreshed.status == RunStatus.RUNNING


# ---- web_search trust wrapping ----

def test_wide_node_wraps_web_search_results_as_untrusted(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """web_search.result_trust=untrusted_content -> ToolMessage content wrapped via trust.wrap_untrusted."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")

    scripted = [
        _ai_with_tool_call("web_search", {"query": "untrusted"}, tc_id="tc_1"),
        _ai_submit([]),
    ]
    node, _ = wide_node_factory(scripted)

    import surveyforge.agents.researcher_wide as rw_mod
    original_wrap = rw_mod.wrap_untrusted
    wrap_calls: list[dict[str, Any]] = []

    def tracking_wrap(content: str, *, source_tool: str, evidence_id: str) -> str:
        wrap_calls.append({"source_tool": source_tool, "evidence_id": evidence_id})
        return original_wrap(content, source_tool=source_tool, evidence_id=evidence_id)

    rw_mod.wrap_untrusted = tracking_wrap  # type: ignore[assignment]

    try:
        run_id = _make_run(conn)
        state = make_initial_state(topic="x")
        state["outline"] = _outline_one_section()
        config: RunnableConfig = {"configurable": {"thread_id": run_id}}
        node(state, config)

        # web_search result was wrapped (trusted_internal tools wouldn't be)
        assert len(wrap_calls) == 1
        assert wrap_calls[0]["source_tool"] == "web_search"
        assert wrap_calls[0]["evidence_id"].startswith("E-wide-S1-T")
    finally:
        rw_mod.wrap_untrusted = original_wrap  # type: ignore[assignment]


# ---- multi-section serial ----

def test_wide_node_processes_all_sections_serially(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """Two sections in outline -> ReAct runs twice, each section gets its own notes."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    scripted = [
        # Section 1: directly submit (with correct section_id)
        _ai_submit(
            [{"paper_id": "arxiv:1.1", "title": "P1", "source": "arxiv",
              "why_relevant": "x", "handoff_to_deep": True}],
            section_id="S1",
        ),
        # Section 2: directly submit (DIFFERENT section_id — proves per-section isolation)
        _ai_submit(
            [{"paper_id": "arxiv:2.1", "title": "P2", "source": "arxiv",
              "why_relevant": "y", "handoff_to_deep": False}],
            section_id="S2",
        ),
    ]
    node, _ = wide_node_factory(scripted)

    outline = _outline_one_section()
    outline.append(PlannerSection(
        section_id="S2", title="Eval", research_questions=["q1", "q2"],
        must_find_evidence=["c1"],
    ).model_dump())

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = outline
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    assert set(result["section_notes"].keys()) == {"S1", "S2"}
    # Only S1's paper was flagged for deep
    assert "arxiv:1.1" in result["deep_read_queue"]
    assert "arxiv:2.1" not in result["deep_read_queue"]


# ---- JSON fallback path (Architecture Decision #10) ----

def test_wide_node_json_fallback_when_llm_emits_content_not_tool_call(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """Defense: if LLM returns ResearcherWideOutput JSON in content (no tool_call),
    parse it as completion rather than treating as forced exit. Handles model-
    quality drift where LLM ignores the 'always use submit_results' instruction."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")

    fallback_output = json.dumps({
        "section_id": "S1",
        "query": "fallback query",
        "candidate_papers": [
            {"paper_id": "arxiv:2401.fallback", "title": "FB", "source": "arxiv",
             "why_relevant": "x", "handoff_to_deep": True},
        ],
        "notes": "model output JSON in content, not tool_call",
    })
    # AIMessage with content but NO tool_calls
    scripted = [AIMessage(
        content=fallback_output,
        tool_calls=[],
        usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
    )]
    node, _ = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    # Treated as completion, NOT context_overflow
    assert "arxiv:2401.fallback" in result["deep_read_queue"]
    rm = RunManager(conn)
    assert rm.get(run_id).error_category is None  # NOT set


def test_wide_node_json_fallback_unparseable_content_treated_as_overflow(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """If content is non-empty but unparseable as ResearcherWideOutput JSON,
    forced exit with context_overflow."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    scripted = [AIMessage(
        content="I think we should use BERT for this task.",  # plain prose, not JSON
        tool_calls=[],
        usage_metadata={"input_tokens": 500, "output_tokens": 50, "total_tokens": 550},
    )]
    node, _ = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "context_overflow"


# ---- real ToolGateway integration (not all-fake) ----

def test_wide_node_real_gateway_writes_tool_calls(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch,
    patch_agent_transaction: Any, respx_mock: Any,
) -> None:
    """Use the REAL _register_wide_tools (no monkeypatch override) + respx-mocked
    HTTP. Verify the tool_calls table actually receives a row — proves the
    gateway integration isn't accidentally bypassed by mocks."""
    import httpx

    from surveyforge.agents.researcher_wide import make_researcher_wide_node
    from surveyforge.tools import arxiv_search

    patch_agent_transaction("surveyforge.agents.researcher_wide")
    monkeypatch.setenv("SERPER_API_KEY", "test-not-used")

    # Mock arxiv API HTTP response (used by the real arxiv_search wrapper)
    arxiv_atom = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.99999v1</id>
    <updated>2024-01-15T00:00:00Z</updated>
    <published>2024-01-15T00:00:00Z</published>
    <title>Real Gateway Test Paper</title>
    <summary>abstract</summary>
    <author><name>RG Author</name></author>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2401.99999v1"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""
    respx_mock.get(arxiv_search.ARXIV_API_URL).mock(
        return_value=httpx.Response(200, content=arxiv_atom)
    )

    # Real router with mock LLM (we still mock the LLM so we don't pay for live calls)
    router = LLMRouter({
        AgentRole.RESEARCHER_WIDE: RoleBinding(
            provider=ProviderName.DEEPSEEK, model="deepseek-chat",
        ),
    })
    invoker = MagicMock()
    bound_llm = MagicMock()
    bound_llm.bind_tools.return_value = invoker
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=bound_llm))

    invoker.invoke.side_effect = [
        _ai_with_tool_call("arxiv_search", {"query": "real-gw-test", "max_results": 5}, tc_id="tc_real"),
        _ai_submit([{"paper_id": "arxiv:2401.99999", "title": "Real Gateway Test Paper",
                     "source": "arxiv", "why_relevant": "test", "handoff_to_deep": True}]),
    ]

    # NOTE: do NOT monkeypatch _register_wide_tools — let the real one run
    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_wide_node(router, registry, budget_manager)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    node(state, config)

    # Verify tool_calls row exists with real arxiv_search dispatch
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tool_name, agent_role, cache_hit, result_trust "
            "FROM tool_calls WHERE run_id = %s ORDER BY created_at",
            (run_id,),
        )
        rows = cur.fetchall()
    # Expect at least one arxiv_search row (the real wrapper got dispatched)
    assert any(r[0] == "arxiv_search" and r[1] == "researcher_wide" for r in rows), (
        f"arxiv_search tool_call not found in {rows}"
    )


# ---- factory shape ----

def test_make_researcher_wide_node_returns_callable() -> None:
    router = LLMRouter({
        AgentRole.RESEARCHER_WIDE: RoleBinding(
            provider=ProviderName.DEEPSEEK, model="deepseek-chat",
        ),
    })
    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_wide_node(router, registry, budget_manager)
    assert callable(node)


def test_make_researcher_wide_node_accepts_rate_limited_router() -> None:
    """Same RouterProtocol satisfaction as Task 3 polish — production uses RateLimitedRouter."""
    from surveyforge.llm.rate_limit import RateLimitConfig, RateLimitedRouter
    router = RateLimitedRouter(
        bindings={
            AgentRole.RESEARCHER_WIDE: RoleBinding(
                provider=ProviderName.DEEPSEEK, model="deepseek-chat",
            ),
        },
        config=RateLimitConfig(),
    )
    registry = PromptRegistry()
    budget_manager = BudgetManager()
    node = make_researcher_wide_node(router, registry, budget_manager)
    assert callable(node)


# ---- Step 4.0g: SUBMIT_TOOL_NAME / completion_tools cross-check ----

def test_submit_tool_name_matches_prompt_completion_tools() -> None:
    """Cross-check: agent's SUBMIT_TOOL_NAME constant must appear in the prompt's
    `completion_tools` front-matter field. If they drift, prompt + agent are
    referring to different tool names — silent contract break."""
    from surveyforge.agents import researcher_wide as rw_mod
    registry = PromptRegistry()
    template = registry.load(AgentRole.RESEARCHER_WIDE)
    assert rw_mod.SUBMIT_TOOL_NAME in template.completion_tools


def test_extract_paper_ids_raises_on_unknown_tool():
    """Unknown tool name → ValueError, not silent empty-set fallback.

    Defends against Task 5 (Researcher-Deep) or later additions that bind a
    new tool but forget to extend `_extract_paper_ids_from_tool_result`'s
    if-chain. Without this raise, the new tool's results would silently never
    be added to seen_paper_ids → deep_read_queue (paper_id loss bug)."""
    from unittest.mock import MagicMock

    from surveyforge.agents.researcher_wide import _extract_paper_ids_from_tool_result

    fake_result = MagicMock()
    fake_result.output.model_dump.return_value = {"papers": [{"paper_id": "arxiv:1"}]}

    with pytest.raises(ValueError, match="unknown tool"):
        _extract_paper_ids_from_tool_result("future_tool_not_in_chain", fake_result)


def test_wide_node_submit_results_mismatch_continues_loop_then_caps(
    conn: psycopg.Connection, wide_node_factory, patch_agent_transaction
):
    """LLM submits with WRONG section_id repeatedly — node feeds error back,
    loop continues until 8-turn cap. Cap-exit driven by mismatches is classified
    as `schema_invalid` (NOT context_overflow), so production observability can
    distinguish 'LLM has a contract bug' from 'LLM ran out of exploration turns'.
    Verifies (a) wrong-section candidates do NOT corrupt section_notes for the
    current section; (b) re-prompt mechanism works; (c) error_category precision."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    bad_candidates = [
        {"paper_id": "arxiv:wrong.1", "title": "Wrong section paper", "source": "arxiv",
         "why_relevant": "should be ignored", "handoff_to_deep": True},
    ]
    # 8 turns, all submit_results with WRONG section_id (current section is S1; LLM keeps emitting S2)
    scripted = [_ai_submit(bad_candidates, section_id="S2") for _ in range(8)]
    # Make IDs unique so multiple identical AIMessages don't collide on tc_id
    for i, msg in enumerate(scripted):
        msg.tool_calls[0]["id"] = f"tc_submit_S2_{i}"
    node, _ = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()  # one section: "S1"
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    # Wrong-section candidates were NOT written to S1's notes (mismatch rejected)
    assert "arxiv:wrong.1" not in [
        p.get("paper_id") for p in result["section_notes"].get("S1", [])
    ]
    assert result["section_notes"].get("S1", []) == []  # forced exit shape

    # Forced exit (8-turn cap with no acceptable submit) → context_overflow recorded
    rm = RunManager(conn)
    refreshed = rm.get(run_id)
    # Repeated section_id mismatches drive the cap → cap-exit classified as
    # schema_invalid (Task 4 polish #3). Loose context_overflow OR schema_invalid
    # would mask whether the contract precision is intact.
    assert refreshed.error_category == "schema_invalid"
    from surveyforge.runtime.runs import RunStatus
    assert refreshed.status == RunStatus.RUNNING  # non-terminal


def test_wide_forced_exit_handoff_caps_at_three_papers_per_section(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """Forced-exit (8-turn cap) with > 3 candidate paper_ids must NOT dump all of
    them into deep_read_queue. Cap at MAX_HANDOFF_TO_DEEP_PER_SECTION (=3) to
    keep Deep within Semantic Scholar's effective per-second throughput.

    Without this cap, a broad-topic forced-exit (the empirical ~30 candidates per
    section observed in the 2026-05-02 live run) makes Deep s2_lookup serially,
    hits 429s, and abandons section_notes. Bounding to 3 papers per section
    keeps the post-Wide queue tractable. Sorted for determinism — set ordering
    is not stable across Python invocations."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    from surveyforge.agents.researcher_wide import MAX_HANDOFF_TO_DEEP_PER_SECTION

    # 8 arxiv_search calls, each returning a non-overlapping batch so total
    # seen_paper_ids = 10 unique paper_ids (well above the cap of 3).
    # 8th turn never produces submit_results → 8-turn cap triggers forced-exit.
    paper_ids_returned = [f"arxiv:2401.{i:05d}" for i in range(10)]
    scripted = [_ai_with_tool_call("arxiv_search", {"query": f"q{i}"}, tc_id=f"tc_{i}")
                for i in range(8)]
    node, gateway_calls = wide_node_factory(
        scripted,
        fake_tool_output={
            "arxiv_search": {"papers": [
                {"paper_id": pid, "arxiv_id": pid.split(":", 1)[1], "title": f"P{pid}",
                 "authors": [], "abstract": "", "published": "2024-01-01T00:00:00Z",
                 "categories": [], "pdf_url": None}
                for pid in paper_ids_returned
            ]},
        },
    )

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    # Sanity: all 8 turns dispatched, all 10 paper_ids seen in section_notes stubs
    assert len(gateway_calls) == 8
    stubs = result["section_notes"]["S1"]
    assert len(stubs) == 10  # full diagnostic set preserved in section_notes
    assert {s["paper_id"] for s in stubs} == set(paper_ids_returned)

    # Cap enforced: deep_read_queue has EXACTLY 3 entries (NOT all 10)
    assert len(result["deep_read_queue"]) == MAX_HANDOFF_TO_DEEP_PER_SECTION
    assert MAX_HANDOFF_TO_DEEP_PER_SECTION == 3, (
        "test scaffolding assumes cap=3; update assertions if cap changes"
    )

    # Determinism: the 3 entries are the first 3 of sorted(seen_paper_ids).
    # Sort matters because set iteration order is not stable across runs.
    expected = sorted(paper_ids_returned)[:MAX_HANDOFF_TO_DEEP_PER_SECTION]
    assert result["deep_read_queue"] == expected


def test_wide_node_json_fallback_mismatch_treated_as_schema_invalid(
    conn: psycopg.Connection, wide_node_factory, patch_agent_transaction
):
    """JSON fallback path: LLM emits valid-shape ResearcherWideOutput in plain
    content (no tool_calls), BUT section_id is wrong. No tool_call_id to feed
    back, so forced exit with schema_invalid (NOT context_overflow)."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")

    mismatch_output = json.dumps({
        "section_id": "S2",  # wrong — current section is S1
        "query": "fallback",
        "candidate_papers": [
            {"paper_id": "arxiv:wrong.fb", "title": "FB", "source": "arxiv",
             "why_relevant": "x", "handoff_to_deep": True},
        ],
        "notes": "json fallback with wrong section_id",
    })
    scripted = [AIMessage(
        content=mismatch_output,
        tool_calls=[],
        usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
    )]
    node, _ = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()  # S1
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    # Wrong-section candidates NOT in S1's notes
    assert "arxiv:wrong.fb" not in [
        p.get("paper_id") for p in result["section_notes"].get("S1", [])
    ]
    # Forced exit, schema_invalid (parsed but contract violation, not budget overflow)
    rm = RunManager(conn)
    assert rm.get(run_id).error_category == "schema_invalid"


def test_wide_normal_completion_handoff_caps_at_three_papers_per_section(
    conn: psycopg.Connection, wide_node_factory: Any, patch_agent_transaction: Any,
) -> None:
    """Normal completion (LLM emits successful submit_results) with > 3 candidates
    all flagged handoff_to_deep=True must NOT dump every paper_id into
    deep_read_queue. The MAX_HANDOFF_TO_DEEP_PER_SECTION cap (=3) applies to
    BOTH forced-exit AND normal-completion paths — without this, a verbose model
    that picks 20 candidates per section would defeat the bounded-smoke design
    (Task 7 polish 2, 2026-05-02).

    section_notes recording remains UNCAPPED (full diagnostic set preserved);
    only the deep_read_queue handoff is bounded. Iteration order preserved
    (the first 3 LLM-supplied candidates land in deep_read_queue)."""
    patch_agent_transaction("surveyforge.agents.researcher_wide")
    from surveyforge.agents.researcher_wide import MAX_HANDOFF_TO_DEEP_PER_SECTION

    # 5 candidate papers, all tagged handoff_to_deep=True. Iteration order is
    # what the LLM submitted (we control it here for deterministic assertion).
    candidate_papers = [
        {"paper_id": f"arxiv:2401.{i:05d}", "title": f"P{i}", "source": "arxiv",
         "why_relevant": "x", "handoff_to_deep": True}
        for i in range(5)
    ]
    scripted = [_ai_submit(candidate_papers)]
    node, _ = wide_node_factory(scripted)

    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline_one_section()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = node(state, config)

    # section_notes recording is UNCAPPED — all 5 candidates land in S1's notes
    notes = result["section_notes"]["S1"]
    assert len(notes) == 5
    assert {n["paper_id"] for n in notes} == {f"arxiv:2401.{i:05d}" for i in range(5)}

    # deep_read_queue is CAPPED at 3 (NOT all 5)
    assert len(result["deep_read_queue"]) == MAX_HANDOFF_TO_DEEP_PER_SECTION
    assert MAX_HANDOFF_TO_DEEP_PER_SECTION == 3, (
        "test scaffolding assumes cap=3; update assertions if cap changes"
    )

    # Iteration order preserved: first 3 candidates land in the queue
    assert result["deep_read_queue"] == [f"arxiv:2401.{i:05d}" for i in range(3)]
