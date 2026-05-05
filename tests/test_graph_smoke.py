"""Graph wire-up smoke tests — topology + stub-node behavior.

Most tests use `InMemorySaver` (LangGraph builtin) to skip Docker bring-up.
ONE dedicated test exercises the default `PostgresSaver` path against a
testcontainer Postgres — verifying lifecycle (autocommit / prepare_threshold /
row_factory / `setup()`) AND a real `graph.invoke()` checkpoint write. That
single test is the W2 graph-level checkpoint integration coverage.

Note: Task 3/4/5 live agent tests do NOT cover graph-level checkpoint
persistence — they are node-level (each agent in isolation, real LLM but
no `compile() -> invoke()` cycle, no `PostgresSaver`). Full end-to-end
checkpoint/resume validation belongs to Task 7.

Stub Synthesizer + Writer behaviors verified directly (DB-backed) to prove
the host-managed read-from-evidence + write-to-state pipeline works.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import psycopg
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from litweave.graph import build_graph
from litweave.llm.providers import ProviderName
from litweave.llm.roles import AgentRole
from litweave.llm.router import LLMRouter, RoleBinding
from litweave.prompts.loader import PromptRegistry
from litweave.runtime.budget import BudgetManager
from litweave.runtime.evidence import EvidenceItem, EvidenceStore
from litweave.runtime.runs import RunManager
from litweave.schemas.planner import PlannerSection
from litweave.state import make_initial_state
from litweave.synthesis.stub import make_synthesize_stub_node
from litweave.writing.stub import make_write_stub_node


def _make_run(conn: psycopg.Connection) -> str:
    rm = RunManager(conn)
    return rm.create(
        topic="test", idempotency_key=f"key-{time.perf_counter_ns()}"
    ).run_id


def _outline() -> list[dict]:
    return [
        PlannerSection(
            section_id="S1",
            title="Background",
            research_questions=["Q1", "Q2"],
            must_find_evidence=["c1"],
        ).model_dump()
    ]


def _install_marker_nodes(
    monkeypatch: pytest.MonkeyPatch,
    visited: list[str] | None = None,
) -> None:
    """Replace the 5 graph node factories with marker callables.

    Used by both real-invoke smoke tests (one with InMemorySaver, one with
    testcontainer PostgresSaver). The two tests differ only in checkpointer +
    assertions, NOT in node behavior — so the marker scaffolding is shared
    here to prevent drift if a future graph node change requires updating
    both tests.

    If `visited` is provided, each marker appends its name to the list so the
    caller can assert visit order. If None, no order tracking (Postgres
    round-trip test only cares about checkpoint persistence, not order).

    Each marker also mutates state with the minimum needed for the next
    node — Deep deliberately does nothing (real Deep persists evidence to DB
    via tool_gateway; the marker bypasses that since the synth marker
    populates structured_extracts directly downstream).
    """
    def _make_marker(name, mutate_state):
        def _node(state, config):
            if visited is not None:
                visited.append(name)
            mutate_state(state)
            return state
        return _node

    def _planner_mut(s):
        s["plan_outline"] = MagicMock(
            sections=[MagicMock(
                section_id="S1", title="Background",
                research_questions=["Q1", "Q2"],
                must_find_evidence=["c1"],
            )],
        )

    def _wide_mut(s):
        s["candidate_pool"] = {"S1": [{"paper_id": "arxiv:p1"}]}
        s["deep_read_queue"] = {"S1": ["arxiv:p1"]}

    def _deep_mut(s):
        # Real Deep persists evidence to DB via tool_gateway; the marker
        # bypasses that — synth marker below populates structured_extracts
        # directly so write marker has something to format.
        pass

    def _synth_mut(s):
        s["structured_extracts"] = {"S1": {
            "section_id": "S1",
            "papers_cited": ["arxiv:p1"],
            "claims": [
                {"evidence_id": "E-test-S1-0",
                 "paper_id": "arxiv:p1",
                 "claim": "smoke claim",
                 "confidence": 0.9},
            ],
        }}

    def _write_mut(s):
        s["section_drafts"] = {
            "S1": "## Background\n\n- smoke claim [E-test-S1-0]"
        }

    monkeypatch.setattr(
        "litweave.graph.make_planner_node",
        lambda *a, **kw: _make_marker("planner", _planner_mut),
    )
    monkeypatch.setattr(
        "litweave.graph.make_researcher_wide_node",
        lambda *a, **kw: _make_marker("researcher_wide", _wide_mut),
    )
    monkeypatch.setattr(
        "litweave.graph.make_researcher_deep_node",
        lambda *a, **kw: _make_marker("researcher_deep", _deep_mut),
    )
    monkeypatch.setattr(
        "litweave.graph.make_synthesizer_node",
        lambda *a, **kw: _make_marker("synthesize", _synth_mut),
    )
    monkeypatch.setattr(
        "litweave.graph.make_write_stub_node",
        lambda: _make_marker("write", _write_mut),
    )


# ---- topology ----

def test_build_graph_returns_compiled_state_graph(monkeypatch):
    """build_graph with all-mocked deps + InMemorySaver returns a compiled graph."""
    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
        AgentRole.RESEARCHER_WIDE: RoleBinding(provider=ProviderName.DEEPSEEK, model="deepseek-chat"),
        AgentRole.RESEARCHER_DEEP: RoleBinding(provider=ProviderName.MINIMAX, model="minimax"),
        AgentRole.SYNTHESIZER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    g = build_graph(
        router=router,
        registry=PromptRegistry(),
        budget_manager=BudgetManager(),
        checkpointer=InMemorySaver(),
    )
    assert g is not None


def test_build_graph_topology_has_5_nodes(monkeypatch):
    """Topology assertion: planner / wide / deep / synthesize / write linearly chained."""
    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
        AgentRole.RESEARCHER_WIDE: RoleBinding(provider=ProviderName.DEEPSEEK, model="deepseek-chat"),
        AgentRole.RESEARCHER_DEEP: RoleBinding(provider=ProviderName.MINIMAX, model="minimax"),
        AgentRole.SYNTHESIZER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    g = build_graph(
        router=router, registry=PromptRegistry(), budget_manager=BudgetManager(),
        checkpointer=InMemorySaver(),
    )
    # CompiledStateGraph exposes nodes via `.nodes` (LangGraph v0.2+)
    assert {"planner", "researcher_wide", "researcher_deep", "synthesize", "write"}.issubset(
        set(g.nodes)
    )


def test_build_graph_raises_when_database_url_missing(monkeypatch):
    """Default checkpointer requires LITWEAVE_DATABASE_URL."""
    monkeypatch.delenv("LITWEAVE_DATABASE_URL", raising=False)
    # Reset module-level pool so a stale prior test doesn't short-circuit the
    # env-var check via the cached pool.
    from litweave import graph as graph_mod
    graph_mod._reset_checkpointer_pool_for_tests()

    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
        AgentRole.RESEARCHER_WIDE: RoleBinding(provider=ProviderName.DEEPSEEK, model="deepseek-chat"),
        AgentRole.RESEARCHER_DEEP: RoleBinding(provider=ProviderName.MINIMAX, model="minimax"),
        AgentRole.SYNTHESIZER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))
    with pytest.raises(RuntimeError, match="LITWEAVE_DATABASE_URL"):
        build_graph(
            router=router, registry=PromptRegistry(), budget_manager=BudgetManager(),
            # Don't pass checkpointer — forces _make_postgres_checkpointer()
        )


def test_build_graph_default_router_loads_llm_routing_yaml(monkeypatch):
    """No `router=` arg -> `build_graph()` loads `config/llm_routing.yaml`.

    The actual file shipped in Plan #1 is `config/llm_routing.yaml`. A path
    typo (e.g., default falling back to `config/routing.yaml`) would only
    surface at `litweave run` runtime, not during unit tests that always
    inject a `router=`. This test guards the default path explicitly so the
    typo is caught at PR time.
    """
    captured_paths: list[str] = []

    def _fake_load_routing_yaml(path: str):
        captured_paths.append(path)
        # Return minimum needed for RateLimitedRouter — empty bindings ok
        # because we never .get_llm() in this test.
        return {}

    monkeypatch.setattr(
        "litweave.graph.load_routing_yaml", _fake_load_routing_yaml,
    )
    # Stub RateLimitedRouter so we don't need real LLM provider keys.
    monkeypatch.setattr(
        "litweave.graph.RateLimitedRouter",
        MagicMock(return_value=MagicMock()),
    )

    g = build_graph(
        # NO router= -> exercises the default-load branch
        registry=PromptRegistry(),
        budget_manager=BudgetManager(),
        checkpointer=InMemorySaver(),
    )
    assert g is not None
    assert captured_paths == ["config/llm_routing.yaml"], (
        f"build_graph default routing path drifted: {captured_paths}"
    )


def test_build_graph_default_postgres_checkpointer_round_trips_invoke(
    postgres_url: str, monkeypatch,
):
    """Default checkpointer path: PostgresSaver against testcontainer URL,
    real `graph.invoke()` round-trip, >=1 row written to `checkpoints`.

    This is the W2 graph-level checkpoint integration test — the ONLY test
    that exercises the real PostgresSaver lifecycle. It guards against:
      - psycopg connection kwargs missing/wrong (autocommit, prepare_threshold,
        row_factory=dict_row); a missing dict_row would fail at first read,
        not at setup
      - PostgresSaver API drift (e.g., `from_conn_string` vs constructor)
      - Pool lifecycle bugs (closed-too-early, leaked, double-init)
      - Whether `g.invoke()` actually writes a checkpoint row (silent
        no-op would mean resume/replay never works in production)

    Marker nodes — no real LLMs / DB writes from agents. Real synth/write
    stubs are also replaced because they read from `evidence_items` which
    isn't populated in a marker run.

    Reuses `postgres_url` (session-scoped) so testcontainer bring-up cost
    is amortized across the whole pytest session.
    """
    monkeypatch.setenv("LITWEAVE_DATABASE_URL", postgres_url)
    # Reset the module-level pool so this test gets a fresh pool against the
    # testcontainer URL (not a stale pool from a prior session/test).
    from litweave import graph as graph_mod
    graph_mod._reset_checkpointer_pool_for_tests()

    # Marker nodes — no real LLMs, no DB-from-agents. This Postgres test does
    # not assert visit order (only checkpoint persistence), so no `visited=`.
    _install_marker_nodes(monkeypatch)

    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
        AgentRole.RESEARCHER_WIDE: RoleBinding(provider=ProviderName.DEEPSEEK, model="deepseek-chat"),
        AgentRole.RESEARCHER_DEEP: RoleBinding(provider=ProviderName.MINIMAX, model="minimax"),
        AgentRole.SYNTHESIZER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))

    g = build_graph(
        router=router,
        registry=PromptRegistry(),
        budget_manager=BudgetManager(),
        # NO checkpointer= -> forces _make_postgres_checkpointer()
    )
    assert g is not None

    # First assertion: setup created the canonical `checkpoints` table.
    with psycopg.connect(postgres_url) as c, c.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        tables = {row[0] for row in cur.fetchall()}
    assert "checkpoints" in tables, (
        f"PostgresSaver.setup() did not create `checkpoints` table; "
        f"got tables={sorted(tables)}"
    )

    # Second assertion (the real lifecycle check): run the graph and verify
    # PostgresSaver actually persisted at least one checkpoint row for our
    # thread_id. If `row_factory=dict_row` is missing, this is where it
    # fails (TypeError: tuple indices must be integers, not str).
    thread_id = f"smoke-pg-{int(time.perf_counter_ns())}"
    initial_state = make_initial_state(topic="pg smoke")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    result = g.invoke(initial_state, config=config)
    assert result.get("section_drafts", {}).get("S1") is not None

    with psycopg.connect(postgres_url) as c, c.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
            (thread_id,),
        )
        row = cur.fetchone()
    assert row is not None
    (count,) = row
    assert count >= 1, (
        f"PostgresSaver did not write any checkpoint rows for thread_id="
        f"{thread_id} — graph compiled but checkpoint persistence is broken"
    )

    # Cleanup: close the dedicated pool so the next test that touches the
    # default path re-opens against the same testcontainer cleanly.
    graph_mod._reset_checkpointer_pool_for_tests()


# ---- linear-flow invoke smoke ----

def test_build_graph_invoke_runs_linear_flow_with_marker_nodes(monkeypatch):
    """`graph.invoke(...)` runs planner -> wide -> deep -> synthesize -> write.

    Replaces the 3 LLM agent factories with marker callables that mutate state
    and record visit order. Real synthesize/write stubs are also replaced
    with markers (the stubs themselves have dedicated DB-backed unit tests
    below; this smoke validates ONLY topology + state propagation).

    Asserts:
      - Visit order is exactly [planner, wide, deep, synthesize, write]
      - Final state has `section_drafts["S1"]` (added by the write marker)
      - No external services (LLMs, DBs) are called

    This guards against silently-broken edges that the topology test
    (which only inspects `g.nodes`) cannot catch.
    """
    visited: list[str] = []
    _install_marker_nodes(monkeypatch, visited=visited)

    router = LLMRouter({
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
        AgentRole.RESEARCHER_WIDE: RoleBinding(provider=ProviderName.DEEPSEEK, model="deepseek-chat"),
        AgentRole.RESEARCHER_DEEP: RoleBinding(provider=ProviderName.MINIMAX, model="minimax"),
        AgentRole.SYNTHESIZER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    })
    monkeypatch.setattr(router, "get_llm", MagicMock(return_value=MagicMock()))

    g = build_graph(
        router=router,
        registry=PromptRegistry(),
        budget_manager=BudgetManager(),
        checkpointer=InMemorySaver(),
    )

    initial_state = make_initial_state(topic="smoke topic")
    config: RunnableConfig = {"configurable": {"thread_id": "smoke-thread-1"}}
    result = g.invoke(initial_state, config=config)

    assert visited == [
        "planner", "researcher_wide", "researcher_deep", "synthesize", "write",
    ], f"linear edge order broken: {visited}"
    assert result.get("section_drafts", {}).get("S1") is not None, (
        "section_drafts did not propagate to final state"
    )
    assert "smoke claim" in result["section_drafts"]["S1"]


# ---- synthesize stub ----

def test_synthesize_stub_dedupes_by_paper_id(
    conn: psycopg.Connection, patch_agent_transaction
):
    patch_agent_transaction("litweave.synthesis.stub")
    run_id = _make_run(conn)
    # Insert 3 evidence rows: 2 papers, 3 cards (paper p1 has 2 cards, paper p2 has 1)
    store = EvidenceStore(conn)
    for i, (pid, claim) in enumerate([
        ("arxiv:p1", "claim a"),
        ("arxiv:p1", "claim b"),  # same paper, different claim
        ("arxiv:p2", "claim c"),
    ]):
        store.save(EvidenceItem(
            evidence_id=f"E-{run_id}-S1-{i}",
            run_id=run_id,
            paper_id=pid,
            section_id="S1",
            claim=claim,
            source_span=None,
            source_locator=None,
            confidence=0.9,
            created_by=AgentRole.RESEARCHER_DEEP,
        ))

    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    node = make_synthesize_stub_node()
    result = node(state, config)

    s1 = result["structured_extracts"]["S1"]
    assert s1["section_id"] == "S1"
    assert s1["papers_cited"] == ["arxiv:p1", "arxiv:p2"]  # dedup, first-seen order
    assert len(s1["claims"]) == 3  # all 3 cards preserved


def test_synthesize_stub_handles_empty_evidence(
    conn: psycopg.Connection, patch_agent_transaction
):
    """No evidence_items rows -> `claims` is empty list, NOT KeyError."""
    patch_agent_transaction("litweave.synthesis.stub")
    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    node = make_synthesize_stub_node()
    result = node(state, config)

    s1 = result["structured_extracts"]["S1"]
    assert s1["claims"] == []
    assert s1["papers_cited"] == []


# ---- write stub ----

def test_write_stub_emits_section_draft_with_citations(
    conn: psycopg.Connection, patch_agent_transaction
):
    patch_agent_transaction("litweave.writing.stub")
    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    state["structured_extracts"] = {
        "S1": {
            "section_id": "S1",
            "papers_cited": ["arxiv:p1"],
            "claims": [
                {"evidence_id": f"E-{run_id}-S1-0",
                 "paper_id": "arxiv:p1",
                 "claim": "X causes Y",
                 "confidence": 0.9},
            ],
        }
    }
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    node = make_write_stub_node()
    result = node(state, config)

    draft = result["section_drafts"]["S1"]
    assert "## Background" in draft
    assert "X causes Y" in draft
    assert f"[E-{run_id}-S1-0]" in draft  # citation marker present


def test_write_stub_handles_section_with_no_evidence(
    conn: psycopg.Connection, patch_agent_transaction
):
    patch_agent_transaction("litweave.writing.stub")
    run_id = _make_run(conn)
    state = make_initial_state(topic="x")
    state["outline"] = _outline()
    state["structured_extracts"] = {"S1": {"section_id": "S1", "papers_cited": [], "claims": []}}
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    node = make_write_stub_node()
    result = node(state, config)

    draft = result["section_drafts"]["S1"]
    assert "## Background" in draft
    assert "No evidence available" in draft  # fallback line
