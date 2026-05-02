"""W2 bounded integration smoke — single section, BOTH Planner AND Wide mocked.

This is the stable auto-collected W2 integration gate. It isolates the
Deep + Synth-stub + Write-stub integration from upstream model
nondeterminism by mocking BOTH Planner (returns 1 fixed section) AND Wide
(returns 3 fixed real long-context-benchmark paper_ids), so the only real
LLM/network surface exercised is Deep's per-paper ReAct loop +
abstract-fetch fallback path + evidence persistence.

Topic choice ("long-context LLM benchmarks") + Wide mock paper_ids
(RULER `arxiv:2402.13718`, LongBench `arxiv:2308.14508`, Counting-Stars
`arxiv:2402.16671`): the section's `must_find_evidence` requires
"needle-in-haystack benchmark or RULER or LongBench"; the 3 mocked papers
are real arxiv papers exactly on this topic, so MiniMax (Deep's reader
LLM) can extract relevant evidence and emit ≥ 1 EvidenceCard. All 3
abstracts are accessible via arxiv (verified, see polish 7 fallback path).

Why mock Wide (Task 7 polish 9, 2026-05-02):
- Bounded smoke v9 (commit `547f6e8`) ran end-to-end with arxiv fallback
  firing for all 3 papers, but FAILED because real Wide's forced-exit
  dumped 3 unrelated math papers ("Long paths and cycles in subgraphs of
  the cube" etc.) into deep_read_queue. MiniMax correctly judged "no
  relevant evidence" → `evidence_count=0` → assertion failed.
- This is NOT a Deep code bug — Deep correctly handled the case (success
  path: queue cleared, drafts emitted with fallback "No evidence
  available" line). The test's `evidence_count >= 1` contract is right;
  it just depends on Wide producing relevant papers, which is unreliable
  on broad/contested topics. Fixing Wide's forced-exit selection logic is
  W3+ scope.
- Mocking Wide lets bounded smoke deterministically exercise Deep+Synth+
  Write integration without paying the cost of Wide's 8-turn ReAct or its
  forced-exit nondeterminism.

Why this design (Task 7 polish, 2026-05-02):
- W2 stub Synth/Write don't benefit from multi-section coverage (drafts are
  one-per-section bullet markdown either way; integration value is the same
  whether 1 or 7 sections).
- Multi-section live e2e is unstable: combinatorial blast of LLM/API calls
  exposes Wide forced-exit + S2 rate limit + model drift even on a single
  run. Real W2 spec deliverable ("单章草稿能跑通") is satisfied by a single
  section being drafted end-to-end with real evidence.
- Wide-real-LLM coverage moves to `tests/agents/integration/
  test_researcher_wide_live.py` (already passes ~54s) and the manual
  full e2e in `test_section_draft_live.py`.

Pre-reqs (same as test_section_draft_live.py): MODELS_API_KEY real key,
Docker daemon up for testcontainer, network to SJTU/arxiv/S2 endpoints.

Expected wall time: 30-60s (Wide's 8-turn ReAct is gone; only Deep's
per-paper ReAct + abstract fetch + evidence persist remains). If > 300s,
the test fails.

Asserts (in order, all must pass):
  (a) outline has exactly 1 section (mocked Planner)
  (b) section_drafts has key "S1" and is non-empty (Writer stub produced
      a draft for the single section)
  (c) section_notes["S1"] is non-empty (mocked Wide injects 3 entries —
      survives Deep's queue-consumption semantics; deep_read_queue itself
      may be [] after Deep finishes processing every shortlisted paper)
  (d) tool_calls table has >= 1 row each for s2_lookup (Deep abstract
      pre-fetch) AND evidence_store_write (Deep persisted EvidenceCard).
      arxiv_search is NOT expected anymore — Wide is fully bypassed by the
      mock so no real arxiv_search calls happen. arxiv_lookup may also
      appear as Deep's fallback path on SS-throttled IPs (not asserted —
      depends on whether s2 actually 429'd this run).
  (e) evidence_items table has >= 1 row (the strict assertion that
      motivated polish 9 — with deterministic relevant papers, MiniMax
      should reliably extract at least one EvidenceCard)
  (f) wall time < 300s hard cap; warn at > 180s (soft)
"""
from __future__ import annotations

import os
import re
import time

import psycopg
import pytest
from langchain_core.runnables import RunnableConfig
from psycopg_pool import ConnectionPool

from surveyforge.graph import build_graph
from surveyforge.runtime.runs import RunManager
from surveyforge.schemas.planner import PlannerSection
from surveyforge.state import make_initial_state

_PLACEHOLDER_KEY_PREFIXES = (
    "fake-", "fake_", "PASTE_YOUR_", "your-key", "test-", "dummy-", "placeholder",
)
_EVIDENCE_MARKER_RE = re.compile(r"\[(E-[A-Za-z0-9_\-]+)\]")


def _is_placeholder_key(key: str) -> bool:
    return any(key.startswith(p) for p in _PLACEHOLDER_KEY_PREFIXES)


@pytest.mark.integration
def test_w2_bounded_smoke_single_section_e2e(
    postgres_url: str,
    initialized_pool: ConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded single-section e2e smoke: mocked Planner + mocked Wide + real
    Deep + stub Synth + stub Write. The stable W2 integration gate.

    The Wide mock injects 3 real long-context-benchmark paper_ids
    (RULER `arxiv:2402.13718`, LongBench `arxiv:2308.14508`, Counting-Stars
    `arxiv:2402.16671`) so MiniMax can reliably extract evidence on the
    section's `must_find_evidence: ["needle-in-haystack benchmark or RULER
    or LongBench"]` requirement.

    See module docstring for the design rationale (single section is enough
    for stub Synth/Write integration coverage; mocking Wide isolates Deep
    from broad-topic forced-exit nondeterminism; multi-section is moved to
    `test_section_draft_live.py` as opportunistic / manual).
    """
    api_key = os.environ.get("MODELS_API_KEY") or os.environ.get("SJTU_MODELS_API_KEY", "")
    if not api_key:
        pytest.skip("MODELS_API_KEY not set")
    if _is_placeholder_key(api_key):
        pytest.skip(
            f"MODELS_API_KEY appears to be a placeholder ({api_key[:12]}...)"
        )

    # Wire testcontainer URL into both production DB pools (runtime.db +
    # graph._make_postgres_checkpointer). Reset both so stale connections from
    # prior tests don't leak.
    monkeypatch.setenv("SURVEYFORGE_DATABASE_URL", postgres_url)
    from surveyforge.graph import _reset_checkpointer_pool_for_tests
    from surveyforge.runtime.db import reset_pool, transaction
    reset_pool()
    _reset_checkpointer_pool_for_tests()

    # Mock Planner: inject 1-section outline so we don't depend on real
    # Planner producing a tractable outline for the topic. The mock still
    # writes the planning stage transition so observability looks normal.
    def _planner_node_mock(state, config):  # type: ignore[no-untyped-def]
        from surveyforge.runtime.db import transaction as _tx
        from surveyforge.runtime.runs import RunManager as _RM
        run_id = config["configurable"]["thread_id"]
        with _tx() as conn:
            _RM(conn).update_stage(run_id, "planning")
        # Topic + section content match `test_researcher_wide_live.py` (which
        # PASSES reliably ~54s on current gateway state). Broad-topic RLHF
        # variants showed Wide forced-exit even on single-section runs; see
        # the module docstring for the full rationale.
        state["outline"] = [
            PlannerSection(
                section_id="S1",
                title="Long-context benchmark methodologies",
                research_questions=[
                    "What benchmarks evaluate context length scaling?",
                    "How do they measure recall at long context?",
                ],
                must_find_evidence=[
                    "needle-in-haystack benchmark or RULER or LongBench"
                ],
            ).model_dump()
        ]
        return state

    monkeypatch.setattr(
        "surveyforge.graph.make_planner_node",
        lambda *args, **kwargs: _planner_node_mock,
    )

    # Mock Wide: inject deterministic 3-paper handoff so Deep + Synth + Write
    # integration is testable in isolation from Wide's broad-topic forced-exit
    # nondeterminism (see module docstring polish 9 rationale). The 3
    # paper_ids are real long-context benchmarks chosen so MiniMax can
    # extract relevant evidence for the section's `must_find_evidence`
    # requirement. Each section_notes entry is shaped like a CandidatePaper
    # with `handoff_to_deep=True` + `_forced_exit_stub=False` (the
    # normal-completion happy-path shape Deep expects from Wide).
    _MOCK_WIDE_PAPER_IDS = [
        "arxiv:2402.13718",  # RULER
        "arxiv:2308.14508",  # LongBench
        "arxiv:2402.16671",  # Counting-Stars (NIAH variant)
    ]
    _MOCK_WIDE_TITLES = {
        "arxiv:2402.13718": "RULER: long-context benchmark (mock Wide handoff)",
        "arxiv:2308.14508": "LongBench: long-context benchmark (mock Wide handoff)",
        "arxiv:2402.16671": "Counting-Stars / NIAH variant (mock Wide handoff)",
    }

    def _wide_node_mock(state, config):  # type: ignore[no-untyped-def]
        from surveyforge.runtime.db import transaction as _tx
        from surveyforge.runtime.runs import RunManager as _RM
        run_id = config["configurable"]["thread_id"]
        # Mirror real Wide's stage transition for observability parity.
        with _tx() as conn:
            _RM(conn).update_stage(run_id, "research_wide")
        section_notes = dict(state.get("section_notes", {}))
        section_notes["S1"] = [
            {
                "paper_id": pid,
                "title": _MOCK_WIDE_TITLES[pid],
                "source": "arxiv",
                "why_relevant": (
                    "Real long-context benchmark paper injected by bounded "
                    "smoke Wide mock; abstract retrievable via arxiv fallback."
                ),
                "handoff_to_deep": True,
                # Critical: this is the normal-completion shape (Wide ran to
                # completion and chose these 3), NOT the forced-exit shape.
                "_forced_exit_stub": False,
            }
            for pid in _MOCK_WIDE_PAPER_IDS
        ]
        state["section_notes"] = section_notes
        # Preserve any preexisting deep_read_queue ordering by appending; in
        # practice the queue is empty before Wide runs so this just sets it.
        existing_queue = list(state.get("deep_read_queue", []))
        state["deep_read_queue"] = existing_queue + list(_MOCK_WIDE_PAPER_IDS)
        return state

    monkeypatch.setattr(
        "surveyforge.graph.make_researcher_wide_node",
        lambda *args, **kwargs: _wide_node_mock,
    )

    try:
        with transaction() as conn:
            run = RunManager(conn).create(
                topic="long-context LLM benchmarks",
                idempotency_key=f"e2e-bounded-{time.time_ns()}",
            )

        # Real Deep + Synth-stub + Write-stub; Planner AND Wide are mocked
        # (polish 9). The only real LLM/network surface is Deep's per-paper
        # ReAct + abstract fetch + evidence persist.
        graph = build_graph()
        initial_state = make_initial_state(topic="long-context LLM benchmarks")
        config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

        wall_start = time.perf_counter()
        result = graph.invoke(initial_state, config=config)
        wall_elapsed_s = time.perf_counter() - wall_start

        print(
            f"\n[w2-bounded] run_id={run.run_id} "
            f"wall_time_s={wall_elapsed_s:.1f} "
            f"sections={len(result.get('outline', []))}"
        )

        # --- diagnostic dump (BEFORE assertions; testcontainer Postgres is
        # destroyed at session end so a post-failure DB query is impossible) ---

        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT status, current_stage, error_category "
                "FROM runs WHERE run_id = %s",
                (run.run_id,),
            )
            runs_row = cur.fetchone()
        print(
            f"[w2-bounded] runs: status={runs_row[0] if runs_row else None} "
            f"current_stage={runs_row[1] if runs_row else None} "
            f"error_category={runs_row[2] if runs_row else None}"
        )

        section_notes = result.get("section_notes", {})
        print(
            "[w2-bounded] section_notes per_section_counts="
            + str({k: len(v) for k, v in section_notes.items()})
        )
        for sid in sorted(section_notes.keys()):
            notes = section_notes[sid]
            handoff_count = sum(1 for n in notes if n.get("handoff_to_deep"))
            forced_exit_count = sum(1 for n in notes if n.get("_forced_exit_stub"))
            print(
                f"  {sid}: total={len(notes)} handoff_to_deep={handoff_count} "
                f"_forced_exit_stub={forced_exit_count}"
            )

        deep_queue = result.get("deep_read_queue", [])
        print(f"[w2-bounded] deep_read_queue len={len(deep_queue)} sample={deep_queue[:5]}")

        # Tool_calls breakdown grouped by error_category / cache_hit / truncated
        # so we can distinguish s2 429 from LLM 429 from arxiv timeouts when the
        # provider_429 flag appears on runs.error_category.
        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT tool_name, agent_role, error_category, cache_hit, truncated, COUNT(*) "
                "FROM tool_calls WHERE run_id = %s "
                "GROUP BY tool_name, agent_role, error_category, cache_hit, truncated "
                "ORDER BY tool_name, agent_role, error_category NULLS FIRST",
                (run.run_id,),
            )
            tool_rows = cur.fetchall()
        print(
            "[w2-bounded] tool_calls breakdown "
            "(tool_name, agent_role, error_category, cache_hit, truncated, count):"
        )
        for row in tool_rows:
            print(f"  {row}")
        if not tool_rows:
            print("  (no tool_calls rows for this run)")

        # S2_lookup output samples — paper presence + abstract length per row.
        # Helps distinguish "s2 returned but no abstract" from "s2 429" from
        # "Deep never reached LLM call" failure modes.
        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT error_category, output FROM tool_calls "
                "WHERE run_id = %s AND tool_name = 's2_lookup' "
                "ORDER BY created_at LIMIT 3",
                (run.run_id,),
            )
            s2_samples = cur.fetchall()
        print("[w2-bounded] s2_lookup samples (first 3 by created_at):")
        for i, (err_cat, output) in enumerate(s2_samples):
            if output is None:
                print(f"  sample {i}: error_category={err_cat!r} output=NULL")
                continue
            paper_id = output.get("paper_id") if isinstance(output, dict) else None
            abstract = output.get("abstract") if isinstance(output, dict) else None
            abstract_len = len(abstract) if isinstance(abstract, str) else None
            top_keys = list(output.keys())[:6] if isinstance(output, dict) else type(output).__name__
            print(
                f"  sample {i}: error_category={err_cat!r} "
                f"paper_id={paper_id!r} abstract_len={abstract_len} top_keys={top_keys}"
            )
        if not s2_samples:
            print("  (no s2_lookup rows for this run — Deep never invoked s2_lookup)")

        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM evidence_items WHERE run_id = %s",
                (run.run_id,),
            )
            evidence_count_row = cur.fetchone()
        evidence_count = evidence_count_row[0] if evidence_count_row else 0
        print(f"[w2-bounded] evidence_count={evidence_count}")

        section_drafts = result.get("section_drafts", {}) or {}
        print(f"[w2-bounded] section_drafts keys={sorted(section_drafts.keys())}")
        for sid in sorted(section_drafts.keys()):
            markers = _EVIDENCE_MARKER_RE.findall(section_drafts[sid] or "")
            print(f"[w2-bounded] section_draft {sid} marker_count={len(markers)} first3={markers[:3]}")

        # --- assertions ---
        #
        # NOTE on "persistent state" framing: do NOT assert on
        # `deep_read_queue` length — `researcher_deep.py` rebuilds
        # `state["deep_read_queue"]` with every successfully-processed paper_id
        # REMOVED (see researcher_deep.py:360-368). On the success path,
        # `deep_read_queue` may be `[]` (everything got processed), so a
        # `>= 1` assertion would contradict Deep's queue-consumption semantics
        # and fail on the happy path. Instead we assert on persistent state:
        # `section_notes` (Wide's output, never consumed) + `tool_calls` rows
        # (audit trail, never deleted) + `evidence_items` rows (Deep's
        # persisted output).

        # (a) outline has exactly 1 section (Planner was mocked)
        outline = result.get("outline", [])
        assert len(outline) == 1, (
            f"mocked Planner should yield exactly 1 section; got {len(outline)}"
        )
        assert outline[0]["section_id"] == "S1"

        # (b) section_drafts has key "S1" and is non-empty (Writer stub
        # produced a draft for the single section)
        assert "S1" in section_drafts, (
            f"section_drafts missing S1; got {sorted(section_drafts.keys())}"
        )
        assert section_drafts["S1"], "section_drafts[S1] is empty/falsy"

        # (c) section_notes["S1"] is non-empty — this is Wide's PERSISTENT
        # output (survives Deep's queue-consumption semantics, unlike
        # deep_read_queue itself). Each entry is a CandidatePaper-shaped dict
        # OR a forced-exit stub.
        assert "S1" in section_notes, (
            f"section_notes missing S1; got {sorted(section_notes.keys())}"
        )
        assert len(section_notes["S1"]) >= 1, (
            f"section_notes[S1] is empty — Wide produced no candidates. "
            f"section_notes={section_notes}"
        )

        # (d) tool_calls has rows for the expected Deep critical-path tools:
        # s2_lookup (Deep abstract pre-fetch) AND evidence_store_write (Deep
        # persisted >= 1 EvidenceCard via the real evidence_store_write impl
        # wired in Task 5; see researcher_deep.py:341
        # `write_gateway.call(..., "evidence_store_write", ...)`).
        #
        # arxiv_search is NO LONGER expected (polish 9): Wide is fully
        # bypassed by the mock so no real arxiv_search calls happen. Wide
        # search coverage is provided by
        # `tests/agents/integration/test_researcher_wide_live.py`.
        # arxiv_lookup may also appear as Deep's fallback path on
        # SS-throttled IPs — not asserted (depends on whether s2 actually
        # 429'd this run).
        tools_called = {row[0] for row in tool_rows}
        expected_tools = {"s2_lookup", "evidence_store_write"}
        missing_tools = expected_tools - tools_called
        assert not missing_tools, (
            f"missing expected tool_calls {missing_tools}; got {tools_called}. "
            f"s2_lookup=Deep abstract pre-fetch, "
            f"evidence_store_write=Deep evidence persistence — both are "
            f"required for the W2 happy path. arxiv_search is NOT in this "
            f"set anymore: Wide is fully mocked (polish 9), so no real "
            f"arxiv_search calls happen in bounded smoke."
        )

        # (e) evidence_items has >= 1 row — sanity check that the
        # evidence_store_write tool_calls rows actually persisted state.
        # NOTE: do NOT assert on [E-...] markers in draft — Deep MAY produce 0
        # evidence if S2 still flakes; the test asserts that the pipeline runs,
        # not perfect output. For strict citation integrity, use the manual
        # full e2e test (test_section_draft_live.py).
        assert evidence_count >= 1, (
            f"evidence_items table has 0 rows — Deep did not persist any evidence. "
            f"deep_queue_len={len(deep_queue)}, tool_rows={tool_rows}"
        )

        # (f) wall_time hard cap: 5 min. If > 3 min, warn (not fail).
        if wall_elapsed_s > 180:
            print(
                f"[w2-bounded] WARNING: wall time {wall_elapsed_s:.0f}s > 180s — "
                f"approaching the 5-min hard cap"
            )
        assert wall_elapsed_s < 300, (
            f"wall time {wall_elapsed_s:.0f}s exceeded 5 min hard cap — "
            f"something is wrong (rate limits / retry storm / model timeout)"
        )
    finally:
        reset_pool()
        _reset_checkpointer_pool_for_tests()
