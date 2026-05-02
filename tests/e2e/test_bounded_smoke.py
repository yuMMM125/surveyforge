"""W2 bounded integration smoke — single section, real LLMs, <3 min budget.

This is the stable auto-collected W2 integration gate. It bypasses the
multi-section nondeterminism of `test_section_draft_live.py` (now manual)
by mocking Planner to return a single fixed section, then exercising real
Wide + Deep + Synth-stub + Write-stub end-to-end.

Why this design (Task 7 polish, 2026-05-02):
- W2 stub Synth/Write don't benefit from multi-section coverage (drafts are
  one-per-section bullet markdown either way; integration value is the same
  whether 1 or 7 sections).
- Multi-section live e2e is unstable: combinatorial blast of LLM/API calls
  exposes Wide forced-exit + S2 rate limit + model drift even on a single
  run. Real W2 spec deliverable ("单章草稿能跑通") is satisfied by a single
  section being drafted end-to-end with real evidence.
- Wide handoff is now capped at 3 papers/section on BOTH forced-exit AND
  normal-completion paths (per `MAX_HANDOFF_TO_DEEP_PER_SECTION` in
  researcher_wide.py), so Deep gets a bounded queue and won't get
  S2-throttled regardless of which exit shape Wide takes.

Pre-reqs (same as test_section_draft_live.py): SJTU_MODELS_API_KEY real key,
Docker daemon up for testcontainer, network to SJTU/arxiv/S2 endpoints.
SERPER_API_KEY soft prereq (Wide may pick web_search; missing tolerated).

Expected wall time: 60-180s. If > 300s, the test fails (something is wrong).

Asserts (in order, all must pass):
  (a) outline has exactly 1 section (mocked Planner)
  (b) section_drafts has key "S1" and is non-empty (Writer stub produced
      a draft for the single section)
  (c) section_notes["S1"] is non-empty (Wide's persistent output — survives
      Deep's queue-consumption semantics; deep_read_queue itself may be []
      after Deep finishes processing every shortlisted paper)
  (d) tool_calls table has >= 1 row each for arxiv_search (Wide called
      arxiv successfully), s2_lookup (Deep abstract pre-fetch), and
      evidence_store_write (Deep persisted >= 1 EvidenceCard via the real
      gateway — see researcher_deep.py ~line 341)
  (e) evidence_items table has >= 1 row (sanity — same persisted state as
      the evidence_store_write tool_calls rows)
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
    """Bounded single-section e2e smoke: mocked Planner + real Wide + Deep +
    stub Synth + stub Write. The stable W2 integration gate.

    See module docstring for the design rationale (single section is enough
    for stub Synth/Write integration coverage; multi-section is moved to
    `test_section_draft_live.py` as opportunistic / manual).
    """
    api_key = os.environ.get("SJTU_MODELS_API_KEY", "")
    if not api_key:
        pytest.skip("SJTU_MODELS_API_KEY not set")
    if _is_placeholder_key(api_key):
        pytest.skip(
            f"SJTU_MODELS_API_KEY appears to be a placeholder ({api_key[:12]}...)"
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
        state["outline"] = [
            PlannerSection(
                section_id="S1",
                title="Background",
                research_questions=[
                    "What is RLHF?",
                    "Why does it matter for AI alignment?",
                ],
                must_find_evidence=["Original RLHF paper"],
            ).model_dump()
        ]
        return state

    monkeypatch.setattr(
        "surveyforge.graph.make_planner_node",
        lambda *args, **kwargs: _planner_node_mock,
    )

    try:
        with transaction() as conn:
            run = RunManager(conn).create(
                topic="Survey of RLHF progress",
                idempotency_key=f"e2e-bounded-{time.time_ns()}",
            )

        # Real Wide + Deep + Synth-stub + Write-stub; only Planner is mocked.
        graph = build_graph()
        initial_state = make_initial_state(topic="Survey of RLHF progress")
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

        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT tool_name, agent_role, COUNT(*) "
                "FROM tool_calls WHERE run_id = %s "
                "GROUP BY tool_name, agent_role "
                "ORDER BY tool_name, agent_role",
                (run.run_id,),
            )
            tool_rows = cur.fetchall()
        print("[w2-bounded] tool_calls breakdown (tool_name, agent_role, count):")
        for row in tool_rows:
            print(f"  {row}")
        if not tool_rows:
            print("  (no tool_calls rows for this run)")

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

        # (d) tool_calls has rows for the expected W2 critical-path tools:
        # arxiv_search (Wide engaged arxiv), s2_lookup (Deep abstract pre-fetch),
        # AND evidence_store_write (Deep persisted >= 1 EvidenceCard via the
        # real evidence_store_write impl wired in Task 5; see
        # researcher_deep.py:341 `write_gateway.call(..., "evidence_store_write", ...)`).
        # web_search is allowed-but-optional (Wide may pick it).
        tools_called = {row[0] for row in tool_rows}
        expected_tools = {"arxiv_search", "s2_lookup", "evidence_store_write"}
        missing_tools = expected_tools - tools_called
        assert not missing_tools, (
            f"missing expected tool_calls {missing_tools}; got {tools_called}. "
            f"arxiv_search=Wide search, s2_lookup=Deep abstract pre-fetch, "
            f"evidence_store_write=Deep evidence persistence — all 3 are "
            f"required for the W2 happy path."
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
