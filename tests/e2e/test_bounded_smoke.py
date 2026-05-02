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
- Wide forced-exit handoff is now capped at 3 papers/section (per
  `MAX_FORCED_EXIT_HANDOFF_PER_SECTION` in researcher_wide.py), so even when
  Wide forced-exits, Deep gets a bounded queue and won't get S2-throttled.

Pre-reqs (same as test_section_draft_live.py): SJTU_MODELS_API_KEY real key,
Docker daemon up for testcontainer, network to SJTU/arxiv/S2 endpoints.
SERPER_API_KEY soft prereq (Wide may pick web_search; missing tolerated).

Expected wall time: 60-180s. If > 300s, the test fails (something is wrong).
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

        # (a) outline has exactly 1 section (Planner was mocked)
        outline = result.get("outline", [])
        assert len(outline) == 1, (
            f"mocked Planner should yield exactly 1 section; got {len(outline)}"
        )
        assert outline[0]["section_id"] == "S1"

        # (b) deep_read_queue: Wide produced some output (>= 1) AND the cap
        # (MAX_FORCED_EXIT_HANDOFF_PER_SECTION = 3) wasn't violated. With a
        # single section, even forced-exit yields at most 3 entries.
        from surveyforge.agents.researcher_wide import (
            MAX_FORCED_EXIT_HANDOFF_PER_SECTION,
        )
        assert len(deep_queue) >= 1, (
            f"deep_read_queue empty — Wide produced no output for the section. "
            f"section_notes={section_notes}"
        )
        assert len(deep_queue) <= MAX_FORCED_EXIT_HANDOFF_PER_SECTION + 5, (
            # Allow some slack: normal completion can submit up to N candidates.
            # The +5 is a generous upper bound; if Wide submits more than 8
            # candidates for a single section the prompt has changed in a
            # surprising way and the test should surface it.
            f"deep_read_queue unexpectedly large: {len(deep_queue)} "
            f"(forced-exit cap is {MAX_FORCED_EXIT_HANDOFF_PER_SECTION}); "
            f"check Wide's candidate-submission behavior"
        )

        # (c) evidence_items has >= 1 row (Deep persisted at least 1 EvidenceCard).
        # NOTE: do NOT assert on [E-...] markers in draft — Deep MAY produce 0
        # evidence if S2 still flakes; the test asserts that the pipeline runs,
        # not perfect output. For strict citation integrity, use the manual
        # full e2e test (test_section_draft_live.py).
        assert evidence_count >= 1, (
            f"evidence_items table has 0 rows — Deep did not persist any evidence. "
            f"deep_queue_len={len(deep_queue)}, tool_rows={tool_rows}"
        )

        # (d) section_drafts has key "S1" and is non-empty
        assert "S1" in section_drafts, (
            f"section_drafts missing S1; got {sorted(section_drafts.keys())}"
        )
        assert section_drafts["S1"], "section_drafts[S1] is empty/falsy"

        # (e) tool_calls has >= 1 row for arxiv_search (Wide called arxiv successfully)
        tools_called = {row[0] for row in tool_rows}
        assert "arxiv_search" in tools_called, (
            f"arxiv_search not in tool_calls; got {tools_called}. "
            f"Wide never dispatched arxiv_search — check Wide invocation."
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
