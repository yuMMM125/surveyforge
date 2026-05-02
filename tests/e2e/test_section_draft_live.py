"""W2 manual / opportunistic full e2e — multi-section graph against real LLMs.

Marked `@pytest.mark.manual`; default `pytest` AND default `-m integration`
collection BOTH skip this file. Run with:

    uv run pytest tests/e2e/test_section_draft_live.py -m manual -v -s

This is NOT a CI gate. As of 2026-05-02 (Task 7 polish), multi-section live e2e
is opportunistic / manual-only because real-LLM runs revealed three distinct
nondeterminism sources that compound on broad topics like "Survey of RLHF
progress":

  1. Wide forced-exit cascades — broad topics make the model's 8-turn ReAct
     budget inadequate; some/all sections forced-exit with `context_overflow`
     instead of submitting a triaged shortlist.
  2. S2 rate-limit storms — when Wide forced-exits, it hands off many candidate
     paper_ids; Deep then s2_lookups them serially and gets 429'd. The
     `MAX_HANDOFF_TO_DEEP_PER_SECTION` cap mitigates this but doesn't
     fully eliminate the failure mode on broad topics.
  3. Model behavior drift — DeepSeek/GLM/MiniMax behavior on the same prompt
     varies day-to-day; an outline that succeeded yesterday may forced-exit
     today.

For the auto-collected integration smoke, see `tests/e2e/test_bounded_smoke.py`.

Run this test ad-hoc when:
  - You want to demo the full multi-section pipeline live.
  - You're investigating a real-world failure mode and want full prompts/traces.
  - You have ~8-25 min wall-time budget AND real LLM/API quota to spend.

Expect ~8-25 min wall time + real LLM/API quota consumption per run.

Pre-reqs (per Task 7 AD #14):
  - `MODELS_API_KEY` in env (real key, not placeholder) - REQUIRED
  - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — recommended
    (run still completes if missing; manual Phase C trace check skipped)
  - `SERPER_API_KEY` — soft prereq; missing tolerated (Wide may waste 1-2 ReAct
    turns if model picks `web_search`, but converges within the 8-turn cap)
  - testcontainer-pulled `postgres:16-alpine` reachable (Docker daemon up)
  - Network access to `https://models.sjtu.edu.cn/api/v1`,
    `https://export.arxiv.org/api/query`, `https://api.semanticscholar.org/`

Expected wall time: 8-25 minutes for a multi-section run on
"Survey of RLHF progress". The Planner schema enforces 3-7 sections;
Wide and Deep iterate ALL sections sequentially:
  - Planner: 1 GLM call
  - Wide: <= 8 DeepSeek ReAct turns x N sections (where N in [3, 7])
  - Deep: <= 3 MiniMax structured calls x N sections (one per shortlisted paper)
If the test exceeds 30 minutes, rate limits or retries are misconfigured.

Asserts (in order, all must pass):
  (a) outline >= 3 sections (Planner schema floor); each has non-empty title
      + >= 2 research_questions + >= 1 must_find_evidence
  (b) evidence_items has >= 3 rows total for this run (across all sections)
  (c) section_drafts has one entry per outline section; >= 1 of them contains
      at least one [E-...] citation marker
  (d) every [E-...] across all drafts resolves to a real evidence_id row
      (citation integrity — zero orphans)
  (e) tool_calls has >= 1 row each for arxiv_search (Wide), s2_lookup (Deep),
      AND evidence_store_write (Deep — one row per EvidenceCard persisted via
      the real evidence_store_write impl wired in Task 5; see researcher_deep.py
      line ~341 `write_gateway.call(..., "evidence_store_write", ...)`).
      web_search is allowed-but-optional (Wide may pick it).

NOT asserted (covered by spike log + Langfuse manual inspection):
  - per-role token counts (W2 has no model_calls writer; Langfuse-only)
  - Langfuse trace tree shape (manual dashboard inspection)
  - exact wall time (logged to stdout but not strict-asserted)
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
from surveyforge.state import make_initial_state

_PLACEHOLDER_KEY_PREFIXES = (
    "fake-", "fake_", "PASTE_YOUR_", "your-key", "test-", "dummy-", "placeholder",
)
_EVIDENCE_MARKER_RE = re.compile(r"\[(E-[A-Za-z0-9_\-]+)\]")


def _is_placeholder_key(key: str) -> bool:
    return any(key.startswith(p) for p in _PLACEHOLDER_KEY_PREFIXES)


@pytest.mark.manual
def test_w2_end_to_end_multi_section_draft_for_rlhf(
    postgres_url: str,
    initialized_pool: ConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W2 manual / opportunistic full e2e: multi-section drafts for
    "Survey of RLHF progress" via real Planner + Wide + Deep + stub Synthesize
    + stub Write.

    NOT a CI gate. Use `tests/e2e/test_bounded_smoke.py` for the auto-collected
    integration smoke. This test is opportunistic / manual-only because
    multi-section live e2e is nondeterministic on broad topics:
    Wide forced-exit cascades + S2 rate-limit storms + day-to-day model behavior
    drift compound. Run ad-hoc when you want to demo the full pipeline; expect
    ~8-25 min wall time + real LLM/API quota.

    Mirrors the harness pattern of test_planner_live / test_researcher_wide_live /
    test_researcher_deep_live; the new piece is invoking the FULL graph
    (`build_graph()`) instead of a single agent factory. Outline is multi-section
    because Planner schema enforces Field(min_length=3, max_length=7).
    """
    api_key = os.environ.get("MODELS_API_KEY") or os.environ.get("SJTU_MODELS_API_KEY", "")
    if not api_key:
        pytest.skip("MODELS_API_KEY not set")
    if _is_placeholder_key(api_key):
        pytest.skip(
            f"MODELS_API_KEY appears to be a placeholder ({api_key[:12]}...)"
        )

    # Wire testcontainer URL into the production DB code path. Both runtime.db
    # (RunManager / EvidenceStore / ToolGateway) and graph._make_postgres_checkpointer
    # need to land on the same Postgres. Reset both pools so neither inherits a
    # stale connection from a prior test session.
    monkeypatch.setenv("LITWEAVE_DATABASE_URL", postgres_url)
    from surveyforge.graph import _reset_checkpointer_pool_for_tests
    from surveyforge.runtime.db import reset_pool, transaction
    reset_pool()
    _reset_checkpointer_pool_for_tests()

    try:
        # Build the graph with all defaults — real RateLimitedRouter from
        # config/llm_routing.yaml, real PromptRegistry, real BudgetManager,
        # real PostgresSaver checkpointer. This is the EXACT path `surveyforge run`
        # uses; the only difference is we skip the argparse + RunManager.create
        # boilerplate and create the run manually for assertion convenience.
        with transaction() as conn:
            run = RunManager(conn).create(
                topic="Survey of RLHF progress",
                idempotency_key=f"e2e-w2-{time.time_ns()}",
            )

        graph = build_graph()
        initial_state = make_initial_state(topic="Survey of RLHF progress")
        config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

        wall_start = time.perf_counter()
        result = graph.invoke(initial_state, config=config)
        wall_elapsed_s = time.perf_counter() - wall_start

        # Sanity log so manual review can spot anomalies without re-running.
        print(
            f"\n[w2-e2e] run_id={run.run_id} "
            f"wall_time_s={wall_elapsed_s:.1f} "
            f"sections={len(result.get('outline', []))}"
        )

        # --- diagnostic dump: state + DB queries BEFORE any assertion ---
        # Testcontainer Postgres is destroyed when pytest exits, so any
        # post-failure DB query is impossible. These prints capture every
        # signal we'd want for diagnosing a partial-pipeline failure
        # (Wide produced no candidates / Deep rejected every output / s2 broke /
        # Writer fabricated citations / etc.).

        # (1) runs table — final lifecycle state
        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT status, current_stage, error_category "
                "FROM runs WHERE run_id = %s",
                (run.run_id,),
            )
            runs_row = cur.fetchone()
        print(
            f"[w2-e2e] runs: status={runs_row[0] if runs_row else None} "
            f"current_stage={runs_row[1] if runs_row else None} "
            f"error_category={runs_row[2] if runs_row else None}"
        )

        # (2) section_notes detail per section (Wide output)
        section_notes = result.get("section_notes", {})
        print(
            "[w2-e2e] section_notes per_section_counts="
            + str({k: len(v) for k, v in section_notes.items()})
        )
        for sid in sorted(section_notes.keys()):
            notes = section_notes[sid]
            handoff_count = sum(1 for n in notes if n.get("handoff_to_deep"))
            forced_exit_count = sum(1 for n in notes if n.get("_forced_exit_stub"))
            source_dist: dict[str, int] = {}
            for n in notes:
                src = n.get("source", "<unknown>")
                source_dist[src] = source_dist.get(src, 0) + 1
            sample_paper_ids = [n.get("paper_id") for n in notes[:3]]
            print(
                f"  {sid}: total={len(notes)} handoff_to_deep={handoff_count} "
                f"_forced_exit_stub={forced_exit_count} sources={source_dist} "
                f"first3_paper_ids={sample_paper_ids}"
            )

        # (3) deep_read_queue (Wide → Deep handoff)
        deep_queue = result.get("deep_read_queue", [])
        print(f"[w2-e2e] deep_read_queue len={len(deep_queue)} sample={deep_queue[:5]}")

        # (4) tool_calls breakdown grouped by error_category / cache_hit / truncated
        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT tool_name, agent_role, error_category, cache_hit, truncated, COUNT(*) "
                "FROM tool_calls WHERE run_id = %s "
                "GROUP BY tool_name, agent_role, error_category, cache_hit, truncated "
                "ORDER BY tool_name, agent_role, error_category NULLS FIRST",
                (run.run_id,),
            )
            tool_breakdown = cur.fetchall()
        print(
            "[w2-e2e] tool_calls breakdown "
            "(tool_name, agent_role, error_category, cache_hit, truncated, count):"
        )
        for row in tool_breakdown:
            print(f"  {row}")
        if not tool_breakdown:
            print("  (no tool_calls rows for this run — Wide and Deep never invoked any tool)")

        # (5) Deep critical path — s2_lookup vs evidence persistence
        # Both s2_lookup AND evidence_store_write go through tool_gateway in W2:
        # `researcher_deep.py:341` calls
        # `write_gateway.call(RESEARCHER_DEEP, "evidence_store_write", ...)` for
        # each EvidenceCard. So s2_lookup>0 + evidence_store_write>0 is the
        # happy-path shape. s2_lookup>0 + evidence_store_write=0 means Deep
        # rejected every output (no EvidenceCards produced) and is worth
        # diagnosing — usually s2 returned no abstracts and Deep had nothing
        # to extract.
        s2_count = sum(r[5] for r in tool_breakdown if r[0] == "s2_lookup")
        esw_count = sum(r[5] for r in tool_breakdown if r[0] == "evidence_store_write")
        print(
            f"[w2-e2e] deep_path: s2_lookup_calls={s2_count} "
            f"evidence_store_write_calls={esw_count}"
        )
        if s2_count > 0 and esw_count == 0:
            with psycopg.connect(postgres_url) as c, c.cursor() as cur:
                cur.execute(
                    "SELECT output FROM tool_calls WHERE run_id = %s "
                    "AND tool_name = 's2_lookup' ORDER BY created_at LIMIT 3",
                    (run.run_id,),
                )
                s2_samples = cur.fetchall()
            print("[w2-e2e] s2_lookup output samples (first 3) — paper presence + abstract length:")
            for i, (output,) in enumerate(s2_samples):
                if output is None:
                    print(f"  sample {i}: output=NULL (truncated or error?)")
                    continue
                paper_id = output.get("paper_id") if isinstance(output, dict) else None
                abstract = output.get("abstract") if isinstance(output, dict) else None
                abstract_len = len(abstract) if isinstance(abstract, str) else None
                top_keys = list(output.keys())[:6] if isinstance(output, dict) else type(output).__name__
                print(
                    f"  sample {i}: paper_id={paper_id!r} "
                    f"abstract_len={abstract_len} top_keys={top_keys}"
                )

        # (6) section_drafts marker counts (Writer output)
        drafts_for_diag = result.get("section_drafts", {}) or {}
        print(f"[w2-e2e] section_drafts keys={sorted(drafts_for_diag.keys())}")
        for sid in sorted(drafts_for_diag.keys()):
            markers = _EVIDENCE_MARKER_RE.findall(drafts_for_diag[sid] or "")
            print(f"  {sid}: marker_count={len(markers)} first3={markers[:3]}")

        # (7) evidence_per_section — moved BEFORE assertion (b) so we always see it
        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT section_id, COUNT(*) FROM evidence_items "
                "WHERE run_id = %s GROUP BY section_id ORDER BY section_id",
                (run.run_id,),
            )
            evidence_per_section_pre = dict(cur.fetchall())
        print(f"[w2-e2e] evidence_per_section={evidence_per_section_pre}")

        # --- (a) outline >= 3 sections (Planner schema floor) ---
        outline = result["outline"]
        assert len(outline) >= 3, (
            f"Planner schema requires >= 3 sections (Field(min_length=3, max_length=7)); "
            f"got {len(outline)}: {[s.get('section_id') for s in outline]}"
        )
        for section in outline:
            assert section.get("title"), (
                f"section_id={section.get('section_id')} has empty title"
            )
            assert len(section.get("research_questions", [])) >= 2, (
                f"section_id={section.get('section_id')} has "
                f"<2 research_questions: {section.get('research_questions')}"
            )
            assert len(section.get("must_find_evidence", [])) >= 1, (
                f"section_id={section.get('section_id')} has "
                f"<1 must_find_evidence: {section.get('must_find_evidence')}"
            )

        # --- (b) evidence_items has >= 3 rows total (across all sections) ---
        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT evidence_id, section_id FROM evidence_items "
                "WHERE run_id = %s",
                (run.run_id,),
            )
            evidence_rows = cur.fetchall()
        evidence_ids_in_db = {row[0] for row in evidence_rows}
        assert len(evidence_ids_in_db) >= 3, (
            f"expected >= 3 evidence rows total across all sections, "
            f"got {len(evidence_ids_in_db)}: {sorted(evidence_ids_in_db)[:5]}"
        )
        # (evidence_per_section already printed above as part of pre-assertion
        # diagnostic block; no need to re-print here.)

        # --- (c) section_drafts has one entry per section; >= 1 has citations ---
        section_drafts = result.get("section_drafts", {})
        section_ids_in_outline = {s["section_id"] for s in outline}
        assert section_drafts, (
            f"section_drafts must not be empty; got {section_drafts}"
        )
        # Writer stub creates one entry per outline section (with the
        # "_No evidence available_" fallback if claims are empty), so
        # section_drafts keys should match outline section_ids.
        section_ids_in_drafts = set(section_drafts.keys())
        missing_drafts = section_ids_in_outline - section_ids_in_drafts
        assert not missing_drafts, (
            f"section_drafts missing entries for outline sections: "
            f"{missing_drafts} (Writer stub should produce one per section)"
        )

        # >= 1 draft must contain at least one [E-...] marker
        drafts_with_markers = {
            sid: _EVIDENCE_MARKER_RE.findall(d)
            for sid, d in section_drafts.items()
        }
        sections_with_citations = {sid: m for sid, m in drafts_with_markers.items() if m}
        assert sections_with_citations, (
            f"no section_drafts contain [E-...] citation markers; "
            f"draft section_ids={sorted(section_ids_in_drafts)}, "
            f"per_section_evidence_counts={evidence_per_section_pre}"
        )

        # --- (d) citation integrity: every [E-...] resolves to a real row ---
        all_markers_in_drafts: set[str] = set()
        for marker_list in sections_with_citations.values():
            all_markers_in_drafts.update(marker_list)
        orphans = all_markers_in_drafts - evidence_ids_in_db
        assert not orphans, (
            f"orphan citations in section_drafts (Writer fabricated or stale): "
            f"{sorted(orphans)[:5]} (run_id={run.run_id})"
        )

        # --- (e) tool_calls table has the expected W2 calls ---
        with psycopg.connect(postgres_url) as c, c.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT tool_name FROM tool_calls WHERE run_id = %s",
                (run.run_id,),
            )
            tools_called = {row[0] for row in cur.fetchall()}
        # All three of arxiv_search (Wide), s2_lookup (Deep abstract pre-fetch),
        # and evidence_store_write (Deep evidence persistence — wired in Task 5,
        # researcher_deep.py:341) go through tool_gateway and are required for
        # the W2 happy path. web_search is allowed-but-optional (model may or
        # may not pick it).
        expected_tools = {"arxiv_search", "s2_lookup", "evidence_store_write"}
        missing_tools = expected_tools - tools_called
        assert not missing_tools, (
            f"missing expected tool_calls: {missing_tools}; got {tools_called}"
        )

        # Soft wall-time check (warning, not failure).
        if wall_elapsed_s > 1800:
            print(
                f"\n[w2-e2e] WARNING: wall time {wall_elapsed_s:.0f}s > 30min — "
                f"rate limits / retries may be misconfigured"
            )
    finally:
        reset_pool()
        _reset_checkpointer_pool_for_tests()
