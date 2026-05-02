"""W2 end-to-end live integration test — full graph against real LLMs.

Marked `@pytest.mark.integration`; default `pytest` skips this file. Run with:

    uv run pytest tests/e2e -m integration -v

Pre-reqs (per Task 7 AD #14):
  - `SJTU_MODELS_API_KEY` in env (real key, not placeholder) — REQUIRED
  - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — recommended
    (run still completes if missing; manual Phase C trace check skipped)
  - `SERPER_API_KEY` — soft prereq; missing tolerated (Wide may waste 1-2 ReAct
    turns if model picks `web_search`, but converges within the 8-turn cap)
  - testcontainer-pulled `postgres:16-alpine` reachable (Docker daemon up)
  - Network access to `https://models.sjtu.edu.cn/api/v1`,
    `https://export.arxiv.org/api/query`, `https://api.semanticscholar.org/`

Expected wall time: 8-20 minutes for a multi-section run on
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
  (e) tool_calls has >= 1 row each for arxiv_search (Wide) + s2_lookup (Deep).
      NOT evidence_store_write — Deep persists via EvidenceStore.save() host-
      managed, not tool_gateway. web_search is allowed-but-optional.

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


@pytest.mark.integration
def test_w2_end_to_end_multi_section_draft_for_rlhf(
    postgres_url: str,
    initialized_pool: ConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W2 deliverable: multi-section drafts for "Survey of RLHF progress"
    via real Planner + Wide + Deep + stub Synthesize + stub Write.

    Mirrors the harness pattern of test_planner_live / test_researcher_wide_live /
    test_researcher_deep_live; the new piece is invoking the FULL graph
    (`build_graph()`) instead of a single agent factory. Outline is multi-section
    because Planner schema enforces Field(min_length=3, max_length=7).
    """
    api_key = os.environ.get("SJTU_MODELS_API_KEY", "")
    if not api_key:
        pytest.skip("SJTU_MODELS_API_KEY not set")
    if _is_placeholder_key(api_key):
        pytest.skip(
            f"SJTU_MODELS_API_KEY appears to be a placeholder ({api_key[:12]}...)"
        )

    # Wire testcontainer URL into the production DB code path. Both runtime.db
    # (RunManager / EvidenceStore / ToolGateway) and graph._make_postgres_checkpointer
    # need to land on the same Postgres. Reset both pools so neither inherits a
    # stale connection from a prior test session.
    monkeypatch.setenv("SURVEYFORGE_DATABASE_URL", postgres_url)
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
        # Logging: per-section evidence breakdown helps debug Wide/Deep skew
        per_section_counts: dict[str, int] = {}
        for _eid, sid in evidence_rows:
            per_section_counts[sid] = per_section_counts.get(sid, 0) + 1
        print(f"[w2-e2e] evidence_per_section={per_section_counts}")

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
            f"per_section_evidence_counts={per_section_counts}"
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
        # NOTE: evidence_store_write is NOT in this set — Deep persists via
        # EvidenceStore.save() host-managed, not tool_gateway. web_search
        # is allowed-but-optional (model may or may not pick it).
        expected_tools = {"arxiv_search", "s2_lookup"}
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
