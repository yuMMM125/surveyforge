# SurveyForge

> Multi-agent academic survey generation system (AI/ML focus). Inspired by Stanford STORM (NAACL 2024), reimplemented in LangGraph with academic-specific improvements.

🚧 Under active development. Detailed docs to come.

## Quickstart

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env  # fill in API keys
pytest
```

## License

Apache-2.0

---

## W2 status (2026-05-02)

W2 ships a 5-node linear LangGraph pipeline + minimal CLI:

```text
START → planner → researcher_wide → researcher_deep → synthesize → write → END
```

Per-node roles (multi-section per Planner schema, 3-7 sections per outline):

- **Planner** (GLM glm-5.1): topic → 3-7 sections, each with research questions + evidence checklist
- **Researcher-Wide** (DeepSeek deepseek-chat, ≤ 8 ReAct turns per section): real arXiv search, hand-off shortlist to Deep
- **Researcher-Deep** (MiniMax minimax): per-section deep read + Semantic Scholar lookup → structured `EvidenceCard` rows
- **Synthesize stub** (W2-only): dedupe evidence by paper_id, populate `structured_extracts`
- **Write stub** (W2-only): emit markdown bullet draft per section with `[E-...]` inline citations

What is verified:

- **Code-level**: 294 unit tests pass, ruff clean, `mypy --strict` clean across all 39 source files.
- **Node-level live**: `tests/agents/integration/test_planner_live.py`, `test_researcher_wide_live.py`, and `test_researcher_deep_live.py` all independently PASS against the real SJTU gateway (GLM / DeepSeek / MiniMax) with real arXiv + Semantic Scholar.
- **Bounded graph smoke** (`tests/e2e/test_bounded_smoke.py`): architecturally complete and passes statically — Wide handoff cap (forced-exit + normal both ≤3), `s2_lookup` retry-on-429 with backoff + `Retry-After` honoring + `SEMANTIC_SCHOLAR_API_KEY` env-var support (commit `d3d451c`), full tool-calls diagnostic dump.

What is blocked:

- The bounded smoke's `evidence_store_write` assertion is gated on Semantic Scholar API access. Anonymous public quota persistently throttles `s2_lookup` from the dev IP (4 attempts × 1+2+4s backoff all return 429). The retry / header-injection design is unit-tested; live PASS is pending an `SEMANTIC_SCHOLAR_API_KEY` value (key application in flight). Once provisioned, set it in `.env` and rerun `uv run pytest tests/e2e -m integration -v`.

What is deferred (opportunistic):

- **Manual full multi-section e2e** (`tests/e2e/test_section_draft_live.py`, `@pytest.mark.manual`): same SS rate-limit blocker, plus Wide forced-exit nondeterminism on broad topics (e.g. RLHF). Run ad-hoc when both SS access and a long wall-time budget are available.

Demo (uses `python-dotenv`, so a `.env` with `SJTU_MODELS_API_KEY` and
`SURVEYFORGE_DATABASE_URL` works for both shells; will likely hit the same SS
rate limit until `SEMANTIC_SCHOLAR_API_KEY` is configured):

bash:

```bash
docker compose up -d postgres
export SURVEYFORGE_DATABASE_URL=postgresql://surveyforge:surveyforge@localhost:5432/surveyforge
export SJTU_MODELS_API_KEY=...
uv run surveyforge run --topic "Survey of RLHF progress"
```

PowerShell:

```powershell
docker compose up -d postgres
$env:SURVEYFORGE_DATABASE_URL = "postgresql://surveyforge:surveyforge@localhost:5432/surveyforge"
$env:SJTU_MODELS_API_KEY = "..."
uv run surveyforge run --topic "Survey of RLHF progress"
```

Deferred to W3-W5:

- **Real Synthesizer** (W3): schema-guided method/dataset/metric extraction with comparison matrix
- **Real Writer** (W4): multi-paragraph long-form drafting + academic citation insertion
- **Critic** + retry loop (W5): section-level + final-survey audit, including citation hallucination check
- **DB `model_calls` logging** (W3+): per-call token / latency persistence; W2 uses Langfuse-only trace metrics
