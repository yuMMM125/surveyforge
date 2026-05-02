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

## W2 status (2026-05-DD)

W2 ships a 5-node linear LangGraph pipeline + minimal CLI:

```text
START → planner → researcher_wide → researcher_deep → synthesize → write → END
```

What runs end-to-end (multi-section per Planner schema, 3-7 sections per outline):

- **Planner** (GLM glm-5.1): topic → 3-7 sections, each with research questions + evidence checklist
- **Researcher-Wide** (DeepSeek deepseek-chat, ≤ 8 ReAct turns per section): real arXiv search, hand-off shortlist to Deep
- **Researcher-Deep** (MiniMax minimax): per-section deep read + Semantic Scholar lookup → structured `EvidenceCard` rows
- **Synthesize stub** (W2-only): dedupe evidence by paper_id, populate `structured_extracts`
- **Write stub** (W2-only): emit markdown bullet draft per section with `[E-...]` inline citations

Demo (uses `python-dotenv`, so a `.env` with `SJTU_MODELS_API_KEY` and
`SURVEYFORGE_DATABASE_URL` works for both shells):

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

W2 acceptance structure (multi-tier):

- **Node-level live tests** (auto, ~30-60s each):
  ```bash
  uv run pytest tests/agents/integration -m integration -v
  ```

- **Bounded graph smoke** (auto, ~60-180s, single section + real LLMs/APIs):
  ```bash
  uv run pytest tests/e2e -m integration -v
  ```

- **Manual full e2e** (opportunistic, ~8-25 min, multi-section + real LLMs/APIs; flakey on broad topics + S2 rate limits):
  ```bash
  uv run pytest tests/e2e/test_section_draft_live.py -m manual -v -s
  ```

- **Manual CLI demo** (the W2 deliverable):
  ```bash
  uv run surveyforge run --topic "Survey of RLHF progress"
  ```
