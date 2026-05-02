# SurveyForge

SurveyForge is a technical preview of a multi-agent academic survey generation
system for AI/ML literature review workflows.

It is inspired by the research direction behind systems such as STORM, but the
implementation focuses on a production-style agent foundation: typed prompt
contracts, guarded tool access, runtime persistence, evidence storage, and
auditable execution traces.

## What It Does

SurveyForge takes a survey topic and runs a LangGraph pipeline:

```text
START -> planner -> researcher_wide -> researcher_deep -> synthesize -> write -> END
```

The current pipeline can:

- plan a topic into structured survey sections;
- search for candidate papers through arXiv;
- look up paper metadata and abstracts through Semantic Scholar;
- route tool calls through a policy-enforced gateway;
- persist run state, tool calls, checkpoints, and evidence rows in PostgreSQL;
- produce a minimal section draft from stored evidence;
- expose a small CLI for `run` and `status`.

## Architecture Highlights

- **LangGraph orchestration**: a linear graph with Planner, Researcher-Wide,
  Researcher-Deep, Synthesize, and Write nodes.
- **Prompt contract pack**: YAML front matter, role-specific prompt files,
  allowed-tool declarations, completion-tool support, and prompt registry tests.
- **Runtime contract pack**: PostgreSQL-backed run lifecycle, checkpointing,
  evidence storage, tool-call audit rows, budget checks, error classification,
  and trust-boundary helpers.
- **Tool gateway**: role-based tool policies, Pydantic input/output validation,
  cache-aware `input_hash`, redaction-safe hashing, and tool-call persistence.
- **Evidence-first design**: factual claims are expected to point back to
  persisted `EvidenceItem` records instead of free-floating model prose.
- **Provider routing**: OpenAI-compatible model access through a configurable
  gateway, with per-role routing and rate limiting.

## Current Status

This repository is currently a **technical preview**, not a finished survey
writer.

Implemented and verified locally:

- prompt/schema contracts for Planner, Researcher-Wide, and Researcher-Deep;
- PostgreSQL runtime foundation and LangGraph checkpoint wiring;
- real arXiv search wrapper;
- Semantic Scholar paper lookup wrapper with retry-on-429, `Retry-After`
  handling, and optional `SEMANTIC_SCHOLAR_API_KEY` support;
- Serper-based web search wrapper;
- Planner, Researcher-Wide, and Researcher-Deep nodes;
- minimal Synthesize and Write stubs;
- CLI draft preview support;
- non-integration test suite, ruff, and mypy strict checks.

W2 status (as of polish 5, 2026-05-02):

- Semantic Scholar is now an optional enhancement. Researcher-Deep falls back
  to the arxiv API on transient S2 failures (HTTP 429/5xx/network) for
  `arxiv:*` papers, so the bounded smoke and CLI demo no longer require an
  SS key. The SS key remains supported via `SEMANTIC_SCHOLAR_API_KEY` env
  var for higher-quality abstracts on s2-only papers and to lift the
  anonymous-IP rate limit.
- All known external blockers resolved as of polish 5. The bounded smoke
  (`tests/e2e/test_bounded_smoke.py`) is now a stable auto gate that
  exercises real Wide + Deep + stub Synth/Write end-to-end without
  requiring an SS key.

## Repository Layout

```text
config/
  llm_routing.yaml              # role -> provider/model routing
src/surveyforge/
  agents/                       # Planner, Researcher-Wide, Researcher-Deep
  llm/                          # provider registry, router, rate limits
  prompts/                      # role prompts + shared rules
  runtime/                      # DB, run manager, gateway, evidence, trust
  schemas/                      # structured outputs and citation IDs
  tools/                        # arXiv, Semantic Scholar, Serper wrappers
  graph.py                      # LangGraph construction
  cli.py                        # surveyforge run/status
tests/
  agents/
  e2e/
  llm/
  prompts/
  runtime/
  schemas/
  tools/
```


## Setup

Install dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env
```

Fill in `.env`:

```env
MODELS_BASE_URL=
MODELS_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
SERPER_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

`MODELS_BASE_URL` is optional. If omitted, SurveyForge uses the default
OpenAI-compatible gateway configured in `src/surveyforge/llm/providers.py`.
When switching providers, update both `MODELS_BASE_URL` and the model aliases in
`config/llm_routing.yaml`.

`SEMANTIC_SCHOLAR_API_KEY` is optional for code paths and unit tests, but is
recommended for live integration runs. `SERPER_API_KEY` is optional unless the
model chooses `web_search` during a live run.

Start PostgreSQL:

```bash
docker compose up -d postgres
```

For local CLI runs, make sure `SURVEYFORGE_DATABASE_URL` is set. You can put it
in `.env` or export it in your shell:

```env
SURVEYFORGE_DATABASE_URL=postgresql://surveyforge:surveyforge@localhost:5432/surveyforge
```

## Usage

Run a topic:

```bash
uv run surveyforge run --topic "long-context LLM benchmarks"
```

Check a run:

```bash
uv run surveyforge status <run_id>
```

On Windows PowerShell, use the same `uv run ...` commands after filling `.env`.
If `uv` is not on PATH, call it with its absolute path.

## Testing

Default non-integration suite:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Live model/API tests are opt-in:

```bash
uv run pytest tests/agents/integration -m integration -v
uv run pytest tests/e2e -m integration -v -s
```

Manual full-graph live test:

```bash
uv run pytest tests/e2e/test_section_draft_live.py -m manual -v -s
```

The manual full-graph test is opportunistic. It is useful for demos and
diagnostics, but it is not treated as a stable default gate because it
depends on external model behavior across multiple sections.

## Roadmap

Near-term work:

- harden provider/API retry behavior and live-run diagnostics;
- add richer synthesis beyond the current evidence dedupe stub;
- replace the writer stub with long-form academic drafting;
- add critic/review loops for citation integrity and coverage;
- persist model-call token and latency records in the database;
- expand evaluation tasks for factuality, citation grounding, and section
  completeness.

## License

MIT License. See [LICENSE](LICENSE).
