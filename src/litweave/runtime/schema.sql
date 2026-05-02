-- SurveyForge runtime schema (per spec § 2.7.2). PostgreSQL 16+.
-- Idempotent: safe to re-run on an existing DB; objects are CREATE IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    idempotency_key  TEXT NOT NULL UNIQUE,
    topic            TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN (
        'PENDING','RUNNING','SUCCEEDED','FAILED','CANCELLED','NEEDS_HUMAN_REVIEW'
    )),
    current_stage    TEXT,
    error_category   TEXT,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind ON artifacts(run_id, kind);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id    TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    section_id     TEXT NOT NULL,
    paper_id       TEXT NOT NULL,
    claim          TEXT NOT NULL,
    source_span    TEXT,
    source_locator TEXT,
    confidence     REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_by     TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_evidence_run_section ON evidence_items(run_id, section_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id   TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    tool_name      TEXT NOT NULL,
    tool_version   TEXT NOT NULL,
    agent_role     TEXT NOT NULL,
    input_hash     TEXT NOT NULL,
    output         JSONB,
    output_hash    TEXT,
    result_trust   TEXT NOT NULL CHECK (result_trust IN ('trusted_internal','untrusted_content')),
    latency_ms     INTEGER,
    cache_hit      BOOLEAN NOT NULL DEFAULT FALSE,
    truncated      BOOLEAN NOT NULL DEFAULT FALSE,
    error_category TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_cache ON tool_calls(tool_name, tool_version, input_hash);

CREATE TABLE IF NOT EXISTS model_calls (
    model_call_id     TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    agent_role        TEXT NOT NULL,
    gateway_id        TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    latency_ms        INTEGER,
    finish_reason     TEXT,
    error_category    TEXT,
    langfuse_trace_id TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_calls_run ON model_calls(run_id, created_at);

CREATE TABLE IF NOT EXISTS eval_results (
    eval_id    TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    evaluator  TEXT NOT NULL,
    score      REAL,
    notes      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);

INSERT INTO schema_version (version) VALUES (1)
ON CONFLICT (version) DO NOTHING;
