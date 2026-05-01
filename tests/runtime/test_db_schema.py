"""Schema-level tests: 7 tables exist, schema_version inserted, key constraints fire."""
from __future__ import annotations

import psycopg
import pytest

EXPECTED_TABLES = {
    "schema_version",
    "runs",
    "artifacts",
    "evidence_items",
    "tool_calls",
    "model_calls",
    "eval_results",
}


def test_schema_creates_seven_tables(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT table_name FROM information_schema.tables
               WHERE table_schema = 'public'"""
        )
        tables = {row[0] for row in cur.fetchall()}
    assert EXPECTED_TABLES.issubset(tables), f"missing: {EXPECTED_TABLES - tables}"


def test_schema_version_row_inserted(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_version")
        rows = cur.fetchall()
    assert (1,) in rows


def test_runs_idempotency_key_unique(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (run_id, idempotency_key, topic, status) "
            "VALUES (%s, %s, %s, %s)",
            ("r1", "key-A", "topic1", "PENDING"),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO runs (run_id, idempotency_key, topic, status) "
                "VALUES (%s, %s, %s, %s)",
                ("r2", "key-A", "topic2", "PENDING"),
            )


def test_runs_status_check_constraint(conn: psycopg.Connection):
    with conn.cursor() as cur, pytest.raises(psycopg.errors.CheckViolation):
        cur.execute(
            "INSERT INTO runs (run_id, idempotency_key, topic, status) "
            "VALUES (%s, %s, %s, %s)",
            ("r1", "k1", "t1", "INVALID_STATUS"),
        )


def test_evidence_items_confidence_range(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (run_id, idempotency_key, topic, status) "
            "VALUES ('r1','k1','t1','PENDING')"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                """INSERT INTO evidence_items
                   (evidence_id, run_id, section_id, paper_id, claim, confidence)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                ("E-1", "r1", "S1", "arxiv:1", "claim", 1.5),
            )


def test_artifacts_cascades_with_run_delete(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (run_id, idempotency_key, topic, status) "
            "VALUES ('r1','k1','t1','PENDING')"
        )
        cur.execute(
            "INSERT INTO artifacts (artifact_id, run_id, kind, payload) "
            "VALUES ('a1', 'r1', 'planner_output', '{}'::jsonb)"
        )
        cur.execute("DELETE FROM runs WHERE run_id = 'r1'")
        cur.execute("SELECT COUNT(*) FROM artifacts WHERE run_id = 'r1'")
        assert cur.fetchone() == (0,)


def test_tool_calls_has_full_observability_columns(conn: psycopg.Connection):
    """Spec § 2.7.3 requires agent_role / cache_hit / output_hash / truncated logging."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = 'tool_calls'"""
        )
        cols = {row[0] for row in cur.fetchall()}
    expected = {
        "tool_call_id", "run_id", "tool_name", "tool_version",
        "agent_role", "input_hash", "output", "output_hash",
        "result_trust", "latency_ms", "cache_hit", "truncated",
        "error_category", "created_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"
