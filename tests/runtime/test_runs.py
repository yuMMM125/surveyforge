"""RunManager round-trip + idempotency + state transition tests."""
from __future__ import annotations

import psycopg
import pytest

from litweave.runtime.runs import IdempotencyConflict, RunManager, RunStatus


def test_run_status_has_six_values():
    assert {s.value for s in RunStatus} == {
        "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "NEEDS_HUMAN_REVIEW",
    }


def test_create_inserts_run_in_pending(conn: psycopg.Connection):
    rm = RunManager(conn)
    run = rm.create(topic="RLHF survey", idempotency_key="key-1")
    assert run.status == RunStatus.PENDING
    assert run.topic == "RLHF survey"
    assert run.idempotency_key == "key-1"
    assert run.cancel_requested is False
    assert run.run_id.startswith("run_")


def test_idempotency_key_conflict_surfaces_existing_run_id(conn: psycopg.Connection):
    rm = RunManager(conn)
    first = rm.create(topic="t1", idempotency_key="dup-key")
    with pytest.raises(IdempotencyConflict) as exc:
        rm.create(topic="t2", idempotency_key="dup-key")
    assert exc.value.existing_run_id == first.run_id
    assert exc.value.idempotency_key == "dup-key"


def test_update_stage_transitions_to_running(conn: psycopg.Connection):
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k1")
    rm.update_stage(run.run_id, "planning")
    refreshed = rm.get(run.run_id)
    assert refreshed.status == RunStatus.RUNNING
    assert refreshed.current_stage == "planning"


def test_succeed_marks_status_succeeded(conn: psycopg.Connection):
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k1")
    rm.update_stage(run.run_id, "writing")
    rm.succeed(run.run_id)
    assert rm.get(run.run_id).status == RunStatus.SUCCEEDED


def test_fail_records_error_category(conn: psycopg.Connection):
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k1")
    rm.fail(run.run_id, error_category="provider_429")
    refreshed = rm.get(run.run_id)
    assert refreshed.status == RunStatus.FAILED
    assert refreshed.error_category == "provider_429"


def test_request_cancel_sets_bit_without_changing_status(conn: psycopg.Connection):
    """request_cancel only flips the bit; CLI consumer transitions to CANCELLED."""
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k1")
    rm.update_stage(run.run_id, "research_wide")
    rm.request_cancel(run.run_id)
    refreshed = rm.get(run.run_id)
    assert refreshed.cancel_requested is True
    assert refreshed.status == RunStatus.RUNNING  # NOT auto-flipped


def test_get_unknown_run_raises_keyerror(conn: psycopg.Connection):
    rm = RunManager(conn)
    with pytest.raises(KeyError, match="run_xyz"):
        rm.get("run_xyz")


def test_update_stage_unknown_run_raises_keyerror(conn: psycopg.Connection):
    rm = RunManager(conn)
    with pytest.raises(KeyError):
        rm.update_stage("run_xyz", "planning")


def test_note_error_category_sets_column_without_changing_status(conn: psycopg.Connection):
    """non-terminal error_category setter — status stays at whatever it was."""
    rm = RunManager(conn)
    run = rm.create(topic="t", idempotency_key="k1")
    rm.update_stage(run.run_id, "research_wide")
    # status is now RUNNING after update_stage
    rm.note_error_category(run.run_id, "context_overflow")
    refreshed = rm.get(run.run_id)
    assert refreshed.error_category == "context_overflow"
    assert refreshed.status == RunStatus.RUNNING  # NOT flipped to FAILED


def test_note_error_category_unknown_run_raises_keyerror(conn: psycopg.Connection):
    rm = RunManager(conn)
    with pytest.raises(KeyError, match="run_xyz"):
        rm.note_error_category("run_xyz", "context_overflow")
