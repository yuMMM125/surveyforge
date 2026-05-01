"""Run lifecycle (RunStatus + RunManager) per spec § 2.7.1."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import psycopg


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


@dataclass(frozen=True)
class Run:
    run_id: str
    idempotency_key: str
    topic: str
    status: RunStatus
    current_stage: str | None
    error_category: str | None
    cancel_requested: bool
    created_at: datetime
    updated_at: datetime


class IdempotencyConflict(Exception):
    """Raised when create() reuses an existing idempotency_key.

    Carries the existing run_id so the caller (CLI / API) can surface it
    instead of erroring opaquely.
    """

    def __init__(self, idempotency_key: str, existing_run_id: str) -> None:
        super().__init__(
            f"idempotency_key {idempotency_key!r} already used by run {existing_run_id!r}"
        )
        self.existing_run_id = existing_run_id
        self.idempotency_key = idempotency_key


_RUN_COLS = (
    "run_id, idempotency_key, topic, status, current_stage, "
    "error_category, cancel_requested, created_at, updated_at"
)


class RunManager:
    """CRUD over the `runs` table. Caller passes a live `psycopg.Connection`.

    Each call uses the connection's autocommit/transaction state — the manager
    does not open its own transactions, so callers can compose calls into a
    larger unit of work via `db.transaction()`.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def create(self, topic: str, idempotency_key: str) -> Run:
        # Use ON CONFLICT DO NOTHING so the duplicate-key path doesn't raise
        # UniqueViolation. Raising would force us to rollback the connection,
        # which would also wipe any unrelated work the caller did earlier in
        # the same transaction. RETURNING yields zero rows when the conflict
        # suppresses the insert; we then look up the existing run_id and raise
        # IdempotencyConflict without touching the caller's transaction state.
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        with self._conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO runs (run_id, idempotency_key, topic, status)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING {_RUN_COLS}""",
                (run_id, idempotency_key, topic, RunStatus.PENDING.value),
            )
            row = cur.fetchone()
        if row is not None:
            return self._row_to_run(row)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT run_id FROM runs WHERE idempotency_key = %s",
                (idempotency_key,),
            )
            existing = cur.fetchone()
        existing_id = existing[0] if existing else "<unknown>"
        raise IdempotencyConflict(idempotency_key, existing_id)

    def update_stage(self, run_id: str, stage: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """UPDATE runs SET current_stage = %s, status = %s, updated_at = now()
                   WHERE run_id = %s""",
                (stage, RunStatus.RUNNING.value, run_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"run {run_id!r} not found")

    def succeed(self, run_id: str) -> None:
        self._set_status(run_id, RunStatus.SUCCEEDED)

    def fail(self, run_id: str, error_category: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """UPDATE runs SET status = %s, error_category = %s, updated_at = now()
                   WHERE run_id = %s""",
                (RunStatus.FAILED.value, error_category, run_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"run {run_id!r} not found")

    def request_cancel(self, run_id: str) -> None:
        """Flip cancel_requested. Status is NOT auto-changed — the graph runner
        polls this flag and transitions to CANCELLED when it reaches a safe point.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET cancel_requested = TRUE, updated_at = now() WHERE run_id = %s",
                (run_id,),
            )
            if cur.rowcount == 0:
                raise KeyError(f"run {run_id!r} not found")

    def get(self, run_id: str) -> Run:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_RUN_COLS} FROM runs WHERE run_id = %s",
                (run_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"run {run_id!r} not found")
        return self._row_to_run(row)

    def _set_status(self, run_id: str, status: RunStatus) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET status = %s, updated_at = now() WHERE run_id = %s",
                (status.value, run_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"run {run_id!r} not found")

    @staticmethod
    def _row_to_run(row: tuple[Any, ...]) -> Run:
        return Run(
            run_id=row[0],
            idempotency_key=row[1],
            topic=row[2],
            status=RunStatus(row[3]),
            current_stage=row[4],
            error_category=row[5],
            cancel_requested=row[6],
            created_at=row[7],
            updated_at=row[8],
        )
