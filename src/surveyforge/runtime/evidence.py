"""EvidenceItem schema + EvidenceStore CRUD per spec § 2.5 (W2 minimal)."""
from __future__ import annotations

from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict, Field

from surveyforge.llm.roles import AgentRole
from surveyforge.schemas.paper_id import PaperId

_EVIDENCE_COLS = (
    "evidence_id, run_id, section_id, paper_id, claim, "
    "source_span, source_locator, confidence, created_by"
)


class EvidenceItem(BaseModel):
    """W2 minimal evidence record per spec § 2.5.

    `paper_id: PaperId` reuses the prefix-form validator from
    `schemas/paper_id.py` — required because `EvidenceStore` is the persistence
    boundary. Anything that writes (CLI, migrations, debug scripts, tests) goes
    through this model and gets prefix + non-empty-suffix checks for free; at
    `load()` time the validator fires too, so bad data injected via raw SQL
    surfaces loudly instead of silently corrupting downstream agents.

    `source_locator` is W2-minimal `str | None` — free-form hint such as
    `"page 3"` / `"§2.4 paragraph 2"`. W3 will upgrade this column to JSONB
    with structured `{type, url, page, span, chunk_id}` shape for the
    citation_verifier; until then treat it as opaque text.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str           # "E-{run_id}-{n}"
    run_id: str
    paper_id: PaperId          # arxiv:* / s2:* / web:* — validated here, not just upstream
    section_id: str            # from PlannerOutput
    claim: str
    source_span: str | None
    source_locator: str | None  # W2: opaque text; W3: JSONB structured locator
    confidence: float = Field(ge=0.0, le=1.0)
    created_by: AgentRole


class EvidenceStore:
    """Repository for EvidenceItem persistence to `evidence_items` table.

    Like `RunManager`, this does NOT open its own transactions — the caller
    composes via `db.transaction()`. The connection is owned externally; the
    store is a thin wrapper that knows the column layout.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def save(self, item: EvidenceItem) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO evidence_items ({_EVIDENCE_COLS})
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    item.evidence_id, item.run_id, item.section_id, item.paper_id,
                    item.claim, item.source_span, item.source_locator,
                    item.confidence, item.created_by.value,
                ),
            )

    def load(self, evidence_id: str) -> EvidenceItem:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_EVIDENCE_COLS} FROM evidence_items WHERE evidence_id = %s",
                (evidence_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"evidence {evidence_id!r} not found")
        return self._row_to_item(row)

    def list_by_section(self, run_id: str, section_id: str) -> list[EvidenceItem]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT {_EVIDENCE_COLS} FROM evidence_items
                    WHERE run_id = %s AND section_id = %s
                    ORDER BY created_at""",
                (run_id, section_id),
            )
            rows = cur.fetchall()
        return [self._row_to_item(row) for row in rows]

    @staticmethod
    def _row_to_item(row: tuple[Any, ...]) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=row[0], run_id=row[1], section_id=row[2],
            paper_id=row[3], claim=row[4], source_span=row[5],
            source_locator=row[6], confidence=row[7],
            created_by=AgentRole(row[8]),
        )
