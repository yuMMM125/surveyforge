"""EvidenceItem + EvidenceStore round-trip tests per spec § 2.5."""
from __future__ import annotations

import time

import psycopg
import pytest
from pydantic import ValidationError

from litweave.llm.roles import AgentRole
from litweave.runtime.evidence import EvidenceItem, EvidenceStore
from litweave.runtime.runs import RunManager


def _make_run(conn: psycopg.Connection) -> str:
    rm = RunManager(conn)
    return rm.create(topic="test", idempotency_key=f"key-{time.perf_counter_ns()}").run_id


def _make_item(run_id: str, n: int = 1, **overrides) -> EvidenceItem:
    base = dict(
        evidence_id=f"E-{run_id}-{n}",
        run_id=run_id,
        paper_id=f"arxiv:240{n}.12345",
        section_id="S1",
        claim=f"Claim {n}",
        source_span=f"Quote {n}",
        source_locator=f"page {n}",
        confidence=0.9,
        created_by=AgentRole.RESEARCHER_DEEP,
    )
    base.update(overrides)
    return EvidenceItem(**base)


# ---- EvidenceItem ----

def test_evidence_item_validates_confidence_range():
    with pytest.raises(ValidationError):
        _make_item("r1", confidence=1.5)
    with pytest.raises(ValidationError):
        _make_item("r1", confidence=-0.1)


def test_evidence_item_allows_null_source_span_and_locator():
    item = _make_item("r1", source_span=None, source_locator=None)
    assert item.source_span is None
    assert item.source_locator is None


def test_evidence_item_requires_created_by_to_be_agent_role():
    with pytest.raises(ValidationError):
        _make_item("r1", created_by="not_a_role")


def test_evidence_item_rejects_paper_id_without_prefix():
    """EvidenceStore is the persistence boundary — bad paper_ids must be rejected
    here, not just upstream in CandidatePaper / EvidenceCard."""
    with pytest.raises(ValidationError, match="must start with"):
        _make_item("r1", paper_id="2401.12345")  # no arxiv: prefix


def test_evidence_item_rejects_paper_id_with_empty_suffix():
    """`arxiv:` alone (empty suffix after prefix) must fail — protects against
    the W2 P2 polish issue surfacing inside the evidence layer."""
    with pytest.raises(ValidationError, match="empty or whitespace-only suffix"):
        _make_item("r1", paper_id="arxiv:")


def test_load_raises_validation_error_on_corrupted_paper_id(conn: psycopg.Connection):
    """If a bad paper_id is written via raw SQL (bypassing the model), load()
    surfaces it as a ValidationError — this is the second leg of the
    persistence-boundary safety contract."""
    run_id = _make_run(conn)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO evidence_items
               (evidence_id, run_id, section_id, paper_id, claim,
                source_span, source_locator, confidence, created_by)
               VALUES ('E-bad-1', %s, 'S1', 'no-prefix-here', 'claim',
                       NULL, NULL, 0.5, 'researcher_deep')""",
            (run_id,),
        )
    store = EvidenceStore(conn)
    with pytest.raises(ValidationError, match="must start with"):
        store.load("E-bad-1")


# ---- EvidenceStore round-trip ----

def test_save_and_load_round_trip(conn: psycopg.Connection):
    run_id = _make_run(conn)
    store = EvidenceStore(conn)
    item = _make_item(run_id, n=1)
    store.save(item)
    loaded = store.load(item.evidence_id)
    assert loaded == item


def test_load_unknown_raises_keyerror(conn: psycopg.Connection):
    store = EvidenceStore(conn)
    with pytest.raises(KeyError, match="E-missing"):
        store.load("E-missing")


def test_list_by_section_filters_by_run_and_section(conn: psycopg.Connection):
    run_id = _make_run(conn)
    other_run = _make_run(conn)
    store = EvidenceStore(conn)
    store.save(_make_item(run_id, n=1, section_id="S1"))
    store.save(_make_item(run_id, n=2, section_id="S1"))
    store.save(_make_item(run_id, n=3, section_id="S2"))
    store.save(_make_item(other_run, n=1, section_id="S1"))
    s1 = store.list_by_section(run_id, "S1")
    assert {e.evidence_id for e in s1} == {f"E-{run_id}-1", f"E-{run_id}-2"}
    s2 = store.list_by_section(run_id, "S2")
    assert [e.evidence_id for e in s2] == [f"E-{run_id}-3"]


def test_list_by_section_empty_when_no_match(conn: psycopg.Connection):
    run_id = _make_run(conn)
    store = EvidenceStore(conn)
    assert store.list_by_section(run_id, "S99") == []


def test_save_duplicate_evidence_id_raises(conn: psycopg.Connection):
    run_id = _make_run(conn)
    store = EvidenceStore(conn)
    item = _make_item(run_id, n=1)
    store.save(item)
    with pytest.raises(psycopg.errors.UniqueViolation):
        store.save(item)
