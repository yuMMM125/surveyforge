"""Schema-level tests per spec § 2.6.4.

Verifies Pydantic round-trip + required fields + paper_id prefix validation
for each W2 role's output schema, plus the 3 canned fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from surveyforge.schemas.citations import Citation, EvidenceRef
from surveyforge.schemas.planner import PlannerOutput, PlannerSection
from surveyforge.schemas.research import (
    CandidatePaper,
    EvidenceCard,
    ResearcherDeepOutput,
    ResearcherWideOutput,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---- citations.py ----

def test_citation_minimal():
    c = Citation(paper_id="arxiv:2401.12345")
    assert c.paper_id == "arxiv:2401.12345"
    assert c.quote is None


def test_citation_with_quote():
    c = Citation(paper_id="arxiv:2401.12345", quote="A direct quote.")
    assert c.quote == "A direct quote."


def test_evidence_ref_requires_all_fields():
    e = EvidenceRef(evidence_id="E-r1-1", paper_id="arxiv:2401.12345", section_id="S1")
    assert e.section_id == "S1"
    with pytest.raises(ValidationError):
        EvidenceRef(evidence_id="E-1", paper_id="arxiv:1")  # missing section_id


# ---- planner.py ----

def test_planner_section_minimal():
    s = PlannerSection(
        section_id="S1",
        title="Background",
        research_questions=["What is the topic?"],
        must_find_evidence=["Key result X"],
    )
    assert s.section_id == "S1"


def test_planner_section_requires_section_id():
    with pytest.raises(ValidationError):
        PlannerSection(
            title="x",
            research_questions=[],
            must_find_evidence=[],
        )


def test_planner_output_full():
    out = PlannerOutput(
        topic="RLHF survey",
        sections=[
            PlannerSection(
                section_id="S1", title="Background",
                research_questions=["q1"], must_find_evidence=["c1"],
            ),
        ],
        rationale="Single section for minimal test.",
    )
    assert len(out.sections) == 1


# ---- research.py: CandidatePaper basics ----

def test_candidate_paper_arxiv_prefix():
    p = CandidatePaper(
        paper_id="arxiv:2401.12345",
        title="A paper",
        source="arxiv",
        why_relevant="Directly answers RQ1.",
        handoff_to_deep=True,
    )
    assert p.paper_id.startswith("arxiv:")


def test_candidate_paper_invalid_source_rejected():
    with pytest.raises(ValidationError):
        CandidatePaper(
            paper_id="arxiv:1",
            title="x",
            source="patent",  # not in Literal["arxiv", "s2", "web"]
            why_relevant="x",
            handoff_to_deep=False,
        )


# ---- research.py: paper_id prefix validators (P2#3 fix) ----

def test_candidate_paper_rejects_paper_id_without_prefix():
    """Decision 0.3: paper_id must use prefix form."""
    with pytest.raises(ValidationError, match="must start with"):
        CandidatePaper(
            paper_id="2401.12345",  # missing arxiv: prefix
            title="x",
            source="arxiv",
            why_relevant="x",
            handoff_to_deep=False,
        )


def test_candidate_paper_rejects_prefix_source_mismatch():
    """source=arxiv but paper_id=s2:... — must reject."""
    with pytest.raises(ValidationError, match="prefix must match source"):
        CandidatePaper(
            paper_id="s2:abc",   # prefix s2 doesn't match source arxiv
            title="x",
            source="arxiv",
            why_relevant="x",
            handoff_to_deep=False,
        )


def test_candidate_paper_web_prefix_with_web_source():
    """Sanity: web:<hash> with source=web is valid."""
    p = CandidatePaper(
        paper_id="web:deadbeef1234",
        title="A blog post",
        source="web",
        why_relevant="Reference site.",
        handoff_to_deep=False,
    )
    assert p.paper_id.startswith("web:")


# ---- research.py: EvidenceCard ----

def test_evidence_card_confidence_in_range():
    e = EvidenceCard(
        evidence_id="E-1", paper_id="arxiv:1", section_id="S1",
        claim="X causes Y.", source_span="...quote...", confidence=0.8,
    )
    assert 0.0 <= e.confidence <= 1.0


def test_evidence_card_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        EvidenceCard(
            evidence_id="E-1", paper_id="arxiv:1", section_id="S1",
            claim="x", source_span=None, confidence=1.5,
        )


def test_evidence_card_rejects_paper_id_without_prefix():
    with pytest.raises(ValidationError, match="must start with"):
        EvidenceCard(
            evidence_id="E-1", paper_id="2401.12345", section_id="S1",
            claim="x", source_span=None, confidence=0.5,
        )


def test_citation_rejects_paper_id_without_prefix():
    with pytest.raises(ValidationError, match="must start with"):
        Citation(paper_id="bare-id-no-prefix")


def test_evidence_ref_rejects_paper_id_without_prefix():
    with pytest.raises(ValidationError, match="must start with"):
        EvidenceRef(evidence_id="E-1", paper_id="bare-id", section_id="S1")


# ---- research.py: outputs ----

def test_researcher_wide_output_minimal():
    out = ResearcherWideOutput(
        section_id="S1",
        query="LLM agents survey",
        candidate_papers=[],
        notes="No results.",
    )
    assert out.candidate_papers == []


def test_researcher_deep_output_separates_supported_and_insufficient():
    out = ResearcherDeepOutput(
        section_id="S1",
        paper_ids_processed=["arxiv:1", "arxiv:2"],
        evidence_cards=[
            EvidenceCard(
                evidence_id="E-1", paper_id="arxiv:1", section_id="S1",
                claim="Supported", source_span="quote", confidence=0.9,
            ),
        ],
        insufficient_evidence_paper_ids=["arxiv:2"],
    )
    assert "arxiv:2" in out.insufficient_evidence_paper_ids


# ---- fixture round-trips ----

def test_planner_happy_fixture_validates():
    fixture = json.loads((FIXTURES / "planner_happy.json").read_text(encoding="utf-8"))
    out = PlannerOutput.model_validate(fixture)
    assert out.topic == "Survey of RLHF progress"
    assert len(out.sections) == 2


def test_researcher_wide_schema_violation_fixture_rejected():
    fixture = json.loads(
        (FIXTURES / "researcher_wide_schema_violation.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValidationError):
        ResearcherWideOutput.model_validate(fixture)


def test_researcher_deep_edge_empty_pdf_fixture():
    fixture = json.loads(
        (FIXTURES / "researcher_deep_edge_empty_pdf.json").read_text(encoding="utf-8")
    )
    out = ResearcherDeepOutput.model_validate(fixture)
    assert out.evidence_cards == []
    assert out.insufficient_evidence_paper_ids == ["arxiv:2401.99999"]
