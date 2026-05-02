"""Researcher (Wide/Deep) output schemas per spec § 2.6.4."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from litweave.schemas.paper_id import PaperId


class CandidatePaper(BaseModel):
    """A paper candidate found by Researcher-Wide (snippet-level only).

    The `paper_id` prefix MUST match the `source` value (e.g., `source="arxiv"`
    requires `paper_id` to start with `"arxiv:"`). Enforced at validation time
    so cross-source collisions and typos surface as ValidationError.
    """

    paper_id: PaperId
    title: str
    source: Literal["arxiv", "s2", "web"]
    why_relevant: str
    handoff_to_deep: bool

    @model_validator(mode="after")
    def _paper_id_prefix_matches_source(self) -> CandidatePaper:
        expected = f"{self.source}:"
        if not self.paper_id.startswith(expected):
            raise ValueError(
                f"paper_id={self.paper_id!r} prefix must match source={self.source!r} "
                f"(expected start: {expected!r})"
            )
        return self


class ResearcherWideOutput(BaseModel):
    """One section's Wide pass output (after the ReAct loop terminates).

    Wide is **triage-only**: it identifies candidate papers but does NOT extract
    evidence cards or write to EvidenceStore. EvidenceCard production is
    Researcher-Deep's job (see ResearcherDeepOutput below).
    """

    section_id: str
    query: str
    candidate_papers: list[CandidatePaper]
    notes: str


class EvidenceCard(BaseModel):
    """A single piece of evidence linking a paper to a claim.

    Producer: Researcher-Deep (sole producer; Wide does not emit EvidenceCard).
    Consumers: Writer (cites via EvidenceRef), Critic (verifies), Judge (rates).
    """

    evidence_id: str
    paper_id: PaperId
    section_id: str
    claim: str
    source_span: str | None
    confidence: float = Field(ge=0.0, le=1.0)


class ResearcherDeepOutput(BaseModel):
    """One section's Deep pass output (Deep is the only role that emits EvidenceCard)."""

    section_id: str
    paper_ids_processed: list[PaperId]
    evidence_cards: list[EvidenceCard]
    insufficient_evidence_paper_ids: list[PaperId]
