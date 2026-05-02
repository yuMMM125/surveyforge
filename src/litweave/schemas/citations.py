"""Citation + evidence-reference schemas shared across roles."""
from __future__ import annotations

from pydantic import BaseModel

from litweave.schemas.paper_id import PaperId


class Citation(BaseModel):
    """A reference to a paper (no evidence body — that's EvidenceRef / EvidenceCard)."""

    paper_id: PaperId          # prefixed: "arxiv:2401.12345" / "s2:abc123" / "web:<hash>"
    quote: str | None = None   # exact quote from source if available


class EvidenceRef(BaseModel):
    """A pointer to an EvidenceItem in EvidenceStore (full Item lives in runtime/evidence.py).

    Used in Writer / Critic / Judge to reference Researcher's evidence without
    re-embedding the full content.
    """

    evidence_id: str
    paper_id: PaperId
    section_id: str
