"""Pydantic output schemas for prompt contracts (per spec § 2.6.4).

Each schema corresponds to one AgentRole's output. PromptRegistry resolves
front-matter `schema:` values to the classes here at load time.

Public API: import directly from `litweave.schemas`:

    from litweave.schemas import PlannerOutput, CandidatePaper, EvidenceCard
"""
from litweave.schemas.citations import Citation, EvidenceRef
from litweave.schemas.paper_id import (
    VALID_PAPER_ID_PREFIXES,
    PaperId,
    validate_paper_id_prefix,
)
from litweave.schemas.planner import PlannerOutput, PlannerSection
from litweave.schemas.research import (
    CandidatePaper,
    EvidenceCard,
    ResearcherDeepOutput,
    ResearcherWideOutput,
)

__all__ = (
    "VALID_PAPER_ID_PREFIXES",
    "CandidatePaper",
    "Citation",
    "EvidenceCard",
    "EvidenceRef",
    "PaperId",
    "PlannerOutput",
    "PlannerSection",
    "ResearcherDeepOutput",
    "ResearcherWideOutput",
    "validate_paper_id_prefix",
)
