"""Planner output schema (PlannerOutput) per spec § 2.6.4."""
from __future__ import annotations

from pydantic import BaseModel


class PlannerSection(BaseModel):
    """One section of the survey outline."""

    section_id: str               # stable id of the form "S1", "S2", ...
    title: str                    # human-readable section name
    research_questions: list[str] # 2-4 questions Researcher must answer
    must_find_evidence: list[str] # 1-3 specific claims that MUST have supporting evidence


class PlannerOutput(BaseModel):
    """Complete Planner output: topic + sections + rationale."""

    topic: str
    sections: list[PlannerSection]
    rationale: str
