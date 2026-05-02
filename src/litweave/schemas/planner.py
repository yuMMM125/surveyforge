"""Planner output schema (PlannerOutput) per spec § 2.6.4.

Length constraints (`Field(min_length=..., max_length=...)`) mirror the prompt's
explicit ranges (3-7 sections, 2-4 research_questions, 1-3 must_find_evidence).
The schema enforces what the prompt promises, so out-of-range Planner output
fails validation early instead of silently breaking Researcher routing.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerSection(BaseModel):
    """One section of the survey outline."""

    section_id: str                                              # stable id "S1", "S2", ...
    title: str                                                   # human-readable section name
    research_questions: list[str] = Field(min_length=2, max_length=4)
    must_find_evidence: list[str] = Field(min_length=1, max_length=3)


class PlannerOutput(BaseModel):
    """Complete Planner output: topic + sections + rationale."""

    topic: str
    sections: list[PlannerSection] = Field(min_length=3, max_length=7)
    rationale: str
