"""Agent roles served by the LLMRouter (10 roles per spec § 2.1)."""
from __future__ import annotations

from enum import StrEnum


class AgentRole(StrEnum):
    PLANNER = "planner"
    RESEARCHER_WIDE = "researcher_wide"
    RESEARCHER_DEEP = "researcher_deep"
    SYNTHESIZER = "synthesizer"
    WRITER = "writer"
    CRITIC_SECTION = "critic_section"
    CRITIC_FINAL = "critic_final"
    JUDGE_DEFAULT = "judge_default"
    JUDGE_FINAL = "judge_final"
    LITE_WORKER = "lite_worker"
