"""Per-role token budget enforcement (per spec § 2.7.5).

In-memory only for W2 (Architecture Decision #3 — single-process simplification).
W3+ should swap to Redis or a Postgres-backed limiter for multi-process
correctness; the in-memory state in BudgetManager is per-process.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from surveyforge.llm.roles import AgentRole


class OverflowFallback(StrEnum):
    NONE = "none"
    SNIPPET_ONLY = "snippet_only"
    SECTION_BATCHING = "section_batching"
    TOP_K_RERANK = "top_k_rerank"
    CLAIM_BY_CLAIM = "claim_by_claim"
    AGGREGATE_FINAL = "aggregate_final"
    SECTION_BATCHED_FALLBACK = "section_batched_fallback"


@dataclass(frozen=True)
class BudgetSpec:
    max_input_tokens: int
    reserved_output_tokens: int
    fallback: OverflowFallback


BUDGET_PER_ROLE: dict[AgentRole, BudgetSpec] = {
    AgentRole.PLANNER:          BudgetSpec(8_000,   2_000,  OverflowFallback.NONE),
    AgentRole.RESEARCHER_WIDE:  BudgetSpec(24_000,  4_000,  OverflowFallback.SNIPPET_ONLY),
    AgentRole.RESEARCHER_DEEP:  BudgetSpec(160_000, 16_000, OverflowFallback.SECTION_BATCHING),
    AgentRole.SYNTHESIZER:      BudgetSpec(96_000,  16_000, OverflowFallback.TOP_K_RERANK),
    AgentRole.WRITER:           BudgetSpec(160_000, 16_000, OverflowFallback.SECTION_BATCHING),
    AgentRole.CRITIC_SECTION:   BudgetSpec(96_000,  8_000,  OverflowFallback.CLAIM_BY_CLAIM),
    AgentRole.CRITIC_FINAL:     BudgetSpec(160_000, 16_000, OverflowFallback.SECTION_BATCHING),
    AgentRole.JUDGE_DEFAULT:    BudgetSpec(220_000, 16_000, OverflowFallback.SECTION_BATCHED_FALLBACK),
    AgentRole.JUDGE_FINAL:      BudgetSpec(96_000,  8_000,  OverflowFallback.AGGREGATE_FINAL),
    AgentRole.LITE_WORKER:      BudgetSpec(8_000,   2_000,  OverflowFallback.NONE),
}


class BudgetExceeded(Exception):
    """Projected input tokens exceed the role's budget; fallback strategy attached."""

    def __init__(
        self,
        role: AgentRole,
        projected_input_tokens: int,
        max_input_tokens: int,
        fallback: OverflowFallback,
    ) -> None:
        super().__init__(
            f"role {role.value} projected {projected_input_tokens} > "
            f"max {max_input_tokens} — fallback: {fallback.value}"
        )
        self.role = role
        self.projected_input_tokens = projected_input_tokens
        self.max_input_tokens = max_input_tokens
        self.fallback = fallback


@dataclass
class RoleUsage:
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    overflow_count: int = 0


class BudgetManager:
    """Per-role token budget tracking — in-memory only for W2.

    Spec § 2.7.5 requires recording `estimated_input_tokens` /
    `actual_usage_tokens` / `tokenizer_version` / `context_overflow_fallback_triggered`.
    The tokenizer version is captured per-call by the LLM router; this class
    tracks per-role rolling totals so we can flag estimator drift in W3.

    Instances are per-run scope (constructed at graph start, GC'd at end); the
    `_usage` dict accumulates without bound by design — that's fine because the
    object's lifetime is bounded by one run. For long-lived process reuse, swap
    to the Redis/Postgres limiter mentioned in the module docstring.
    """

    def __init__(self) -> None:
        self._usage: dict[AgentRole, RoleUsage] = defaultdict(RoleUsage)

    def check(self, role: AgentRole, projected_input_tokens: int) -> BudgetSpec:
        """Raise BudgetExceeded if projection > max; return BudgetSpec on success."""
        spec = BUDGET_PER_ROLE[role]
        if projected_input_tokens > spec.max_input_tokens:
            self._usage[role].overflow_count += 1
            raise BudgetExceeded(
                role=role,
                projected_input_tokens=projected_input_tokens,
                max_input_tokens=spec.max_input_tokens,
                fallback=spec.fallback,
            )
        return spec

    def record_usage(
        self,
        role: AgentRole,
        actual_input_tokens: int,
        actual_output_tokens: int,
    ) -> None:
        u = self._usage[role]
        u.actual_input_tokens += actual_input_tokens
        u.actual_output_tokens += actual_output_tokens

    def get_usage(self, role: AgentRole) -> RoleUsage:
        return self._usage[role]
