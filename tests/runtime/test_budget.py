"""BudgetManager + BUDGET_PER_ROLE tests per spec § 2.7.5."""
from __future__ import annotations

import pytest

from litweave.llm.roles import AgentRole
from litweave.runtime.budget import (
    BUDGET_PER_ROLE,
    BudgetExceeded,
    BudgetManager,
    OverflowFallback,
)


def test_budget_per_role_covers_all_ten_roles():
    assert set(BUDGET_PER_ROLE.keys()) == set(AgentRole)


def test_planner_budget_matches_spec():
    spec = BUDGET_PER_ROLE[AgentRole.PLANNER]
    assert spec.max_input_tokens == 8_000
    assert spec.reserved_output_tokens == 2_000
    assert spec.fallback == OverflowFallback.NONE


def test_researcher_wide_budget_matches_spec():
    spec = BUDGET_PER_ROLE[AgentRole.RESEARCHER_WIDE]
    assert spec.max_input_tokens == 24_000
    assert spec.fallback == OverflowFallback.SNIPPET_ONLY


def test_judge_default_uses_section_batched_fallback():
    """Largest budget + custom fallback per § 4.3.1."""
    spec = BUDGET_PER_ROLE[AgentRole.JUDGE_DEFAULT]
    assert spec.max_input_tokens == 220_000
    assert spec.fallback == OverflowFallback.SECTION_BATCHED_FALLBACK


def test_check_returns_spec_when_within_budget():
    bm = BudgetManager()
    spec = bm.check(AgentRole.PLANNER, projected_input_tokens=4_000)
    assert spec.max_input_tokens == 8_000


def test_check_raises_with_fallback_hint_when_over_budget():
    bm = BudgetManager()
    with pytest.raises(BudgetExceeded) as exc:
        bm.check(AgentRole.RESEARCHER_WIDE, projected_input_tokens=30_000)
    assert exc.value.role == AgentRole.RESEARCHER_WIDE
    assert exc.value.max_input_tokens == 24_000
    assert exc.value.projected_input_tokens == 30_000
    assert exc.value.fallback == OverflowFallback.SNIPPET_ONLY


def test_overflow_count_increments_per_role():
    bm = BudgetManager()
    with pytest.raises(BudgetExceeded):
        bm.check(AgentRole.PLANNER, 10_000)
    with pytest.raises(BudgetExceeded):
        bm.check(AgentRole.PLANNER, 9_000)
    assert bm.get_usage(AgentRole.PLANNER).overflow_count == 2
    # Other roles' counters untouched
    assert bm.get_usage(AgentRole.WRITER).overflow_count == 0


def test_record_usage_accumulates():
    bm = BudgetManager()
    bm.record_usage(AgentRole.RESEARCHER_DEEP, 5_000, 1_000)
    bm.record_usage(AgentRole.RESEARCHER_DEEP, 3_000, 500)
    u = bm.get_usage(AgentRole.RESEARCHER_DEEP)
    assert u.actual_input_tokens == 8_000
    assert u.actual_output_tokens == 1_500
