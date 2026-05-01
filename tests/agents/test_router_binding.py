"""Tests for LLMRouter.binding(role) accessor (Plan #1 follow-up)."""
from __future__ import annotations

import pytest

from surveyforge.llm.providers import ProviderName
from surveyforge.llm.rate_limit import RateLimitConfig, RateLimitedRouter
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import LLMRouter, RoleBinding


def _make_router() -> LLMRouter:
    return LLMRouter({
        AgentRole.PLANNER: RoleBinding(
            provider=ProviderName.GLM, model="glm-5.1", temperature=0.0,
        ),
    })


def test_binding_returns_role_binding_for_configured_role() -> None:
    router = _make_router()
    binding = router.binding(AgentRole.PLANNER)
    assert binding.provider == ProviderName.GLM
    assert binding.model == "glm-5.1"
    assert isinstance(binding, RoleBinding)


def test_binding_raises_keyerror_for_unconfigured_role() -> None:
    router = _make_router()
    with pytest.raises(KeyError, match="researcher_wide"):
        router.binding(AgentRole.RESEARCHER_WIDE)


def test_rate_limited_router_binding_mirrors_llm_router() -> None:
    """RateLimitedRouter.binding() must mirror LLMRouter.binding() so both
    classes satisfy RouterProtocol identically (production code uses the
    rate-limited variant; agent factories type-hint the protocol)."""
    router = RateLimitedRouter(
        bindings={
            AgentRole.PLANNER: RoleBinding(
                provider=ProviderName.GLM, model="glm-5.1", temperature=0.0,
            ),
        },
        config=RateLimitConfig(),
    )
    binding = router.binding(AgentRole.PLANNER)
    assert binding.provider == ProviderName.GLM
    assert binding.model == "glm-5.1"
    assert isinstance(binding, RoleBinding)
    # Same KeyError contract as LLMRouter
    with pytest.raises(KeyError, match="researcher_wide"):
        router.binding(AgentRole.RESEARCHER_WIDE)
