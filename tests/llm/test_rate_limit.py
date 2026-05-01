import time
from unittest.mock import MagicMock

import pytest

from surveyforge.llm.providers import ProviderName
from surveyforge.llm.rate_limit import (
    ProviderRateLimiter,
    RateLimitCallback,
    RateLimitConfig,
    RateLimitedRouter,
)
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import RoleBinding


def test_token_bucket_allows_burst_within_capacity():
    rl = ProviderRateLimiter(rpm=60, burst=5)
    start = time.monotonic()
    for _ in range(5):
        rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05  # burst should be near-instant


def test_token_bucket_blocks_after_burst_exhausted(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "surveyforge.llm.rate_limit.time.sleep", lambda s: sleeps.append(s)
    )
    rl = ProviderRateLimiter(rpm=60, burst=2)
    rl.acquire()
    rl.acquire()
    rl.acquire()  # third should request a sleep
    assert sleeps and sleeps[0] > 0


def test_rate_limit_config_per_provider():
    cfg = RateLimitConfig(
        per_provider={
            ProviderName.DEEPSEEK: (60, 5),  # rpm, burst
            ProviderName.QWEN: (120, 10),
        }
    )
    assert cfg.get(ProviderName.DEEPSEEK) == (60, 5)
    assert cfg.get(ProviderName.MINIMAX) == cfg.default  # falls back


def test_callback_fires_acquire_per_invoke():
    """Critical: acquire() must fire on every chat-model start, not just once."""
    calls: list[str] = []
    limiter = MagicMock(spec=ProviderRateLimiter)
    limiter.acquire = lambda: calls.append("acquired")

    cb = RateLimitCallback(limiter)
    # Simulate 3 .invoke() calls — LangChain fires on_chat_model_start each time
    cb.on_chat_model_start(serialized={}, messages=[])
    cb.on_chat_model_start(serialized={}, messages=[])
    cb.on_chat_model_start(serialized={}, messages=[])

    assert len(calls) == 3, "acquire must fire ONCE PER invoke, not just at LLM construction"


def test_callback_also_fires_on_llm_start():
    """Non-chat code paths (legacy LLM interface) should also debit."""
    calls: list[str] = []
    limiter = MagicMock(spec=ProviderRateLimiter)
    limiter.acquire = lambda: calls.append("acquired")

    cb = RateLimitCallback(limiter)
    cb.on_llm_start(serialized={}, prompts=[])
    assert len(calls) == 1


def test_rate_limited_router_attaches_callback(fake_env):
    bindings = {
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    }
    cfg = RateLimitConfig(per_provider={ProviderName.GLM: (60, 5)})
    router = RateLimitedRouter(bindings, cfg)
    llm = router.get_llm(AgentRole.PLANNER)

    cbs = list(llm.callbacks or [])
    rl_cbs = [c for c in cbs if isinstance(c, RateLimitCallback)]
    assert len(rl_cbs) == 1, "every constructed LLM must carry exactly one RateLimitCallback"


def test_rate_limited_router_caches_per_role(fake_env):
    bindings = {
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
    }
    cfg = RateLimitConfig(per_provider={ProviderName.GLM: (60, 5)})
    router = RateLimitedRouter(bindings, cfg)
    a = router.get_llm(AgentRole.PLANNER)
    b = router.get_llm(AgentRole.PLANNER)
    assert a is b


def test_rate_limited_router_unknown_role_raises(fake_env):
    cfg = RateLimitConfig()
    router = RateLimitedRouter({}, cfg)
    with pytest.raises(KeyError, match="planner"):
        router.get_llm(AgentRole.PLANNER)
