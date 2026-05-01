"""Per-provider token-bucket rate limiting at INVOKE level via callbacks.

Critical: rate-limit tokens are debited inside on_chat_model_start (and
on_llm_start) callbacks, which LangChain fires on every .invoke / .ainvoke /
.batch / .stream entry point — not at router lookup. This means cached LLM
instances and tight inner loops both correctly throttle real provider requests.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from surveyforge.llm.providers import ProviderName, build_chat_model
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import RoleBinding, load_routing_yaml


class ProviderRateLimiter:
    """Token-bucket limiter for a single provider.

    rpm = requests per minute (steady-state refill rate).
    burst = max requests allowed back-to-back (bucket capacity).
    """

    def __init__(
        self, rpm: int, burst: int, *, provider: ProviderName | None = None
    ) -> None:
        if rpm <= 0 or burst <= 0:
            raise ValueError("rpm and burst must be positive")
        self.provider = provider
        self.rpm = rpm
        self.capacity = burst
        self._tokens = float(burst)
        self._refill_per_sec = rpm / 60.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then debit one. Thread-safe."""
        with self._lock:
            self._refill()
            while self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._refill_per_sec
                time.sleep(max(wait, 0.001))
                self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(
            self.capacity, self._tokens + elapsed * self._refill_per_sec
        )
        self._last = now


@dataclass
class RateLimitConfig:
    """Maps provider → (rpm, burst). `default` used for unset providers."""

    per_provider: dict[ProviderName, tuple[int, int]] = field(default_factory=dict)
    default: tuple[int, int] = (30, 3)

    def get(self, provider: ProviderName) -> tuple[int, int]:
        return self.per_provider.get(provider, self.default)


class RateLimitCallback(BaseCallbackHandler):
    """Acquires a rate-limit token before each LLM API call.

    LangChain invokes on_chat_model_start / on_llm_start synchronously at the
    start of every model call (invoke / ainvoke / batch / stream), BEFORE the
    actual HTTP request to the provider. Blocking inside acquire() therefore
    blocks the real request rate — correct semantics for limiting upstream RPM.
    """

    raise_error = True  # propagate exceptions from the limiter

    # NOTE: acquire() uses time.sleep() which blocks the calling thread/event loop.
    # For .ainvoke()/.astream() under high concurrency, this serializes async calls
    # on the loop. Acceptable for the single-agent pipeline; if W2+ fanout introduces
    # parallel async dispatch, replace with asyncio.sleep + an async-aware bucket.

    def __init__(self, limiter: ProviderRateLimiter) -> None:
        super().__init__()
        self._limiter = limiter

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self._limiter.acquire()

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._limiter.acquire()


class RateLimitedRouter:
    """LLM router that rate-limits at INVOKE level via callback binding.

    Each cached ChatOpenAI carries a RateLimitCallback bound at construction time.
    The callback fires at every actual API request (not just router lookups).
    """

    def __init__(
        self,
        bindings: dict[AgentRole, RoleBinding],
        config: RateLimitConfig,
    ) -> None:
        self._bindings = bindings
        self._config = config
        self._limiters: dict[ProviderName, ProviderRateLimiter] = {}
        self._cache: dict[AgentRole, ChatOpenAI] = {}
        # Guards _limiters / _cache against concurrent get_llm() calls (e.g. when a
        # future Researcher-Wide fanout uses ThreadPoolExecutor). ProviderRateLimiter
        # itself is thread-safe; this lock only protects the registry-construction race.
        self._dict_lock = threading.Lock()

    @classmethod
    def from_yaml(
        cls, path: str | Path, config: RateLimitConfig
    ) -> RateLimitedRouter:
        return cls(load_routing_yaml(path), config)

    def get_llm(self, role: AgentRole, **kwargs: Any) -> ChatOpenAI:
        if role not in self._bindings:
            raise KeyError(f"No binding configured for role: {role.value}")

        is_overridden = bool(kwargs)
        with self._dict_lock:
            if not is_overridden and role in self._cache:
                return self._cache[role]

        binding = self._bindings[role]
        limiter = self._limiter_for(binding.provider)

        existing = list(kwargs.pop("callbacks", []) or [])
        existing.append(RateLimitCallback(limiter))

        llm = build_chat_model(
            binding.provider,
            model=kwargs.pop("model_override", None) or binding.model,
            temperature=kwargs.pop("temperature", binding.temperature),
            max_tokens=kwargs.pop("max_tokens", binding.max_tokens),
            callbacks=existing,
            **kwargs,
        )

        if not is_overridden:
            with self._dict_lock:
                self._cache[role] = llm
        return llm

    def _limiter_for(self, provider: ProviderName) -> ProviderRateLimiter:
        with self._dict_lock:
            if provider not in self._limiters:
                rpm, burst = self._config.get(provider)
                self._limiters[provider] = ProviderRateLimiter(
                    rpm, burst, provider=provider
                )
            return self._limiters[provider]
