"""LLMRouter: AgentRole → ChatOpenAI factory with caching and overrides."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from surveyforge.llm.providers import PROVIDERS, ProviderName, build_chat_model
from surveyforge.llm.roles import AgentRole


@dataclass(frozen=True)
class RoleBinding:
    """Binding from AgentRole → provider + model + per-call defaults.

    `supports_fc` is None by default (falls back to provider-level config).
    Set explicitly per-binding when a non-default model on a given provider has
    different FC support (e.g. lite_worker uses qwen3coder which DOES support FC,
    while QWEN provider defaults to FC=False).
    """

    provider: ProviderName
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None
    supports_fc: bool | None = None  # None → fall back to PROVIDERS[provider].supports_function_calling

    def fc_enabled(self) -> bool:
        if self.supports_fc is not None:
            return self.supports_fc
        return PROVIDERS[self.provider].supports_function_calling


class LLMRouter:
    """Resolves an AgentRole to a configured ChatOpenAI instance.

    Instances are cached when no per-call overrides are supplied.
    """

    def __init__(self, bindings: dict[AgentRole, RoleBinding]) -> None:
        self._bindings = bindings
        self._cache: dict[AgentRole, ChatOpenAI] = {}

    def get_llm(
        self,
        role: AgentRole,
        *,
        model_override: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatOpenAI:
        if role not in self._bindings:
            raise KeyError(f"No binding configured for role: {role.value}")

        is_overridden = bool(
            model_override or temperature is not None or max_tokens is not None or kwargs
        )
        if not is_overridden and role in self._cache:
            return self._cache[role]

        binding = self._bindings[role]
        llm = build_chat_model(
            binding.provider,
            model=model_override or binding.model,
            temperature=temperature if temperature is not None else binding.temperature,
            max_tokens=max_tokens if max_tokens is not None else binding.max_tokens,
            **kwargs,
        )

        if not is_overridden:
            self._cache[role] = llm
        return llm
