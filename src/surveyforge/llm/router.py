"""LLMRouter: AgentRole → ChatOpenAI factory with caching and overrides."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMRouter:
        return cls(bindings=load_routing_yaml(path))

    def get_llm(
        self,
        role: AgentRole,
        *,
        model_override: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """Resolve a role to a ChatOpenAI instance.

        Cache + override contract:
        - With NO overrides: read-through cache (build on miss, cached forever).
        - With ANY explicit non-None override (model / temperature / max_tokens / **kwargs):
          build a fresh instance, return it, and DO NOT touch the cache (no read,
          no write). The base cached instance is preserved across transient overrides.
        - Note: passing the same value as the binding's default (e.g. `temperature=0.0`
          when the binding's default is also 0.0) still counts as an override and
          triggers a fresh build. Pass no kwargs to get the cached instance.

        Raises KeyError if `role` has no binding configured.
        """
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


def load_routing_yaml(path: str | Path) -> dict[AgentRole, RoleBinding]:
    """Parse llm_routing.yaml into RoleBinding objects, validating completeness."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected mapping at top level")

    bindings: dict[AgentRole, RoleBinding] = {}
    valid_roles = {r.value for r in AgentRole}
    for key, entry in raw.items():
        if key not in valid_roles:
            raise ValueError(f"Unknown role in {path}: {key!r}")
        bindings[AgentRole(key)] = RoleBinding(
            provider=ProviderName(entry["provider"]),
            model=entry["model"],
            temperature=float(entry.get("temperature", 0.0)),
            max_tokens=entry.get("max_tokens"),
            supports_fc=entry.get("supports_fc"),  # None if not set → fall back to provider default
        )

    missing = set(AgentRole) - bindings.keys()
    if missing:
        raise ValueError(
            f"Missing role bindings in {path}: {sorted(r.value for r in missing)}"
        )
    return bindings
