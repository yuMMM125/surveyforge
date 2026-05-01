"""Provider registry: each domestic LLM exposes an OpenAI-compatible endpoint."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class ProviderName(StrEnum):
    DEEPSEEK = "deepseek"
    GLM = "glm"
    MINIMAX = "minimax"
    QWEN = "qwen"


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str
    context_window: int  # tokens (claimed value; stress-tested in W2)
    default_model: str
    supports_function_calling: bool = True  # set False for providers where bind_tools returns 400
    notes: str = ""


# Values verified via Phase 0 spike (2026-05-01) against SJTU model gateway.
# All 4 model families served via single OpenAI-compatible endpoint with one shared key.
# Reference: _private/spikes/provider-compatibility.md
SJTU_GATEWAY_URL = "https://models.sjtu.edu.cn/api/v1"

PROVIDERS: dict[ProviderName, ProviderConfig] = {
    ProviderName.DEEPSEEK: ProviderConfig(
        base_url=SJTU_GATEWAY_URL,
        api_key_env="SJTU_MODELS_API_KEY",
        context_window=32_000,           # claimed; W2 will stress-test
        default_model="deepseek-chat",   # spike: hello/JSON/FC all PASS
        supports_function_calling=True,
        notes="V3.2 regular-mode alias. Use `deepseek-reasoner` for higher-stakes reasoning.",
    ),
    ProviderName.GLM: ProviderConfig(
        base_url=SJTU_GATEWAY_URL,
        api_key_env="SJTU_MODELS_API_KEY",
        context_window=128_000,
        default_model="glm-5.1",
        supports_function_calling=True,
        notes="GLM-5.1; spec target; hello/JSON/FC PASS.",
    ),
    ProviderName.MINIMAX: ProviderConfig(
        base_url=SJTU_GATEWAY_URL,
        api_key_env="SJTU_MODELS_API_KEY",
        context_window=192_000,
        default_model="minimax",         # M2.7 alias; FC PASS; raw JSON-only NOT clean (uses tool-call path)
        supports_function_calling=True,
        notes=(
            "MiniMax-M2.7 alias. FC works; raw JSON-only prompts emit <think> "
            "blocks, so structured_call must prefer tool-calling path."
        ),
    ),
    ProviderName.QWEN: ProviderConfig(
        base_url=SJTU_GATEWAY_URL,
        api_key_env="SJTU_MODELS_API_KEY",
        context_window=256_000,
        default_model="qwen",            # qwen3.5-27b alias; JSON PASS; FC FAIL via this gateway
        supports_function_calling=False,  # HTTP 400 on bind_tools — structured_call must skip FC for QWEN
        notes=(
            "Qwen aliases (`qwen`, `qwen3.5-27b`, `qwen3vl`): tool-calling returns HTTP 400 "
            "on this gateway. Use JSON-content path. For FC-needed lite tasks, override binding "
            "to model='qwen3coder' and supports_fc=true (per-binding) — see llm_routing.yaml lite_worker."
        ),
    ),
}


def build_chat_model(
    provider: ProviderName,
    model: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Build a ChatOpenAI client for any of the four providers.

    Raises KeyError if the API key env var is unset.
    """
    cfg = PROVIDERS[provider]
    api_key = os.environ[cfg.api_key_env]  # raises KeyError if missing
    return ChatOpenAI(
        model=model or cfg.default_model,
        base_url=cfg.base_url,
        api_key=SecretStr(api_key),
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore[call-arg]
        **kwargs,
    )
