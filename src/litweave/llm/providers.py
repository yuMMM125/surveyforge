"""Provider registry for OpenAI-compatible model gateways."""
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
    context_window: int  # token window claimed by the configured gateway
    default_model: str
    supports_function_calling: bool = True  # set False for providers where bind_tools returns 400
    notes: str = ""
    api_key_env_aliases: tuple[str, ...] = ()


# Defaults target the currently tested OpenAI-compatible gateway. Public users
# can override MODELS_BASE_URL and MODELS_API_KEY to use another compatible
# provider without editing source code.
DEFAULT_MODEL_GATEWAY_URL = "https://models.sjtu.edu.cn/api/v1"
MODEL_BASE_URL_ENV = "MODELS_BASE_URL"
MODEL_API_KEY_ENV = "MODELS_API_KEY"
LEGACY_SJTU_MODEL_API_KEY_ENV = "SJTU_MODELS_API_KEY"
MODEL_API_KEY_ENV_ALIASES = (LEGACY_SJTU_MODEL_API_KEY_ENV,)

PROVIDERS: dict[ProviderName, ProviderConfig] = {
    ProviderName.DEEPSEEK: ProviderConfig(
        base_url=DEFAULT_MODEL_GATEWAY_URL,
        api_key_env=MODEL_API_KEY_ENV,
        api_key_env_aliases=MODEL_API_KEY_ENV_ALIASES,
        context_window=32_000,
        default_model="deepseek-chat",
        supports_function_calling=True,
        notes="V3.2 regular-mode alias. Use `deepseek-reasoner` for higher-stakes reasoning.",
    ),
    ProviderName.GLM: ProviderConfig(
        base_url=DEFAULT_MODEL_GATEWAY_URL,
        api_key_env=MODEL_API_KEY_ENV,
        api_key_env_aliases=MODEL_API_KEY_ENV_ALIASES,
        context_window=128_000,
        default_model="glm-5.1",
        supports_function_calling=True,
        notes="GLM-5.1; hello/JSON/function-calling path verified on the default gateway.",
    ),
    ProviderName.MINIMAX: ProviderConfig(
        base_url=DEFAULT_MODEL_GATEWAY_URL,
        api_key_env=MODEL_API_KEY_ENV,
        api_key_env_aliases=MODEL_API_KEY_ENV_ALIASES,
        context_window=192_000,
        default_model="minimax",
        supports_function_calling=True,
        notes=(
            "MiniMax-M2.7 alias. Function-calling works; raw JSON-only prompts may emit "
            "<think> blocks, so structured_call should prefer the tool-calling path."
        ),
    ),
    ProviderName.QWEN: ProviderConfig(
        base_url=DEFAULT_MODEL_GATEWAY_URL,
        api_key_env=MODEL_API_KEY_ENV,
        api_key_env_aliases=MODEL_API_KEY_ENV_ALIASES,
        context_window=256_000,
        default_model="qwen",
        supports_function_calling=False,
        notes=(
            "Qwen aliases (`qwen`, `qwen3.5-27b`, `qwen3vl`): tool-calling returns "
            "HTTP 400 on the default gateway. Use the JSON-content path. For "
            "function-calling utility tasks, override the binding to "
            "model='qwen3coder' and supports_fc=true."
        ),
    ),
}


def _read_api_key(cfg: ProviderConfig) -> str:
    """Read the provider API key from the public env var, with legacy fallback."""
    for env_name in (cfg.api_key_env, *cfg.api_key_env_aliases):
        value = os.environ.get(env_name)
        if value:
            return value
    names = " or ".join((cfg.api_key_env, *cfg.api_key_env_aliases))
    raise KeyError(names)


def _read_base_url(cfg: ProviderConfig) -> str:
    """Read the gateway base URL, allowing deployments to swap providers."""
    return os.environ.get(MODEL_BASE_URL_ENV) or cfg.base_url


def build_chat_model(
    provider: ProviderName,
    model: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Build a ChatOpenAI client for any configured provider.

    Raises KeyError if no configured API key env var is set.
    """
    cfg = PROVIDERS[provider]
    api_key = _read_api_key(cfg)
    return ChatOpenAI(
        model=model or cfg.default_model,
        base_url=_read_base_url(cfg),
        api_key=SecretStr(api_key),
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore[call-arg]
        **kwargs,
    )
