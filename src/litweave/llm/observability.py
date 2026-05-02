"""Langfuse integration: optional tracing for all LLM calls."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
except ImportError:  # langfuse not installed — kept for optional-dep resilience
    CallbackHandler = None


@dataclass(frozen=True)
class LangfuseSettings:
    public_key: str
    secret_key: str
    host: str

    @classmethod
    def from_env(cls) -> LangfuseSettings:
        return cls(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )


def is_enabled() -> bool:
    s = LangfuseSettings.from_env()
    return bool(s.public_key and s.secret_key) and CallbackHandler is not None


def get_callback_handler() -> CallbackHandler | None:
    """Return a Langfuse CallbackHandler if enabled, else None.

    Pass the handler into LangChain invocations via:
        llm.invoke(..., config={"callbacks": [handler]})
    """
    if not is_enabled():
        return None
    s = LangfuseSettings.from_env()
    return CallbackHandler(
        public_key=s.public_key,
        secret_key=s.secret_key,
        host=s.host,
    )
