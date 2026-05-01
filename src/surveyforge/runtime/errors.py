"""ErrorCategory enum + classify_exception per spec § 2.7.6.

Late-imports `httpx` / `pydantic` / `jsonschema` so this module stays a
zero-dependency anchor — usable from any layer (graph, agents, tools)
without dragging in HTTP/schema libraries when they aren't needed.
"""
from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    PROVIDER_429 = "provider_429"
    PROVIDER_5XX = "provider_5xx"
    SCHEMA_INVALID = "schema_invalid"
    CONTEXT_OVERFLOW = "context_overflow"
    EMPTY_EVIDENCE = "empty_evidence"
    CITATION_PROBLEM = "citation_problem"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"


def classify_exception(exc: BaseException) -> ErrorCategory | None:
    """Map known exception types to spec § 2.7.6 categories.

    Returns None for unmapped exceptions so the caller can decide whether to
    treat as fatal, escalate to human review, or wrap with more context.
    `context_overflow`, `empty_evidence`, `citation_problem`, and
    `prompt_injection_detected` are application-level signals — the producing
    site sets them directly rather than going through this classifier.
    """
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore[assignment]
    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return ErrorCategory.PROVIDER_429
        if 500 <= status < 600:
            return ErrorCategory.PROVIDER_5XX
        return None

    try:
        from pydantic import ValidationError as _PydanticValidationError
    except ImportError:
        _PydanticValidationError = None  # type: ignore[assignment,misc]
    if _PydanticValidationError is not None and isinstance(exc, _PydanticValidationError):
        return ErrorCategory.SCHEMA_INVALID

    try:
        import jsonschema
    except ImportError:
        jsonschema = None  # type: ignore[assignment]
    if jsonschema is not None and isinstance(exc, jsonschema.ValidationError):
        return ErrorCategory.SCHEMA_INVALID

    return None
