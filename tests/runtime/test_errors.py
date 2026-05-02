"""ErrorCategory + classify_exception tests per spec § 2.7.6."""
from __future__ import annotations

import httpx
import jsonschema
import pytest
from pydantic import BaseModel, ValidationError

from litweave.runtime.errors import ErrorCategory, classify_exception


def test_error_category_has_seven_values():
    assert {c.value for c in ErrorCategory} == {
        "provider_429",
        "provider_5xx",
        "schema_invalid",
        "context_overflow",
        "empty_evidence",
        "citation_problem",
        "prompt_injection_detected",
    }


def _httpx_error(status_code: int) -> httpx.HTTPStatusError:
    """Construct a real HTTPStatusError without a network round-trip."""
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_classify_httpx_429_to_provider_429():
    assert classify_exception(_httpx_error(429)) == ErrorCategory.PROVIDER_429


@pytest.mark.parametrize("code", [500, 502, 503, 504, 599])
def test_classify_httpx_5xx_to_provider_5xx(code: int):
    assert classify_exception(_httpx_error(code)) == ErrorCategory.PROVIDER_5XX


def test_classify_httpx_4xx_other_returns_none():
    """4xx that isn't 429 (e.g., 400/404) is not currently mapped."""
    assert classify_exception(_httpx_error(404)) is None


def test_classify_pydantic_validation_error_to_schema_invalid():
    class _M(BaseModel):
        x: int
    try:
        _M.model_validate({"x": "not-an-int"})
    except ValidationError as exc:
        assert classify_exception(exc) == ErrorCategory.SCHEMA_INVALID
    else:
        pytest.fail("expected ValidationError")


def test_classify_jsonschema_validation_error_to_schema_invalid():
    schema = {"type": "object", "required": ["x"]}
    try:
        jsonschema.validate({}, schema)
    except jsonschema.ValidationError as exc:
        assert classify_exception(exc) == ErrorCategory.SCHEMA_INVALID
    else:
        pytest.fail("expected ValidationError")


def test_classify_unknown_exception_returns_none():
    assert classify_exception(ValueError("not classified")) is None
    assert classify_exception(KeyError("not classified")) is None
    assert classify_exception(RuntimeError("not classified")) is None
