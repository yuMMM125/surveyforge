from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from litweave.llm.structured_output import (
    StructuredCallError,
    structured_call,
)


def _ai(content: str = "", tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


def test_structured_call_uses_tool_call_when_provided():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = _ai(
        tool_calls=[{"name": "extract", "args": {"city": "Tokyo"}, "id": "x"}]
    )
    result = structured_call(
        llm,
        messages=[HumanMessage(content="weather in Tokyo")],
        schema={"type": "object", "properties": {"city": {"type": "string"}}},
        tool_name="extract",
    )
    assert result == {"city": "Tokyo"}


def test_structured_call_falls_back_to_json_in_content():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = _ai(content='{"city": "Paris"}')
    result = structured_call(
        llm,
        messages=[HumanMessage(content="weather in Paris")],
        schema={"type": "object", "properties": {"city": {"type": "string"}}},
        tool_name="extract",
    )
    assert result == {"city": "Paris"}


def test_structured_call_extracts_json_from_code_fence():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = _ai(
        content='Sure thing.\n```json\n{"city": "Berlin"}\n```\nDone.'
    )
    result = structured_call(
        llm,
        messages=[HumanMessage(content="weather")],
        schema={"type": "object", "properties": {"city": {"type": "string"}}},
        tool_name="extract",
    )
    assert result == {"city": "Berlin"}


def test_structured_call_retries_on_invalid_then_succeeds():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = [
        _ai(content="I'm not sure, here's some prose."),
        _ai(content='{"city": "Madrid"}'),
    ]
    result = structured_call(
        llm,
        messages=[HumanMessage(content="weather")],
        schema={"type": "object", "properties": {"city": {"type": "string"}}},
        tool_name="extract",
        max_retries=1,
    )
    assert result == {"city": "Madrid"}
    assert llm.invoke.call_count == 2


def test_structured_call_raises_after_max_retries():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = _ai(content="no JSON anywhere")
    with pytest.raises(StructuredCallError):
        structured_call(
            llm,
            messages=[HumanMessage(content="weather")],
            schema={"type": "object", "properties": {"city": {"type": "string"}}},
            tool_name="extract",
            max_retries=1,
        )


def test_structured_call_rejects_schema_invalid_then_retries():
    """Schema-invalid output (parses as JSON but missing required field) must trigger retry."""
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "year": {"type": "integer"}},
        "required": ["city", "year"],
    }
    llm.invoke.side_effect = [
        _ai(content='{"city": "Madrid"}'),  # missing required field "year"
        _ai(content='{"city": "Madrid", "year": 2024}'),
    ]
    result = structured_call(
        llm,
        messages=[HumanMessage(content="info")],
        schema=schema,
        tool_name="extract",
        max_retries=1,
    )
    assert result == {"city": "Madrid", "year": 2024}
    assert llm.invoke.call_count == 2


def test_structured_call_rejects_wrong_type():
    """Wrong type (year as string instead of int) must be rejected as schema-invalid."""
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    schema = {
        "type": "object",
        "properties": {"year": {"type": "integer"}},
        "required": ["year"],
    }
    llm.invoke.return_value = _ai(content='{"year": "2024"}')
    with pytest.raises(StructuredCallError):
        structured_call(
            llm,
            messages=[HumanMessage(content="info")],
            schema=schema,
            tool_name="extract",
            max_retries=0,
        )


def test_structured_call_skips_bind_tools_when_supports_fc_false():
    """When supports_fc=False (e.g. Qwen via gateway), skip bind_tools entirely.

    Spike (2026-05-01): qwen / qwen3.5-27b / qwen3vl return HTTP 400 on bind_tools.
    structured_call must use JSON-content path directly without binding tools,
    or it wastes an API round trip on a guaranteed-failure call.
    """
    llm = MagicMock()
    llm.bind_tools.side_effect = AssertionError("bind_tools must NOT be called when supports_fc=False")
    llm.invoke.return_value = _ai(content='{"score": 4}')
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }
    result = structured_call(
        llm,
        messages=[HumanMessage(content="rate this")],
        schema=schema,
        tool_name="rate",
        max_retries=0,
        supports_fc=False,
    )
    assert result == {"score": 4}
    llm.bind_tools.assert_not_called()


def test_structured_call_no_fc_still_validates_schema():
    """Even on the FC-skipped path, schema validation must run."""
    llm = MagicMock()
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }
    llm.invoke.side_effect = [
        _ai(content='{"score": "high"}'),  # wrong type — must trigger retry
        _ai(content='{"score": 5}'),
    ]
    result = structured_call(
        llm,
        messages=[HumanMessage(content="rate")],
        schema=schema,
        tool_name="rate",
        max_retries=1,
        supports_fc=False,
    )
    assert result == {"score": 5}
    assert llm.invoke.call_count == 2
