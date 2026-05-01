"""Tests for structured_call(config=...) plumbing (Plan #1 follow-up)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from surveyforge.llm.structured_output import structured_call

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
}


def test_structured_call_threads_config_to_invoke_when_fc_disabled() -> None:
    """When supports_fc=False, llm.invoke is called directly and must receive config."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content='{"x": 42}')
    captured_config: dict[str, Any] = {"metadata": {"run_id": "run_abc"}, "tags": ["planning"]}

    result = structured_call(
        llm,
        [HumanMessage(content="give me x=42")],
        schema=_SCHEMA,
        supports_fc=False,
        config=captured_config,  # type: ignore[arg-type]
    )

    assert result == {"x": 42}
    # llm.invoke was called with config=our_config kwarg
    llm.invoke.assert_called_once()
    _, kwargs = llm.invoke.call_args
    assert kwargs["config"] == captured_config


def test_structured_call_default_config_none_does_not_break_invoke() -> None:
    """When config not passed, llm.invoke gets config=None — must still work."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content='{"x": 7}')

    result = structured_call(
        llm,
        [HumanMessage(content="x")],
        schema=_SCHEMA,
        supports_fc=False,
    )

    assert result == {"x": 7}
    _, kwargs = llm.invoke.call_args
    assert kwargs["config"] is None


def test_structured_call_threads_config_to_invoke_when_fc_enabled() -> None:
    """FC path: `llm.bind_tools(...).invoke(messages, config=config)` must
    also receive the config kwarg, not just the JSON-only fallback path."""
    invoker = MagicMock()
    # FC path expects an AIMessage with .tool_calls (list of dicts with name/args/id)
    invoker.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "structured_output", "args": {"x": 99}, "id": "tc_1"}],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = invoker

    captured_config: dict[str, Any] = {
        "metadata": {"run_id": "run_xyz", "stage": "planning"},
        "tags": ["planning", "planner"],
    }

    result = structured_call(
        llm,
        [HumanMessage(content="give me x=99")],
        schema=_SCHEMA,
        supports_fc=True,
        config=captured_config,  # type: ignore[arg-type]
    )

    assert result == {"x": 99}
    # bind_tools was called once
    llm.bind_tools.assert_called_once()
    # the resulting invoker was called with config=our_config
    invoker.invoke.assert_called_once()
    _, kwargs = invoker.invoke.call_args
    assert kwargs["config"] == captured_config
