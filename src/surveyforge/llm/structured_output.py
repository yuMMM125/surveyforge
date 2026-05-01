"""Robust structured-output wrapper.

Tries native function-calling first; on missing/invalid tool calls,
re-prompts with JSON schema instruction and parses the response.
**Validates output against the JSON Schema** — successful JSON parse is NOT
sufficient; required fields, types, and constraints must all match.
"""
from __future__ import annotations

import json
import re
from typing import Any

import jsonschema
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
BARE_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


class StructuredCallError(RuntimeError):
    pass


def _extract_json(text: str) -> Any | None:
    if not text:
        return None
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = BARE_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _validate(value: Any, schema: dict[str, Any]) -> str | None:
    """Return None if valid, else a short human-readable error message."""
    try:
        jsonschema.validate(value, schema)
        return None
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        return f"at {path}: {exc.message}"


def _fallback_system_prompt(schema: dict[str, Any]) -> SystemMessage:
    return SystemMessage(
        content=(
            "Reply with ONLY a single JSON object that conforms to this JSON Schema. "
            "Do not include prose, markdown, or code fences.\n\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )
    )


def structured_call(
    llm: BaseChatModel,
    messages: list[Any],
    schema: dict[str, Any],
    *,
    tool_name: str = "structured_output",
    max_retries: int = 2,
    supports_fc: bool = True,
) -> Any:
    """Call llm and return parsed JSON validated against schema.

    Strategy per attempt:
    1. If `supports_fc` is True, bind a single tool (`tool_name`, parameters=schema)
       and invoke. If the model returns a tool call, take its args.
       If `supports_fc` is False (e.g. Qwen via SJTU gateway returns HTTP 400 on
       bind_tools), skip this branch and use the JSON-only system prompt directly.
    2. If no tool call returned, extract JSON from message content.
    3. Validate the candidate against `schema` via jsonschema.
    4. If valid, return it.
    5. If invalid (parse error or schema violation), construct a corrective
       message and retry up to `max_retries`.

    Raises StructuredCallError if no attempt yields schema-valid output.
    """
    if supports_fc:
        tools_payload = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Return structured output.",
                    "parameters": schema,
                },
            }
        ]
        invoker = llm.bind_tools(tools_payload)
        attempt_messages = list(messages)
    else:
        invoker = llm
        attempt_messages = [_fallback_system_prompt(schema), *messages]

    last_error: str = "no candidate JSON found"

    for attempt in range(max_retries + 1):
        resp: AIMessage = invoker.invoke(attempt_messages)

        candidate: Any | None = None
        if supports_fc:
            for tc in resp.tool_calls or []:
                if tc.get("name") == tool_name:
                    candidate = tc["args"]
                    break
        if candidate is None:
            candidate = _extract_json(
                resp.content if isinstance(resp.content, str) else ""
            )

        if candidate is not None:
            err = _validate(candidate, schema)
            if err is None:
                return candidate
            last_error = f"schema invalid: {err}"
        else:
            last_error = "no candidate JSON found"

        if attempt < max_retries:
            attempt_messages = [
                _fallback_system_prompt(schema),
                *messages,
                HumanMessage(
                    content=(
                        f"Your previous reply was rejected ({last_error}). "
                        "Reply with ONLY a JSON object that conforms to the schema."
                    )
                ),
            ]

    raise StructuredCallError(
        f"Could not extract schema-valid structured output after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
