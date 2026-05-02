"""Live API tests for structured_call against GLM (FC=True) and Qwen (FC=False).

Splits the spike's Next Action 4 tests out of test_glm_hello.py and test_qwen_hello.py
because those files exist before structured_output.py is created (Task 5/7 vs Task 9).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from litweave.llm.providers import ProviderName, build_chat_model
from litweave.llm.structured_output import structured_call

pytestmark = pytest.mark.integration

# `skip_if_no_key` fixture is provided by tests/llm/integration/conftest.py.


def test_glm_structured_output_path(skip_if_no_key):
    """Spike Next Action: GLM structured-output path via structured_call (FC enabled)."""
    llm = build_chat_model(ProviderName.GLM)
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["city", "country"],
    }
    result = structured_call(
        llm,
        messages=[HumanMessage(content="Where is the Eiffel Tower? Reply with city + country.")],
        schema=schema,
        tool_name="locate",
        max_retries=1,
        supports_fc=True,
    )
    assert "paris" in result["city"].lower()
    assert "france" in result["country"].lower()


def test_qwen_judge_json_path_no_tools(skip_if_no_key):
    """Spike Next Action: Qwen FC fails (HTTP 400) — judge path uses JSON-content
    via structured_call(supports_fc=False).
    """
    llm = build_chat_model(ProviderName.QWEN)
    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
    }
    result = structured_call(
        llm,
        messages=[HumanMessage(
            content=(
                "Rate this sentence on grammatical correctness from 1 to 5 with a brief reason: "
                "'Cats sleeps on mats.'"
            )
        )],
        schema=schema,
        tool_name="rate",
        max_retries=2,
        supports_fc=False,  # critical: FC is broken on this gateway for QWEN
    )
    assert isinstance(result["score"], int)
    assert 1 <= result["score"] <= 5
    assert isinstance(result["reason"], str) and len(result["reason"]) > 0
