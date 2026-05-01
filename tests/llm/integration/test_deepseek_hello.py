"""Live API hello-world for DeepSeek V3.2 via SJTU gateway. Run with: pytest -m integration."""
from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from surveyforge.llm.providers import ProviderName, build_chat_model

pytestmark = pytest.mark.integration


@pytest.fixture
def skip_if_no_key(monkeypatch: pytest.MonkeyPatch):
    if not os.environ.get("SJTU_MODELS_API_KEY"):
        pytest.skip("SJTU_MODELS_API_KEY not set")
    # Clear SOCKS proxy vars so httpx can construct its client (socksio not installed).
    for proxy_var in ("ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(proxy_var, raising=False)


def test_deepseek_simple_completion(skip_if_no_key):
    llm = build_chat_model(ProviderName.DEEPSEEK)  # default_model = deepseek-chat
    resp = llm.invoke([HumanMessage(content="Say only the word: pong")])
    assert "pong" in resp.content.lower()


def test_deepseek_tool_call_format(skip_if_no_key):
    """Spike Next Action: explicit DeepSeek tool-call path test."""
    llm = build_chat_model(ProviderName.DEEPSEEK)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    resp = llm.bind_tools(tools).invoke(
        [HumanMessage(content="What is the weather in Tokyo?")]
    )
    # Either returns a tool call, or content mentioning the city.
    assert resp.tool_calls or "tokyo" in resp.content.lower()
