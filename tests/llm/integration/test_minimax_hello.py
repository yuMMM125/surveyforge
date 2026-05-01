"""Live API hello-world for MiniMax-M2.7 via SJTU gateway."""
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


def test_minimax_simple_completion(skip_if_no_key):
    llm = build_chat_model(ProviderName.MINIMAX)  # default_model = minimax (M2.7 alias)
    resp = llm.invoke([HumanMessage(content="Say only the word: pong")])
    assert "pong" in resp.content.lower()


def test_minimax_tool_call_path(skip_if_no_key):
    """Spike Next Action: MiniMax FC path is the canonical structured-output route
    (raw JSON-only prompts emit <think> blocks and are unreliable).
    """
    llm = build_chat_model(ProviderName.MINIMAX)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_capital",
                "description": "Return the capital city of a country",
                "parameters": {
                    "type": "object",
                    "properties": {"country": {"type": "string"}},
                    "required": ["country"],
                },
            },
        }
    ]
    resp = llm.bind_tools(tools).invoke(
        [HumanMessage(content="What is the capital of France?")]
    )
    # MiniMax should emit a tool call — that's the supported structured path here.
    assert resp.tool_calls or "paris" in resp.content.lower()
