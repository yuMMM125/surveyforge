"""Live API hello-world for Qwen3.5-27B via SJTU gateway."""
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


def test_qwen_simple_completion(skip_if_no_key):
    llm = build_chat_model(ProviderName.QWEN)  # default_model = qwen
    resp = llm.invoke([HumanMessage(content="Say only the word: pong")])
    assert "pong" in resp.content.lower()
