"""Live API hello-world for Qwen3.5-27B via SJTU gateway."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from surveyforge.llm.providers import ProviderName, build_chat_model

pytestmark = pytest.mark.integration


def test_qwen_simple_completion(skip_if_no_key):
    llm = build_chat_model(ProviderName.QWEN)  # default_model = qwen
    resp = llm.invoke([HumanMessage(content="Say only the word: pong")])
    assert "pong" in resp.content.lower()
