"""Live API hello-world for GLM-5.1 via SJTU gateway."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from litweave.llm.providers import ProviderName, build_chat_model

pytestmark = pytest.mark.integration


def test_glm_simple_completion(skip_if_no_key):
    llm = build_chat_model(ProviderName.GLM)  # default_model = glm-5.1
    resp = llm.invoke([HumanMessage(content="Say only the word: pong")])
    assert "pong" in resp.content.lower()


def test_glm_long_context_smoke(skip_if_no_key):
    """Verify GLM accepts mid-large input prompt (~1k tokens) without erroring.

    GLM-5.1 is a reasoning model on this gateway: visible content is emitted
    only after internal reasoning tokens. We use max_tokens=1024 to leave
    room for both reasoning and a short visible reply ('ack'). The assertion
    targets the input-side capacity, not output verbosity.
    """
    llm = build_chat_model(ProviderName.GLM, max_tokens=1024)
    long_text = "filler. " * 500
    resp = llm.invoke(
        [HumanMessage(content=f"{long_text}\n\nReply with only: ack")]
    )
    assert "ack" in resp.content.lower()
