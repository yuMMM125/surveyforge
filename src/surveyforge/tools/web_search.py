"""web_search tool wrapper per spec § 3.

Direct wrapper over Serper API (google.serper.dev/search). Reads SERPER_API_KEY
from the environment; raises RuntimeError loudly if missing rather than silently
falling back. Results are `untrusted_content` — open-web text could carry
prompt-injection payloads; downstream prompts MUST wrap via `runtime.trust.wrap_untrusted`.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from surveyforge.llm.roles import AgentRole
from surveyforge.runtime.tool_gateway import ToolGateway, ToolPolicy
from surveyforge.schemas.paper_id import PaperId

SERPER_API_URL = "https://google.serper.dev/search"
SERPER_API_KEY_ENV = "SERPER_API_KEY"
TOOL_NAME = "web_search"
TOOL_VERSION = "0.1.0"


def _url_to_paper_id(url: str) -> str:
    """Stable `web:<hash>` paper_id derived from URL (16 hex = 64-bit dedup space).

    Contract: URL string equality IS the dedup contract — same URL string
    always produces the same paper_id. This function does NOT normalize URLs
    (no trailing-slash trimming, no UTM/tracking-param stripping). Downstream
    consumers that need cross-call dedup are responsible for canonicalizing
    URLs before passing them in. W3 may add normalization here when cross-call
    dedup matters; W2's primary use case (within-Serper-response dedup) is
    fine because Serper returns one canonical link per organic result.

    Empty strings are an invalid input — call sites must filter linkless
    results before invoking this helper. `search_web` does this at the
    result-comprehension level.

    Birthday-collision boundary at ~4 billion distinct URLs, far beyond
    W2's expected scale (10s-100s of papers per run).
    """
    return f"web:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}"


class WebSearchInput(BaseModel):
    query: str
    num_results: int = Field(default=5, ge=1, le=20)


class WebResult(BaseModel):
    """One web search result.

    `paper_id` is the canonical prefix-form id (`web:<sha256(url)[:16]>`) ready
    for direct use as `CandidatePaper.paper_id` — same rationale as ArxivPaper.
    Hash is stable across runs as long as the URL is, so dedup across calls works.
    """

    model_config = ConfigDict(frozen=True)

    paper_id: PaperId        # canonical "web:<hash>" — copy directly into CandidatePaper
    title: str
    url: str
    snippet: str
    position: int | None


class WebSearchOutput(BaseModel):
    results: list[WebResult]


def search_web(query: str, num_results: int = 5) -> dict[str, Any]:
    """Direct impl: POST Serper API, parse organic results."""
    api_key = os.environ.get(SERPER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{SERPER_API_KEY_ENV} not set. Get a key at https://serper.dev "
            f"and `export {SERPER_API_KEY_ENV}=<key>` for web_search to work."
        )
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            SERPER_API_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
        )
        response.raise_for_status()
    data = response.json()
    return {
        "results": [
            {
                "paper_id": _url_to_paper_id(r["link"]),  # safe after filter below
                "title": r.get("title", ""),
                "url": r["link"],
                "snippet": r.get("snippet", ""),
                "position": r.get("position"),
            }
            # Skip linkless results — without this, _url_to_paper_id("") would
            # produce a deterministic web:e3b0c442... collision id shared by every
            # linkless result across runs, polluting cache + dedup.
            for r in data.get("organic", [])
            if r.get("link")
        ]
    }


def make_policy() -> ToolPolicy:
    return ToolPolicy(
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        allowed_roles=(AgentRole.RESEARCHER_WIDE,),
        input_schema=WebSearchInput,
        output_schema=WebSearchOutput,
        result_trust="untrusted_content",  # open-web text — must wrap before LLM consumption
    )


def register(gateway: ToolGateway) -> None:
    gateway.register(make_policy(), search_web)
