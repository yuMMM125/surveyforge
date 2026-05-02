"""s2_lookup tool wrapper per spec § 3.

Direct wrapper over Semantic Scholar Graph API. Looks up paper metadata by
prefixed paper_id (`arxiv:` or `s2:` only — `doi:` excluded to keep paper_id
namespace consistent with global PaperId contract); 404 returns `paper=None`
rather than raising, so callers can distinguish missing-paper from network-error.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from surveyforge.llm.roles import AgentRole
from surveyforge.runtime.tool_gateway import ToolGateway, ToolPolicy
from surveyforge.schemas.paper_id import PaperId

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
TOOL_NAME = "s2_lookup"
TOOL_VERSION = "0.1.0"

_S2_FIELDS = "paperId,externalIds,title,abstract,authors,year,venue,citationCount"

SEMANTIC_SCHOLAR_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"
MAX_RETRY_ATTEMPTS = 3                      # retry up to 3 times on 429
DEFAULT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)   # used when Retry-After header absent


class S2LookupInput(BaseModel):
    """Restrict paper_id to `arxiv:` / `s2:` only — `web:` is excluded
    because S2 doesn't index web pages, and `doi:` is excluded because the
    global PaperId contract (`schemas/paper_id.py`) doesn't support DOI as a
    paper-id prefix. If W3+ needs DOI lookup, add a separate `doi: str | None`
    field rather than overloading `paper_id`.

    Layered validation: the `PaperId` Annotated validator (from
    `surveyforge.schemas.paper_id`) runs first — it enforces prefix-form +
    rejects empty/whitespace-only suffix. Then the `field_validator` below
    further restricts the supported prefixes to arxiv:/s2: (rejecting web:).
    """

    paper_id: PaperId

    @field_validator("paper_id")
    @classmethod
    def _restrict_to_s2_supported_prefixes(cls, v: str) -> str:
        # PaperId already validated prefix-and-non-empty-suffix; this layer
        # narrows further to arxiv:/s2: (S2 API doesn't accept web: lookups).
        if not (v.startswith("arxiv:") or v.startswith("s2:")):
            raise ValueError(
                f"s2_lookup paper_id must start with arxiv: or s2:, got {v!r}"
            )
        return v


class S2Paper(BaseModel):
    """One Semantic Scholar paper record.

    `paper_id` is the canonical prefix-form id (`s2:<paperId>`) ready for
    direct use as `CandidatePaper.paper_id` — same rationale as ArxivPaper.
    """

    model_config = ConfigDict(frozen=True)

    paper_id: PaperId        # canonical "s2:<paperId>" — copy directly into CandidatePaper
    s2_paper_id: str         # raw S2 id without prefix
    arxiv_id: str | None
    doi: str | None          # raw DOI value, NOT a paper_id (PaperId contract excludes doi: prefix)
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int


class S2LookupOutput(BaseModel):
    paper: S2Paper | None


def _compute_retry_delay(response: httpx.Response, attempt_index: int) -> float:
    """Pick retry delay for a 429 response.

    Prefer the server-provided `Retry-After` header (RFC 7231: integer seconds
    OR HTTP-date — we only handle integer-seconds, the common case). Fall back
    to exponential backoff `DEFAULT_BACKOFF_SECONDS[attempt_index]`. Negative
    or non-numeric Retry-After values are ignored — fall back too.
    """
    retry_after_raw = response.headers.get("Retry-After")
    if retry_after_raw is not None:
        try:
            ra = float(retry_after_raw)
            if ra >= 0:
                return ra
        except ValueError:
            pass  # non-numeric / HTTP-date → fall through
    return DEFAULT_BACKOFF_SECONDS[attempt_index]


def _build_request_headers() -> dict[str, str]:
    """Inject `x-api-key` if `SEMANTIC_SCHOLAR_API_KEY` env var is set.

    SS Graph API v1 takes the key via `x-api-key` header. If unset, the
    request goes anonymous (1 RPS shared rate limit per IP — the failure
    mode that motivated this retry logic).
    """
    headers: dict[str, str] = {}
    api_key = os.environ.get(SEMANTIC_SCHOLAR_API_KEY_ENV)
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _build_s2_url(paper_id: str) -> str:
    """Translate prefixed paper_id to S2 API path component.

    Only `arxiv:` and `s2:` are supported (input validator enforces this);
    `doi:` was deliberately excluded to keep paper_id namespace consistent
    with the global PaperId contract. If DOI lookup becomes needed, add a
    separate `lookup_paper_by_doi` function — don't overload paper_id.
    """
    prefix, _, suffix = paper_id.partition(":")
    if prefix == "arxiv":
        return f"{S2_API_BASE}/paper/arXiv:{suffix}"
    if prefix == "s2":
        return f"{S2_API_BASE}/paper/{suffix}"
    raise ValueError(f"unsupported paper_id prefix: {prefix!r}")


def lookup_paper(paper_id: str) -> dict[str, Any]:
    """Direct impl: GET S2 paper metadata; return dict matching S2LookupOutput.

    Retry policy (per Task 7 polish 3, 2026-05-02):
      - HTTP 429 (rate limit): retry up to MAX_RETRY_ATTEMPTS times. Delay
        comes from `Retry-After` header if numeric, else exp backoff
        (1s, 2s, 4s). After all retries exhausted, raise the last response's
        HTTPStatusError so ToolGateway tags it provider_429.
      - HTTP 404 (paper not found): return `{"paper": None}` — caller can
        distinguish missing-paper from network-error. NEVER retry 404.
      - Other 4xx / 5xx / network errors: raise immediately. NO retry —
        retrying these is either incorrect (4xx is client-side wrong) or
        belongs to a higher-level resilience policy (5xx).
      - HTTP 200: parse and return.

    `SEMANTIC_SCHOLAR_API_KEY` env var, if set, is sent via `x-api-key`
    header to lift the anonymous-IP 1-RPS rate limit.
    """
    # Validate input via Pydantic schema — keeps the constraint enforced even when
    # callers invoke the impl directly (not just via ToolGateway).
    validated = S2LookupInput(paper_id=paper_id)
    url = _build_s2_url(validated.paper_id)
    headers = _build_request_headers()

    with httpx.Client(timeout=30.0) as client:
        for attempt in range(MAX_RETRY_ATTEMPTS + 1):  # 1 initial + MAX retries
            response = client.get(url, params={"fields": _S2_FIELDS}, headers=headers)
            if response.status_code != 429:
                break  # 200 / 404 / other — exit retry loop
            if attempt >= MAX_RETRY_ATTEMPTS:
                break  # exhausted retries; fall through to raise_for_status below
            delay = _compute_retry_delay(response, attempt)
            time.sleep(delay)

    if response.status_code == 404:
        return {"paper": None}
    response.raise_for_status()  # raises HTTPStatusError on the final 429 (after retries) or other non-2xx
    data = response.json()
    s2_paper_id = data["paperId"]
    return {
        "paper": {
            "paper_id": f"s2:{s2_paper_id}",  # canonical prefix-form for downstream PaperId contract
            "s2_paper_id": s2_paper_id,
            "arxiv_id": data.get("externalIds", {}).get("ArXiv"),
            "doi": data.get("externalIds", {}).get("DOI"),
            "title": data.get("title", ""),
            "abstract": data.get("abstract"),
            "authors": [a.get("name", "") for a in data.get("authors", [])],
            "year": data.get("year"),
            "venue": data.get("venue"),
            "citation_count": data.get("citationCount", 0),
        }
    }


def make_policy() -> ToolPolicy:
    return ToolPolicy(
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        # Wide (search/lookup) + Deep (abstract pre-fetch). Per least-privilege:
        # `arxiv_search` and `web_search` stay Wide-only because Deep doesn't call them.
        allowed_roles=(AgentRole.RESEARCHER_WIDE, AgentRole.RESEARCHER_DEEP),
        input_schema=S2LookupInput,
        output_schema=S2LookupOutput,
        result_trust="trusted_internal",
    )


def register(gateway: ToolGateway) -> None:
    gateway.register(make_policy(), lookup_paper)
