"""arxiv_lookup tool wrapper — single-paper abstract fetch by arxiv id.

Used as the W2 fallback for Deep when `s2_lookup` returns transient errors
(429/5xx) on `arxiv:*` papers. Could also be used standalone in W3+ if Deep
needs paper metadata without the broader S2 enrichment (citations, venues).

Endpoint: `http://export.arxiv.org/api/query?id_list=<arxiv_id>` returns an
atom feed; parser extracts the `<summary>` element as the abstract.

Per Task 7 polish 5 (2026-05-02): Semantic Scholar API key application was
rejected, so anonymous-IP throttling on `s2_lookup` would otherwise block
the bounded smoke / CLI demo. arxiv_lookup serves as the transparent fallback
for arxiv:* papers — at the cost of losing s2-enrichment fields (citation
count, venue, doi) the abstract path stays available without an SS key.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from litweave.llm.roles import AgentRole
from litweave.runtime.tool_gateway import ToolGateway, ToolPolicy
from litweave.schemas.paper_id import PaperId

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
"""arxiv API endpoint. MUST be https: as of 2026-05-02 arxiv enforces a 301
Moved Permanently from http → https (root cause of bounded smoke v8 failure
— see Task 7 polish 8 in spike log). The httpx client below also enables
`follow_redirects=True` as defensive belt-and-suspenders so future arxiv
URL changes don't break the fallback path again."""
TOOL_NAME = "arxiv_lookup"
TOOL_VERSION = "0.1.0"

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Retry policy mirrors `s2_lookup` (1 initial + MAX_RETRY_ATTEMPTS retries on
# 429, exp backoff 1/2/4s, Retry-After header preferred). The `_compute_retry_delay`
# helper is duplicated rather than imported from `s2_lookup` because tool
# wrappers are intended to be loosely coupled — a future change to s2's retry
# tuning shouldn't silently change arxiv's behavior. The helper is small enough
# (5 lines of logic) that the duplication cost is lower than the coupling cost.
MAX_RETRY_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class ArxivLookupInput(BaseModel):
    """Restrict paper_id to `arxiv:` only — `s2:` and `web:` are out of scope.

    The arxiv API does not index s2 or web ids, and this tool exists
    specifically to be the arxiv-side fallback for s2_lookup transient
    errors. The validator rejects non-arxiv prefixes early so the caller
    (Deep's `_fetch_abstract`) doesn't need to second-guess routing.
    """

    paper_id: PaperId

    @field_validator("paper_id")
    @classmethod
    def _restrict_to_arxiv_prefix(cls, v: str) -> str:
        if not v.startswith("arxiv:"):
            raise ValueError(
                f"arxiv_lookup paper_id must start with arxiv:, got {v!r}"
            )
        return v


class ArxivPaperAbstract(BaseModel):
    """Minimal arxiv-fetched paper record.

    Deliberately narrower than `s2_lookup.S2Paper`: arxiv API does not
    expose citation count / venue / canonical DOI, and the only field Deep
    actually consumes from the abstract pre-fetch is `abstract`. Title is
    kept for diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    paper_id: PaperId        # canonical "arxiv:<id>" form
    title: str
    abstract: str | None     # may legitimately be missing/empty in atom feed


class ArxivLookupOutput(BaseModel):
    paper: ArxivPaperAbstract | None


def _compute_retry_delay(response: httpx.Response, attempt_index: int) -> float:
    """Pick retry delay for a 429 response.

    Prefer the server-provided `Retry-After` header (RFC 7231: integer seconds
    OR HTTP-date — we only handle integer-seconds, the common case). Fall back
    to exponential backoff `DEFAULT_BACKOFF_SECONDS[attempt_index]`. Negative
    or non-numeric Retry-After values are ignored — fall back too.

    Duplicated from `s2_lookup._compute_retry_delay` rather than imported so
    the two tools' retry tuning can diverge independently (see module docstring).
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


def _parse_atom_feed(xml_content: str) -> dict[str, Any] | None:
    """Parse arxiv id_list response. Returns None if the feed has no `<entry>`.

    Tolerant of whitespace; `<summary>` may be missing or empty (in which case
    `abstract` is returned as None, distinguishing 'no abstract' from
    'lookup failed').
    """
    root = ET.fromstring(xml_content)
    entry = root.find("atom:entry", _ATOM_NS)
    if entry is None:
        return None
    title_elem = entry.find("atom:title", _ATOM_NS)
    summary_elem = entry.find("atom:summary", _ATOM_NS)
    title = (title_elem.text or "").strip() if title_elem is not None and title_elem.text else ""
    abstract: str | None
    if summary_elem is None or summary_elem.text is None:
        abstract = None
    else:
        stripped = summary_elem.text.strip()
        abstract = stripped if stripped else None
    return {"title": title, "abstract": abstract}


def _build_arxiv_id_param(paper_id: str) -> str:
    """Translate prefixed paper_id `arxiv:<id>` → bare `<id>` for `id_list=` query."""
    prefix, _, suffix = paper_id.partition(":")
    if prefix != "arxiv":
        raise ValueError(f"unsupported paper_id prefix: {prefix!r}")
    return suffix


def lookup_paper(paper_id: str) -> dict[str, Any]:
    """Direct impl: GET arxiv id_list, parse atom feed, return dict matching ArxivLookupOutput.

    Retry policy (parallel to s2_lookup, commit `d3d451c`):
      - HTTP 429: retry up to MAX_RETRY_ATTEMPTS times with exp backoff
        (Retry-After preferred). After exhaustion, raise HTTPStatusError.
      - HTTP 200 with empty `<entry>` (arxiv's typical "unknown id" shape):
        return `{"paper": None}`.
      - HTTP 404 (rare for arxiv but treated for parity with s2_lookup):
        return `{"paper": None}`.
      - Other 4xx / 5xx / network errors: raise (caller `classify_exception`s).

    Note: arxiv has no API-key path equivalent to `SEMANTIC_SCHOLAR_API_KEY`;
    rate limits on the public id_list endpoint are very generous compared to
    SS anonymous quota. Retry-on-429 is here for parity / completeness, not
    because arxiv 429s are common in practice.
    """
    validated = ArxivLookupInput(paper_id=paper_id)
    arxiv_id = _build_arxiv_id_param(validated.paper_id)

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for attempt in range(MAX_RETRY_ATTEMPTS + 1):  # 1 initial + MAX retries
            response = client.get(ARXIV_API_BASE, params={"id_list": arxiv_id})
            if response.status_code != 429:
                break
            if attempt >= MAX_RETRY_ATTEMPTS:
                break
            delay = _compute_retry_delay(response, attempt)
            time.sleep(delay)

    if response.status_code == 404:
        return {"paper": None}
    response.raise_for_status()

    parsed = _parse_atom_feed(response.text)
    if parsed is None:
        return {"paper": None}
    return {
        "paper": {
            "paper_id": validated.paper_id,  # canonical prefix-form already validated
            "title": parsed["title"],
            "abstract": parsed["abstract"],
        }
    }


def make_policy() -> ToolPolicy:
    return ToolPolicy(
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        # Deep-only: this is an internal fallback path for the abstract pre-fetch,
        # not a general-purpose lookup. Wide already has `arxiv_search`; adding
        # arxiv_lookup to Wide's allowed_roles would create two redundant arxiv
        # paths the LLM might oscillate between.
        allowed_roles=(AgentRole.RESEARCHER_DEEP,),
        input_schema=ArxivLookupInput,
        output_schema=ArxivLookupOutput,
        result_trust="trusted_internal",
    )


def register(gateway: ToolGateway) -> None:
    gateway.register(make_policy(), lookup_paper)
