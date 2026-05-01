"""s2_lookup tool wrapper per spec § 3.

Direct wrapper over Semantic Scholar Graph API. Looks up paper metadata by
prefixed paper_id (`arxiv:` or `s2:` only — `doi:` excluded to keep paper_id
namespace consistent with global PaperId contract); 404 returns `paper=None`
rather than raising, so callers can distinguish missing-paper from network-error.
"""
from __future__ import annotations

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


class S2LookupInput(BaseModel):
    """Restrict paper_id to `arxiv:` / `s2:` only — `web:` is excluded
    because S2 doesn't index web pages, and `doi:` is excluded because the
    global PaperId contract (`schemas/paper_id.py`) doesn't support DOI as a
    paper-id prefix. If W3+ needs DOI lookup, add a separate `doi: str | None`
    field rather than overloading `paper_id`.
    """

    paper_id: str

    @field_validator("paper_id")
    @classmethod
    def _has_supported_prefix(cls, v: str) -> str:
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
    """Direct impl: GET S2 paper metadata; return dict matching S2LookupOutput."""
    # Validate input via Pydantic schema — keeps the constraint enforced even when
    # callers invoke the impl directly (not just via ToolGateway).
    validated = S2LookupInput(paper_id=paper_id)
    url = _build_s2_url(validated.paper_id)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params={"fields": _S2_FIELDS})
    if response.status_code == 404:
        return {"paper": None}
    response.raise_for_status()
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
        allowed_roles=(AgentRole.RESEARCHER_WIDE,),
        input_schema=S2LookupInput,
        output_schema=S2LookupOutput,
        result_trust="trusted_internal",
    )


def register(gateway: ToolGateway) -> None:
    gateway.register(make_policy(), lookup_paper)
