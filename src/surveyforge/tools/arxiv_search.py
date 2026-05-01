"""arxiv_search tool wrapper per spec § 3.

Direct wrapper over arxiv.org/api/query Atom feed using httpx + stdlib
xml.etree.ElementTree. Registers with a ToolGateway via `register(gateway)`;
the registered policy uses real Pydantic schemas (replacing the placeholder
`_OpaqueArgs`/`_OpaqueOutput` from `runtime.tool_gateway.TOOL_REGISTRY`).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from surveyforge.llm.roles import AgentRole
from surveyforge.runtime.tool_gateway import ToolGateway, ToolPolicy
from surveyforge.schemas.paper_id import PaperId

ARXIV_API_URL = "https://export.arxiv.org/api/query"
TOOL_NAME = "arxiv_search"
TOOL_VERSION = "0.1.0"

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def _strip_version(arxiv_id: str) -> str:
    """Strip trailing `vN` from an arxiv id so the canonical paper_id treats
    versions of the same paper as the same entity.

    `2401.12345v2` → `2401.12345` ; `cs/9901001v1` → `cs/9901001` ; ids without
    a version suffix pass through unchanged. The raw `arxiv_id` field on
    `ArxivPaper` keeps the full version-bearing id for traceability — only the
    canonical `paper_id` (used as evidence_store / dedup key) is stripped.
    """
    return _VERSION_SUFFIX_RE.sub("", arxiv_id)


class ArxivSearchInput(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=20)


class ArxivPaper(BaseModel):
    """One arxiv search result.

    `paper_id` is the canonical prefix-form id (`arxiv:<arxiv_id>`) ready for
    direct use as `CandidatePaper.paper_id` — Researcher-Wide should NOT
    reconstruct it from `arxiv_id` (LLM-side construction is a known source
    of `schema_invalid` retries due to dropped/wrong prefix).
    """

    model_config = ConfigDict(frozen=True)

    paper_id: PaperId        # canonical "arxiv:<id>" — copy directly into CandidatePaper
    arxiv_id: str            # raw id without prefix; kept for compatibility / debugging
    title: str
    authors: list[str]
    abstract: str
    published: datetime  # ISO-8601 from Atom <published>; Pydantic parses the string emitted by _parse_atom_feed
    categories: list[str]
    pdf_url: str | None


class ArxivSearchOutput(BaseModel):
    papers: list[ArxivPaper]


def _parse_atom_feed(xml_content: str) -> list[dict[str, Any]]:
    """Parse arxiv Atom XML response into list of paper dicts (matches ArxivPaper schema)."""
    root = ET.fromstring(xml_content)
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        id_elem = entry.find("atom:id", _ATOM_NS)
        title_elem = entry.find("atom:title", _ATOM_NS)
        summary_elem = entry.find("atom:summary", _ATOM_NS)
        published_elem = entry.find("atom:published", _ATOM_NS)
        authors: list[str] = []
        for a in entry.findall("atom:author", _ATOM_NS):
            name_elem = a.find("atom:name", _ATOM_NS)
            if name_elem is not None:
                authors.append((name_elem.text or "").strip())
        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", _ATOM_NS)]
        pdf_url: str | None = None
        for link in entry.findall("atom:link", _ATOM_NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break

        if id_elem is None or id_elem.text is None:
            continue
        arxiv_id = id_elem.text.rsplit("/", 1)[-1]
        papers.append({
            "paper_id": f"arxiv:{_strip_version(arxiv_id)}",  # canonical: version-stripped, so v1/v2 of same paper dedup
            "arxiv_id": arxiv_id,
            "title": (title_elem.text or "").strip() if title_elem is not None else "",
            "authors": authors,
            "abstract": (summary_elem.text or "").strip() if summary_elem is not None else "",
            "published": (published_elem.text or "") if published_elem is not None else "",
            "categories": categories,
            "pdf_url": pdf_url,
        })
    return papers


def search_papers(query: str, max_results: int = 10) -> dict[str, Any]:
    """Direct impl: GET arxiv API, parse Atom feed, return dict matching ArxivSearchOutput."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            ARXIV_API_URL,
            params={"search_query": f"all:{query}", "max_results": max_results},
        )
        response.raise_for_status()
    return {"papers": _parse_atom_feed(response.text)}


def make_policy() -> ToolPolicy:
    return ToolPolicy(
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        allowed_roles=(AgentRole.RESEARCHER_WIDE,),
        input_schema=ArxivSearchInput,
        output_schema=ArxivSearchOutput,
        result_trust="trusted_internal",
    )


def register(gateway: ToolGateway) -> None:
    """Register arxiv_search policy + impl with a ToolGateway instance."""
    gateway.register(make_policy(), search_papers)
