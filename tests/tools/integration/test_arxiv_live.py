"""Live arxiv integration test — hits real arxiv.org/api/query (no key needed).

Marked `pytest.mark.integration`; default `pytest` skips this file.
Run with `pytest tests/tools/integration -m integration`.
"""
from __future__ import annotations

import pytest

from surveyforge.tools import arxiv_search


@pytest.mark.integration
def test_arxiv_live_search_returns_real_papers():
    """Smoke: arxiv API is up + our parser handles the real Atom feed shape."""
    result = arxiv_search.search_papers(query="long context language models", max_results=3)
    assert len(result["papers"]) >= 1
    paper = result["papers"][0]
    assert paper["arxiv_id"]
    assert paper["title"]
    assert isinstance(paper["authors"], list)
    assert len(paper["authors"]) >= 1
    assert paper["abstract"]
    assert paper["pdf_url"] is None or paper["pdf_url"].startswith("http")
