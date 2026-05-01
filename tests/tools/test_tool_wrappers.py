"""Tool wrapper tests per spec § 3.

Three wrappers (arxiv_search / s2_lookup / web_search), each tested at two layers:
  (a) impl directly via respx-mocked HTTP — fast, no DB
  (b) gateway-mediated via real testcontainers Postgres — verifies tool_calls logging
"""
from __future__ import annotations

import time

import httpx
import psycopg
import pytest
import respx

from surveyforge.llm.roles import AgentRole
from surveyforge.runtime.runs import RunManager
from surveyforge.runtime.tool_gateway import ToolGateway
from surveyforge.tools import arxiv_search, s2_lookup, web_search

# ---- helpers ----

def _make_run(conn: psycopg.Connection) -> str:
    rm = RunManager(conn)
    return rm.create(topic="test", idempotency_key=f"key-{time.perf_counter_ns()}").run_id


@pytest.fixture
def respx_mock():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


# ---- arxiv_search ----

ARXIV_ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-15T00:00:00Z</updated>
    <published>2024-01-15T00:00:00Z</published>
    <title>A Test Paper About LLM Agents</title>
    <summary>This is a test abstract describing the paper.</summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Coauthor</name></author>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2401.12345v1"/>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2401.12345v1"/>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.99999v2</id>
    <updated>2024-02-20T12:00:00Z</updated>
    <published>2024-02-20T12:00:00Z</published>
    <title>Second Test Paper</title>
    <summary>Another abstract.</summary>
    <author><name>Carol Solo</name></author>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2402.99999v2"/>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2402.99999v2"/>
    <category term="stat.ML"/>
  </entry>
</feed>
"""


def test_arxiv_search_impl_parses_atom_feed_into_papers(respx_mock):
    respx_mock.get(arxiv_search.ARXIV_API_URL).mock(
        return_value=httpx.Response(200, content=ARXIV_ATOM_FIXTURE)
    )
    result = arxiv_search.search_papers(query="LLM agents", max_results=10)
    assert len(result["papers"]) == 2
    p1 = result["papers"][0]
    assert p1["paper_id"] == "arxiv:2401.12345v1"  # canonical prefix form for downstream
    assert p1["arxiv_id"] == "2401.12345v1"
    assert p1["title"] == "A Test Paper About LLM Agents"
    assert p1["authors"] == ["Alice Researcher", "Bob Coauthor"]
    assert "test abstract" in p1["abstract"]
    assert p1["categories"] == ["cs.LG", "cs.AI"]
    assert p1["pdf_url"] == "http://arxiv.org/pdf/2401.12345v1"


def test_arxiv_search_max_results_field_bounds(respx_mock):
    """`max_results` clamped to [1, 20] at the input schema layer."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        arxiv_search.ArxivSearchInput(query="x", max_results=0)
    with pytest.raises(ValidationError):
        arxiv_search.ArxivSearchInput(query="x", max_results=21)
    arxiv_search.ArxivSearchInput(query="x", max_results=1)   # ok
    arxiv_search.ArxivSearchInput(query="x", max_results=20)  # ok


def test_arxiv_search_impl_handles_empty_feed(respx_mock):
    empty_feed = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>'
    respx_mock.get(arxiv_search.ARXIV_API_URL).mock(
        return_value=httpx.Response(200, content=empty_feed)
    )
    result = arxiv_search.search_papers(query="zzz no results", max_results=10)
    assert result["papers"] == []


def test_arxiv_search_through_gateway_logs_tool_call(conn: psycopg.Connection, respx_mock):
    """End-to-end: register wrapper with gateway, call, verify tool_calls row."""
    respx_mock.get(arxiv_search.ARXIV_API_URL).mock(
        return_value=httpx.Response(200, content=ARXIV_ATOM_FIXTURE)
    )
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    arxiv_search.register(gw)
    res = gw.call(AgentRole.RESEARCHER_WIDE, "arxiv_search", query="LLM agents")
    assert res.cache_hit is False
    assert res.result_trust == "trusted_internal"
    assert len(res.output.papers) == 2

    with conn.cursor() as cur:
        cur.execute(
            "SELECT tool_name, agent_role, cache_hit, result_trust "
            "FROM tool_calls WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    assert row == ("arxiv_search", "researcher_wide", False, "trusted_internal")


def test_arxiv_search_role_not_in_allowed_roles_raises(conn: psycopg.Connection, respx_mock):
    """Planner is NOT in allowed_roles for arxiv_search."""
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    arxiv_search.register(gw)
    from surveyforge.runtime.tool_gateway import ToolRoleDenied
    with pytest.raises(ToolRoleDenied):
        gw.call(AgentRole.PLANNER, "arxiv_search", query="x")


def test_arxiv_search_http_error_propagates(respx_mock):
    respx_mock.get(arxiv_search.ARXIV_API_URL).mock(
        return_value=httpx.Response(503, text="upstream down")
    )
    with pytest.raises(httpx.HTTPStatusError):
        arxiv_search.search_papers(query="x", max_results=10)


# ---- s2_lookup ----

S2_PAPER_FIXTURE = {
    "paperId": "abc123",
    "externalIds": {"ArXiv": "2401.12345", "DOI": "10.0000/test"},
    "title": "S2 Test Paper",
    "abstract": "S2 abstract.",
    "authors": [{"name": "Dave Author"}, {"name": "Eve Coauthor"}],
    "year": 2024,
    "venue": "NeurIPS 2024",
    "citationCount": 42,
}


def test_s2_lookup_by_arxiv_id_returns_paper(respx_mock):
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        return_value=httpx.Response(200, json=S2_PAPER_FIXTURE)
    )
    result = s2_lookup.lookup_paper(paper_id="arxiv:2401.12345")
    assert result["paper"]["paper_id"] == "s2:abc123"  # canonical prefix form for downstream
    assert result["paper"]["s2_paper_id"] == "abc123"
    assert result["paper"]["title"] == "S2 Test Paper"
    assert result["paper"]["authors"] == ["Dave Author", "Eve Coauthor"]
    assert result["paper"]["year"] == 2024
    assert result["paper"]["citation_count"] == 42


def test_s2_lookup_404_returns_paper_none(respx_mock):
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:9999.00000").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    result = s2_lookup.lookup_paper(paper_id="arxiv:9999.00000")
    assert result["paper"] is None


def test_s2_lookup_through_gateway_logs_tool_call(conn: psycopg.Connection, respx_mock):
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        return_value=httpx.Response(200, json=S2_PAPER_FIXTURE)
    )
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    s2_lookup.register(gw)
    res = gw.call(AgentRole.RESEARCHER_WIDE, "s2_lookup", paper_id="arxiv:2401.12345")
    assert res.output.paper is not None
    assert res.output.paper.s2_paper_id == "abc123"
    with conn.cursor() as cur:
        cur.execute("SELECT tool_name FROM tool_calls WHERE run_id = %s", (run_id,))
        assert cur.fetchone() == ("s2_lookup",)


def test_s2_lookup_paper_id_only_accepts_arxiv_or_s2_prefix(respx_mock):
    """`s2_lookup` accepts arxiv: / s2: only; bare ids, doi:, web: all rejected.

    Keeping the namespace tight matches global PaperId contract (`schemas/paper_id.py`)
    — if W3+ needs DOI lookup, add a separate `lookup_paper_by_doi` function.
    """
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="2401.12345")  # no prefix
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="doi:10.0000/test")  # doi: not supported
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="web:abc123")  # web: not supported (S2 isn't web-indexed)


# ---- web_search ----

SERPER_FIXTURE = {
    "organic": [
        {
            "title": "Result 1",
            "link": "https://example.com/1",
            "snippet": "First result snippet text.",
            "position": 1,
        },
        {
            "title": "Result 2",
            "link": "https://example.com/2",
            "snippet": "Second result snippet.",
            "position": 2,
        },
    ]
}


def test_web_search_impl_parses_serper_response(respx_mock, monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key-not-real")
    respx_mock.post(web_search.SERPER_API_URL).mock(
        return_value=httpx.Response(200, json=SERPER_FIXTURE)
    )
    result = web_search.search_web(query="LLM agents", num_results=5)
    assert len(result["results"]) == 2
    r1 = result["results"][0]
    assert r1["paper_id"].startswith("web:")  # canonical prefix form for downstream
    assert len(r1["paper_id"]) == len("web:") + 16  # 16-hex-char hash
    assert r1["title"] == "Result 1"
    assert r1["url"] == "https://example.com/1"
    assert r1["snippet"] == "First result snippet text."
    # Stable hash: same URL → same paper_id across calls
    result2 = web_search.search_web(query="LLM agents", num_results=5)
    assert result["results"][0]["paper_id"] == result2["results"][0]["paper_id"]


def test_web_search_num_results_field_bounds():
    """`num_results` clamped to [1, 20] — same bounds as arxiv max_results for consistency."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        web_search.WebSearchInput(query="x", num_results=0)
    with pytest.raises(ValidationError):
        web_search.WebSearchInput(query="x", num_results=21)
    web_search.WebSearchInput(query="x", num_results=1)   # ok
    web_search.WebSearchInput(query="x", num_results=20)  # ok


def test_web_search_skips_linkless_results(respx_mock, monkeypatch):
    """Serper occasionally returns results without a `link` field (or empty link).
    `search_web` must filter these out — without the filter, `_url_to_paper_id("")`
    returns a deterministic `web:e3b0c442...` paper_id shared by every linkless
    result across all runs, which would pollute the dedup space."""
    monkeypatch.setenv("SERPER_API_KEY", "test-key-not-real")
    fixture_with_linkless = {
        "organic": [
            {"title": "Has link", "link": "https://example.com/1", "snippet": "ok", "position": 1},
            {"title": "No link", "snippet": "missing link key", "position": 2},     # missing 'link'
            {"title": "Empty link", "link": "", "snippet": "empty value", "position": 3},  # empty string
            {"title": "Has link 2", "link": "https://example.com/2", "snippet": "ok", "position": 4},
        ]
    }
    respx_mock.post(web_search.SERPER_API_URL).mock(
        return_value=httpx.Response(200, json=fixture_with_linkless)
    )
    result = web_search.search_web(query="x", num_results=5)
    titles = [r["title"] for r in result["results"]]
    assert titles == ["Has link", "Has link 2"]  # linkless entries skipped


def test_web_search_missing_api_key_raises_runtime_error(respx_mock, monkeypatch):
    """SERPER_API_KEY env var is required; missing → loud failure with actionable message."""
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SERPER_API_KEY"):
        web_search.search_web(query="x", num_results=5)


def test_web_search_through_gateway_marks_result_untrusted(
    conn: psycopg.Connection, respx_mock, monkeypatch,
):
    """web_search results are `untrusted_content` (open-web text); verify the flag persists."""
    monkeypatch.setenv("SERPER_API_KEY", "test-key-not-real")
    respx_mock.post(web_search.SERPER_API_URL).mock(
        return_value=httpx.Response(200, json=SERPER_FIXTURE)
    )
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    web_search.register(gw)
    res = gw.call(AgentRole.RESEARCHER_WIDE, "web_search", query="x")
    assert res.result_trust == "untrusted_content"
    with conn.cursor() as cur:
        cur.execute("SELECT result_trust FROM tool_calls WHERE run_id = %s", (run_id,))
        assert cur.fetchone() == ("untrusted_content",)


def test_web_search_api_key_not_in_input_hash(respx_mock, monkeypatch):
    """Sanity: SERPER_API_KEY value never participates in cache key (sanitize_args drops it)."""
    monkeypatch.setenv("SERPER_API_KEY", "test-key-not-real")
    from surveyforge.runtime.tool_gateway import compute_input_hash
    h_clean = compute_input_hash({"query": "x", "num_results": 5})
    h_with_key = compute_input_hash({"query": "x", "num_results": 5, "api_key": "secret"})
    assert h_clean == h_with_key
