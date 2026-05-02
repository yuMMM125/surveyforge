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
from surveyforge.tools import arxiv_lookup, arxiv_search, s2_lookup, web_search

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
    assert p1["paper_id"] == "arxiv:2401.12345"   # canonical: version stripped for dedup
    assert p1["arxiv_id"] == "2401.12345v1"        # raw: keeps version for traceability
    assert p1["title"] == "A Test Paper About LLM Agents"
    assert p1["authors"] == ["Alice Researcher", "Bob Coauthor"]
    assert "test abstract" in p1["abstract"]
    assert p1["categories"] == ["cs.LG", "cs.AI"]
    assert p1["pdf_url"] == "http://arxiv.org/pdf/2401.12345v1"


def test_arxiv_search_paper_id_strips_version_suffix(respx_mock):
    """Same paper at different versions (v1, v2, v3) must dedup to the same
    canonical paper_id. The raw `arxiv_id` field keeps the version-bearing
    id for traceability."""
    multi_version_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-15T00:00:00Z</updated>
    <published>2024-01-15T00:00:00Z</published>
    <title>Paper v1</title>
    <summary>v1 abstract.</summary>
    <author><name>Alice</name></author>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2401.12345v1"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v3</id>
    <updated>2024-03-20T00:00:00Z</updated>
    <published>2024-03-20T00:00:00Z</published>
    <title>Paper v3</title>
    <summary>v3 abstract.</summary>
    <author><name>Alice</name></author>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2401.12345v3"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""
    respx_mock.get(arxiv_search.ARXIV_API_URL).mock(
        return_value=httpx.Response(200, content=multi_version_feed)
    )
    result = arxiv_search.search_papers(query="x", max_results=10)
    assert len(result["papers"]) == 2
    # Both versions canonicalize to the same paper_id (dedup-friendly)
    assert result["papers"][0]["paper_id"] == "arxiv:2401.12345"
    assert result["papers"][1]["paper_id"] == "arxiv:2401.12345"
    # But raw arxiv_id preserves the version (traceability)
    assert result["papers"][0]["arxiv_id"] == "2401.12345v1"
    assert result["papers"][1]["arxiv_id"] == "2401.12345v3"


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


def test_s2_lookup_allows_researcher_deep(conn: psycopg.Connection, respx_mock):
    """Deep uses s2_lookup as the W2 abstract pre-fetch tool."""
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        return_value=httpx.Response(200, json=S2_PAPER_FIXTURE)
    )
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    s2_lookup.register(gw)
    res = gw.call(AgentRole.RESEARCHER_DEEP, "s2_lookup", paper_id="arxiv:2401.12345")
    assert res.output.paper is not None


def test_s2_lookup_paper_id_only_accepts_arxiv_or_s2_prefix(respx_mock):
    """`s2_lookup` accepts arxiv: / s2: only; bare ids, doi:, web: all rejected.

    Layered validation:
    - `PaperId` (Annotated, runs first) rejects: missing prefix, empty suffix
    - `field_validator` (runs second) rejects: web: prefix (PaperId allows it
      but S2 API doesn't support web-indexed lookups)
    """
    from pydantic import ValidationError
    # PaperId layer: missing prefix
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="2401.12345")
    # PaperId layer: empty/whitespace suffix
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="arxiv:")  # empty suffix
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="s2:   ")  # whitespace-only suffix
    # PaperId layer: doi: not in valid prefixes
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="doi:10.0000/test")
    # field_validator layer: web: allowed by PaperId but S2 doesn't index web
    with pytest.raises(ValidationError):
        s2_lookup.lookup_paper(paper_id="web:abc123")


# ---- s2_lookup retry-on-429 + SEMANTIC_SCHOLAR_API_KEY ----

def test_s2_lookup_retries_on_429_then_succeeds_on_200(respx_mock, monkeypatch):
    """First 429 (no Retry-After) → backoff 1.0s → second 200 succeeds."""
    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "surveyforge.tools.s2_lookup.time.sleep",
        lambda s: recorded_sleeps.append(s),
    )
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        side_effect=[
            httpx.Response(429),  # no Retry-After → use DEFAULT_BACKOFF_SECONDS[0]=1.0
            httpx.Response(200, json=S2_PAPER_FIXTURE),
        ]
    )
    result = s2_lookup.lookup_paper(paper_id="arxiv:2401.12345")
    assert result["paper"]["s2_paper_id"] == "abc123"
    assert recorded_sleeps == [1.0]


def test_s2_lookup_retries_use_retry_after_header_when_present(respx_mock, monkeypatch):
    """`Retry-After: 7` overrides DEFAULT_BACKOFF_SECONDS[0]=1.0."""
    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "surveyforge.tools.s2_lookup.time.sleep",
        lambda s: recorded_sleeps.append(s),
    )
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json=S2_PAPER_FIXTURE),
        ]
    )
    result = s2_lookup.lookup_paper(paper_id="arxiv:2401.12345")
    assert result["paper"]["s2_paper_id"] == "abc123"
    assert recorded_sleeps == [7.0]


def test_s2_lookup_exhausted_retries_raises_http_error(respx_mock, monkeypatch):
    """4 consecutive 429s (1 initial + 3 retries) → HTTPStatusError raised.

    Sleep called 3 times with the full DEFAULT_BACKOFF_SECONDS sequence
    (1.0, 2.0, 4.0) — no sleep after the final attempt because we give up
    instead of waiting before raising.
    """
    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "surveyforge.tools.s2_lookup.time.sleep",
        lambda s: recorded_sleeps.append(s),
    )
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
        ]
    )
    with pytest.raises(httpx.HTTPStatusError):
        s2_lookup.lookup_paper(paper_id="arxiv:2401.12345")
    assert recorded_sleeps == [1.0, 2.0, 4.0]


def test_s2_lookup_exhausted_retries_through_gateway_records_provider_429(
    conn: psycopg.Connection, respx_mock, monkeypatch,
):
    """All-429 scenario via ToolGateway: tool_calls row tagged provider_429."""
    monkeypatch.setattr(
        "surveyforge.tools.s2_lookup.time.sleep",
        lambda s: None,
    )
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
        ]
    )
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    s2_lookup.register(gw)
    with pytest.raises(httpx.HTTPStatusError):
        gw.call(AgentRole.RESEARCHER_DEEP, "s2_lookup", paper_id="arxiv:2401.12345")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT error_category FROM tool_calls WHERE run_id = %s",
            (run_id,),
        )
        rows = cur.fetchall()
    assert rows == [("provider_429",)]


def test_s2_lookup_404_does_not_retry(respx_mock, monkeypatch):
    """404 returns `{"paper": None}` immediately — never retried (paper-missing
    is a stable signal, not a transient failure)."""
    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "surveyforge.tools.s2_lookup.time.sleep",
        lambda s: recorded_sleeps.append(s),
    )
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:9999.00000").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    result = s2_lookup.lookup_paper(paper_id="arxiv:9999.00000")
    assert result["paper"] is None
    assert recorded_sleeps == []


def test_s2_lookup_500_does_not_retry(respx_mock, monkeypatch):
    """5xx is NOT retried at this layer — that belongs to a higher-level
    resilience policy. We raise immediately so the gateway can tag it
    provider_5xx for downstream routing."""
    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "surveyforge.tools.s2_lookup.time.sleep",
        lambda s: recorded_sleeps.append(s),
    )
    respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        return_value=httpx.Response(500, text="internal error")
    )
    with pytest.raises(httpx.HTTPStatusError):
        s2_lookup.lookup_paper(paper_id="arxiv:2401.12345")
    assert recorded_sleeps == []


def test_s2_lookup_passes_x_api_key_header_when_env_set(respx_mock, monkeypatch):
    """`SEMANTIC_SCHOLAR_API_KEY=test-key-123` → request carries
    `x-api-key: test-key-123` header (lifts anonymous-IP 1-RPS limit)."""
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key-123")
    route = respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        return_value=httpx.Response(200, json=S2_PAPER_FIXTURE)
    )
    s2_lookup.lookup_paper(paper_id="arxiv:2401.12345")
    sent_request = route.calls.last.request
    assert sent_request.headers.get("x-api-key") == "test-key-123"


def test_s2_lookup_no_x_api_key_header_when_env_missing(respx_mock, monkeypatch):
    """No env var → no `x-api-key` header (fall back to anonymous request)."""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    route = respx_mock.get(f"{s2_lookup.S2_API_BASE}/paper/arXiv:2401.12345").mock(
        return_value=httpx.Response(200, json=S2_PAPER_FIXTURE)
    )
    s2_lookup.lookup_paper(paper_id="arxiv:2401.12345")
    sent_request = route.calls.last.request
    assert "x-api-key" not in sent_request.headers


# ---- arxiv_lookup (W2 polish 5: SS fallback for arxiv:* papers) ----

ARXIV_LOOKUP_ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <updated>2024-04-01T00:00:00Z</updated>
    <published>2024-04-01T00:00:00Z</published>
    <title>An Arxiv Lookup Test Paper</title>
    <summary>This is the abstract returned by the arxiv id_list endpoint.</summary>
    <author><name>Lookup Author</name></author>
  </entry>
</feed>
"""

ARXIV_LOOKUP_EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>
"""


def test_arxiv_lookup_success_returns_abstract(respx_mock):
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(200, content=ARXIV_LOOKUP_ATOM_FIXTURE)
    )
    result = arxiv_lookup.lookup_paper(paper_id="arxiv:1234.5678")
    assert result["paper"] is not None
    assert result["paper"]["paper_id"] == "arxiv:1234.5678"
    assert result["paper"]["title"] == "An Arxiv Lookup Test Paper"
    assert "abstract returned by the arxiv id_list endpoint" in result["paper"]["abstract"]


def test_arxiv_lookup_unknown_id_returns_paper_none(respx_mock):
    """arxiv typically returns 200 with an empty feed (no <entry>) for unknown ids."""
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(200, content=ARXIV_LOOKUP_EMPTY_FEED)
    )
    result = arxiv_lookup.lookup_paper(paper_id="arxiv:9999.99999")
    assert result == {"paper": None}


def test_arxiv_lookup_404_returns_paper_none(respx_mock):
    """Real HTTP 404 (rare for arxiv) treated for parity with s2_lookup 404 handling."""
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(404, text="not found")
    )
    result = arxiv_lookup.lookup_paper(paper_id="arxiv:9999.99999")
    assert result == {"paper": None}


def test_arxiv_lookup_500_propagates(respx_mock):
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(500, text="upstream broken")
    )
    with pytest.raises(httpx.HTTPStatusError):
        arxiv_lookup.lookup_paper(paper_id="arxiv:1234.5678")


def test_arxiv_lookup_retries_on_429_then_succeeds(respx_mock, monkeypatch):
    """First 429 (no Retry-After) → backoff 1.0s → second 200 succeeds."""
    recorded_sleeps: list[float] = []
    monkeypatch.setattr(
        "surveyforge.tools.arxiv_lookup.time.sleep",
        lambda s: recorded_sleeps.append(s),
    )
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, content=ARXIV_LOOKUP_ATOM_FIXTURE),
        ]
    )
    result = arxiv_lookup.lookup_paper(paper_id="arxiv:1234.5678")
    assert result["paper"] is not None
    assert result["paper"]["title"] == "An Arxiv Lookup Test Paper"
    assert recorded_sleeps == [1.0]


def test_arxiv_lookup_through_gateway_logs_tool_call(conn: psycopg.Connection, respx_mock):
    """End-to-end: register wrapper with gateway, call as Deep, verify tool_calls row."""
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(200, content=ARXIV_LOOKUP_ATOM_FIXTURE)
    )
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    arxiv_lookup.register(gw)
    res = gw.call(AgentRole.RESEARCHER_DEEP, "arxiv_lookup", paper_id="arxiv:1234.5678")
    assert res.cache_hit is False
    assert res.result_trust == "trusted_internal"
    assert res.output.paper is not None
    assert res.output.paper.title == "An Arxiv Lookup Test Paper"

    with conn.cursor() as cur:
        cur.execute(
            "SELECT tool_name, agent_role, cache_hit, result_trust "
            "FROM tool_calls WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    assert row == ("arxiv_lookup", "researcher_deep", False, "trusted_internal")


def test_arxiv_lookup_role_not_in_allowed_raises(conn: psycopg.Connection, respx_mock):
    """arxiv_lookup is Deep-only — Wide / Planner not in allowed_roles."""
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    arxiv_lookup.register(gw)
    from surveyforge.runtime.tool_gateway import ToolRoleDenied
    with pytest.raises(ToolRoleDenied):
        gw.call(AgentRole.RESEARCHER_WIDE, "arxiv_lookup", paper_id="arxiv:1234.5678")


def test_arxiv_lookup_paper_id_only_accepts_arxiv_prefix(respx_mock):
    """Non-arxiv prefixes rejected at input validation — arxiv API doesn't index s2/web."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        arxiv_lookup.lookup_paper(paper_id="s2:abc123")
    with pytest.raises(ValidationError):
        arxiv_lookup.lookup_paper(paper_id="web:abc123")
    with pytest.raises(ValidationError):
        arxiv_lookup.lookup_paper(paper_id="1234.5678")  # missing prefix


def test_arxiv_lookup_handles_missing_summary_in_atom(respx_mock):
    """Atom feed with `<entry>` but no `<summary>` → paper.abstract is None.

    Documented choice: distinguish 'no abstract present in feed' (None) from
    'lookup failed' (paper=None). A None abstract still indicates the paper
    exists; the caller can decide whether to skip or surface the title.
    """
    feed_no_summary = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>Title But No Abstract</title>
  </entry>
</feed>
"""
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(200, content=feed_no_summary)
    )
    result = arxiv_lookup.lookup_paper(paper_id="arxiv:1234.5678")
    assert result["paper"] is not None
    assert result["paper"]["title"] == "Title But No Abstract"
    assert result["paper"]["abstract"] is None


def test_arxiv_lookup_uses_https_endpoint():
    """`ARXIV_API_BASE` MUST be https. arxiv enforces 301 redirect from
    http→https as of 2026-05-02 (root cause of bounded smoke v8 failure
    — see Task 7 polish 8 in spike log). A regression to http would only
    fail at live integration, not unit tests, because respx mocks intercept
    on URL match without simulating 301. This test guards the constant
    explicitly so the regression can never escape PR review again.
    """
    assert arxiv_lookup.ARXIV_API_BASE.startswith("https://"), (
        f"ARXIV_API_BASE must use https; got {arxiv_lookup.ARXIV_API_BASE!r}. "
        "arxiv enforces http→https 301 redirect since at least 2026-05; "
        "without `follow_redirects=True` (also enabled defensively in the "
        "wrapper's httpx.Client) a plain http URL raises HTTPStatusError(301)."
    )


def test_arxiv_lookup_follows_301_redirect_defensively(respx_mock):
    """Defensive belt-and-suspenders: even if a future arxiv URL change
    introduces a new redirect, `follow_redirects=True` on the httpx.Client
    keeps the wrapper working. Mock a 301 → 200 chain at the wrapper level.

    Pairs with `test_arxiv_lookup_uses_https_endpoint` — the URL constant
    + the client kwarg are both defenses against the bounded smoke v8
    failure mode.
    """
    redirect_target = "https://export.arxiv.org/api/query/somewhere-else"
    respx_mock.get(arxiv_lookup.ARXIV_API_BASE).mock(
        return_value=httpx.Response(
            301,
            headers={"Location": redirect_target},
        )
    )
    respx_mock.get(redirect_target).mock(
        return_value=httpx.Response(200, content=ARXIV_LOOKUP_ATOM_FIXTURE)
    )
    result = arxiv_lookup.lookup_paper(paper_id="arxiv:1234.5678")
    assert result["paper"] is not None
    assert result["paper"]["title"] == "An Arxiv Lookup Test Paper"


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
