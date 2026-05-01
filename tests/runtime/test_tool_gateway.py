"""ToolGateway + ToolPolicy + TOOL_REGISTRY tests per spec § 2.7.3."""
from __future__ import annotations

import time
from typing import Any

import psycopg
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from surveyforge.llm.roles import AgentRole
from surveyforge.runtime.runs import RunManager
from surveyforge.runtime.tool_gateway import (
    TOOL_REGISTRY,
    ToolGateway,
    ToolNotRegistered,
    ToolPolicy,
    ToolRoleDenied,
    compute_input_hash,
    sanitize_args,
)

# ---- helpers ----

class _EchoIn(BaseModel):
    query: str


class _EchoOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    echoed: str
    n_chars: int


def _make_run(conn: psycopg.Connection) -> str:
    rm = RunManager(conn)
    return rm.create(topic="test", idempotency_key=f"key-{time.perf_counter_ns()}").run_id


def _echo_policy(**overrides: Any) -> ToolPolicy:
    base = dict(
        tool_name="echo", tool_version="1.0.0",
        allowed_roles=(AgentRole.RESEARCHER_WIDE,),
        input_schema=_EchoIn, output_schema=_EchoOut,
        result_trust="trusted_internal",
    )
    base.update(overrides)
    return ToolPolicy(**base)


def _echo_impl(query: str) -> dict[str, Any]:
    return {"echoed": query, "n_chars": len(query)}


# ---- TOOL_REGISTRY shape ----

def test_tool_registry_has_nine_entries():
    assert len(TOOL_REGISTRY) == 9
    assert set(TOOL_REGISTRY.keys()) == {
        "arxiv_search", "s2_lookup", "web_search", "pdf_reader", "citation_verifier",
        "evidence_store_read", "evidence_store_write", "metadata_helper", "format_helper",
    }


def test_tool_registry_trust_attribution():
    untrusted = {"pdf_reader", "web_search"}
    for name, policy in TOOL_REGISTRY.items():
        if name in untrusted:
            assert policy.result_trust == "untrusted_content", name
        else:
            assert policy.result_trust == "trusted_internal", name


def test_evidence_store_write_is_not_idempotent():
    """Each write produces a new evidence_id row — must not cache."""
    assert TOOL_REGISTRY["evidence_store_write"].idempotent is False


# ---- input_hash helpers ----

def test_compute_input_hash_is_deterministic_under_key_order():
    h1 = compute_input_hash({"a": 1, "b": 2, "c": 3})
    h2 = compute_input_hash({"c": 3, "a": 1, "b": 2})
    assert h1 == h2


def test_sanitize_args_strips_secret_fields():
    cleaned = sanitize_args({
        "query": "x",
        "api_key": "AKIA...",
        "auth_token": "tok",
        "Authorization": "Bearer xyz",
        "password": "p",
        "_secret": "s",
        "nested": {"inner_key": "drop", "keep": "yes"},
    })
    assert "query" in cleaned
    assert "api_key" not in cleaned
    assert "auth_token" not in cleaned
    assert "Authorization" not in cleaned
    assert "password" not in cleaned
    assert "_secret" not in cleaned
    assert cleaned["nested"] == {"keep": "yes"}


def test_sanitize_args_strips_inside_lists_of_dicts():
    cleaned = sanitize_args({"items": [{"name": "a", "api_key": "x"}, {"name": "b"}]})
    assert cleaned["items"] == [{"name": "a"}, {"name": "b"}]


def test_compute_input_hash_strips_secrets_before_hashing():
    h_with_secret = compute_input_hash({"q": "x", "api_key": "AKIA"})
    h_without = compute_input_hash({"q": "x"})
    assert h_with_secret == h_without


# ---- ToolGateway.call ----

def test_unknown_tool_raises_tool_not_registered(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    with pytest.raises(ToolNotRegistered):
        gw.call(AgentRole.RESEARCHER_WIDE, "nonexistent", q="x")


def test_role_not_allowed_raises_tool_role_denied(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    gw.register(_echo_policy(), _echo_impl)
    with pytest.raises(ToolRoleDenied):
        gw.call(AgentRole.PLANNER, "echo", query="hi")


def test_invalid_input_raises_validation_error(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    gw.register(_echo_policy(), _echo_impl)
    with pytest.raises(ValidationError):
        gw.call(AgentRole.RESEARCHER_WIDE, "echo", wrong_arg="x")


def test_call_invokes_impl_and_logs_to_tool_calls(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    gw.register(_echo_policy(), _echo_impl)
    res = gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hello")
    assert res.cache_hit is False
    assert res.truncated is False
    assert res.result_trust == "trusted_internal"
    assert res.output.echoed == "hello"
    with conn.cursor() as cur:
        cur.execute(
            """SELECT tool_name, tool_version, agent_role, cache_hit, truncated, result_trust
               FROM tool_calls WHERE run_id = %s""",
            (run_id,),
        )
        rows = cur.fetchall()
    assert rows == [("echo", "1.0.0", "researcher_wide", False, False, "trusted_internal")]


def test_cache_hit_on_second_call_with_same_args(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    n_invocations = [0]
    def counting_impl(query: str) -> dict[str, Any]:
        n_invocations[0] += 1
        return {"echoed": query, "n_chars": len(query)}
    gw.register(_echo_policy(), counting_impl)
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    res2 = gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    assert n_invocations[0] == 1  # second call hit cache
    assert res2.cache_hit is True
    assert res2.output.echoed == "hi"
    with conn.cursor() as cur:
        cur.execute("SELECT cache_hit FROM tool_calls WHERE run_id = %s ORDER BY created_at", (run_id,))
        flags = [r[0] for r in cur.fetchall()]
    assert flags == [False, True]  # first call MISS, second call HIT


def test_cache_skipped_when_idempotent_false(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    n_invocations = [0]
    def counting_impl(query: str) -> dict[str, Any]:
        n_invocations[0] += 1
        return {"echoed": query, "n_chars": len(query)}
    gw.register(_echo_policy(idempotent=False), counting_impl)
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    assert n_invocations[0] == 2  # both invocations actually run


def test_cache_skipped_when_ttl_is_none(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    n_invocations = [0]
    def counting_impl(query: str) -> dict[str, Any]:
        n_invocations[0] += 1
        return {"echoed": query, "n_chars": len(query)}
    gw.register(_echo_policy(cache_ttl_seconds=None), counting_impl)
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    assert n_invocations[0] == 2


def test_oversized_output_is_truncated_and_flagged(conn: psycopg.Connection):
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    big_payload = "x" * 100  # 100 bytes
    def big_impl(query: str) -> dict[str, Any]:
        return {"echoed": big_payload, "n_chars": len(big_payload)}
    gw.register(_echo_policy(max_result_bytes=50), big_impl)
    res = gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    assert res.truncated is True
    with conn.cursor() as cur:
        cur.execute("SELECT output, truncated FROM tool_calls WHERE run_id = %s", (run_id,))
        output, truncated = cur.fetchone()
    assert truncated is True
    assert output["_truncated"] is True
    assert "_byte_size" in output


def test_truncated_results_are_not_returned_by_cache_lookup(conn: psycopg.Connection):
    """Once truncated, the next identical call re-invokes (does not return sentinel)."""
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    big_payload = "x" * 100
    n_invocations = [0]
    def big_impl(query: str) -> dict[str, Any]:
        n_invocations[0] += 1
        return {"echoed": big_payload, "n_chars": len(big_payload)}
    gw.register(_echo_policy(max_result_bytes=50), big_impl)
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    assert n_invocations[0] == 2  # truncated row didn't satisfy second cache lookup


def test_register_overwrites_previous_policy(conn: psycopg.Connection):
    """Re-registering a tool name replaces the old policy + impl (no error)."""
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    gw.register(_echo_policy(), _echo_impl)
    gw.register(_echo_policy(tool_version="2.0.0"), _echo_impl)
    gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    with conn.cursor() as cur:
        cur.execute("SELECT tool_version FROM tool_calls WHERE run_id = %s", (run_id,))
        assert cur.fetchone() == ("2.0.0",)


def test_output_validation_failure_logs_schema_invalid(conn: psycopg.Connection):
    """When the impl returns output that doesn't match output_schema, the failed
    call is logged with error_category='schema_invalid' before the ValidationError
    propagates."""
    run_id = _make_run(conn)
    gw = ToolGateway(conn, run_id)
    def bad_impl(query: str) -> dict[str, Any]:
        return {"unexpected_field": query}  # missing required 'echoed' / 'n_chars'
    gw.register(_echo_policy(), bad_impl)
    with pytest.raises(ValidationError):
        gw.call(AgentRole.RESEARCHER_WIDE, "echo", query="hi")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT error_category, output, cache_hit, truncated "
            "FROM tool_calls WHERE run_id = %s",
            (run_id,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    error_category, output, cache_hit, truncated = rows[0]
    assert error_category == "schema_invalid"
    assert output is None
    assert cache_hit is False
    assert truncated is False
