"""CLI argparse + subcommand dispatch tests.

Direct `cli.main(argv=...)` invocation (NOT subprocess). Allows monkeypatching
`build_graph` to a mock; subprocess would force real DB + LLMs which belong
in Task 7's live integration test."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import psycopg
import pytest

from surveyforge import cli
from surveyforge.runtime.runs import RunManager


def _make_run(conn: psycopg.Connection, idempotency_key: str | None = None) -> str:
    rm = RunManager(conn)
    return rm.create(
        topic="test",
        idempotency_key=idempotency_key or f"key-{time.perf_counter_ns()}",
    ).run_id


# ---- run subcommand ----

def test_cli_run_creates_run_and_invokes_graph(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, capsys,
):
    """`surveyforge run --topic X` creates a run + invokes graph + exits 0 + prints run_id."""
    patch_agent_transaction("surveyforge.cli")

    # Mock build_graph to return a graph whose .invoke returns a fake state.
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {
        "topic": "test topic",
        "section_drafts": {"S1": "## Background\n\n- claim [E-1]"},
    }
    monkeypatch.setattr(cli, "build_graph", MagicMock(return_value=fake_graph))

    rc = cli.main(["run", "--topic", "test topic", "--idempotency-key", "test-cli-1"])
    assert rc == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "run_id: run_" in out
    assert "sections: ['S1']" in out
    # AD #13 (Task 7): success path must print the draft body, not just
    # the section keys. Without this, the spec § 8 "viewable section draft"
    # deliverable is not met.
    assert "[section_id: S1]" in out
    assert "## Background" in out
    assert "claim [E-1]" in out


def test_cli_run_idempotency_conflict_returns_usage_error(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, capsys,
):
    """Re-using an idempotency_key surfaces the existing run_id + exit 3."""
    patch_agent_transaction("surveyforge.cli")
    existing_run_id = _make_run(conn, idempotency_key="dup-key")

    monkeypatch.setattr(cli, "build_graph", MagicMock())  # not invoked

    rc = cli.main(["run", "--topic", "different topic", "--idempotency-key", "dup-key"])
    assert rc == cli.EXIT_USAGE

    err = capsys.readouterr().err
    assert "dup-key" in err
    assert existing_run_id in err


def test_cli_run_graph_invoke_raises_returns_failed(
    conn: psycopg.Connection, monkeypatch, patch_agent_transaction, capsys,
):
    patch_agent_transaction("surveyforge.cli")
    fake_graph = MagicMock()
    fake_graph.invoke.side_effect = RuntimeError("simulated graph failure")
    monkeypatch.setattr(cli, "build_graph", MagicMock(return_value=fake_graph))

    rc = cli.main(["run", "--topic", "x", "--idempotency-key", "fail-key-1"])
    assert rc == cli.EXIT_FAILED
    assert "simulated graph failure" in capsys.readouterr().err


# ---- status subcommand ----

def test_cli_status_returns_run_metadata(
    conn: psycopg.Connection, patch_agent_transaction, capsys,
):
    patch_agent_transaction("surveyforge.cli")
    run_id = _make_run(conn)

    rc = cli.main(["status", run_id])
    assert rc == cli.EXIT_OK

    out = capsys.readouterr().out
    assert f"run_id: {run_id}" in out
    assert "status: PENDING" in out


def test_cli_status_unknown_run_returns_usage_error(
    conn: psycopg.Connection, patch_agent_transaction, capsys,
):
    patch_agent_transaction("surveyforge.cli")
    rc = cli.main(["status", "run_does_not_exist"])
    assert rc == cli.EXIT_USAGE
    assert "run not found" in capsys.readouterr().err


# ---- W3+ stub subcommands ----

@pytest.mark.parametrize("stub_cmd", ["resume", "cancel", "export"])
def test_cli_stub_subcommands_print_deferred_message(stub_cmd: str, capsys):
    rc = cli.main([stub_cmd, "run_some_id"])
    assert rc == cli.EXIT_USAGE
    assert "deferred to W3+" in capsys.readouterr().err
