"""SurveyForge CLI — `surveyforge run / status` (resume / cancel / export = W3+ stubs).

Per Task 6 Architecture Decision #5: exit codes 0/1/2/3 (success/failed/cancelled/
usage). `IdempotencyConflict` on `run --idempotency-key K` -> exit 3 with the
existing run_id surfaced (caller can re-use that run_id for `status`).
"""
from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from surveyforge.graph import build_graph
from surveyforge.runtime.db import transaction
from surveyforge.runtime.runs import IdempotencyConflict, RunManager
from surveyforge.state import make_initial_state

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CANCELLED = 2
EXIT_USAGE = 3


def main(argv: list[str] | None = None) -> int:
    """SurveyForge CLI entry point.

    Calls `load_dotenv()` at startup so the user can put SJTU_MODELS_API_KEY,
    LANGFUSE_*, SURVEYFORGE_DATABASE_URL etc. in `.env` instead of exporting
    them every shell session. NOTE: this dotenv load is process-local — it
    does NOT export to the parent shell, and it does NOT propagate to other
    Python processes (e.g., the schema-init one-liner in Task 7 Step 7.0b(c)
    has its own `load_dotenv()`).
    """
    load_dotenv()  # auto-load .env so users don't need to export vars every session
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args.topic, args.idempotency_key)
    if args.cmd == "status":
        return _cmd_status(args.run_id)
    if args.cmd in {"resume", "cancel", "export"}:
        print(f"{args.cmd}: deferred to W3+; not implemented in W2", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_USAGE  # unreachable; argparse `required=True` blocks empty cmd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="surveyforge",
        description="Multi-agent academic survey generator (W2 minimal CLI)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Start a new survey run")
    run.add_argument("--topic", required=True, help="Survey topic")
    run.add_argument(
        "--idempotency-key", default=None,
        help="Idempotency key (auto-generated if omitted)",
    )

    status = sub.add_parser("status", help="Show run status")
    status.add_argument("run_id", help="Run id (e.g., run_abc123def456)")

    for stub_cmd in ("resume", "cancel", "export"):
        s = sub.add_parser(stub_cmd, help=f"({stub_cmd} — deferred to W3+)")
        s.add_argument("run_id", nargs="?")

    return p


def _cmd_run(topic: str, idempotency_key: str | None) -> int:
    if not idempotency_key:
        idempotency_key = f"cli-{uuid.uuid4().hex[:12]}"

    with transaction() as conn:
        rm = RunManager(conn)
        try:
            run = rm.create(topic=topic, idempotency_key=idempotency_key)
        except IdempotencyConflict as exc:
            print(
                f"idempotency_key {exc.idempotency_key!r} already used by run "
                f"{exc.existing_run_id!r}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    initial_state = make_initial_state(topic=topic)
    config: RunnableConfig = {"configurable": {"thread_id": run.run_id}}

    try:
        graph = build_graph()
        result = graph.invoke(initial_state, config=config)
    except Exception as exc:
        # AD #12 (W2-minimal): all graph-invoke exceptions are lumped into
        # `schema_invalid` to keep CLI error handling small. Task 7 will refine
        # this to call `classify_exception(exc)` so transport errors / 429 / 5xx
        # get distinct error_category values.
        with transaction() as conn:
            RunManager(conn).fail(run.run_id, error_category="schema_invalid")
        print(f"run {run.run_id} failed: {exc!s}", file=sys.stderr)
        return EXIT_FAILED

    with transaction() as conn:
        RunManager(conn).succeed(run.run_id)

    print(f"run_id: {run.run_id}")
    drafts = result.get("section_drafts", {})
    print(f"sections: {sorted(drafts.keys())}")
    # AD #13: show each draft body so the deliverable "viewable section draft"
    # is satisfied literally. Drafts can be long; readers can pipe to a pager
    # or redirect to a file. Empty drafts (sections with no evidence) still
    # print the header + "_No evidence available..._" fallback line.
    for section_id in sorted(drafts.keys()):
        print("---")
        print(f"[section_id: {section_id}]")
        print()
        print(drafts[section_id])
    return EXIT_OK


def _cmd_status(run_id: str) -> int:
    try:
        with transaction() as conn:
            run = RunManager(conn).get(run_id)
    except KeyError:
        print(f"run not found: {run_id}", file=sys.stderr)
        return EXIT_USAGE

    print(f"run_id: {run.run_id}")
    print(f"status: {run.status.value}")
    print(f"current_stage: {run.current_stage}")
    if run.error_category:
        print(f"error_category: {run.error_category}")
    print(f"created_at: {run.created_at.isoformat()}")
    print(f"updated_at: {run.updated_at.isoformat()}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
