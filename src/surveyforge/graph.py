"""W2 SurveyForge graph: linear pipeline wire-up.

START -> planner -> researcher_wide -> researcher_deep -> synthesize -> write -> END

Per Task 1 Architecture Decision #11, the graph's checkpointer is
`langgraph.checkpoint.postgres.PostgresSaver` pointed at the same Postgres
instance as `RunManager` (`SURVEYFORGE_DATABASE_URL`). LangGraph manages its
own checkpoint tables, separate from our spec § 2.7.2 7-table schema.
`thread_id == run_id` (spec § 2.7.2 contract).

PostgresSaver requires `autocommit=True` + `prepare_threshold=0` +
`row_factory=dict_row` on the underlying psycopg connection (per
langgraph-checkpoint-postgres docs: checkpoint writes use savepoints +
INSERT-or-UPDATE patterns that conflict with implicit transactions and
cached prepared plans; reads do dict-subscript access). We therefore CANNOT
reuse `runtime.db.get_pool()` (non-autocommit by design — `init_db` needs
the multi-DDL transaction). A dedicated module-level pool with the right
kwargs is constructed lazily on first call.

W2 is intentionally linear — no conditional edges. W5 adds Critic retry edges;
W6 adds Judge edges; W7 adds per-section parallelism.
"""
from __future__ import annotations

import atexit
import os
from typing import Any, cast

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from surveyforge.agents import (
    make_planner_node,
    make_researcher_deep_node,
    make_researcher_wide_node,
)
from surveyforge.llm.rate_limit import RateLimitConfig, RateLimitedRouter
from surveyforge.llm.router import RouterProtocol, load_routing_yaml
from surveyforge.prompts.loader import PromptRegistry
from surveyforge.runtime.budget import BudgetManager
from surveyforge.runtime.db import ENV_DATABASE_URL
from surveyforge.state import SurveyState
from surveyforge.synthesis.stub import make_synthesize_stub_node
from surveyforge.writing.stub import make_write_stub_node

# Dedicated autocommit pool for langgraph PostgresSaver — separate from
# `runtime.db._pool` because that one is non-autocommit (required for
# `init_db`'s multi-statement DDL transaction). atexit closes on interpreter
# exit so long-lived processes (CLI, server) don't leak. Typed against
# `psycopg.Connection[DictRow]` because PostgresSaver requires `dict_row`
# rows (see kwargs in `_make_postgres_checkpointer`).
_checkpointer_pool: ConnectionPool[psycopg.Connection[DictRow]] | None = None


def build_graph(
    *,
    router: RouterProtocol | None = None,
    registry: PromptRegistry | None = None,
    budget_manager: BudgetManager | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    routing_yaml_path: str | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build + compile the W2 graph.

    All deps optional with sensible env-driven defaults; tests inject mocks
    via the kwargs to bypass real LLM/DB. Pattern matches Task 3/4/5 factory
    closures (deps captured at compile time, not at node-invocation time).
    """
    if router is None:
        # NOTE: actual file is `config/llm_routing.yaml` (Plan #1 deliverable);
        # do NOT change to `config/routing.yaml` — it does not exist.
        path = routing_yaml_path or os.environ.get(
            "SURVEYFORGE_ROUTING_YAML", "config/llm_routing.yaml"
        )
        bindings = load_routing_yaml(path)
        router = RateLimitedRouter(bindings, RateLimitConfig())
    if registry is None:
        registry = PromptRegistry()
    if budget_manager is None:
        budget_manager = BudgetManager()
    if checkpointer is None:
        checkpointer = _make_postgres_checkpointer()

    g: StateGraph[SurveyState, Any, Any, Any] = StateGraph(SurveyState)
    # Cast factory closures to `Any` for `add_node`: `Callable[[SurveyState,
    # RunnableConfig], SurveyState]` is structurally a `_NodeWithConfig`
    # protocol, but mypy cannot match a positional-arg Callable to that
    # Protocol in strict mode. The runtime contract is enforced at the
    # factory level (see PlannerNode etc. type aliases).
    g.add_node("planner", cast(Any, make_planner_node(router, registry)))
    g.add_node(
        "researcher_wide",
        cast(Any, make_researcher_wide_node(router, registry, budget_manager)),
    )
    g.add_node(
        "researcher_deep",
        cast(Any, make_researcher_deep_node(router, registry, budget_manager)),
    )
    g.add_node("synthesize", cast(Any, make_synthesize_stub_node()))
    g.add_node("write", cast(Any, make_write_stub_node()))

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher_wide")
    g.add_edge("researcher_wide", "researcher_deep")
    g.add_edge("researcher_deep", "synthesize")
    g.add_edge("synthesize", "write")
    g.add_edge("write", END)

    return g.compile(checkpointer=checkpointer)


def _make_postgres_checkpointer() -> BaseCheckpointSaver[Any]:
    """Build PostgresSaver against `SURVEYFORGE_DATABASE_URL`.

    Constructs a dedicated autocommit ConnectionPool (NOT runtime.db's pool —
    that one is non-autocommit by design). Calls `setup()` to create
    langgraph's checkpoint tables idempotently. Pool registered for atexit
    close so long-lived processes (CLI, server) don't leak.

    NOTE: do NOT use `PostgresSaver.from_conn_string(url)` here — that
    classmethod is a `@contextmanager` and returns a generator-based context
    manager, not a `PostgresSaver`. Calling `.setup()` on it raises
    `AttributeError`. We construct the saver via `PostgresSaver(pool)` with
    a properly-configured pool instead.
    """
    global _checkpointer_pool
    url = os.environ.get(ENV_DATABASE_URL)
    if not url:
        raise RuntimeError(
            f"{ENV_DATABASE_URL} not set; required for PostgresSaver checkpointer. "
            "For local dev: `docker compose up -d postgres` + export the URL."
        )
    if _checkpointer_pool is None:
        # PostgresSaver requires three psycopg connection kwargs:
        #   - autocommit=True: its writes use savepoints + INSERT-or-UPDATE
        #     patterns that conflict with implicit transactions
        #   - prepare_threshold=0: some statements vary parameter shape, which
        #     breaks psycopg's automatic prepared-plan cache
        #   - row_factory=dict_row: PostgresSaver internally calls
        #     `cur.fetchone()["..."]` (dict subscript), NOT `cur.fetchone()[idx]`
        #     (tuple index). With the default tuple-row factory, every read
        #     raises `TypeError: tuple indices must be integers, not str`.
        _checkpointer_pool = ConnectionPool(
            url,
            min_size=1,
            max_size=4,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            open=True,
        )
        atexit.register(_checkpointer_pool.close)
    saver = PostgresSaver(_checkpointer_pool)
    saver.setup()  # idempotent — CREATE TABLE IF NOT EXISTS + version migration
    return saver


def _reset_checkpointer_pool_for_tests() -> None:
    """Test-only helper: close + null the module-level pool so a new test can
    point `SURVEYFORGE_DATABASE_URL` at a different testcontainer URL without
    inheriting a stale connection. Production code must NOT call this."""
    global _checkpointer_pool
    if _checkpointer_pool is not None:
        _checkpointer_pool.close()
        _checkpointer_pool = None
