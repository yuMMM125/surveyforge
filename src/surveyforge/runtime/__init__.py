"""Runtime Contract Pack (per spec § 2.7).

Run lifecycle, tool gateway, budget enforcement, evidence store, trust boundary,
error classification, observability metadata, and PostgreSQL persistence layer.
"""
from surveyforge.runtime.budget import (
    BUDGET_PER_ROLE,
    BudgetExceeded,
    BudgetManager,
    BudgetSpec,
    OverflowFallback,
    RoleUsage,
)
from surveyforge.runtime.db import (
    ENV_DATABASE_URL,
    get_pool,
    init_db,
    reset_pool,
    transaction,
)
from surveyforge.runtime.errors import ErrorCategory, classify_exception
from surveyforge.runtime.evidence import EvidenceItem, EvidenceStore
from surveyforge.runtime.observability import with_run_metadata
from surveyforge.runtime.runs import (
    IdempotencyConflict,
    Run,
    RunManager,
    RunStatus,
)
from surveyforge.runtime.tool_gateway import (
    TOOL_REGISTRY,
    ToolGateway,
    ToolNotRegistered,
    ToolPolicy,
    ToolResult,
    ToolRoleDenied,
    compute_input_hash,
    sanitize_args,
)
from surveyforge.runtime.trust import extract_untrusted, wrap_untrusted

__all__ = (  # ruff RUF022 isort-sorted (UPPER → PascalCase → lowercase)
    "BUDGET_PER_ROLE",
    "ENV_DATABASE_URL",
    "TOOL_REGISTRY",
    "BudgetExceeded",
    "BudgetManager",
    "BudgetSpec",
    "ErrorCategory",
    "EvidenceItem",
    "EvidenceStore",
    "IdempotencyConflict",
    "OverflowFallback",
    "RoleUsage",
    "Run",
    "RunManager",
    "RunStatus",
    "ToolGateway",
    "ToolNotRegistered",
    "ToolPolicy",
    "ToolResult",
    "ToolRoleDenied",
    "classify_exception",
    "compute_input_hash",
    "extract_untrusted",
    "get_pool",
    "init_db",
    "reset_pool",
    "sanitize_args",
    "transaction",
    "with_run_metadata",
    "wrap_untrusted",
)
