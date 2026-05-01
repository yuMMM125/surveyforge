"""ToolGateway + ToolPolicy + TOOL_REGISTRY per spec § 2.7.3.

Every external + runtime tool call goes through ToolGateway.call so:
- per-role allowlist matches the Prompt Contract Pack `allowed_tools` declaration
- input args validated against the tool's `input_schema`
- duplicate calls hit the `tool_calls` cache when idempotent + within TTL
- output validated against `output_schema` (catches drifting external APIs)
- result truncated to `max_result_bytes` (logged with a flag; truncated rows
  are not returned by future cache lookups)
- result_trust tagged so downstream prompts can wrap untrusted_content
- every call (hit OR miss OR output-validation-failure) logged to `tool_calls`
  for traceability
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

import psycopg
from pydantic import BaseModel, ConfigDict, ValidationError

from surveyforge.llm.roles import AgentRole
from surveyforge.runtime.errors import ErrorCategory, classify_exception

# Field names whose values must be sanitized out before hashing input args.
# Match by case-insensitive substring — covers credentials passed through args.
SECRET_FIELD_PATTERNS: tuple[str, ...] = (
    "_key", "_token", "_secret", "password", "authorization",
)

def _json_default(obj: Any) -> str:
    """Disambiguate non-primitive values in canonical-JSON serialization.

    Without this, `Path("/a/b")` and the string `"/a/b"` would canonicalize
    identically (both via `str()`), producing false cache-key collisions.
    Real tool input schemas (Task 2) constrain inputs to JSON primitives so
    this fallback rarely fires; it exists to make the W2 placeholder-schema
    behavior collision-safe.
    """
    return f"<{type(obj).__name__}:{obj!s}>"


class ToolNotRegistered(KeyError):
    """ToolGateway.call referenced a tool that wasn't register()-ed."""


class ToolRoleDenied(PermissionError):
    """Role is not in this tool's allowed_roles."""


class ToolPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tool_name: str
    tool_version: str  # bump invalidates cache
    allowed_roles: tuple[AgentRole, ...]
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    # W2: declared for future use; not enforced by ToolGateway.call (Bundle 1c will wrap).
    timeout_seconds: int = 30
    # W2: declared for future use; not enforced by ToolGateway.call (Bundle 1c will wrap).
    max_retries: int = 2
    cache_ttl_seconds: int | None = 3600
    idempotent: bool = True  # False → never cache
    result_trust: Literal["trusted_internal", "untrusted_content"]
    max_result_bytes: int = 5_000_000


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str
    output: BaseModel
    cache_hit: bool
    truncated: bool
    result_trust: Literal["trusted_internal", "untrusted_content"]
    latency_ms: int


def sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Drop fields whose names match any secret pattern, recursively."""
    def _strip(v: Any) -> Any:
        if isinstance(v, dict):
            return {
                k: _strip(val)
                for k, val in v.items()
                if not any(p in k.lower() for p in SECRET_FIELD_PATTERNS)
            }
        if isinstance(v, list):
            return [_strip(x) for x in v]
        return v
    cleaned = _strip(args)
    assert isinstance(cleaned, dict)
    return cleaned


def compute_input_hash(args: dict[str, Any]) -> str:
    """SHA-256 of canonical JSON (sort_keys=True) of sanitized args."""
    sanitized = sanitize_args(args)
    canonical = json.dumps(sanitized, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolGateway:
    """Per-run tool gateway. Constructed once per graph run; logs all calls
    against the run_id so post-hoc analysis can reconstruct what each agent did.
    """

    def __init__(self, conn: psycopg.Connection, run_id: str) -> None:
        self._conn = conn
        self._run_id = run_id
        self._policies: dict[str, ToolPolicy] = {}
        self._impls: dict[str, Callable[..., Any]] = {}

    def register(self, policy: ToolPolicy, impl: Callable[..., Any]) -> None:
        self._policies[policy.tool_name] = policy
        self._impls[policy.tool_name] = impl

    def call(self, role: AgentRole, tool_name: str, **args: Any) -> ToolResult:
        if tool_name not in self._policies:
            raise ToolNotRegistered(tool_name)
        policy = self._policies[tool_name]
        if role not in policy.allowed_roles:
            raise ToolRoleDenied(
                f"role {role.value} cannot call tool {tool_name} "
                f"(allowed: {[r.value for r in policy.allowed_roles]})"
            )

        validated_in = policy.input_schema.model_validate(args)
        input_hash = compute_input_hash(validated_in.model_dump())

        # Cache lookup (only if policy is idempotent and TTL > 0)
        if policy.idempotent and policy.cache_ttl_seconds:
            cached = self._lookup_cache(policy, input_hash)
            if cached is not None:
                output_payload, output_hash = cached
                output = policy.output_schema.model_validate(output_payload)
                tool_call_id = self._record_call(
                    tool_name=tool_name, tool_version=policy.tool_version,
                    agent_role=role, input_hash=input_hash,
                    output=output_payload, output_hash=output_hash,
                    result_trust=policy.result_trust, latency_ms=0,
                    cache_hit=True, truncated=False, error_category=None,
                )
                return ToolResult(
                    tool_call_id=tool_call_id, output=output, cache_hit=True,
                    truncated=False, result_trust=policy.result_trust, latency_ms=0,
                )

        # Miss — invoke the real implementation
        start = time.perf_counter()
        try:
            raw_output = self._impls[tool_name](**validated_in.model_dump())
        except Exception as exc:
            # Log impl failure to tool_calls so retry routing (spec § 2.7.6) has
            # an error_category signal. Re-raise unchanged — the gateway's job
            # is to record + classify, not to decide whether to retry.
            latency_ms = int((time.perf_counter() - start) * 1000)
            category = classify_exception(exc)
            self._record_call(
                tool_name=tool_name, tool_version=policy.tool_version,
                agent_role=role, input_hash=input_hash,
                output=None, output_hash=None,
                result_trust=policy.result_trust, latency_ms=latency_ms,
                cache_hit=False, truncated=False,
                error_category=category.value if category else None,
            )
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)

        try:
            validated_out = policy.output_schema.model_validate(raw_output)
        except ValidationError:
            # Log the failed call so tool_calls reflects every invocation, then
            # propagate. Schema drift in external tools is an operational signal
            # we want preserved for post-hoc analysis.
            self._record_call(
                tool_name=tool_name, tool_version=policy.tool_version,
                agent_role=role, input_hash=input_hash,
                output=None, output_hash=None,
                result_trust=policy.result_trust, latency_ms=latency_ms,
                cache_hit=False, truncated=False,
                error_category=ErrorCategory.SCHEMA_INVALID.value,
            )
            raise
        output_payload = validated_out.model_dump()

        encoded = json.dumps(output_payload, separators=(",", ":"), default=_json_default).encode("utf-8")
        truncated = len(encoded) > policy.max_result_bytes
        if truncated:
            output_payload = {"_truncated": True, "_byte_size": len(encoded)}

        output_hash = hashlib.sha256(
            json.dumps(output_payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
        ).hexdigest()

        tool_call_id = self._record_call(
            tool_name=tool_name, tool_version=policy.tool_version,
            agent_role=role, input_hash=input_hash,
            output=output_payload, output_hash=output_hash,
            result_trust=policy.result_trust, latency_ms=latency_ms,
            cache_hit=False, truncated=truncated, error_category=None,
        )
        return ToolResult(
            tool_call_id=tool_call_id, output=validated_out, cache_hit=False,
            truncated=truncated, result_trust=policy.result_trust, latency_ms=latency_ms,
        )

    def _lookup_cache(
        self, policy: ToolPolicy, input_hash: str,
    ) -> tuple[dict[str, Any], str] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT output, output_hash FROM tool_calls
                   WHERE tool_name = %s
                     AND tool_version = %s
                     AND input_hash = %s
                     AND truncated = FALSE
                     AND error_category IS NULL
                     AND output IS NOT NULL
                     AND created_at >= now() - make_interval(secs => %s)
                   ORDER BY created_at DESC LIMIT 1""",
                (policy.tool_name, policy.tool_version, input_hash, policy.cache_ttl_seconds),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return row[0], row[1]

    def _record_call(
        self,
        *,
        tool_name: str,
        tool_version: str,
        agent_role: AgentRole,
        input_hash: str,
        output: dict[str, Any] | None,
        output_hash: str | None,
        result_trust: Literal["trusted_internal", "untrusted_content"],
        latency_ms: int,
        cache_hit: bool,
        truncated: bool,
        error_category: str | None,
    ) -> str:
        tool_call_id = f"tc_{uuid.uuid4().hex[:12]}"
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO tool_calls (
                       tool_call_id, run_id, tool_name, tool_version,
                       agent_role, input_hash, output, output_hash,
                       result_trust, latency_ms, cache_hit, truncated, error_category
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    tool_call_id, self._run_id, tool_name, tool_version,
                    agent_role.value, input_hash,
                    json.dumps(output) if output is not None else None,
                    output_hash, result_trust, latency_ms,
                    cache_hit, truncated, error_category,
                ),
            )
        return tool_call_id


# ---- TOOL_REGISTRY (W2 ships only metadata; concrete impls register in Task 2) ----

TOOL_REGISTRY: dict[str, ToolPolicy] = {}


def _register(policy: ToolPolicy) -> None:
    TOOL_REGISTRY[policy.tool_name] = policy


class _OpaqueArgs(BaseModel):
    """Placeholder input schema for registry-level metadata.

    Real schemas attach in Task 2 when concrete tool wrappers register via
    `gateway.register(real_policy, real_impl)`. `extra="allow"` accepts any
    keys so the W2 registry can hold policy metadata without a working impl.
    """
    model_config = ConfigDict(extra="allow")


class _OpaqueOutput(BaseModel):
    """Placeholder output schema for registry-level metadata. See `_OpaqueArgs`."""
    model_config = ConfigDict(extra="allow")


_register(ToolPolicy(
    tool_name="arxiv_search", tool_version="0.1.0",
    allowed_roles=(AgentRole.RESEARCHER_WIDE,),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    result_trust="trusted_internal",
))
_register(ToolPolicy(
    tool_name="s2_lookup", tool_version="0.1.0",
    allowed_roles=(AgentRole.RESEARCHER_WIDE,),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    result_trust="trusted_internal",
))
_register(ToolPolicy(
    tool_name="web_search", tool_version="0.1.0",
    allowed_roles=(AgentRole.RESEARCHER_WIDE,),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    result_trust="untrusted_content",
))
_register(ToolPolicy(
    tool_name="pdf_reader", tool_version="0.1.0",
    allowed_roles=(AgentRole.RESEARCHER_DEEP,),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    result_trust="untrusted_content",
))
_register(ToolPolicy(
    tool_name="citation_verifier", tool_version="0.1.0",
    allowed_roles=(AgentRole.RESEARCHER_DEEP, AgentRole.CRITIC_SECTION, AgentRole.CRITIC_FINAL),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    result_trust="trusted_internal",
))
_register(ToolPolicy(
    tool_name="evidence_store_read", tool_version="0.1.0",
    allowed_roles=(
        AgentRole.SYNTHESIZER, AgentRole.WRITER,
        AgentRole.CRITIC_SECTION, AgentRole.CRITIC_FINAL,
        AgentRole.JUDGE_DEFAULT, AgentRole.JUDGE_FINAL,
    ),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    cache_ttl_seconds=None,  # local DB read; no cache layer needed
    result_trust="trusted_internal",
))
_register(ToolPolicy(
    tool_name="evidence_store_write", tool_version="0.1.0",
    allowed_roles=(AgentRole.RESEARCHER_DEEP,),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    idempotent=False,  # each write produces a new row
    cache_ttl_seconds=None,
    result_trust="trusted_internal",
))
_register(ToolPolicy(
    tool_name="metadata_helper", tool_version="0.1.0",
    allowed_roles=tuple(AgentRole),  # available to any role
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    result_trust="trusted_internal",
))
_register(ToolPolicy(
    tool_name="format_helper", tool_version="0.1.0",
    allowed_roles=tuple(AgentRole),
    input_schema=_OpaqueArgs, output_schema=_OpaqueOutput,
    result_trust="trusted_internal",
))
