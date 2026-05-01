"""PromptRegistry: load and validate role prompts from src/surveyforge/prompts/.

Per spec § 2.6: each prompt is a Markdown file with YAML front-matter declaring
role / version / schema / allowed_tools / forbidden. The registry validates:

1. front-matter has all required fields
2. front-matter `role:` matches the enum the caller passed
3. `schema:` resolves to a real Pydantic class in surveyforge.schemas
4. `allowed_tools` are all in KNOWN_TOOLS (Task 0 hard-codes; Task 1's
   runtime/tool_gateway.py will replace this with a canonical TOOL_REGISTRY)
5. body has no TODO / FIXME / <placeholder>
6. citation-emitting roles include the shared Citation Rules block via
   `<<citation_rules>>` placeholder, which is substituted at load time
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from surveyforge.llm.roles import AgentRole

# ---- Constants ----

# Task 0 hard-codes the known-tools set so registry tests can run before Task 1
# (runtime/tool_gateway.py) ships. Task 1 will canonicalize via runtime.tool_gateway.TOOL_REGISTRY.
KNOWN_TOOLS: frozenset[str] = frozenset({
    # external (registered by tools/ wrappers in Task 2)
    "arxiv_search",
    "s2_lookup",
    "web_search",
    "pdf_reader",
    "citation_verifier",
    # runtime capabilities (registered by runtime/tool_gateway.py in Task 1)
    "evidence_store_read",
    "evidence_store_write",
    "metadata_helper",
    "format_helper",
})

CITATION_EMITTING_ROLES: frozenset[AgentRole] = frozenset({
    # Researcher-Wide is **NOT** here — Wide is triage-only (CandidatePaper output,
    # no claim/evidence_id emission). Wide uses <<source_integrity_rules>> instead.
    AgentRole.RESEARCHER_DEEP,
    # W3+ adds: SYNTHESIZER, WRITER, CRITIC_SECTION, CRITIC_FINAL, JUDGE_DEFAULT, JUDGE_FINAL
})

# Schema name → module path under surveyforge.schemas (allows registry to resolve
# `schema:` front-matter values to actual Pydantic classes).
SCHEMA_MODULE: dict[str, str] = {
    "PlannerOutput": "surveyforge.schemas.planner",
    "PlannerSection": "surveyforge.schemas.planner",
    "ResearcherWideOutput": "surveyforge.schemas.research",
    "ResearcherDeepOutput": "surveyforge.schemas.research",
    "CandidatePaper": "surveyforge.schemas.research",
    "EvidenceCard": "surveyforge.schemas.research",
    "Citation": "surveyforge.schemas.citations",
    "EvidenceRef": "surveyforge.schemas.citations",
}

PLACEHOLDER_MARKERS = ("TODO", "FIXME", "<placeholder>")
REQUIRED_FRONT_MATTER = ("role", "version", "schema", "allowed_tools", "forbidden")

# Generic shared-include syntax: `<<name>>` → contents of `shared/<name>.md`.
SHARED_INCLUDE_RE = re.compile(r"<<([a-z_][a-z0-9_]*)>>")


class PromptContractError(ValueError):
    """Raised when a prompt file violates the Prompt Contract."""


@dataclass(frozen=True)
class PromptTemplate:
    """A loaded, validated prompt ready for runtime substitution."""

    role: AgentRole
    version: str
    schema_name: str
    schema_class: type[BaseModel]
    allowed_tools: tuple[str, ...]
    forbidden: tuple[str, ...]
    body: str

    def format(self, **kwargs: Any) -> str:
        return self.body.format(**kwargs)


class PromptRegistry:
    """Loads + validates prompts from src/surveyforge/prompts/."""

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or Path(__file__).parent
        self._cache: dict[AgentRole, PromptTemplate] = {}

    def load(self, role: AgentRole) -> PromptTemplate:
        if role in self._cache:
            return self._cache[role]

        path = self._dir / f"{role.value}.md"
        if not path.exists():
            raise PromptContractError(
                f"No prompt file for role {role.value!r} (looked at {path})"
            )

        text = path.read_text(encoding="utf-8")
        front_matter_str, body = self._split_front_matter(text, role)
        meta = yaml.safe_load(front_matter_str) or {}

        for required in REQUIRED_FRONT_MATTER:
            if required not in meta:
                raise PromptContractError(
                    f"{role.value}: front-matter missing field {required!r}"
                )
        if meta["role"] != role.value:
            raise PromptContractError(
                f"{role.value}: front-matter declares role={meta['role']!r}, "
                f"expected {role.value!r}"
            )

        schema_name = meta["schema"]
        if schema_name not in SCHEMA_MODULE:
            raise PromptContractError(
                f"{role.value}: schema {schema_name!r} not in SCHEMA_MODULE "
                f"(known: {sorted(SCHEMA_MODULE)})"
            )
        schema_class = getattr(
            importlib.import_module(SCHEMA_MODULE[schema_name]), schema_name
        )

        unknown = set(meta["allowed_tools"]) - KNOWN_TOOLS
        if unknown:
            raise PromptContractError(
                f"{role.value}: allowed_tools contains unknown tools {sorted(unknown)}; "
                f"known: {sorted(KNOWN_TOOLS)}"
            )

        for match in list(SHARED_INCLUDE_RE.finditer(body)):
            name = match.group(1)
            shared_path = self._dir / "shared" / f"{name}.md"
            if not shared_path.exists():
                raise PromptContractError(
                    f"{role.value}: body uses <<{name}>> but {shared_path} is missing"
                )
            content = shared_path.read_text(encoding="utf-8").strip()
            body = body.replace(f"<<{name}>>", content)

        for marker in PLACEHOLDER_MARKERS:
            if marker in body:
                raise PromptContractError(
                    f"{role.value}: body contains placeholder marker {marker!r}"
                )
        if SHARED_INCLUDE_RE.search(body):
            raise PromptContractError(
                f"{role.value}: body still has unsubstituted <<...>> include after processing"
            )

        if role in CITATION_EMITTING_ROLES and "Citation Rules" not in body:
            raise PromptContractError(
                f"{role.value}: citation-emitting role missing 'Citation Rules' content "
                f"(use <<citation_rules>> to include shared/citation_rules.md)"
            )

        template = PromptTemplate(
            role=role,
            version=meta["version"],
            schema_name=schema_name,
            schema_class=schema_class,
            allowed_tools=tuple(meta["allowed_tools"]),
            forbidden=tuple(meta["forbidden"]),
            body=body,
        )
        self._cache[role] = template
        return template

    def _split_front_matter(self, text: str, role: AgentRole) -> tuple[str, str]:
        """Parse Markdown YAML front-matter form: ``---\\n<yaml>\\n---\\n<body>``."""
        if not text.startswith("---\n"):
            raise PromptContractError(
                f"{role.value}: expected YAML front-matter starting with `---\\n`"
            )
        try:
            _, front_matter, body = text.split("---\n", 2)
        except ValueError as exc:
            raise PromptContractError(
                f"{role.value}: malformed front-matter (need three `---` delimiters)"
            ) from exc
        return front_matter, body
