"""PromptRegistry tests per spec § 2.6.5."""
from __future__ import annotations

from pathlib import Path

import pytest

from surveyforge.llm.roles import AgentRole
from surveyforge.prompts.loader import (
    KNOWN_TOOLS,
    PromptContractError,
    PromptRegistry,
    PromptTemplate,
)
from surveyforge.schemas.planner import PlannerOutput
from surveyforge.schemas.research import ResearcherDeepOutput, ResearcherWideOutput

W2_ROLES = (AgentRole.PLANNER, AgentRole.RESEARCHER_WIDE, AgentRole.RESEARCHER_DEEP)


@pytest.fixture
def registry() -> PromptRegistry:
    return PromptRegistry()


# ---- W2 role coverage ----

@pytest.mark.parametrize("role", W2_ROLES)
def test_w2_role_loads_successfully(registry: PromptRegistry, role: AgentRole):
    template = registry.load(role)
    assert isinstance(template, PromptTemplate)
    assert template.role == role


# ---- Front-matter validation ----

def test_planner_front_matter(registry: PromptRegistry):
    p = registry.load(AgentRole.PLANNER)
    assert p.version == "0.1.0"
    assert p.schema_name == "PlannerOutput"
    assert p.schema_class is PlannerOutput
    assert p.allowed_tools == ()  # Planner has no tools (tuple, not list)


def test_researcher_wide_front_matter(registry: PromptRegistry):
    rw = registry.load(AgentRole.RESEARCHER_WIDE)
    assert rw.schema_class is ResearcherWideOutput
    assert set(rw.allowed_tools) == {"arxiv_search", "s2_lookup", "web_search"}


def test_researcher_deep_front_matter(registry: PromptRegistry):
    rd = registry.load(AgentRole.RESEARCHER_DEEP)
    assert rd.schema_class is ResearcherDeepOutput
    assert "pdf_reader" in rd.allowed_tools
    assert "citation_verifier" in rd.allowed_tools
    assert "evidence_store_write" in rd.allowed_tools


# ---- Tool allowlist validation ----

@pytest.mark.parametrize("role", W2_ROLES)
def test_allowed_tools_in_known_registry(registry: PromptRegistry, role: AgentRole):
    template = registry.load(role)
    unknown = set(template.allowed_tools) - KNOWN_TOOLS
    assert not unknown, f"{role.value} declares unknown tools: {unknown}"


# ---- Body validation (no placeholders) ----

@pytest.mark.parametrize("role", W2_ROLES)
def test_no_placeholder_marker_in_body(registry: PromptRegistry, role: AgentRole):
    body = registry.load(role).body
    for marker in ("TODO", "FIXME", "<placeholder>"):
        assert marker not in body, f"{role.value} body contains marker {marker!r}"


# ---- Shared rule inclusion ----

def test_researcher_wide_uses_source_integrity_rules_not_citation_rules(registry: PromptRegistry):
    """Wide is triage-only — must use Source Integrity Rules (smaller scope), NOT Citation Rules."""
    body = registry.load(AgentRole.RESEARCHER_WIDE).body
    assert "Source Integrity Rules" in body
    assert "<<source_integrity_rules>>" not in body  # include must be substituted
    assert "Citation Rules" not in body  # Wide does NOT emit evidence_id-bearing claims


def test_researcher_deep_includes_citation_rules(registry: PromptRegistry):
    body = registry.load(AgentRole.RESEARCHER_DEEP).body
    assert "Citation Rules" in body
    assert "<<citation_rules>>" not in body


def test_planner_does_not_require_citation_rules(registry: PromptRegistry):
    """Planner emits no citations, so it must NOT have any <<...>> placeholder."""
    body = registry.load(AgentRole.PLANNER).body
    assert "<<" not in body  # no shared include placeholders


def test_researcher_wide_not_in_citation_emitting_roles(registry: PromptRegistry):
    """Contract: Wide is triage-only, so the loader must NOT enforce Citation Rules on it."""
    from surveyforge.prompts.loader import CITATION_EMITTING_ROLES
    assert AgentRole.RESEARCHER_WIDE not in CITATION_EMITTING_ROLES
    assert AgentRole.RESEARCHER_DEEP in CITATION_EMITTING_ROLES


# ---- Caching ----

def test_load_caches_template(registry: PromptRegistry):
    a = registry.load(AgentRole.PLANNER)
    b = registry.load(AgentRole.PLANNER)
    assert a is b


# ---- Format substitution ----

def test_planner_format_substitutes_topic(registry: PromptRegistry):
    p = registry.load(AgentRole.PLANNER)
    out = p.format(topic="Survey of RLHF progress")
    assert "Survey of RLHF progress" in out
    assert "{topic}" not in out


def test_researcher_wide_format_substitutes_section_context(registry: PromptRegistry):
    rw = registry.load(AgentRole.RESEARCHER_WIDE)
    out = rw.format(
        section_id="S1",
        title="Methods",
        research_questions=["What are the main RLHF methods?"],
        must_find_evidence=["DPO (Rafailov 2023)"],
    )
    assert "S1" in out
    assert "Methods" in out
    assert "DPO" in out
    assert "{section_id}" not in out


# ---- Error paths ----

def test_unknown_role_raises(registry: PromptRegistry):
    """SYNTHESIZER prompt isn't created in W2 — should raise."""
    with pytest.raises(PromptContractError, match="No prompt file"):
        registry.load(AgentRole.SYNTHESIZER)


def test_malformed_front_matter_raises(tmp_path: Path):
    bad = tmp_path / "planner.md"
    bad.write_text("no front-matter here\nbody only\n", encoding="utf-8")
    reg = PromptRegistry(prompts_dir=tmp_path)
    with pytest.raises(PromptContractError, match="front-matter"):
        reg.load(AgentRole.PLANNER)


def test_unknown_tool_in_allowed_tools_raises(tmp_path: Path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "citation_rules.md").write_text("rules", encoding="utf-8")
    bad = tmp_path / "planner.md"
    bad.write_text(
        "---\n"
        "role: planner\n"
        "version: 0.1.0\n"
        "schema: PlannerOutput\n"
        "allowed_tools: [made_up_tool]\n"
        "forbidden: []\n"
        "---\n"
        "Body without placeholder markers.\n",
        encoding="utf-8",
    )
    reg = PromptRegistry(prompts_dir=tmp_path)
    with pytest.raises(PromptContractError, match="unknown tools"):
        reg.load(AgentRole.PLANNER)


def test_role_mismatch_in_front_matter_raises(tmp_path: Path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "citation_rules.md").write_text("rules", encoding="utf-8")
    bad = tmp_path / "planner.md"
    bad.write_text(
        "---\n"
        "role: researcher_wide\n"  # wrong role
        "version: 0.1.0\n"
        "schema: PlannerOutput\n"
        "allowed_tools: []\n"
        "forbidden: []\n"
        "---\n"
        "Body.\n",
        encoding="utf-8",
    )
    reg = PromptRegistry(prompts_dir=tmp_path)
    with pytest.raises(PromptContractError, match="declares role"):
        reg.load(AgentRole.PLANNER)
