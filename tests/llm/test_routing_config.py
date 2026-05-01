from pathlib import Path

import pytest

from surveyforge.llm.providers import ProviderName
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import LLMRouter, load_routing_yaml


def test_load_routing_yaml_returns_all_roles():
    cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_routing.yaml"
    bindings = load_routing_yaml(cfg_path)
    assert set(bindings) == set(AgentRole)


def test_load_routing_yaml_planner_uses_glm():
    cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_routing.yaml"
    bindings = load_routing_yaml(cfg_path)
    assert bindings[AgentRole.PLANNER].provider == ProviderName.GLM


def test_load_routing_yaml_writer_uses_minimax():
    cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_routing.yaml"
    bindings = load_routing_yaml(cfg_path)
    assert bindings[AgentRole.WRITER].provider == ProviderName.MINIMAX
    assert bindings[AgentRole.WRITER].temperature == 0.3


def test_load_routing_yaml_judge_uses_qwen():
    cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_routing.yaml"
    bindings = load_routing_yaml(cfg_path)
    assert bindings[AgentRole.JUDGE_DEFAULT].provider == ProviderName.QWEN


def test_router_from_yaml(fake_env):
    cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_routing.yaml"
    router = LLMRouter.from_yaml(cfg_path)
    llm = router.get_llm(AgentRole.PLANNER)
    assert llm.model_name == "glm-5.1"


def test_load_routing_yaml_lite_worker_overrides_fc():
    """qwen3coder supports FC even though QWEN provider defaults to FC=False."""
    cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_routing.yaml"
    bindings = load_routing_yaml(cfg_path)
    lite = bindings[AgentRole.LITE_WORKER]
    assert lite.provider == ProviderName.QWEN
    assert lite.model == "qwen3coder"
    assert lite.supports_fc is True
    assert lite.fc_enabled() is True


def test_load_routing_yaml_judge_default_inherits_qwen_fc_false():
    """judge_default inherits provider-level FC=False (no explicit override)."""
    cfg_path = Path(__file__).parent.parent.parent / "config" / "llm_routing.yaml"
    bindings = load_routing_yaml(cfg_path)
    judge = bindings[AgentRole.JUDGE_DEFAULT]
    assert judge.provider == ProviderName.QWEN
    assert judge.supports_fc is None     # not set in yaml
    assert judge.fc_enabled() is False   # falls back to provider default


def test_load_routing_yaml_unknown_role_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_role:\n  provider: glm\n  model: glm-5.1\n")
    with pytest.raises(ValueError, match="Unknown role"):
        load_routing_yaml(bad)


def test_load_routing_yaml_missing_role_raises(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("planner:\n  provider: glm\n  model: glm-5.1\n")
    with pytest.raises(ValueError, match="Missing role bindings"):
        load_routing_yaml(partial)
