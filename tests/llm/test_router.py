import pytest
from langchain_openai import ChatOpenAI

from surveyforge.llm.providers import ProviderName
from surveyforge.llm.roles import AgentRole
from surveyforge.llm.router import LLMRouter, RoleBinding


@pytest.fixture
def basic_bindings() -> dict[AgentRole, RoleBinding]:
    return {
        AgentRole.PLANNER: RoleBinding(provider=ProviderName.GLM, model="glm-5.1"),
        AgentRole.RESEARCHER_WIDE: RoleBinding(
            provider=ProviderName.DEEPSEEK, model="deepseek-chat"
        ),
        AgentRole.WRITER: RoleBinding(provider=ProviderName.MINIMAX, model="minimax"),
    }


def test_router_returns_chat_openai_for_role(fake_env, basic_bindings):
    router = LLMRouter(bindings=basic_bindings)
    llm = router.get_llm(AgentRole.PLANNER)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "glm-5.1"


def test_router_unknown_role_raises(fake_env, basic_bindings):
    router = LLMRouter(bindings=basic_bindings)
    with pytest.raises(KeyError, match="critic_section"):
        router.get_llm(AgentRole.CRITIC_SECTION)


def test_router_override_model(fake_env, basic_bindings):
    router = LLMRouter(bindings=basic_bindings)
    llm = router.get_llm(AgentRole.PLANNER, model_override="glm-5-flash")
    assert llm.model_name == "glm-5-flash"


def test_router_passes_temperature(fake_env, basic_bindings):
    router = LLMRouter(bindings=basic_bindings)
    llm = router.get_llm(AgentRole.PLANNER, temperature=0.7)
    assert llm.temperature == 0.7


def test_router_caches_instances(fake_env, basic_bindings):
    router = LLMRouter(bindings=basic_bindings)
    a = router.get_llm(AgentRole.PLANNER)
    b = router.get_llm(AgentRole.PLANNER)
    assert a is b


def test_router_no_cache_when_overriding(fake_env, basic_bindings):
    router = LLMRouter(bindings=basic_bindings)
    a = router.get_llm(AgentRole.PLANNER)
    b = router.get_llm(AgentRole.PLANNER, temperature=0.7)
    assert a is not b


def test_router_override_does_not_evict_cached_default(fake_env, basic_bindings):
    """Transient overrides must not touch the cache — base instance is preserved."""
    router = LLMRouter(bindings=basic_bindings)
    a = router.get_llm(AgentRole.PLANNER)
    _transient = router.get_llm(AgentRole.PLANNER, temperature=0.7)
    c = router.get_llm(AgentRole.PLANNER)
    assert a is c, "cached default must survive transient override calls"
    assert a is not _transient


def test_role_binding_fc_falls_back_to_provider_default():
    """When supports_fc is None, fc_enabled() returns the provider-level default."""
    qwen_binding = RoleBinding(provider=ProviderName.QWEN, model="qwen")
    assert qwen_binding.fc_enabled() is False  # PROVIDERS[QWEN].supports_function_calling = False

    deepseek_binding = RoleBinding(provider=ProviderName.DEEPSEEK, model="deepseek-chat")
    assert deepseek_binding.fc_enabled() is True


def test_role_binding_fc_per_binding_override_wins():
    """qwen3coder supports FC even though its provider (QWEN) defaults to False."""
    lite = RoleBinding(provider=ProviderName.QWEN, model="qwen3coder", supports_fc=True)
    assert lite.fc_enabled() is True

    forced_off = RoleBinding(provider=ProviderName.GLM, model="glm-5.1", supports_fc=False)
    assert forced_off.fc_enabled() is False
