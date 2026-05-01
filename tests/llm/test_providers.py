import pytest

from surveyforge.llm.providers import PROVIDERS, ProviderName, build_chat_model

SJTU_GATEWAY = "https://models.sjtu.edu.cn/api/v1"


def test_all_four_providers_registered():
    assert {ProviderName.DEEPSEEK, ProviderName.GLM, ProviderName.MINIMAX, ProviderName.QWEN} <= set(PROVIDERS)


def test_all_providers_share_sjtu_gateway():
    """Spike (2026-05-01): all 4 model families served via the same SJTU endpoint."""
    for name, cfg in PROVIDERS.items():
        assert cfg.base_url == SJTU_GATEWAY, f"{name} base_url drifted from gateway"
        assert cfg.api_key_env == "SJTU_MODELS_API_KEY", f"{name} api_key_env drifted"


def test_provider_config_has_required_fields():
    for name, cfg in PROVIDERS.items():
        assert cfg.base_url.startswith("https://"), f"{name} missing https base_url"
        assert cfg.api_key_env, f"{name} missing api_key_env"
        assert cfg.context_window > 0
        assert isinstance(cfg.supports_function_calling, bool)


def test_qwen_provider_has_fc_disabled_by_default():
    """Spike: Qwen aliases (`qwen`, `qwen3.5-27b`) return HTTP 400 on bind_tools."""
    assert PROVIDERS[ProviderName.QWEN].supports_function_calling is False


def test_other_providers_have_fc_enabled():
    for name in (ProviderName.DEEPSEEK, ProviderName.GLM, ProviderName.MINIMAX):
        assert PROVIDERS[name].supports_function_calling is True, f"{name} should default FC=True"


def test_context_windows_match_spec_claimed():
    """Claimed values; W2 will stress-test."""
    assert PROVIDERS[ProviderName.DEEPSEEK].context_window == 32_000
    assert PROVIDERS[ProviderName.GLM].context_window == 128_000
    assert PROVIDERS[ProviderName.MINIMAX].context_window == 192_000
    assert PROVIDERS[ProviderName.QWEN].context_window == 256_000


def test_default_models_match_spike():
    assert PROVIDERS[ProviderName.DEEPSEEK].default_model == "deepseek-chat"
    assert PROVIDERS[ProviderName.GLM].default_model == "glm-5.1"
    assert PROVIDERS[ProviderName.MINIMAX].default_model == "minimax"
    assert PROVIDERS[ProviderName.QWEN].default_model == "qwen"


def test_build_chat_model_returns_chat_openai(fake_env):
    from langchain_openai import ChatOpenAI
    llm = build_chat_model(ProviderName.DEEPSEEK)  # uses default_model
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "deepseek-chat"


def test_build_chat_model_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SJTU_MODELS_API_KEY", raising=False)
    with pytest.raises(KeyError, match="SJTU_MODELS_API_KEY"):
        build_chat_model(ProviderName.DEEPSEEK)
