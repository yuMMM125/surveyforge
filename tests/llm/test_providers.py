import pytest

from litweave.llm.providers import (
    DEFAULT_MODEL_GATEWAY_URL,
    LEGACY_SJTU_MODEL_API_KEY_ENV,
    MODEL_API_KEY_ENV,
    MODEL_BASE_URL_ENV,
    PROVIDERS,
    ProviderName,
    build_chat_model,
)


def test_all_four_providers_registered():
    assert {ProviderName.DEEPSEEK, ProviderName.GLM, ProviderName.MINIMAX, ProviderName.QWEN} <= set(PROVIDERS)


def test_all_providers_share_default_gateway():
    """All configured model families default to the same OpenAI-compatible endpoint."""
    for name, cfg in PROVIDERS.items():
        assert cfg.base_url == DEFAULT_MODEL_GATEWAY_URL, f"{name} base_url drifted from gateway"
        assert cfg.api_key_env == MODEL_API_KEY_ENV, f"{name} api_key_env drifted"
        assert LEGACY_SJTU_MODEL_API_KEY_ENV in cfg.api_key_env_aliases


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


def test_context_windows_match_configured_claims():
    """Claimed values from the configured model gateway."""
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
    monkeypatch.delenv(MODEL_API_KEY_ENV, raising=False)
    monkeypatch.delenv(LEGACY_SJTU_MODEL_API_KEY_ENV, raising=False)
    with pytest.raises(KeyError, match=MODEL_API_KEY_ENV):
        build_chat_model(ProviderName.DEEPSEEK)


def test_build_chat_model_accepts_legacy_sjtu_key(monkeypatch):
    monkeypatch.delenv(MODEL_API_KEY_ENV, raising=False)
    monkeypatch.setenv(LEGACY_SJTU_MODEL_API_KEY_ENV, "fake-legacy-sjtu")
    llm = build_chat_model(ProviderName.DEEPSEEK)
    assert llm.model_name == "deepseek-chat"


def test_build_chat_model_accepts_base_url_override(fake_env, monkeypatch):
    monkeypatch.setenv(MODEL_BASE_URL_ENV, "https://models.example.test/v1")
    llm = build_chat_model(ProviderName.DEEPSEEK)
    assert str(llm.openai_api_base) == "https://models.example.test/v1"


def test_build_chat_model_ignores_empty_base_url_override(fake_env, monkeypatch):
    monkeypatch.setenv(MODEL_BASE_URL_ENV, "")
    llm = build_chat_model(ProviderName.DEEPSEEK)
    assert str(llm.openai_api_base) == DEFAULT_MODEL_GATEWAY_URL
