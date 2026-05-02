from unittest.mock import MagicMock, patch

from litweave.llm.observability import (
    LangfuseSettings,
    get_callback_handler,
    is_enabled,
)


def test_settings_loads_from_env(fake_env):
    settings = LangfuseSettings.from_env()
    assert settings.public_key == "fake-pub"
    assert settings.secret_key == "fake-sec"
    assert settings.host == "https://example.test"


def test_is_enabled_true_when_keys_present(fake_env):
    assert is_enabled() is True


def test_is_enabled_false_when_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert is_enabled() is False


def test_get_callback_handler_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert get_callback_handler() is None


def test_get_callback_handler_returns_handler_when_enabled(fake_env):
    with patch("litweave.llm.observability.CallbackHandler") as MockH:
        MockH.return_value = MagicMock(name="handler")
        handler = get_callback_handler()
        assert handler is not None
        MockH.assert_called_once()
