"""Tests for auto_configure_model() — env-var based model detection.

All tests work without actual API access — model constructors are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _clear_env(monkeypatch):
    """Clear all model-related env vars."""
    for var in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "OPENAI_API_KEY",
        "KERNLE_MODEL_PROVIDER",
        "KERNLE_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


class TestAutoConfigureModel:
    """Tests for kernle.models.auto.auto_configure_model()."""

    def test_no_keys_returns_none(self, monkeypatch):
        _clear_env(monkeypatch)
        from kernle.models.auto import auto_configure_model

        assert auto_configure_model() is None

    def test_anthropic_api_key_detected(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_model = MagicMock()
        with patch("kernle.models.anthropic.AnthropicModel", return_value=mock_model) as mock_cls:
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
        mock_cls.assert_called_once_with(model_id="claude-haiku-4-5-20251001")

    def test_claude_api_key_detected(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-claude")

        mock_model = MagicMock()
        with patch("kernle.models.anthropic.AnthropicModel", return_value=mock_model) as mock_cls:
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
        mock_cls.assert_called_once_with(model_id="claude-haiku-4-5-20251001")

    def test_openai_api_key_detected(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        mock_model = MagicMock()
        with patch("kernle.models.openai.OpenAIModel", return_value=mock_model) as mock_cls:
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
        mock_cls.assert_called_once_with(model_id="gpt-4o-mini")

    def test_anthropic_takes_priority_over_openai(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        mock_model = MagicMock()
        with patch("kernle.models.anthropic.AnthropicModel", return_value=mock_model) as mock_cls:
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
        mock_cls.assert_called_once_with(model_id="claude-haiku-4-5-20251001")

    def test_provider_override_forces_openai(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("KERNLE_MODEL_PROVIDER", "openai")

        mock_model = MagicMock()
        with patch("kernle.models.openai.OpenAIModel", return_value=mock_model) as mock_cls:
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
        mock_cls.assert_called_once_with(model_id="gpt-4o-mini")

    def test_provider_override_forces_ollama(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("KERNLE_MODEL_PROVIDER", "ollama")

        mock_model = MagicMock()
        with patch("kernle.models.ollama.OllamaModel", return_value=mock_model) as mock_cls:
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
        mock_cls.assert_called_once_with(model_id="llama3.2:latest")

    def test_model_name_override(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("KERNLE_MODEL", "gpt-4o")

        mock_model = MagicMock()
        with patch("kernle.models.openai.OpenAIModel", return_value=mock_model) as mock_cls:
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
        mock_cls.assert_called_once_with(model_id="gpt-4o")

    def test_unknown_provider_returns_none(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("KERNLE_MODEL_PROVIDER", "unknown_provider")

        from kernle.models.auto import auto_configure_model

        assert auto_configure_model() is None

    def test_provider_case_insensitive(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("KERNLE_MODEL_PROVIDER", "OpenAI")

        mock_model = MagicMock()
        with patch("kernle.models.openai.OpenAIModel", return_value=mock_model):
            from kernle.models.auto import auto_configure_model

            result = auto_configure_model()

        assert result is mock_model
