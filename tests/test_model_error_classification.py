"""Tests for model provider error classification.

Verifies that each provider classifies SDK exceptions into
the correct error_class categories.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kernle.protocols import ModelMessage

# ============================================================================
# OpenAI error classification
# ============================================================================


_OPENAI_FAKE = SimpleNamespace(
    OpenAI=MagicMock(return_value=MagicMock()),
    RateLimitError=type("RateLimitError", (Exception,), {}),
    AuthenticationError=type("AuthenticationError", (Exception,), {}),
    APITimeoutError=type("APITimeoutError", (Exception,), {}),
    APIStatusError=type(
        "APIStatusError",
        (Exception,),
        {
            "__init__": lambda self, msg, status_code=500: (
                super(type(self), self).__init__(msg),
                setattr(self, "status_code", status_code),
            )[-1]
        },
    ),
)


@pytest.fixture
def openai_model():
    """OpenAIModel with a mocked SDK that stays active."""
    with patch.dict("sys.modules", {"openai": _OPENAI_FAKE}):
        from kernle.models.openai import OpenAIModel

        model = OpenAIModel(api_key="test-key")
        yield model, _OPENAI_FAKE


class TestOpenAIErrorClassification:
    """OpenAIModel classifies SDK exceptions into error_class categories."""

    def test_rate_limit_classified(self, openai_model):
        model, fake = openai_model
        exc = fake.RateLimitError("rate limited")
        model._client.chat.completions.create.side_effect = exc
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "rate_limit"

    def test_auth_error_classified(self, openai_model):
        model, fake = openai_model
        exc = fake.AuthenticationError("bad key")
        model._client.chat.completions.create.side_effect = exc
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "auth"

    def test_timeout_classified(self, openai_model):
        model, fake = openai_model
        exc = fake.APITimeoutError("timed out")
        model._client.chat.completions.create.side_effect = exc
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "timeout"

    def test_api_status_error_classified_as_server(self, openai_model):
        model, fake = openai_model
        exc = fake.APIStatusError("server error", status_code=502)
        model._client.chat.completions.create.side_effect = exc
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "server"

    def test_unknown_exception_classified(self, openai_model):
        model, _ = openai_model
        model._client.chat.completions.create.side_effect = RuntimeError("unknown")
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "unknown"

    def test_stream_error_classified(self, openai_model):
        model, _ = openai_model
        model._client.chat.completions.create.side_effect = RuntimeError("stream fail")
        with pytest.raises(Exception) as exc_info:
            list(model.stream([ModelMessage(role="user", content="hi")]))
        assert exc_info.value.error_class == "unknown"


# ============================================================================
# Anthropic error classification
# ============================================================================


_ANTHROPIC_FAKE = SimpleNamespace(
    Anthropic=MagicMock(return_value=MagicMock()),
    RateLimitError=type("RateLimitError", (Exception,), {}),
    AuthenticationError=type("AuthenticationError", (Exception,), {}),
    APITimeoutError=type("APITimeoutError", (Exception,), {}),
    APIStatusError=type(
        "APIStatusError",
        (Exception,),
        {
            "__init__": lambda self, msg, status_code=500: (
                super(type(self), self).__init__(msg),
                setattr(self, "status_code", status_code),
            )[-1]
        },
    ),
)


@pytest.fixture
def anthropic_model():
    """AnthropicModel with a mocked SDK that stays active."""
    with patch.dict("sys.modules", {"anthropic": _ANTHROPIC_FAKE}):
        from kernle.models.anthropic import AnthropicModel

        model = AnthropicModel(api_key="test-key")
        yield model, _ANTHROPIC_FAKE


class TestAnthropicErrorClassification:
    """AnthropicModel classifies SDK exceptions into error_class categories."""

    def test_rate_limit_classified(self, anthropic_model):
        model, fake = anthropic_model
        exc = fake.RateLimitError("rate limited")
        model._client.messages.create.side_effect = exc
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "rate_limit"

    def test_auth_error_classified(self, anthropic_model):
        model, fake = anthropic_model
        exc = fake.AuthenticationError("bad key")
        model._client.messages.create.side_effect = exc
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "auth"

    def test_timeout_classified(self, anthropic_model):
        model, fake = anthropic_model
        exc = fake.APITimeoutError("timed out")
        model._client.messages.create.side_effect = exc
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "timeout"

    def test_unknown_exception_classified(self, anthropic_model):
        model, _ = anthropic_model
        model._client.messages.create.side_effect = RuntimeError("unknown")
        with pytest.raises(Exception) as exc_info:
            model.generate([ModelMessage(role="user", content="hi")])
        assert exc_info.value.error_class == "unknown"

    def test_stream_error_classified(self, anthropic_model):
        model, _ = anthropic_model
        model._client.messages.stream.side_effect = RuntimeError("stream fail")
        with pytest.raises(Exception) as exc_info:
            list(model.stream([ModelMessage(role="user", content="hi")]))
        assert exc_info.value.error_class == "unknown"


# ============================================================================
# Ollama error classification
# ============================================================================


class _MockRequestsModule:
    """Fake requests module with exception classes."""

    class ConnectionError(Exception):
        pass

    class TimeoutError(Exception):  # noqa: A001
        pass

    Timeout = TimeoutError  # alias used by requests

    def post(self, *args, **kwargs):
        pass


class TestOllamaErrorClassification:
    """OllamaModel classifies HTTP errors into error_class categories."""

    def _make_model(self):
        mock_module = _MockRequestsModule()
        with patch.dict("sys.modules", {"requests": mock_module}):
            from kernle.models.ollama import OllamaModel

            model = OllamaModel()
        return model

    def test_connection_error_classified_as_timeout(self):
        model = self._make_model()
        with patch.object(
            model._requests, "post", side_effect=model._requests.ConnectionError("refused")
        ):
            with pytest.raises(Exception) as exc_info:
                model.generate([ModelMessage(role="user", content="hi")])
            assert exc_info.value.error_class == "timeout"

    def test_timeout_classified(self):
        model = self._make_model()
        with patch.object(
            model._requests, "post", side_effect=model._requests.Timeout("timed out")
        ):
            with pytest.raises(Exception) as exc_info:
                model.generate([ModelMessage(role="user", content="hi")])
            assert exc_info.value.error_class == "timeout"

    def test_http_401_classified_as_auth(self):
        model = self._make_model()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        with patch.object(model._requests, "post", return_value=mock_resp):
            with pytest.raises(Exception) as exc_info:
                model.generate([ModelMessage(role="user", content="hi")])
            assert exc_info.value.error_class == "auth"

    def test_http_429_classified_as_rate_limit(self):
        model = self._make_model()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Too Many Requests"
        with patch.object(model._requests, "post", return_value=mock_resp):
            with pytest.raises(Exception) as exc_info:
                model.generate([ModelMessage(role="user", content="hi")])
            assert exc_info.value.error_class == "rate_limit"

    def test_http_500_classified_as_server(self):
        model = self._make_model()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch.object(model._requests, "post", return_value=mock_resp):
            with pytest.raises(Exception) as exc_info:
                model.generate([ModelMessage(role="user", content="hi")])
            assert exc_info.value.error_class == "server"

    def test_http_503_classified_as_server(self):
        model = self._make_model()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        with patch.object(model._requests, "post", return_value=mock_resp):
            with pytest.raises(Exception) as exc_info:
                model.generate([ModelMessage(role="user", content="hi")])
            assert exc_info.value.error_class == "server"

    def test_http_404_classified_as_unknown(self):
        model = self._make_model()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        with patch.object(model._requests, "post", return_value=mock_resp):
            with pytest.raises(Exception) as exc_info:
                model.generate([ModelMessage(role="user", content="hi")])
            assert exc_info.value.error_class == "unknown"

    def test_stream_connection_error_classified(self):
        model = self._make_model()
        with patch.object(
            model._requests, "post", side_effect=model._requests.ConnectionError("refused")
        ):
            with pytest.raises(Exception) as exc_info:
                list(model.stream([ModelMessage(role="user", content="hi")]))
            assert exc_info.value.error_class == "timeout"
