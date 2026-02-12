"""Tests for kernle.models.openai.OpenAIModel."""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kernle.models.openai import OpenAIModel, OpenAIModelError
from kernle.protocols import ModelMessage, ToolDefinition


def _response(*, content="hello", tool_calls=None, model="gpt-4o-mini"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        model=model,
    )


def _chunk(*, content="", finish_reason=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
        usage=usage,
    )


def _usage_only_chunk(prompt_tokens=5, completion_tokens=3):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _install_fake_openai(client):
    fake_openai = SimpleNamespace(OpenAI=MagicMock(return_value=client))
    return patch.dict("sys.modules", {"openai": fake_openai})


class TestOpenAIModel:
    def test_init_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = MagicMock()
        with _install_fake_openai(client):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIModel()

    def test_init_uses_env_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = MagicMock()
        with _install_fake_openai(client):
            model = OpenAIModel(model_id="gpt-4o-mini")
        assert model.model_id == "gpt-4o-mini"

    def test_init_raises_when_openai_package_missing(self):
        original_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("no module named openai")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            with pytest.raises(ImportError, match="pip install openai"):
                OpenAIModel(api_key="test-key")

    def test_generate_returns_parsed_response_with_tool_calls(self):
        tool_call = SimpleNamespace(
            id="tool-1",
            function=SimpleNamespace(name="lookup_user", arguments=json.dumps({"user_id": 42})),
        )
        client = MagicMock()
        client.chat.completions.create.return_value = _response(
            content="done", tool_calls=[tool_call]
        )

        with _install_fake_openai(client):
            model = OpenAIModel(api_key="test-key")
            result = model.generate([ModelMessage(role="user", content="hello")])

        assert result.content == "done"
        assert result.tool_calls == [
            {"id": "tool-1", "name": "lookup_user", "input": {"user_id": 42}}
        ]
        assert result.usage == {"input_tokens": 11, "output_tokens": 7}
        assert result.stop_reason == "stop"

    def test_generate_wraps_openai_errors(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        with _install_fake_openai(client):
            model = OpenAIModel(api_key="test-key")
            with pytest.raises(OpenAIModelError, match="OpenAI API error"):
                model.generate([ModelMessage(role="user", content="hello")])

    def test_stream_yields_text_chunks_and_final_usage(self):
        stream_chunks = [
            _chunk(content="hel"),
            _chunk(
                content="lo",
                finish_reason="stop",
                usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4),
            ),
            _usage_only_chunk(prompt_tokens=9, completion_tokens=4),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(stream_chunks)

        with _install_fake_openai(client):
            model = OpenAIModel(api_key="test-key")
            chunks = list(model.stream([ModelMessage(role="user", content="hello")]))

        assert chunks[0].content == "hel"
        assert chunks[0].is_final is False
        assert chunks[1].is_final is True
        assert chunks[1].usage == {"input_tokens": 9, "output_tokens": 4}
        assert chunks[2].is_final is True
        assert chunks[2].usage == {"input_tokens": 9, "output_tokens": 4}

    def test_stream_wraps_openai_errors(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("stream failed")

        with _install_fake_openai(client):
            model = OpenAIModel(api_key="test-key")
            with pytest.raises(OpenAIModelError, match="OpenAI streaming error"):
                list(model.stream([ModelMessage(role="user", content="hello")]))

    def test_prepare_messages_handles_system_and_tool_fields(self):
        client = MagicMock()
        with _install_fake_openai(client):
            model = OpenAIModel(api_key="test-key")
            messages = model._prepare_messages(
                [
                    ModelMessage(role="user", content="Q1"),
                    ModelMessage(
                        role="assistant",
                        content="Calling tool",
                        tool_calls=[{"id": "tc1", "type": "function"}],
                    ),
                    ModelMessage(role="tool", content="tool output", tool_call_id="tc1"),
                ],
                system="You are helpful.",
            )

        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[2]["tool_calls"] == [{"id": "tc1", "type": "function"}]
        assert messages[3]["tool_call_id"] == "tc1"

    def test_build_kwargs_maps_tools_temperature_and_limits(self):
        client = MagicMock()
        with _install_fake_openai(client):
            model = OpenAIModel(api_key="test-key", max_tokens=999)
            kwargs = model._build_kwargs(
                [{"role": "user", "content": "hello"}],
                tools=[
                    ToolDefinition(
                        name="lookup_user",
                        description="Lookup a user by id",
                        input_schema={
                            "type": "object",
                            "properties": {"user_id": {"type": "number"}},
                        },
                    )
                ],
                temperature=0.3,
                max_tokens=123,
            )

        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["max_tokens"] == 123
        assert kwargs["temperature"] == 0.3
        assert kwargs["tools"][0]["function"]["name"] == "lookup_user"
