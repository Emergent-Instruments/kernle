"""OpenAIModel — ModelProtocol implementation for OpenAI's API.

Wraps the ``openai`` Python SDK. The SDK is imported lazily so that
the module can be imported without having ``openai`` installed (the
import fails only when the class is instantiated).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterator, Optional

from kernle.protocols import (
    KernleError,
    ModelCapabilities,
    ModelChunk,
    ModelMessage,
    ModelResponse,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class OpenAIModelError(KernleError):
    """Raised when the OpenAI SDK reports an error."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class OpenAIModel:
    """ModelProtocol implementation backed by the OpenAI API.

    Requires the ``openai`` package::

        pip install openai
        # or
        pip install kernle[openai]

    Usage::

        model = OpenAIModel()  # uses OPENAI_API_KEY env var
        response = model.generate([ModelMessage(role="user", content="Hello")])
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import openai as _openai  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIModel. "
                "Install it with: pip install openai"
            ) from None

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("An API key is required. Pass api_key= or set OPENAI_API_KEY.")

        self._model_id = model_id
        self._max_tokens = max_tokens
        self._client = _openai.OpenAI(api_key=resolved_key)

    # ---- ModelProtocol properties ----

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self._model_id,
            provider="openai",
            context_window=128_000,
            max_output_tokens=self._max_tokens,
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
        )

    # ---- Generate ----

    def generate(
        self,
        messages: list[ModelMessage],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> ModelResponse:
        """Generate a complete response via the OpenAI chat completions API."""
        api_messages = self._prepare_messages(messages, system)
        kwargs = self._build_kwargs(
            api_messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.debug("OpenAI API generate failed: %s", exc, exc_info=True)
            raise self._classify_error(exc, "OpenAI API error") from exc

        return self._parse_response(response)

    # ---- Stream ----

    def stream(
        self,
        messages: list[ModelMessage],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> Iterator[ModelChunk]:
        """Stream a response chunk by chunk."""
        api_messages = self._prepare_messages(messages, system)
        kwargs = self._build_kwargs(
            api_messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if not chunk.choices:
                    # Final chunk with usage only
                    usage = {}
                    if chunk.usage:
                        usage = {
                            "input_tokens": chunk.usage.prompt_tokens,
                            "output_tokens": chunk.usage.completion_tokens,
                        }
                    yield ModelChunk(content="", is_final=True, usage=usage)
                    continue

                delta = chunk.choices[0].delta
                content = delta.content or ""

                if chunk.choices[0].finish_reason:
                    usage = {}
                    if chunk.usage:
                        usage = {
                            "input_tokens": chunk.usage.prompt_tokens,
                            "output_tokens": chunk.usage.completion_tokens,
                        }
                    yield ModelChunk(content=content, is_final=True, usage=usage)
                else:
                    yield ModelChunk(content=content)
        except Exception as exc:
            logger.debug("OpenAI API stream failed: %s", exc, exc_info=True)
            raise self._classify_error(exc, "OpenAI streaming error") from exc

    # ---- Internal helpers ----

    def _prepare_messages(
        self,
        messages: list[ModelMessage],
        system: Optional[str],
    ) -> list[dict[str, Any]]:
        """Convert ModelMessages to OpenAI chat format."""
        api_messages: list[dict[str, Any]] = []

        # Prepend explicit system param as a system message
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else msg.content
            api_msg: dict[str, Any] = {
                "role": msg.role,
                "content": content,
            }
            if msg.tool_call_id:
                api_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                api_msg["tool_calls"] = msg.tool_calls
            api_messages.append(api_msg)

        return api_messages

    def _build_kwargs(
        self,
        api_messages: list[dict[str, Any]],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for the OpenAI API call."""
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": api_messages,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        return kwargs

    def _parse_response(self, response: Any) -> ModelResponse:
        """Convert an OpenAI response to ModelResponse."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls: list[dict[str, Any]] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                parsed_input = self._parse_tool_call_input(tc.function.arguments)
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": parsed_input,
                    }
                )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=choice.finish_reason,
            model_id=response.model,
        )

    @staticmethod
    def _classify_error(exc: Exception, prefix: str) -> OpenAIModelError:
        """Classify an OpenAI SDK exception into an error class.

        Uses defensive attribute access so this works even when the
        openai package is mocked or partially available.
        """
        try:
            import openai as _openai
        except (ImportError, ModuleNotFoundError):
            return OpenAIModelError("unknown", f"{prefix}: {exc}")

        _checks: list[tuple[str, str, str]] = [
            ("RateLimitError", "rate_limit", "rate limited"),
            ("AuthenticationError", "auth", "auth failed"),
            ("APITimeoutError", "timeout", "timeout"),
        ]
        for attr, cls, label in _checks:
            exc_type = getattr(_openai, attr, None)
            if exc_type is not None and isinstance(exc, exc_type):
                return OpenAIModelError(cls, f"{prefix}: {label}: {exc}")

        api_status = getattr(_openai, "APIStatusError", None)
        if api_status is not None and isinstance(exc, api_status):
            code = getattr(exc, "status_code", "?")
            return OpenAIModelError("server", f"{prefix}: API error ({code}): {exc}")

        return OpenAIModelError("unknown", f"{prefix}: {exc}")

    def _parse_tool_call_input(self, arguments: Any) -> dict[str, Any]:
        """Best-effort parse for tool-call arguments.

        Some model responses contain malformed JSON or non-object JSON values.
        We return a dict in all cases so downstream tool handlers can continue.
        """
        if isinstance(arguments, dict):
            return arguments

        if not isinstance(arguments, str):
            return {"_raw": str(arguments), "_parse_error": "arguments_not_string"}

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw": arguments, "_parse_error": "invalid_json"}

        if isinstance(parsed, dict):
            return parsed

        return {"_value": parsed, "_parse_error": "arguments_not_object"}
