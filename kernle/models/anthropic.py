"""AnthropicModel — ModelProtocol implementation for Anthropic's API.

Wraps the ``anthropic`` Python SDK. The SDK is imported lazily so that
the module can be imported without having ``anthropic`` installed (the
import fails only when the class is instantiated).
"""

from __future__ import annotations

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


class AnthropicModelError(KernleError):
    """Raised when the Anthropic SDK reports an error."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class AnthropicModel:
    """ModelProtocol implementation backed by the Anthropic API.

    Requires the ``anthropic`` package::

        pip install anthropic
        # or
        pip install kernle[anthropic]

    Usage::

        model = AnthropicModel()  # uses ANTHROPIC_API_KEY env var
        response = model.generate([ModelMessage(role="user", content="Hello")])
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5-20250929",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicModel. "
                "Install it with: pip install anthropic"
            ) from None

        resolved_key = (
            api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "An API key is required. Pass api_key= or set CLAUDE_API_KEY / ANTHROPIC_API_KEY."
            )

        self._model_id = model_id
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=resolved_key)

    # ---- ModelProtocol properties ----

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self._model_id,
            provider="anthropic",
            context_window=200_000,
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
        """Generate a complete response via the Anthropic messages API."""
        api_messages, extracted_system = self._prepare_messages(messages, system)
        kwargs = self._build_kwargs(
            api_messages,
            extracted_system,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise self._classify_error(exc, "Anthropic API error") from exc

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
        api_messages, extracted_system = self._prepare_messages(messages, system)
        kwargs = self._build_kwargs(
            api_messages,
            extracted_system,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield ModelChunk(content=text)

                # Final chunk with usage from the accumulated message
                final_message = stream.get_final_message()
                usage = {}
                if final_message.usage:
                    usage = {
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                    }
                yield ModelChunk(
                    content="",
                    is_final=True,
                    usage=usage,
                )
        except Exception as exc:
            raise self._classify_error(exc, "Anthropic streaming error") from exc

    # ---- Internal helpers ----

    def _prepare_messages(
        self,
        messages: list[ModelMessage],
        system: Optional[str],
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Convert ModelMessages to Anthropic format, extracting system messages."""
        extracted_system = system
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # Anthropic API uses a top-level system param, not a system role
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if extracted_system:
                    extracted_system = f"{extracted_system}\n\n{content}"
                else:
                    extracted_system = content
                continue

            api_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            api_messages.append(api_msg)

        return api_messages, extracted_system

    def _build_kwargs(
        self,
        api_messages: list[dict[str, Any]],
        system: Optional[str],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for the Anthropic API call."""
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": api_messages,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        return kwargs

    @staticmethod
    def _classify_error(exc: Exception, prefix: str) -> AnthropicModelError:
        """Classify an Anthropic SDK exception into an error class.

        Uses defensive attribute access so this works even when the
        anthropic package is mocked or partially available.
        """
        import anthropic as _anthropic

        _checks: list[tuple[str, str, str]] = [
            ("RateLimitError", "rate_limit", "rate limited"),
            ("AuthenticationError", "auth", "auth failed"),
            ("APITimeoutError", "timeout", "timeout"),
        ]
        for attr, cls, label in _checks:
            exc_type = getattr(_anthropic, attr, None)
            if exc_type is not None and isinstance(exc, exc_type):
                return AnthropicModelError(cls, f"{prefix}: {label}: {exc}")

        api_status = getattr(_anthropic, "APIStatusError", None)
        if api_status is not None and isinstance(exc, api_status):
            code = getattr(exc, "status_code", "?")
            return AnthropicModelError("server", f"{prefix}: API error ({code}): {exc}")

        return AnthropicModelError("unknown", f"{prefix}: {exc}")

    def _parse_response(self, response: Any) -> ModelResponse:
        """Convert an Anthropic response to ModelResponse."""
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return ModelResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=response.stop_reason,
            model_id=response.model,
        )
