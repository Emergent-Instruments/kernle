"""OllamaModel — ModelProtocol implementation for local Ollama instances.

Uses HTTP requests to the Ollama REST API. No external SDK required
beyond ``requests`` (part of the Python ecosystem, typically available).
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from kernle.protocols import (
    KernleError,
    ModelCapabilities,
    ModelChunk,
    ModelMessage,
    ModelResponse,
    ToolDefinition,
)


class OllamaModelError(KernleError):
    """Raised when the Ollama API reports an error or is unreachable."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class OllamaModel:
    """ModelProtocol implementation backed by a local Ollama instance.

    Requires a running Ollama server (default: ``http://localhost:11434``)
    and the ``requests`` library::

        pip install requests

    Usage::

        model = OllamaModel(model_id="llama3.2:latest")
        response = model.generate([ModelMessage(role="user", content="Hello")])
    """

    def __init__(
        self,
        model_id: str = "llama3.2:latest",
        *,
        base_url: str = "http://localhost:11434",
        context_window: int = 8192,
        timeout: int = 120,
    ) -> None:
        try:
            import requests as _requests  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for OllamaModel. "
                "Install it with: pip install requests"
            ) from None

        self._requests = _requests
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._context_window = context_window
        self._timeout = timeout

    # ---- ModelProtocol properties ----

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self._model_id,
            provider="ollama",
            context_window=self._context_window,
            max_output_tokens=self._context_window,
            supports_tools=False,
            supports_vision=False,
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
        """Generate a complete response via the Ollama chat API."""
        api_messages = self._prepare_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": api_messages,
            "stream": False,
        }
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        if max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = max_tokens

        data = self._post("/api/chat", payload)
        message = data.get("message", {})
        return ModelResponse(
            content=message.get("content", ""),
            usage=self._extract_usage(data),
            stop_reason="stop",
            model_id=data.get("model", self._model_id),
        )

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
        """Stream a response chunk by chunk from the Ollama chat API."""
        api_messages = self._prepare_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": api_messages,
            "stream": True,
        }
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature
        if max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = max_tokens

        url = f"{self._base_url}/api/chat"
        try:
            resp = self._requests.post(url, json=payload, stream=True, timeout=self._timeout)
        except self._requests.ConnectionError as exc:
            raise OllamaModelError(
                "timeout", f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        except self._requests.Timeout as exc:
            raise OllamaModelError(
                "timeout", f"Ollama request timed out after {self._timeout}s: {exc}"
            ) from exc

        if resp.status_code != 200:
            error_class = self._classify_http_status(resp.status_code)
            raise OllamaModelError(
                error_class, f"Ollama returned HTTP {resp.status_code}: {resp.text}"
            )

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            done = data.get("done", False)
            message = data.get("message", {})
            content = message.get("content", "")

            if done:
                yield ModelChunk(
                    content=content,
                    is_final=True,
                    usage=self._extract_usage(data),
                )
            else:
                yield ModelChunk(content=content)

    # ---- Internal helpers ----

    def _prepare_messages(
        self,
        messages: list[ModelMessage],
        system: Optional[str],
    ) -> list[dict[str, Any]]:
        """Convert ModelMessages to Ollama chat format."""
        api_messages: list[dict[str, Any]] = []

        # Prepend explicit system message if provided
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            api_messages.append({"role": msg.role, "content": content})

        return api_messages

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to the Ollama API and return parsed JSON."""
        url = f"{self._base_url}{path}"
        try:
            resp = self._requests.post(url, json=payload, timeout=self._timeout)
        except self._requests.ConnectionError as exc:
            raise OllamaModelError(
                "timeout", f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        except self._requests.Timeout as exc:
            raise OllamaModelError(
                "timeout", f"Ollama request timed out after {self._timeout}s: {exc}"
            ) from exc

        if resp.status_code != 200:
            error_class = self._classify_http_status(resp.status_code)
            raise OllamaModelError(
                error_class, f"Ollama returned HTTP {resp.status_code}: {resp.text}"
            )

        return resp.json()

    @staticmethod
    def _classify_http_status(status_code: int) -> str:
        """Map HTTP status codes to error classes."""
        if status_code == 401:
            return "auth"
        if status_code == 429:
            return "rate_limit"
        if status_code >= 500:
            return "server"
        return "unknown"

    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from an Ollama response."""
        usage: dict[str, int] = {}
        if "prompt_eval_count" in data:
            usage["input_tokens"] = data["prompt_eval_count"]
        if "eval_count" in data:
            usage["output_tokens"] = data["eval_count"]
        return usage
