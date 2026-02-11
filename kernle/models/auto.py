"""Auto-configure a model from environment variables.

Provides a zero-config way to get an inference model for local
development and CLI usage (e.g. ``kernle process exhaust``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Default models — cheap/fast for dev testing
_PROVIDER_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2:latest",
}


def auto_configure_model() -> Optional[object]:
    """Auto-detect and create a model from environment variables.

    Detection priority (when ``KERNLE_MODEL_PROVIDER`` is not set):
    1. ``CLAUDE_API_KEY`` or ``ANTHROPIC_API_KEY`` → Anthropic
    2. ``OPENAI_API_KEY`` → OpenAI
    3. No key → ``None`` (graceful degradation)

    Environment variables:
        KERNLE_MODEL_PROVIDER: Force a specific provider (anthropic, openai, ollama).
        KERNLE_MODEL: Override the default model name for the chosen provider.
        CLAUDE_API_KEY / ANTHROPIC_API_KEY: Anthropic API key.
        OPENAI_API_KEY: OpenAI API key.

    Returns:
        A ModelProtocol instance, or None if no API keys are available.
    """
    forced_provider = os.environ.get("KERNLE_MODEL_PROVIDER", "").lower().strip()
    model_override = os.environ.get("KERNLE_MODEL", "").strip() or None

    if forced_provider:
        provider = forced_provider
    elif os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"):
        provider = "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
    else:
        return None

    model_id = model_override or _PROVIDER_DEFAULTS.get(provider)

    if provider == "anthropic":
        from kernle.models.anthropic import AnthropicModel

        model = AnthropicModel(model_id=model_id)
        logger.info("Auto-configured AnthropicModel (model=%s)", model_id)
        return model

    if provider == "openai":
        from kernle.models.openai import OpenAIModel

        model = OpenAIModel(model_id=model_id)
        logger.info("Auto-configured OpenAIModel (model=%s)", model_id)
        return model

    if provider == "ollama":
        from kernle.models.ollama import OllamaModel

        model = OllamaModel(model_id=model_id)
        logger.info("Auto-configured OllamaModel (model=%s)", model_id)
        return model

    logger.warning("Unknown model provider '%s', skipping auto-configuration", provider)
    return None
