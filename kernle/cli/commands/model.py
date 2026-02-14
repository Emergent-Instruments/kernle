"""Model binding commands for Kernle CLI."""

import json as _json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)

# boot_config keys for persisted model binding
_KEY_PROVIDER = "model_provider"
_KEY_MODEL_ID = "model_id"


def load_persisted_model(k: "Kernle") -> Optional[object]:
    """Load a model from boot_config if one is persisted.

    Returns a ModelProtocol instance, or None if no model is configured
    or if instantiation fails.
    """
    provider = k.boot_get(_KEY_PROVIDER)
    model_id = k.boot_get(_KEY_MODEL_ID)
    if not provider:
        return None

    try:
        if provider == "anthropic":
            from kernle.models.anthropic import AnthropicModel

            return AnthropicModel(model_id=model_id) if model_id else AnthropicModel()

        if provider == "openai":
            from kernle.models.openai import OpenAIModel

            return OpenAIModel(model_id=model_id) if model_id else OpenAIModel()

        if provider == "ollama":
            from kernle.models.ollama import OllamaModel

            return OllamaModel(model_id=model_id) if model_id else OllamaModel()
    except (ValueError, ImportError) as e:
        logger.debug(
            "Failed to load persisted model (%s/%s): %s", provider, model_id, e, exc_info=True
        )
        return None

    logger.debug("Unknown persisted model provider: %s", provider)
    return None


def cmd_model(args, k: "Kernle"):
    """Handle model subcommands."""
    action = args.model_action

    if action == "show":
        # Try in-memory model first, then persisted config
        model = k.entity.model
        if model is None:
            model = load_persisted_model(k)
            if model is not None:
                k.entity.set_model(model)

        if model is None:
            if getattr(args, "json", False):
                print(_json.dumps({"bound": False}))
            else:
                print("Model: (none)")
            return

        caps = model.capabilities
        info = {
            "bound": True,
            "provider": caps.provider,
            "model_id": model.model_id,
        }

        if getattr(args, "json", False):
            print(_json.dumps(info, indent=2))
        else:
            print(f"Model: {caps.provider} / {model.model_id}")

    elif action == "set":
        provider = args.provider
        model_id = getattr(args, "model_id", None)

        from kernle.models.auto import _PROVIDER_DEFAULTS

        # Normalize provider name for storage
        storage_provider = "anthropic" if provider == "claude" else provider
        resolved_model_id = model_id or _PROVIDER_DEFAULTS.get(storage_provider)

        if provider == "claude":
            from kernle.models.anthropic import AnthropicModel

            try:
                model = AnthropicModel(model_id=resolved_model_id)
            except (ValueError, ImportError) as e:
                print(f"Error: {e}")
                return
            k.entity.set_model(model)

        elif provider == "openai":
            try:
                from kernle.models.openai import OpenAIModel

                model = OpenAIModel(model_id=resolved_model_id)
            except (ValueError, ImportError) as e:
                print(f"Error: {e}")
                return
            k.entity.set_model(model)

        elif provider == "ollama":
            try:
                from kernle.models.ollama import OllamaModel

                model = OllamaModel(model_id=resolved_model_id)
            except (ValueError, ImportError) as e:
                print(f"Error: {e}")
                return
            k.entity.set_model(model)

        else:
            print(f"Unknown provider: {provider}")
            return

        # Persist to boot_config
        k.boot_set(_KEY_PROVIDER, storage_provider)
        k.boot_set(_KEY_MODEL_ID, resolved_model_id)
        print(f"Bound model: {storage_provider} / {resolved_model_id}")

    elif action == "clear":
        had_model = k.entity.model is not None
        had_config = k.boot_get(_KEY_PROVIDER) is not None

        if not had_model and not had_config:
            print("No model bound.")
            return

        k.entity.set_model(None)
        k.boot_delete(_KEY_PROVIDER)
        k.boot_delete(_KEY_MODEL_ID)
        print("Model unbound.")

    else:
        print("Usage: kernle model {show|set|clear}")
