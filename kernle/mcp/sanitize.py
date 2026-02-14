"""Shared sanitization utilities for MCP layer.

These functions provide input validation and sanitization
for all MCP tools to ensure consistent security handling.

Canonical implementations live in ``kernle.core.validation``.
This module re-exports them and provides backward-compatible aliases.
"""

from typing import Any, Dict, List, Optional

from kernle.core.validation import (
    sanitize_list,  # noqa: F401 — re-exported
    sanitize_number,  # noqa: F401 — re-exported
    sanitize_string,  # noqa: F401 — re-exported
)

# Backward-compatible aliases — existing callers use these names.
sanitize_array = sanitize_list
validate_number = sanitize_number


def validate_enum(
    value: Any,
    field_name: str,
    valid_values: List[str],
    default: Optional[str] = None,
    required: bool = False,
) -> str:
    """Validate enum values.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        valid_values: List of valid enum values
        default: Default value if value is None
        required: If True, value must be provided (no default)

    Returns:
        Validated enum value

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required")

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if value not in valid_values:
        raise ValueError(f"{field_name} must be one of {valid_values}, got '{value}'")

    return value


def sanitize_source_metadata(
    arguments: Dict[str, Any],
    source_type_values: Optional[List[str]] = None,
    *,
    context_max_length: int = 1000,
    coalesce_empty_to_none: bool = True,
    context_tags_item_max_length: int = 100,
    context_tags_max_items: int = 20,
    source_max_length: int = 500,
    derived_from_item_max_length: int = 200,
    derived_from_max_items: int = 20,
) -> Dict[str, Any]:
    """Sanitize shared source/provenance metadata fields for memory writes.

    Args:
        arguments: Incoming tool arguments.
        source_type_values: Allowed values for source_type.
        context_max_length: Maximum length for context.
        coalesce_empty_to_none: Convert empty strings to None for context/source.
        context_tags_item_max_length: Max length per context tag.
        context_tags_max_items: Max number of context tags.
        source_max_length: Maximum length for source.
        derived_from_item_max_length: Max length for each derived_from item.
        derived_from_max_items: Max number of derived_from items.

    Returns:
        Sanitized metadata dictionary.
    """
    sanitized: Dict[str, Any] = {}
    context = sanitize_string(
        arguments.get("context"), "context", context_max_length, required=False
    )
    sanitized["context"] = (context or None) if coalesce_empty_to_none else context
    sanitized["context_tags"] = (
        sanitize_array(
            arguments.get("context_tags"),
            "context_tags",
            context_tags_item_max_length,
            context_tags_max_items,
        )
        or None
    )
    source = sanitize_string(arguments.get("source"), "source", source_max_length, required=False)
    sanitized["source"] = (source or None) if coalesce_empty_to_none else source
    sanitized["derived_from"] = (
        sanitize_array(
            arguments.get("derived_from"),
            "derived_from",
            derived_from_item_max_length,
            derived_from_max_items,
        )
        or None
    )

    source_type = arguments.get("source_type")
    if source_type:
        if source_type_values is None:
            raise ValueError("source_type validation requires source_type_values")
        sanitized["source_type"] = validate_enum(
            source_type, "source_type", source_type_values, None
        )
    else:
        sanitized["source_type"] = None

    return sanitized
