"""Shared sanitization utilities for MCP layer.

These functions provide input validation and sanitization
for all MCP tools to ensure consistent security handling.
"""

import math
from typing import Any, Dict, List, Optional

from kernle.core.validation import sanitize_string  # noqa: F401 — re-exported


def sanitize_array(
    value: Any, field_name: str, item_max_length: int = 500, max_items: int = 100
) -> List[str]:
    """Sanitize and validate array inputs.

    Args:
        value: The array to sanitize
        field_name: Name of the field for error messages
        item_max_length: Maximum length for each item
        max_items: Maximum number of items allowed

    Returns:
        List of sanitized strings (empty items removed)

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array, got {type(value).__name__}")

    if len(value) > max_items:
        raise ValueError(f"{field_name} too many items (max {max_items}, got {len(value)})")

    if any(item is None for item in value):
        raise ValueError(f"{field_name} must not contain null items")

    sanitized = []
    for i, item in enumerate(value):
        sanitized_item = sanitize_string(
            item, f"{field_name}[{i}]", item_max_length, required=False
        )
        if sanitized_item:  # Only add non-empty items
            sanitized.append(sanitized_item)

    return sanitized


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


def validate_number(
    value: Any,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[float] = None,
) -> float:
    """Validate numeric values.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value if value is None

    Returns:
        Validated number

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required")

    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {type(value).__name__}")

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"{field_name} must be a finite number, got {value}")

    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be <= {max_val}, got {value}")

    return float(value)


def sanitize_source_metadata(
    arguments: Dict[str, Any],
    source_type_values: Optional[List[str]] = None,
    *,
    context_max_length: int = 500,
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
