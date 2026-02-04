"""Shared sanitization utilities for MCP layer.

These functions provide input validation and sanitization
for all MCP tools to ensure consistent security handling.
"""

import re
from typing import Any, List, Optional


def sanitize_string(
    value: Any, field_name: str, max_length: int = 1000, required: bool = True
) -> str:
    """Sanitize and validate string inputs at MCP layer.

    Args:
        value: The value to sanitize
        field_name: Name of the field for error messages
        max_length: Maximum allowed string length
        required: If True, empty strings are rejected

    Returns:
        Sanitized string

    Raises:
        ValueError: If validation fails
    """
    if value is None and not required:
        return ""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if required and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")

    if len(value) > max_length:
        raise ValueError(f"{field_name} too long (max {max_length} characters, got {len(value)})")

    # Remove null bytes and control characters except newlines and tabs
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    return sanitized


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

    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be <= {max_val}, got {value}")

    return float(value)
