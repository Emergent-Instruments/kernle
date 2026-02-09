"""Shared helper functions for CLI commands."""

import argparse
import json
import re
from typing import Any


def validate_input(value: str, field_name: str, max_length: int = 1000) -> str:
    """Validate and sanitize CLI inputs."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if len(value) > max_length:
        raise ValueError(f"{field_name} too long (max {max_length} characters)")

    # Remove null bytes and control characters except newlines
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    return sanitized


def print_json(data: Any) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def validate_budget(value: str) -> int:
    """Validate budget argument for token budget."""
    from kernle.core import MAX_TOKEN_BUDGET, MIN_TOKEN_BUDGET

    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Budget must be an integer, got '{value}'")

    if ivalue < MIN_TOKEN_BUDGET:
        raise argparse.ArgumentTypeError(
            f"Budget must be at least {MIN_TOKEN_BUDGET}, got {ivalue}"
        )
    if ivalue > MAX_TOKEN_BUDGET:
        raise argparse.ArgumentTypeError(f"Budget cannot exceed {MAX_TOKEN_BUDGET}, got {ivalue}")
    return ivalue
