"""Shared helper functions for CLI commands."""

import argparse
import json
from functools import partial
from typing import Any

from kernle.core.validation import sanitize_string

# Thin wrapper keeping the existing call contract (required=True, no `required` kwarg)
validate_input = partial(sanitize_string, required=True)


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
