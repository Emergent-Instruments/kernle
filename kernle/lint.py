"""Memory lint — reject malformed or low-signal beliefs and values before commit.

The pipeline can produce truncated/low-quality statements: value names that read
like partial sentences, belief statements that are templated noise like
"Value: Worked on ...", and other artifacts. Once stored, they contaminate
retrieval forever.

This module provides configurable lint rules that run before save_belief and
save_value. On failure, the memory is redirected to a MemorySuggestion with
resolution_reason indicating the lint failure, instead of being committed
directly. This keeps the agent's structured memory clean while preserving
the content for manual review.

Lint rules are configurable via stack settings (key: "lint_rules") as a
JSON object. Individual rules can be enabled/disabled and thresholds tuned.

Example lint_rules setting:
    {
        "min_length": 10,
        "max_length": 2000,
        "check_fragments": true,
        "check_prefix_artifacts": true,
        "check_templated_noise": true,
        "enabled": true
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default lint configuration
DEFAULT_LINT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_length": 10,
    "max_length": 2000,
    "check_fragments": True,
    "check_prefix_artifacts": True,
    "check_templated_noise": True,
}

# Prefix artifacts: patterns that look like the LLM echoed a field label
# into the content itself. Case-insensitive matching.
PREFIX_ARTIFACT_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(value|belief|goal|note|episode|statement|name)\s*:\s*", re.IGNORECASE),
    re.compile(r"^(title|description|objective|outcome)\s*:\s*", re.IGNORECASE),
]

# Templated noise: content that looks like boilerplate or placeholder text.
TEMPLATED_NOISE_PATTERNS: List[re.Pattern] = [
    # "Worked on ...", "Updated ...", "Fixed ..." with no substance
    re.compile(
        r"^(worked on|updated|fixed|changed|modified|edited|added|removed)\s+\.{2,}", re.IGNORECASE
    ),
    # Placeholder ellipsis content
    re.compile(r"^\.\.\.$"),
    re.compile(r"^…$"),
    # Content that is just "TODO" or "TBD" variants
    re.compile(r"^(todo|tbd|fixme|xxx|placeholder|n/a|none|null|undefined)$", re.IGNORECASE),
    # Repeated single character
    re.compile(r"^(.)\1{4,}$"),
    # Content that is only whitespace and punctuation
    re.compile(r"^[\s\W]+$"),
]

# Fragment indicators: signs of truncated or incomplete text
FRAGMENT_ENDINGS: List[str] = [
    " the",
    " a",
    " an",
    " and",
    " or",
    " but",
    " with",
    " for",
    " to",
    " in",
    " on",
    " at",
    " by",
    " of",
    " is",
    " was",
    " are",
    " were",
    " that",
    " which",
    " when",
    " if",
    " as",
]


# =============================================================================
# Result Type
# =============================================================================


@dataclass
class LintResult:
    """Result of running lint on a memory.

    Attributes:
        passed: Whether the content passed all lint checks.
        failures: List of human-readable failure reason strings.
        rule_name: The name of the first failing rule (for audit log).
    """

    passed: bool
    failures: List[str] = field(default_factory=list)

    @property
    def rule_name(self) -> Optional[str]:
        """Name of the first failing rule, or None if passed."""
        if self.passed or not self.failures:
            return None
        return self.failures[0].split(":")[0].strip().lower().replace(" ", "_")

    @property
    def summary(self) -> str:
        """One-line summary for audit log."""
        if self.passed:
            return "lint_passed"
        return "; ".join(self.failures)


# =============================================================================
# Lint Functions
# =============================================================================


def get_lint_config(settings_getter=None) -> Dict[str, Any]:
    """Get lint configuration from stack settings or use defaults.

    Args:
        settings_getter: Callable that takes a key and returns Optional[str].
                        Typically stack.get_stack_setting.

    Returns:
        Merged lint configuration dict.
    """
    config = dict(DEFAULT_LINT_CONFIG)
    if settings_getter:
        raw = settings_getter("lint_rules")
        if raw:
            try:
                overrides = json.loads(raw)
                if isinstance(overrides, dict):
                    config.update(overrides)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Invalid lint_rules setting, using defaults")
    return config


def lint_text(text: str, config: Optional[Dict[str, Any]] = None) -> LintResult:
    """Run lint rules against a text string.

    This is the core lint function. It checks the text against all enabled
    rules and returns a LintResult.

    Args:
        text: The text to lint (e.g., belief statement or value name).
        config: Lint configuration dict. Uses DEFAULT_LINT_CONFIG if None.

    Returns:
        LintResult with passed=True if all checks pass.
    """
    if config is None:
        config = DEFAULT_LINT_CONFIG

    if not config.get("enabled", True):
        return LintResult(passed=True)

    failures: List[str] = []

    # Strip for analysis but check original length
    stripped = text.strip() if text else ""

    # Rule 1: Minimum length
    min_length = config.get("min_length", 10)
    if len(stripped) < min_length:
        failures.append(f"too_short: content is {len(stripped)} chars, minimum is {min_length}")

    # Rule 2: Maximum length
    max_length = config.get("max_length", 2000)
    if len(stripped) > max_length:
        failures.append(f"too_long: content is {len(stripped)} chars, maximum is {max_length}")

    # Rule 3: Prefix artifacts
    if config.get("check_prefix_artifacts", True):
        for pattern in PREFIX_ARTIFACT_PATTERNS:
            if pattern.match(stripped):
                failures.append(
                    f"prefix_artifact: content starts with field label pattern '{pattern.pattern}'"
                )
                break

    # Rule 4: Templated noise
    if config.get("check_templated_noise", True):
        for pattern in TEMPLATED_NOISE_PATTERNS:
            if pattern.match(stripped):
                failures.append(
                    f"templated_noise: content matches noise pattern '{pattern.pattern}'"
                )
                break

    # Rule 5: Fragment detection (trailing incomplete phrases)
    if config.get("check_fragments", True):
        lower_stripped = stripped.lower()
        for ending in FRAGMENT_ENDINGS:
            if lower_stripped.endswith(ending):
                failures.append(
                    f"trailing_fragment: content ends with incomplete phrase '{ending.strip()}'"
                )
                break

    return LintResult(passed=len(failures) == 0, failures=failures)


def lint_belief(statement: str, config: Optional[Dict[str, Any]] = None) -> LintResult:
    """Lint a belief statement.

    Args:
        statement: The belief's statement text.
        config: Lint configuration dict.

    Returns:
        LintResult.
    """
    return lint_text(statement, config)


def lint_value(name: str, statement: str, config: Optional[Dict[str, Any]] = None) -> LintResult:
    """Lint a value's name and statement.

    Both the name and statement are checked. The name has a reduced minimum
    length (3 chars) since value names are typically short labels.

    Args:
        name: The value's name.
        statement: The value's statement text.
        config: Lint configuration dict.

    Returns:
        LintResult combining failures from both name and statement checks.
    """
    if config is None:
        config = dict(DEFAULT_LINT_CONFIG)

    failures: List[str] = []

    if not config.get("enabled", True):
        return LintResult(passed=True)

    # Lint the name with a lower min length threshold
    name_config = dict(config)
    name_config["min_length"] = max(3, config.get("min_length", 10) // 3)
    name_result = lint_text(name, name_config)
    for f in name_result.failures:
        failures.append(f"name: {f}")

    # Lint the statement with full config
    statement_result = lint_text(statement, config)
    for f in statement_result.failures:
        failures.append(f"statement: {f}")

    return LintResult(passed=len(failures) == 0, failures=failures)
