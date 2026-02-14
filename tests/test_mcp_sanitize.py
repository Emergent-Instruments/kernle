"""Focused tests for MCP argument sanitization helpers."""

import pytest

from kernle.mcp.handlers.memory import validate_memory_auto_capture
from kernle.mcp.sanitize import sanitize_array, sanitize_source_metadata, validate_number


def test_sanitize_array_none_returns_empty():
    """None input is coerced to an empty array."""
    assert sanitize_array(None, "values") == []


def test_sanitize_array_requires_array_and_enforces_limits():
    """sanitize_array rejects non-lists and enforces max-items."""
    with pytest.raises(ValueError, match="must be an array"):
        sanitize_array("not-a-list", "values")

    with pytest.raises(ValueError, match="too many items"):
        sanitize_array(["one", "two", "three"], "values", max_items=2)


def test_sanitize_array_item_constraints():
    """Item-level checks for max length and empty-item filtering are enforced."""
    assert sanitize_array(["", "valid", "a\x00b", "", "x"], "items", item_max_length=5) == [
        "valid",
        "ab",
        "x",
    ]

    with pytest.raises(ValueError, match="must be a string"):
        sanitize_array(["ok", 7], "items")

    with pytest.raises(ValueError, match="too long"):
        sanitize_array(["abcd"], "items", item_max_length=3)


def test_sanitize_source_metadata_coalesces_empty_strings_to_none():
    """Optional provenance fields coalesce empty values to None when enabled."""
    metadata = sanitize_source_metadata(
        {
            "context": "",
            "source": "\n",
            "context_tags": ["", "tag"],
            "derived_from": ["", "ref-1"],
            "source_type": "",
        },
        source_type_values=["direct_experience", "inference", "external", "consolidation"],
    )

    assert metadata["context"] is None
    assert metadata["source"] == "\n"
    assert metadata["context_tags"] == ["tag"]
    assert metadata["derived_from"] == ["ref-1"]
    assert metadata["source_type"] is None


def test_sanitize_source_metadata_invalid_source_type_errors():
    """Invalid or missing source-type constraints are rejected deterministically."""
    with pytest.raises(ValueError, match="source_type validation requires source_type_values"):
        sanitize_source_metadata({"source_type": "external"})

    with pytest.raises(ValueError, match="must be one of"):
        sanitize_source_metadata(
            {"source_type": "not-a-source"},
            source_type_values=["direct_experience", "inference", "external", "consolidation"],
        )


def test_sanitize_source_metadata_coalesce_false_preserves_empty():
    """coalesce_empty_to_none=False preserves empty strings for context/source."""
    metadata = sanitize_source_metadata(
        {"context": "", "source": ""},
        coalesce_empty_to_none=False,
    )
    assert metadata["context"] == ""
    assert metadata["source"] == ""


def test_sanitize_source_metadata_coalesce_false_none_source():
    """source=None with coalesce_empty_to_none=False returns empty string (sanitize_string default)."""
    metadata = sanitize_source_metadata(
        {"source": None},
        coalesce_empty_to_none=False,
    )
    # sanitize_string(None, ..., required=False) returns "" and coalesce is off
    assert metadata["source"] == ""


def test_auto_capture_defaults_source_to_auto():
    """validate_memory_auto_capture defaults source to 'auto' when sanitized to None."""
    result = validate_memory_auto_capture({"text": "hello world"})
    assert result["source"] == "auto"


def test_auto_capture_preserves_explicit_source():
    """validate_memory_auto_capture preserves an explicitly provided source."""
    result = validate_memory_auto_capture({"text": "hello world", "source": "user"})
    assert result["source"] == "user"


# ============================================================================
# validate_number — NaN / Infinity rejection
# ============================================================================


class TestValidateNumberFiniteness:
    """validate_number must reject NaN and Infinity."""

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(float("nan"), "confidence")

    def test_rejects_positive_infinity(self):
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(float("inf"), "priority")

    def test_rejects_negative_infinity(self):
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_number(float("-inf"), "intensity")

    def test_accepts_normal_float(self):
        assert validate_number(0.5, "confidence") == 0.5

    def test_accepts_zero(self):
        assert validate_number(0.0, "confidence") == 0.0

    def test_accepts_integer(self):
        assert validate_number(42, "count") == 42.0


# ============================================================================
# sanitize_array — null item rejection
# ============================================================================


class TestSanitizeArrayNullItems:
    """sanitize_array must reject arrays containing None items."""

    def test_rejects_none_item(self):
        with pytest.raises(ValueError, match="must not contain null items"):
            sanitize_array(["a", None, "b"], "tags")

    def test_rejects_single_none_item(self):
        with pytest.raises(ValueError, match="must not contain null items"):
            sanitize_array([None], "tags")

    def test_accepts_array_without_none(self):
        assert sanitize_array(["a", "b"], "tags") == ["a", "b"]
