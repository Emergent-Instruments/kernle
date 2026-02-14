"""Tests for kernle.core.validation — canonical input sanitization helpers."""

import math

import pytest

from kernle.core.validation import sanitize_list, sanitize_number, sanitize_string


class TestSanitizeString:
    """Unit tests for the canonical sanitize_string function."""

    def test_valid_string_passes(self):
        assert sanitize_string("hello", "field") == "hello"

    def test_strips_control_characters(self):
        result = sanitize_string("hello\x00world\x07!", "field")
        assert result == "helloworld!"

    def test_preserves_newlines_and_tabs(self):
        result = sanitize_string("line1\nline2\ttab", "field")
        assert result == "line1\nline2\ttab"

    def test_rejects_none_when_required(self):
        with pytest.raises(ValueError, match="must be a string"):
            sanitize_string(None, "field", required=True)

    def test_returns_empty_for_none_when_optional(self):
        assert sanitize_string(None, "field", required=False) == ""

    def test_rejects_non_string_types(self):
        with pytest.raises(ValueError, match="must be a string, got int"):
            sanitize_string(42, "field")

    def test_rejects_empty_when_required(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_string("", "field", required=True)

    def test_rejects_whitespace_only_when_required(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_string("   ", "field", required=True)

    def test_allows_empty_when_optional(self):
        assert sanitize_string("", "field", required=False) == ""

    def test_rejects_over_max_length(self):
        with pytest.raises(ValueError, match="too long"):
            sanitize_string("x" * 1001, "field", max_length=1000)

    def test_custom_max_length(self):
        with pytest.raises(ValueError, match="too long"):
            sanitize_string("abcdef", "field", max_length=5)

    def test_exact_max_length_passes(self):
        result = sanitize_string("x" * 1000, "field", max_length=1000)
        assert len(result) == 1000

    def test_control_char_regex_matches_expected_range(self):
        """Verify the control-char regex strips \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f."""
        # Characters that should be stripped
        stripped = "".join(chr(c) for c in range(0x00, 0x09))  # \x00-\x08
        stripped += "\x0b\x0c"
        stripped += "".join(chr(c) for c in range(0x0E, 0x20))  # \x0e-\x1f
        stripped += "\x7f"

        result = sanitize_string(f"A{stripped}B", "field")
        assert result == "AB"

    def test_preserves_tab_newline_carriage_return(self):
        """Tab (\x09), newline (\x0a), and carriage return (\x0d) are preserved."""
        result = sanitize_string("a\tb\nc\rd", "field")
        assert result == "a\tb\nc\rd"


class TestSanitizeNumber:
    """Unit tests for the canonical sanitize_number function."""

    def test_valid_float(self):
        assert sanitize_number(0.5, "field") == 0.5

    def test_valid_int_converted_to_float(self):
        assert sanitize_number(42, "field") == 42.0
        assert isinstance(sanitize_number(42, "field"), float)

    def test_zero_passes(self):
        assert sanitize_number(0, "field") == 0.0

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite number"):
            sanitize_number(float("nan"), "field")

    def test_rejects_infinity(self):
        with pytest.raises(ValueError, match="finite number"):
            sanitize_number(float("inf"), "field")

    def test_rejects_negative_infinity(self):
        with pytest.raises(ValueError, match="finite number"):
            sanitize_number(float("-inf"), "field")

    def test_rejects_math_nan(self):
        with pytest.raises(ValueError, match="finite number"):
            sanitize_number(math.nan, "field")

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="must be a number"):
            sanitize_number("0.5", "field")

    def test_rejects_none_without_default(self):
        with pytest.raises(ValueError, match="is required"):
            sanitize_number(None, "field")

    def test_returns_default_for_none(self):
        assert sanitize_number(None, "field", default=0.5) == 0.5

    def test_min_val_enforced(self):
        with pytest.raises(ValueError, match="must be >="):
            sanitize_number(-1.0, "field", min_val=0.0)

    def test_max_val_enforced(self):
        with pytest.raises(ValueError, match="must be <="):
            sanitize_number(1.5, "field", max_val=1.0)

    def test_boundary_values_pass(self):
        assert sanitize_number(0.0, "field", min_val=0.0, max_val=1.0) == 0.0
        assert sanitize_number(1.0, "field", min_val=0.0, max_val=1.0) == 1.0


class TestSanitizeList:
    """Unit tests for the canonical sanitize_list function."""

    def test_valid_list(self):
        result = sanitize_list(["a", "b", "c"], "field")
        assert result == ["a", "b", "c"]

    def test_none_returns_empty(self):
        assert sanitize_list(None, "field") == []

    def test_rejects_non_list(self):
        with pytest.raises(ValueError, match="must be an array"):
            sanitize_list("not a list", "field")

    def test_rejects_null_items(self):
        with pytest.raises(ValueError, match="must not contain null"):
            sanitize_list(["a", None, "b"], "field")

    def test_rejects_too_many_items(self):
        with pytest.raises(ValueError, match="too many items"):
            sanitize_list(["x"] * 101, "field", max_items=100)

    def test_filters_empty_items(self):
        result = sanitize_list(["a", "", "b"], "field")
        assert result == ["a", "b"]

    def test_strips_control_chars_from_items(self):
        result = sanitize_list(["hello\x00world"], "field")
        assert result == ["helloworld"]

    def test_item_max_length_enforced(self):
        with pytest.raises(ValueError, match="too long"):
            sanitize_list(["x" * 600], "field", item_max_length=500)

    def test_empty_list_passes(self):
        assert sanitize_list([], "field") == []
