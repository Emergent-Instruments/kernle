"""Tests for kernle.core.validation.sanitize_string — canonical input sanitization."""

import pytest

from kernle.core.validation import sanitize_string


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
