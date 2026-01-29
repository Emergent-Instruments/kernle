"""Tests for CLI helper functions."""

import json

import pytest

from kernle.cli.commands.helpers import print_json, validate_input


class TestValidateInput:
    """Test validate_input function."""

    def test_valid_string(self):
        """Test that valid strings pass through."""
        result = validate_input("hello world", "test_field")
        assert result == "hello world"

    def test_non_string_raises_error(self):
        """Test that non-string values raise ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_input(123, "test_field")

    def test_too_long_raises_error(self):
        """Test that too-long strings raise ValueError."""
        long_string = "x" * 1001
        with pytest.raises(ValueError, match="too long"):
            validate_input(long_string, "test_field")

    def test_custom_max_length(self):
        """Test custom max_length parameter."""
        with pytest.raises(ValueError, match="max 50 characters"):
            validate_input("x" * 51, "test_field", max_length=50)

    def test_removes_control_characters(self):
        """Test that control characters are removed."""
        input_with_null = "hello\x00world"
        result = validate_input(input_with_null, "test_field")
        assert result == "helloworld"

    def test_preserves_newlines(self):
        """Test that newlines are preserved."""
        input_with_newline = "hello\nworld"
        result = validate_input(input_with_newline, "test_field")
        assert result == "hello\nworld"


class TestPrintJson:
    """Test print_json function."""

    def test_prints_formatted_json(self, capsys):
        """Test that data is printed as formatted JSON."""
        data = {"key": "value", "number": 42}
        print_json(data)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output == data

    def test_handles_non_serializable(self, capsys):
        """Test that non-JSON-serializable objects are converted with str."""
        from datetime import datetime

        data = {"timestamp": datetime(2026, 1, 15, 10, 30, 0)}
        print_json(data)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "2026-01-15" in output["timestamp"]
