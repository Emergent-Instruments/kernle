"""Tests for kernle.types utility helpers."""

from datetime import datetime, timezone

import pytest

from kernle.types import ParseDatetimeError, parse_datetime


def test_parse_datetime_valid_iso_string():
    parsed = parse_datetime("2026-02-13T12:00:00+00:00")
    assert isinstance(parsed, datetime)
    assert parsed == datetime(2026, 2, 13, 12, 0, tzinfo=timezone.utc)


def test_parse_datetime_accepts_z_suffix():
    parsed = parse_datetime("2026-02-13T12:00:00Z")
    assert parsed == datetime(2026, 2, 13, 12, 0, tzinfo=timezone.utc)


def test_parse_datetime_invalid_returns_parse_error():
    parsed = parse_datetime("not-a-datetime")
    assert isinstance(parsed, ParseDatetimeError)
    assert parsed.value == "not-a-datetime"


def test_parse_datetime_invalid_raises_in_strict_mode():
    with pytest.raises(ParseDatetimeError):
        parse_datetime("not-a-datetime", strict=True)


def test_parse_datetime_empty_is_none():
    assert parse_datetime("") is None
    assert parse_datetime(None) is None
