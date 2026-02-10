"""Tests for kernle.storage.memory_crud module.

Tests the extracted row deserializer helpers.
"""

import sqlite3

from kernle.storage.memory_crud import _from_json, _safe_get


class TestFromJson:
    def test_none_returns_none(self):
        assert _from_json(None) is None

    def test_empty_returns_none(self):
        assert _from_json("") is None

    def test_valid_json(self):
        assert _from_json('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_none(self):
        assert _from_json("not valid json {{{") is None

    def test_json_list(self):
        assert _from_json("[1, 2, 3]") == [1, 2, 3]


class TestSafeGet:
    def test_key_present(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (col TEXT)")
        conn.execute("INSERT INTO t VALUES ('value')")
        row = conn.execute("SELECT * FROM t").fetchone()
        assert _safe_get(row, "col") == "value"
        conn.close()

    def test_null_returns_default(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (col TEXT)")
        conn.execute("INSERT INTO t VALUES (NULL)")
        row = conn.execute("SELECT * FROM t").fetchone()
        assert _safe_get(row, "col", "fallback") == "fallback"
        conn.close()

    def test_missing_key_returns_default(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (col TEXT)")
        conn.execute("INSERT INTO t VALUES ('x')")
        row = conn.execute("SELECT * FROM t").fetchone()
        assert _safe_get(row, "nonexistent", "default") == "default"
        conn.close()
