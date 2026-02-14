"""Tests for LIKE pattern injection prevention (GitHub issue #725).

Verifies that LIKE metacharacters (%, _, \\) in entity names and search
queries are properly escaped, preventing access control bypass or
unintended pattern matching.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.storage.raw_entries import escape_like_pattern
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Episode, Note

STACK_ID = "test-like-injection"


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_like_injection.db"


@pytest.fixture
def storage(tmp_db):
    return SQLiteStorage(stack_id=STACK_ID, db_path=tmp_db)


def _make_episode(access_grants=None, **overrides):
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": STACK_ID,
        "objective": "Test objective",
        "outcome": "Test outcome",
        "created_at": datetime.now(timezone.utc),
    }
    if access_grants is not None:
        defaults["access_grants"] = access_grants
    defaults.update(overrides)
    return Episode(**defaults)


def _make_note(access_grants=None, content="some note", **overrides):
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": STACK_ID,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    }
    if access_grants is not None:
        defaults["access_grants"] = access_grants
    defaults.update(overrides)
    return Note(**defaults)


# ---------------------------------------------------------------------------
# escape_like_pattern unit tests
# ---------------------------------------------------------------------------


class TestEscapeLikePattern:
    """Unit tests for the escape_like_pattern helper."""

    def test_percent_is_escaped(self):
        assert escape_like_pattern("foo%bar") == "foo\\%bar"

    def test_underscore_is_escaped(self):
        assert escape_like_pattern("foo_bar") == "foo\\_bar"

    def test_backslash_is_escaped_first(self):
        # Backslash must be escaped before % and _ to avoid double-escaping
        assert escape_like_pattern("a\\b") == "a\\\\b"

    def test_all_metacharacters_escaped(self):
        assert escape_like_pattern("100%_done\\") == "100\\%\\_done\\\\"

    def test_plain_string_unchanged(self):
        assert escape_like_pattern("normal-entity") == "normal-entity"

    def test_empty_string(self):
        assert escape_like_pattern("") == ""


# ---------------------------------------------------------------------------
# _build_access_filter injection tests
# ---------------------------------------------------------------------------


class TestAccessFilterInjection:
    """Verify that _build_access_filter escapes LIKE metacharacters."""

    def test_percent_in_entity_does_not_match_all_records(self, storage):
        """An entity name containing '%' must not act as a wildcard.

        Before the fix, requesting_entity='%' would generate LIKE '%"%"%'
        which matches every non-null access_grants value.
        """
        # Create an episode granted to a specific entity
        ep_granted = _make_episode(access_grants=["legit-entity"])
        storage.save_episode(ep_granted)

        # Create a private episode (no access_grants)
        ep_private = _make_episode(access_grants=None)
        storage.save_episode(ep_private)

        # Request with a '%' entity -- should match nothing
        results = storage.get_episodes(requesting_entity="%")
        assert (
            len(results) == 0
        ), "Entity name '%' should not match any records via wildcard expansion"

    def test_percent_entity_does_not_match_other_grants(self, storage):
        """Entity '%' should not match records granted to other entities."""
        ep = _make_episode(access_grants=["alice", "bob"])
        storage.save_episode(ep)

        results = storage.get_episodes(requesting_entity="%")
        assert len(results) == 0

    def test_underscore_in_entity_does_not_match_single_char(self, storage):
        """An entity name containing '_' must not act as a single-char wildcard.

        Before the fix, requesting_entity='a_c' could match 'abc' grants.
        """
        ep = _make_episode(access_grants=["abc"])
        storage.save_episode(ep)

        # 'a_c' should NOT match 'abc' -- underscore is not a wildcard
        results = storage.get_episodes(requesting_entity="a_c")
        assert len(results) == 0, "Entity name 'a_c' should not match 'abc' via underscore wildcard"

    def test_normal_entity_name_still_works(self, storage):
        """Normal entity names without metacharacters continue to work."""
        ep = _make_episode(access_grants=["trusted-agent"])
        storage.save_episode(ep)

        results = storage.get_episodes(requesting_entity="trusted-agent")
        assert len(results) == 1
        assert results[0].id == ep.id

    def test_exact_entity_with_percent_in_name_matches(self, storage):
        """An entity whose real name contains '%' should match its own grants."""
        weird_entity = "entity%with%percent"
        ep = _make_episode(access_grants=[weird_entity])
        storage.save_episode(ep)

        results = storage.get_episodes(requesting_entity=weird_entity)
        assert len(results) == 1
        assert results[0].id == ep.id

    def test_exact_entity_with_underscore_in_name_matches(self, storage):
        """An entity whose real name contains '_' should match its own grants."""
        weird_entity = "entity_with_underscores"
        ep = _make_episode(access_grants=[weird_entity])
        storage.save_episode(ep)

        results = storage.get_episodes(requesting_entity=weird_entity)
        assert len(results) == 1
        assert results[0].id == ep.id

    def test_backslash_in_entity_does_not_cause_injection(self, storage):
        """A backslash in entity name must not cause unintended matching.

        Note: entity names containing backslashes have a pre-existing
        limitation where they cannot match their own grants due to JSON
        encoding (json.dumps escapes \\ to \\\\). This test verifies the
        security property: backslashes do not enable wildcard injection.
        """
        ep = _make_episode(access_grants=["alice"])
        storage.save_episode(ep)

        # A backslash should not be treated as a LIKE escape and cause
        # the rest of the pattern to be interpreted differently
        results = storage.get_episodes(requesting_entity="\\")
        assert len(results) == 0

    def test_self_access_sees_everything(self, storage):
        """requesting_entity=None should bypass the filter entirely."""
        ep_private = _make_episode(access_grants=None)
        ep_granted = _make_episode(access_grants=["someone"])
        storage.save_episode(ep_private)
        storage.save_episode(ep_granted)

        results = storage.get_episodes(requesting_entity=None)
        assert len(results) == 2

    def test_escape_clause_in_access_filter_sql(self, storage):
        """Verify the ESCAPE clause is present in the generated SQL."""
        where_clause, params = storage._build_access_filter("test-entity")
        assert "ESCAPE" in where_clause, "Access filter SQL must include ESCAPE clause"

    def test_no_filter_for_none_entity(self, storage):
        """None requesting_entity should produce empty filter."""
        where_clause, params = storage._build_access_filter(None)
        assert where_clause == ""
        assert params == []


# ---------------------------------------------------------------------------
# _build_token_filter injection tests
# ---------------------------------------------------------------------------


class TestTokenFilterInjection:
    """Verify that _build_token_filter escapes LIKE metacharacters in search tokens."""

    def test_token_with_percent_is_escaped_in_sql(self):
        """Tokens containing '%' should be escaped in the LIKE pattern."""
        sql, params = SQLiteStorage._build_token_filter(["foo%bar"], ["content"])
        assert "ESCAPE" in sql, "Token filter SQL must include ESCAPE clause"
        # The param should have the percent escaped
        assert "foo\\%bar" in params[0]

    def test_token_with_underscore_is_escaped_in_sql(self):
        """Tokens containing '_' should be escaped in the LIKE pattern."""
        sql, params = SQLiteStorage._build_token_filter(["foo_bar"], ["content"])
        assert "ESCAPE" in sql
        assert "foo\\_bar" in params[0]

    def test_normal_tokens_produce_correct_patterns(self):
        """Normal tokens without metacharacters produce standard LIKE patterns."""
        sql, params = SQLiteStorage._build_token_filter(["hello", "world"], ["col1"])
        assert len(params) == 2
        assert params[0] == "%hello%"
        assert params[1] == "%world%"
        assert "ESCAPE" in sql

    def test_multiple_columns_multiple_tokens(self):
        """Each token+column combination gets its own LIKE clause."""
        sql, params = SQLiteStorage._build_token_filter(["abc", "def"], ["col1", "col2"])
        # 2 tokens x 2 columns = 4 clauses
        assert len(params) == 4
        assert sql.count("LIKE") == 4
        assert sql.count("ESCAPE") == 4
