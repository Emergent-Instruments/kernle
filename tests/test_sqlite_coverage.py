"""Tests targeting uncovered methods in SQLiteStorage for coverage improvement.

Covers:
- Pure functions: _escape_like_pattern, _tokenize_query, _token_match_score, _build_token_filter
- Boot config CRUD: boot_set, boot_get, boot_list, boot_delete
- Stack settings: get_stack_setting, set_stack_setting, get_all_stack_settings
- Raw processing: mark_raw_processed, delete_raw
- Processing config: get_processing_config, set_processing_config
- Audit & health: log_audit, log_health_check, get_health_check_stats
- Memory marking: mark_episode_processed, mark_belief_processed
- Access tracking: record_access
"""

import pytest

from kernle.storage import Belief, Episode, SQLiteStorage


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db, monkeypatch):
    """Create a SQLiteStorage instance for testing."""
    s = SQLiteStorage(stack_id="test-stack", db_path=temp_db)
    monkeypatch.setattr(s, "has_cloud_credentials", lambda: False)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# 1. Pure functions: _escape_like_pattern
# ---------------------------------------------------------------------------


class TestEscapeLikePattern:
    def test_no_special_chars(self, storage):
        assert storage._escape_like_pattern("hello world") == "hello world"

    def test_escapes_percent(self, storage):
        assert storage._escape_like_pattern("100%") == "100\\%"

    def test_escapes_underscore(self, storage):
        assert storage._escape_like_pattern("file_name") == "file\\_name"

    def test_escapes_backslash(self, storage):
        assert storage._escape_like_pattern("path\\to") == "path\\\\to"

    def test_escapes_all_special_chars(self, storage):
        result = storage._escape_like_pattern("a%b_c\\d")
        assert result == "a\\%b\\_c\\\\d"

    def test_empty_string(self, storage):
        assert storage._escape_like_pattern("") == ""


# ---------------------------------------------------------------------------
# 2. Pure functions: _tokenize_query
# ---------------------------------------------------------------------------


class TestTokenizeQuery:
    def test_basic_tokenization(self):
        tokens = SQLiteStorage._tokenize_query("hello world today")
        assert tokens == ["hello", "world", "today"]

    def test_filters_short_words(self):
        tokens = SQLiteStorage._tokenize_query("I am a big cat")
        assert tokens == ["big", "cat"]

    def test_empty_query(self):
        tokens = SQLiteStorage._tokenize_query("")
        assert tokens == []

    def test_all_short_words(self):
        tokens = SQLiteStorage._tokenize_query("a be it")
        assert tokens == []

    def test_preserves_case(self):
        tokens = SQLiteStorage._tokenize_query("Hello WORLD")
        assert tokens == ["Hello", "WORLD"]


# ---------------------------------------------------------------------------
# 3. Pure functions: _token_match_score
# ---------------------------------------------------------------------------


class TestTokenMatchScore:
    def test_all_tokens_match(self):
        score = SQLiteStorage._token_match_score("hello world today", ["hello", "world"])
        assert score == 1.0

    def test_no_tokens_match(self):
        score = SQLiteStorage._token_match_score("hello world", ["xyz", "abc"])
        assert score == 0.0

    def test_partial_match(self):
        score = SQLiteStorage._token_match_score("hello world", ["hello", "xyz"])
        assert score == 0.5

    def test_empty_tokens_returns_one(self):
        score = SQLiteStorage._token_match_score("any text", [])
        assert score == 1.0

    def test_case_insensitive(self):
        score = SQLiteStorage._token_match_score("Hello World", ["hello", "world"])
        assert score == 1.0


# ---------------------------------------------------------------------------
# 4. Pure functions: _build_token_filter
# ---------------------------------------------------------------------------


class TestBuildTokenFilter:
    def test_single_token_single_column(self):
        sql, params = SQLiteStorage._build_token_filter(["hello"], ["content"])
        assert sql == "(content LIKE ?)"
        assert params == ["%hello%"]

    def test_multiple_tokens_multiple_columns(self):
        sql, params = SQLiteStorage._build_token_filter(["hello", "world"], ["content", "title"])
        assert "content LIKE ?" in sql
        assert "title LIKE ?" in sql
        assert " OR " in sql
        assert len(params) == 4
        assert params == ["%hello%", "%hello%", "%world%", "%world%"]

    def test_empty_tokens(self):
        sql, params = SQLiteStorage._build_token_filter([], ["content"])
        assert sql == "()"
        assert params == []


# ---------------------------------------------------------------------------
# 5. Boot config CRUD: boot_set, boot_get, boot_list, boot_delete
# ---------------------------------------------------------------------------


class TestBootConfig:
    def test_set_and_get(self, storage):
        storage.boot_set("key1", "value1")
        assert storage.boot_get("key1") == "value1"

    def test_get_nonexistent_returns_default(self, storage):
        assert storage.boot_get("missing") is None
        assert storage.boot_get("missing", "fallback") == "fallback"

    def test_set_overwrites_existing(self, storage):
        storage.boot_set("key1", "value1")
        storage.boot_set("key1", "value2")
        assert storage.boot_get("key1") == "value2"

    def test_list_returns_all(self, storage):
        storage.boot_set("alpha", "1")
        storage.boot_set("beta", "2")
        result = storage.boot_list()
        assert result == {"alpha": "1", "beta": "2"}

    def test_list_empty(self, storage):
        assert storage.boot_list() == {}

    def test_delete_existing(self, storage):
        storage.boot_set("key1", "value1")
        assert storage.boot_delete("key1") is True
        assert storage.boot_get("key1") is None

    def test_delete_nonexistent(self, storage):
        assert storage.boot_delete("missing") is False

    def test_set_validates_empty_key(self, storage):
        with pytest.raises(ValueError, match="non-empty string"):
            storage.boot_set("", "value")

    def test_set_strips_whitespace_key(self, storage):
        storage.boot_set("  key  ", "value")
        assert storage.boot_get("key") == "value"


# ---------------------------------------------------------------------------
# 6. Stack settings: get_stack_setting, set_stack_setting, get_all_stack_settings
# ---------------------------------------------------------------------------


class TestStackSettings:
    def test_set_and_get(self, storage):
        storage.set_stack_setting("theme", "dark")
        assert storage.get_stack_setting("theme") == "dark"

    def test_get_nonexistent_returns_none(self, storage):
        assert storage.get_stack_setting("missing") is None

    def test_set_overwrites_existing(self, storage):
        storage.set_stack_setting("theme", "dark")
        storage.set_stack_setting("theme", "light")
        assert storage.get_stack_setting("theme") == "light"

    def test_get_all_stack_settings(self, storage):
        storage.set_stack_setting("alpha", "1")
        storage.set_stack_setting("beta", "2")
        result = storage.get_all_stack_settings()
        assert result == {"alpha": "1", "beta": "2"}

    def test_get_all_empty(self, storage):
        assert storage.get_all_stack_settings() == {}

    def test_settings_scoped_to_stack(self, temp_db):
        """Two stacks using the same DB have independent settings."""
        s1 = SQLiteStorage(stack_id="stack-a", db_path=temp_db)
        s2 = SQLiteStorage(stack_id="stack-b", db_path=temp_db)
        s1.set_stack_setting("key", "from-a")
        s2.set_stack_setting("key", "from-b")
        assert s1.get_stack_setting("key") == "from-a"
        assert s2.get_stack_setting("key") == "from-b"
        s1.close()
        s2.close()


# ---------------------------------------------------------------------------
# 7. Raw processing: mark_raw_processed, delete_raw
# ---------------------------------------------------------------------------


class TestRawProcessing:
    def test_mark_raw_processed(self, storage):
        raw_id = storage.save_raw("some raw content", source="cli")
        result = storage.mark_raw_processed(raw_id, ["episode:ep-1", "note:n-1"])
        assert result is True

    def test_mark_raw_processed_nonexistent(self, storage):
        result = storage.mark_raw_processed("nonexistent-id", ["episode:ep-1"])
        assert result is False

    def test_delete_raw(self, storage):
        raw_id = storage.save_raw("content to delete", source="cli")
        result = storage.delete_raw(raw_id)
        assert result is True

    def test_delete_raw_nonexistent(self, storage):
        result = storage.delete_raw("nonexistent-id")
        assert result is False

    def test_delete_raw_already_deleted(self, storage):
        raw_id = storage.save_raw("content", source="cli")
        storage.delete_raw(raw_id)
        # Second delete should return False (already soft-deleted)
        assert storage.delete_raw(raw_id) is False


# ---------------------------------------------------------------------------
# 8. Processing config: get_processing_config, set_processing_config
# ---------------------------------------------------------------------------


class TestProcessingConfig:
    def test_get_empty(self, storage):
        configs = storage.get_processing_config()
        assert isinstance(configs, list)

    def test_set_and_get(self, storage):
        storage.set_processing_config(
            "raw_to_episode",
            enabled=True,
            model_id="test-model",
            quantity_threshold=5,
            batch_size=10,
        )
        configs = storage.get_processing_config()
        match = [c for c in configs if c["layer_transition"] == "raw_to_episode"]
        assert len(match) == 1
        assert match[0]["enabled"] is True
        assert match[0]["model_id"] == "test-model"
        assert match[0]["quantity_threshold"] == 5

    def test_update_existing_config(self, storage):
        storage.set_processing_config("raw_to_episode", enabled=True, batch_size=10)
        storage.set_processing_config("raw_to_episode", enabled=False, batch_size=20)
        configs = storage.get_processing_config()
        match = [c for c in configs if c["layer_transition"] == "raw_to_episode"]
        assert len(match) == 1
        assert match[0]["enabled"] is False
        assert match[0]["batch_size"] == 20

    def test_set_returns_true(self, storage):
        result = storage.set_processing_config("episode_to_belief")
        assert result is True


# ---------------------------------------------------------------------------
# 9. Audit & health: log_audit, log_health_check, get_health_check_stats
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_log_audit_returns_id(self, storage):
        audit_id = storage.log_audit(
            memory_type="episode",
            memory_id="ep-1",
            operation="forget",
            actor="core:test",
        )
        assert isinstance(audit_id, str)
        assert len(audit_id) > 0

    def test_log_audit_with_details(self, storage):
        audit_id = storage.log_audit(
            memory_type="belief",
            memory_id="b-1",
            operation="protect",
            actor="plugin:test",
            details={"reason": "core identity"},
        )
        assert isinstance(audit_id, str)

    def test_log_audit_without_details(self, storage):
        audit_id = storage.log_audit(
            memory_type="value",
            memory_id="v-1",
            operation="verify",
            actor="core:test",
        )
        assert isinstance(audit_id, str)


class TestHealthCheck:
    def test_log_health_check_returns_id(self, storage):
        event_id = storage.log_health_check(anxiety_score=42, source="cli")
        assert isinstance(event_id, str)
        assert len(event_id) > 0

    def test_get_stats_empty(self, storage):
        stats = storage.get_health_check_stats()
        assert stats["total_checks"] == 0
        assert stats["avg_per_day"] == 0.0
        assert stats["last_check_at"] is None
        assert stats["last_anxiety_score"] is None
        assert stats["checks_by_source"] == {}
        assert stats["checks_by_trigger"] == {}

    def test_get_stats_after_checks(self, storage):
        storage.log_health_check(anxiety_score=30, source="cli", triggered_by="boot")
        storage.log_health_check(anxiety_score=50, source="mcp", triggered_by="heartbeat")
        storage.log_health_check(anxiety_score=10, source="cli", triggered_by="manual")

        stats = storage.get_health_check_stats()
        assert stats["total_checks"] == 3
        assert stats["last_anxiety_score"] == 10
        assert stats["last_check_at"] is not None
        assert stats["checks_by_source"]["cli"] == 2
        assert stats["checks_by_source"]["mcp"] == 1
        assert stats["checks_by_trigger"]["boot"] == 1
        assert stats["checks_by_trigger"]["heartbeat"] == 1
        assert stats["checks_by_trigger"]["manual"] == 1


# ---------------------------------------------------------------------------
# 10. Memory marking: mark_episode_processed, mark_belief_processed
# ---------------------------------------------------------------------------


class TestMemoryMarking:
    def test_mark_episode_processed(self, storage):
        episode = Episode(
            id="ep-1",
            stack_id="test-stack",
            objective="test objective",
            outcome="test outcome",
        )
        storage.save_episode(episode)
        result = storage.mark_episode_processed("ep-1")
        assert result is True

    def test_mark_episode_processed_nonexistent(self, storage):
        result = storage.mark_episode_processed("nonexistent-ep")
        assert result is False

    def test_mark_belief_processed(self, storage):
        belief = Belief(
            id="b-1",
            stack_id="test-stack",
            statement="test belief",
        )
        storage.save_belief(belief)
        result = storage.mark_belief_processed("b-1")
        assert result is True

    def test_mark_belief_processed_nonexistent(self, storage):
        result = storage.mark_belief_processed("nonexistent-b")
        assert result is False


# ---------------------------------------------------------------------------
# 11. Access tracking: record_access
# ---------------------------------------------------------------------------


class TestRecordAccess:
    def test_record_access_episode(self, storage):
        episode = Episode(
            id="ep-access-1",
            stack_id="test-stack",
            objective="test",
            outcome="test",
        )
        storage.save_episode(episode)
        result = storage.record_access("episode", "ep-access-1")
        assert result is True

    def test_record_access_belief(self, storage):
        belief = Belief(
            id="b-access-1",
            stack_id="test-stack",
            statement="test belief",
        )
        storage.save_belief(belief)
        result = storage.record_access("belief", "b-access-1")
        assert result is True

    def test_record_access_unknown_type(self, storage):
        result = storage.record_access("unknown_type", "some-id")
        assert result is False

    def test_record_access_nonexistent_id(self, storage):
        result = storage.record_access("episode", "nonexistent-id")
        assert result is False

    def test_record_access_increments_count(self, storage):
        episode = Episode(
            id="ep-count-1",
            stack_id="test-stack",
            objective="test",
            outcome="test",
        )
        storage.save_episode(episode)
        storage.record_access("episode", "ep-count-1")
        storage.record_access("episode", "ep-count-1")
        # Retrieve episode to verify times_accessed was incremented
        episodes = storage.get_episodes()
        ep = [e for e in episodes if e.id == "ep-count-1"][0]
        assert ep.times_accessed == 2
