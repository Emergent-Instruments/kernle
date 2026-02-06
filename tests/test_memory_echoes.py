"""
Tests for memory echoes (peripheral awareness) in load().

Tests that excluded memories generate echo hints, temporal summaries,
and topic clusters in _meta when the token budget excludes memories.
"""

import uuid
from datetime import datetime, timedelta, timezone

from kernle.core import (
    Kernle,
    _build_memory_echoes,
    _get_memory_hint_text,
    _get_record_tags,
    _truncate_to_words,
)
from kernle.storage import SQLiteStorage
from kernle.storage.base import Belief, Drive, Episode, Goal, Note, Relationship, Value


class TestTruncateToWords:
    """Tests for the _truncate_to_words helper."""

    def test_short_text_not_truncated(self):
        """Text with fewer than max_words should not be truncated."""
        text = "one two three"
        result = _truncate_to_words(text, max_words=8)
        assert result == text
        assert "..." not in result

    def test_exact_word_count_not_truncated(self):
        """Text with exactly max_words should not be truncated."""
        text = "one two three four five six seven eight"
        result = _truncate_to_words(text, max_words=8)
        assert result == text
        assert "..." not in result

    def test_long_text_truncated(self):
        """Text exceeding max_words should be truncated with ellipsis."""
        text = "one two three four five six seven eight nine ten eleven"
        result = _truncate_to_words(text, max_words=8)
        assert result == "one two three four five six seven eight..."
        assert result.endswith("...")

    def test_word_count_correct(self):
        """Truncated text should have exactly max_words words (before ellipsis)."""
        text = "a b c d e f g h i j k l m n o"
        result = _truncate_to_words(text, max_words=8)
        # Remove trailing "..." and count words
        words = result.replace("...", "").split()
        assert len(words) == 8

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert _truncate_to_words("") == ""

    def test_none_returns_empty(self):
        """None should return empty string."""
        assert _truncate_to_words(None) == ""

    def test_single_word(self):
        """Single word should not be truncated."""
        assert _truncate_to_words("hello", max_words=8) == "hello"


class TestGetMemoryHintText:
    """Tests for _get_memory_hint_text helper."""

    def test_episode_hint(self):
        ep = Episode(
            id="ep1",
            agent_id="a",
            objective="Debug the login flow",
            outcome="Fixed the auth bug",
        )
        result = _get_memory_hint_text("episode", ep)
        assert "Debug the login flow" in result
        assert "Fixed the auth bug" in result

    def test_belief_hint(self):
        b = Belief(
            id="b1",
            agent_id="a",
            statement="Testing leads to reliable software",
        )
        result = _get_memory_hint_text("belief", b)
        assert result == "Testing leads to reliable software"

    def test_note_hint(self):
        n = Note(
            id="n1",
            agent_id="a",
            content="Important decision about architecture",
        )
        result = _get_memory_hint_text("note", n)
        assert result == "Important decision about architecture"

    def test_value_hint(self):
        v = Value(
            id="v1",
            agent_id="a",
            name="Quality",
            statement="Software should be tested",
        )
        result = _get_memory_hint_text("value", v)
        assert "Quality" in result
        assert "Software should be tested" in result

    def test_goal_hint(self):
        g = Goal(
            id="g1",
            agent_id="a",
            title="Improve coverage",
            description="Write tests for edge cases",
        )
        result = _get_memory_hint_text("goal", g)
        assert "Improve coverage" in result

    def test_drive_hint(self):
        d = Drive(
            id="d1",
            agent_id="a",
            drive_type="growth",
            focus_areas=["learning", "improvement"],
        )
        result = _get_memory_hint_text("drive", d)
        assert "growth" in result
        assert "learning" in result

    def test_relationship_hint(self):
        r = Relationship(
            id="r1",
            agent_id="a",
            entity_name="Alice",
            entity_type="human",
            relationship_type="collaborator",
            notes="Great at code reviews",
        )
        result = _get_memory_hint_text("relationship", r)
        assert "Alice" in result
        assert "Great at code reviews" in result


class TestGetRecordTags:
    """Tests for _get_record_tags helper."""

    def test_episode_tags(self):
        ep = Episode(
            id="ep1",
            agent_id="a",
            objective="test",
            outcome="pass",
            tags=["testing", "dev"],
            context_tags=["project-x"],
        )
        tags = _get_record_tags("episode", ep)
        assert "testing" in tags
        assert "dev" in tags
        assert "project-x" in tags

    def test_drive_focus_areas(self):
        d = Drive(
            id="d1",
            agent_id="a",
            drive_type="growth",
            focus_areas=["learning", "improvement"],
        )
        tags = _get_record_tags("drive", d)
        assert "learning" in tags
        assert "improvement" in tags

    def test_no_tags(self):
        ep = Episode(
            id="ep1",
            agent_id="a",
            objective="test",
            outcome="pass",
        )
        tags = _get_record_tags("episode", ep)
        assert tags == []


class TestBuildMemoryEchoes:
    """Tests for _build_memory_echoes function."""

    def _make_candidates(self, count, memory_type="note"):
        """Helper to create candidate tuples."""
        candidates = []
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(count):
            record = Note(
                id=f"note-{i}",
                agent_id="a",
                content=f"This is note number {i} with some extra words for testing purposes",
                tags=["tag-a", "tag-b"] if i % 2 == 0 else ["tag-c"],
                created_at=base_time + timedelta(days=i * 30),
            )
            priority = 0.5 - (i * 0.01)
            candidates.append((priority, memory_type, record))
        return candidates

    def test_no_excluded(self):
        """When no excluded candidates, echoes should be empty."""
        result = _build_memory_echoes([])
        assert result["echoes"] == []
        assert result["temporal_summary"] is None
        assert result["topic_clusters"] == []

    def test_echoes_present(self):
        """Echoes should be generated from excluded candidates."""
        excluded = self._make_candidates(5)
        result = _build_memory_echoes(excluded)
        assert len(result["echoes"]) == 5
        for echo in result["echoes"]:
            assert "type" in echo
            assert "id" in echo
            assert "hint" in echo
            assert "salience" in echo

    def test_echoes_max_limit(self):
        """Echoes should be capped at max_echoes."""
        excluded = self._make_candidates(30)
        result = _build_memory_echoes(excluded, max_echoes=20)
        assert len(result["echoes"]) == 20

    def test_hints_truncated(self):
        """Hints should be truncated to ~8 words."""
        record = Note(
            id="n1",
            agent_id="a",
            content="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        excluded = [(0.5, "note", record)]
        result = _build_memory_echoes(excluded)
        hint = result["echoes"][0]["hint"]
        assert hint.endswith("...")
        word_count = len(hint.replace("...", "").split())
        assert word_count == 8

    def test_short_hint_not_truncated(self):
        """Short hints should not be truncated."""
        record = Note(
            id="n1",
            agent_id="a",
            content="Short note",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        excluded = [(0.5, "note", record)]
        result = _build_memory_echoes(excluded)
        assert result["echoes"][0]["hint"] == "Short note"
        assert "..." not in result["echoes"][0]["hint"]

    def test_temporal_summary_date_range(self):
        """Temporal summary should show correct date range."""
        base_time = datetime(2023, 6, 15, tzinfo=timezone.utc)
        records = []
        for i in range(3):
            record = Note(
                id=f"n{i}",
                agent_id="a",
                content=f"note {i}",
                created_at=base_time + timedelta(days=i * 180),
            )
            records.append((0.5, "note", record))

        result = _build_memory_echoes(records)
        summary = result["temporal_summary"]
        assert summary is not None
        assert "2023-06-15" in summary
        # Last record is at base_time + 360 days
        expected_end = (base_time + timedelta(days=360)).strftime("%Y-%m-%d")
        assert expected_end in summary
        assert "3 excluded memories" in summary

    def test_temporal_summary_none_when_no_dates(self):
        """Temporal summary should be None when no records have dates."""
        record = Note(id="n1", agent_id="a", content="no date")
        excluded = [(0.5, "note", record)]
        result = _build_memory_echoes(excluded)
        assert result["temporal_summary"] is None

    def test_topic_clusters_from_tags(self):
        """Topic clusters should be extracted from excluded memory tags."""
        records = []
        for i in range(10):
            tags = ["python", "testing"] if i < 6 else ["deployment"]
            record = Note(
                id=f"n{i}",
                agent_id="a",
                content=f"note {i}",
                tags=tags,
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
            records.append((0.5, "note", record))

        result = _build_memory_echoes(records)
        clusters = result["topic_clusters"]
        assert len(clusters) > 0
        assert len(clusters) <= 6
        # "python" and "testing" appear 6 times each, "deployment" 4 times
        assert "python" in clusters
        assert "testing" in clusters
        assert "deployment" in clusters

    def test_topic_clusters_max_six(self):
        """Topic clusters should be capped at 6."""
        records = []
        for i in range(20):
            tags = [f"tag-{i % 8}"]
            record = Note(
                id=f"n{i}",
                agent_id="a",
                content=f"note {i}",
                tags=tags,
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
            records.append((0.5, "note", record))

        result = _build_memory_echoes(records)
        assert len(result["topic_clusters"]) <= 6

    def test_salience_is_priority_score(self):
        """Echo salience should match the candidate's priority score."""
        record = Note(
            id="n1",
            agent_id="a",
            content="test note",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        excluded = [(0.456, "note", record)]
        result = _build_memory_echoes(excluded)
        assert result["echoes"][0]["salience"] == 0.456

    def test_echo_type_matches_memory_type(self):
        """Echo type field should match the memory type."""
        ep = Episode(
            id="ep1",
            agent_id="a",
            objective="test",
            outcome="pass",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        excluded = [(0.5, "episode", ep)]
        result = _build_memory_echoes(excluded)
        assert result["echoes"][0]["type"] == "episode"


class TestLoadWithEchoes:
    """Integration tests for memory echoes in the load() method."""

    def _create_kernle_with_many_memories(self, tmp_path, memory_count=50):
        """Create a Kernle instance with many memories to force budget exclusion."""
        db_path = tmp_path / "test_echoes.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
        kernle = Kernle(
            agent_id="test_agent",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
        )

        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

        # Create many notes (low priority, likely to be excluded)
        for i in range(memory_count):
            note = Note(
                id=str(uuid.uuid4()),
                agent_id="test_agent",
                content=f"This is a detailed note about topic number {i} with enough words to consume tokens in the budget allocation process",
                note_type="observation",
                tags=["topic-a", "topic-b"] if i % 3 == 0 else ["topic-c", "topic-d"],
                created_at=base_time + timedelta(days=i * 7),
            )
            storage.save_note(note)

        # Add some episodes too
        for i in range(20):
            ep = Episode(
                id=str(uuid.uuid4()),
                agent_id="test_agent",
                objective=f"Complete task {i} which involves various steps and planning",
                outcome=f"Successfully finished with lessons learned about approach {i}",
                outcome_type="success",
                tags=["development", "planning"],
                created_at=base_time + timedelta(days=i * 14),
            )
            storage.save_episode(ep)

        return kernle, storage

    def test_echoes_in_meta_with_budget_constraint(self, tmp_path):
        """Echoes should appear in _meta when memories are excluded by budget."""
        kernle, storage = self._create_kernle_with_many_memories(tmp_path)
        try:
            # Use a small budget to force exclusions
            result = kernle.load(budget=500, sync=False, track_access=False)
            meta = result["_meta"]

            assert meta["excluded_count"] > 0
            assert "echoes" in meta
            assert isinstance(meta["echoes"], list)
            assert len(meta["echoes"]) > 0
            assert len(meta["echoes"]) <= 20
        finally:
            storage.close()

    def test_echo_structure(self, tmp_path):
        """Each echo should have the required fields."""
        kernle, storage = self._create_kernle_with_many_memories(tmp_path)
        try:
            result = kernle.load(budget=500, sync=False, track_access=False)
            echoes = result["_meta"]["echoes"]

            if echoes:
                echo = echoes[0]
                assert "type" in echo
                assert "id" in echo
                assert "hint" in echo
                assert "salience" in echo
                assert isinstance(echo["type"], str)
                assert isinstance(echo["id"], str)
                assert isinstance(echo["hint"], str)
                assert isinstance(echo["salience"], float)
        finally:
            storage.close()

    def test_temporal_summary_present(self, tmp_path):
        """Temporal summary should be present when memories are excluded."""
        kernle, storage = self._create_kernle_with_many_memories(tmp_path)
        try:
            result = kernle.load(budget=500, sync=False, track_access=False)
            meta = result["_meta"]

            assert "temporal_summary" in meta
            if meta["excluded_count"] > 0:
                assert meta["temporal_summary"] is not None
                assert "Memory spans" in meta["temporal_summary"]
                assert "excluded memories" in meta["temporal_summary"]
        finally:
            storage.close()

    def test_topic_clusters_present(self, tmp_path):
        """Topic clusters should be present when memories are excluded."""
        kernle, storage = self._create_kernle_with_many_memories(tmp_path)
        try:
            result = kernle.load(budget=500, sync=False, track_access=False)
            meta = result["_meta"]

            assert "topic_clusters" in meta
            assert isinstance(meta["topic_clusters"], list)
            if meta["excluded_count"] > 0:
                assert len(meta["topic_clusters"]) > 0
                assert len(meta["topic_clusters"]) <= 6
        finally:
            storage.close()

    def test_no_echoes_when_all_fit(self, tmp_path):
        """When all memories fit in budget, echoes should be empty."""
        db_path = tmp_path / "test_no_echoes.db"
        checkpoint_dir = tmp_path / "checkpoints_no_echoes"
        checkpoint_dir.mkdir()

        storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
        kernle = Kernle(
            agent_id="test_agent",
            storage=storage,
            checkpoint_dir=checkpoint_dir,
        )

        # Add just one small note
        note = Note(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Short note",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        storage.save_note(note)

        try:
            # Large budget so everything fits
            result = kernle.load(budget=50000, sync=False, track_access=False)
            meta = result["_meta"]

            assert meta["excluded_count"] == 0
            assert meta["echoes"] == []
            assert meta["temporal_summary"] is None
            assert meta["topic_clusters"] == []
        finally:
            storage.close()

    def test_echo_hints_are_truncated(self, tmp_path):
        """Echo hints should be truncated to ~8 words."""
        kernle, storage = self._create_kernle_with_many_memories(tmp_path)
        try:
            result = kernle.load(budget=500, sync=False, track_access=False)
            echoes = result["_meta"]["echoes"]

            for echo in echoes:
                hint = echo["hint"]
                if "..." in hint:
                    word_count = len(hint.replace("...", "").split())
                    assert word_count <= 8, f"Hint has {word_count} words: {hint}"
        finally:
            storage.close()

    def test_existing_meta_fields_preserved(self, tmp_path):
        """Budget_used, budget_total, excluded_count should still be present."""
        kernle, storage = self._create_kernle_with_many_memories(tmp_path)
        try:
            result = kernle.load(budget=500, sync=False, track_access=False)
            meta = result["_meta"]

            assert "budget_used" in meta
            assert "budget_total" in meta
            assert "excluded_count" in meta
            assert meta["budget_total"] == 500
            assert meta["budget_used"] > 0
        finally:
            storage.close()
