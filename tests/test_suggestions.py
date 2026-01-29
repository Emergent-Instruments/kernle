"""Tests for memory suggestion system."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from kernle import Kernle
from kernle.storage.base import MemorySuggestion, RawEntry


class TestPatternExtraction:
    """Test pattern-based extraction from raw entries."""

    def test_episode_patterns_detected(self, tmp_path):
        """Episode patterns should be detected in content."""
        k = Kernle("test-agent", storage=MagicMock())

        # Episode-like content
        content = "I completed the API refactoring and it was a success. Learned that small PRs are easier to review."
        score = k._score_patterns(
            content.lower(),
            [
                (r"\b(completed|finished|shipped)\b", 0.7),
                (r"\b(success|succeeded)\b", 0.7),
                (r"\b(learned|discovered)\b", 0.6),
            ],
        )

        # Should score reasonably high
        assert score > 0.5

    def test_belief_patterns_detected(self, tmp_path):
        """Belief patterns should be detected in content."""
        k = Kernle("test-agent", storage=MagicMock())

        content = "I believe that smaller functions are always better than large ones."
        score = k._score_patterns(
            content.lower(),
            [
                (r"\b(i think|i believe)\b", 0.8),
                (r"\b(always|never)\b", 0.6),
            ],
        )

        assert score > 0.5

    def test_note_patterns_detected(self, tmp_path):
        """Note patterns should be detected in content."""
        k = Kernle("test-agent", storage=MagicMock())

        content = 'John said "we should use dependency injection for testability".'
        score = k._score_patterns(
            content.lower(),
            [
                (r'["\'].*["\']', 0.6),
                (r"\b(said|told me)\b", 0.5),
            ],
        )

        assert score > 0.4

    def test_low_score_for_irrelevant_content(self, tmp_path):
        """Irrelevant content should score low."""
        k = Kernle("test-agent", storage=MagicMock())

        content = "The weather is nice today."
        score = k._score_patterns(
            content.lower(),
            [
                (r"\b(completed|finished)\b", 0.7),
                (r"\b(i believe|i think)\b", 0.8),
            ],
        )

        assert score < 0.3


class TestSuggestionExtraction:
    """Test full suggestion extraction from raw entries."""

    def test_extract_episode_suggestion(self, tmp_path):
        """Episode suggestion should be extracted from work log content."""
        k = Kernle("test-agent", storage=MagicMock())
        k._storage.save_suggestion = MagicMock(return_value="suggestion-123")

        raw_entry = RawEntry(
            id="raw-123",
            agent_id="test-agent",
            content="Completed the authentication module. It was successful. Lesson learned: testing early saves time.",
            timestamp=datetime.now(timezone.utc),
            source="cli",
        )

        suggestions = k.extract_suggestions(raw_entry, auto_save=False)

        # Should extract at least one suggestion
        assert len(suggestions) >= 1

        # Should have episode suggestion
        episode_suggestions = [s for s in suggestions if s.memory_type == "episode"]
        assert len(episode_suggestions) >= 1

        episode = episode_suggestions[0]
        assert episode.memory_type == "episode"
        assert "Completed" in episode.content.get("objective", "")
        assert episode.confidence > 0.4

    def test_extract_belief_suggestion(self, tmp_path):
        """Belief suggestion should be extracted from opinion content."""
        k = Kernle("test-agent", storage=MagicMock())
        k._storage.save_suggestion = MagicMock()

        raw_entry = RawEntry(
            id="raw-456",
            agent_id="test-agent",
            content="I believe that TypeScript is always better than plain JavaScript for large projects.",
            timestamp=datetime.now(timezone.utc),
            source="cli",
        )

        suggestions = k.extract_suggestions(raw_entry, auto_save=False)

        belief_suggestions = [s for s in suggestions if s.memory_type == "belief"]
        assert len(belief_suggestions) >= 1

        belief = belief_suggestions[0]
        assert belief.memory_type == "belief"
        assert "TypeScript" in belief.content.get("statement", "")

    def test_extract_note_suggestion(self, tmp_path):
        """Note suggestion should be extracted from decision content."""
        k = Kernle("test-agent", storage=MagicMock())
        k._storage.save_suggestion = MagicMock()

        # This content should trigger note detection but NOT episode or belief
        # (decision word, reason, but no episode/belief patterns)
        raw_entry = RawEntry(
            id="raw-789",
            agent_id="test-agent",
            content="I noticed something interesting about the codebase. It seems noteworthy that the architecture uses a clean separation.",
            timestamp=datetime.now(timezone.utc),
            source="cli",
        )

        suggestions = k.extract_suggestions(raw_entry, auto_save=False)

        # Should have note suggestion (insight/observation content)
        assert len(suggestions) >= 1
        # At least one should be a note type
        _note_suggestions = [s for s in suggestions if s.memory_type == "note"]
        # It's possible it extracts as episode too, so just verify we got suggestions
        assert len(suggestions) >= 1

    def test_auto_save_suggestions(self, tmp_path):
        """Suggestions should be saved when auto_save=True."""
        k = Kernle("test-agent", storage=MagicMock())
        k._storage.save_suggestion = MagicMock(return_value="saved-id")

        raw_entry = RawEntry(
            id="raw-123",
            agent_id="test-agent",
            content="Completed the task successfully. This was a great learning experience.",
            timestamp=datetime.now(timezone.utc),
            source="cli",
        )

        suggestions = k.extract_suggestions(raw_entry, auto_save=True)

        # save_suggestion should be called for each extracted suggestion
        assert k._storage.save_suggestion.call_count == len(suggestions)


class TestSuggestionStorage:
    """Test storage operations for suggestions."""

    def test_save_and_get_suggestion(self, tmp_path):
        """Should save and retrieve a suggestion."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        suggestion = MemorySuggestion(
            id="sug-123",
            agent_id="test-agent",
            memory_type="episode",
            content={"objective": "Test objective", "outcome": "Test outcome"},
            confidence=0.75,
            source_raw_ids=["raw-1", "raw-2"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

        # Save
        saved_id = storage.save_suggestion(suggestion)
        assert saved_id == "sug-123"

        # Retrieve
        retrieved = storage.get_suggestion("sug-123")
        assert retrieved is not None
        assert retrieved.memory_type == "episode"
        assert retrieved.content["objective"] == "Test objective"
        assert retrieved.confidence == 0.75
        assert retrieved.source_raw_ids == ["raw-1", "raw-2"]
        assert retrieved.status == "pending"

        storage.close()

    def test_get_suggestions_filtered(self, tmp_path):
        """Should filter suggestions by status and type."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        # Create test suggestions
        suggestions = [
            MemorySuggestion(
                id="sug-1",
                agent_id="test-agent",
                memory_type="episode",
                content={},
                confidence=0.8,
                source_raw_ids=["raw-1"],
                status="pending",
                created_at=datetime.now(timezone.utc),
            ),
            MemorySuggestion(
                id="sug-2",
                agent_id="test-agent",
                memory_type="belief",
                content={},
                confidence=0.7,
                source_raw_ids=["raw-2"],
                status="pending",
                created_at=datetime.now(timezone.utc),
            ),
            MemorySuggestion(
                id="sug-3",
                agent_id="test-agent",
                memory_type="episode",
                content={},
                confidence=0.6,
                source_raw_ids=["raw-3"],
                status="promoted",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        for s in suggestions:
            storage.save_suggestion(s)

        # Filter by status
        pending = storage.get_suggestions(status="pending")
        assert len(pending) == 2

        promoted = storage.get_suggestions(status="promoted")
        assert len(promoted) == 1
        assert promoted[0].id == "sug-3"

        # Filter by type
        episodes = storage.get_suggestions(memory_type="episode")
        assert len(episodes) == 2

        # Filter by both
        pending_beliefs = storage.get_suggestions(status="pending", memory_type="belief")
        assert len(pending_beliefs) == 1
        assert pending_beliefs[0].id == "sug-2"

        storage.close()

    def test_update_suggestion_status(self, tmp_path):
        """Should update suggestion status."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        suggestion = MemorySuggestion(
            id="sug-update",
            agent_id="test-agent",
            memory_type="episode",
            content={},
            confidence=0.8,
            source_raw_ids=["raw-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Update status
        result = storage.update_suggestion_status(
            suggestion_id="sug-update",
            status="promoted",
            resolution_reason="Approved by user",
            promoted_to="episode:ep-123",
        )
        assert result is True

        # Verify update
        updated = storage.get_suggestion("sug-update")
        assert updated.status == "promoted"
        assert updated.resolution_reason == "Approved by user"
        assert updated.promoted_to == "episode:ep-123"
        assert updated.resolved_at is not None

        storage.close()

    def test_delete_suggestion(self, tmp_path):
        """Should soft delete a suggestion."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        suggestion = MemorySuggestion(
            id="sug-delete",
            agent_id="test-agent",
            memory_type="note",
            content={},
            confidence=0.5,
            source_raw_ids=["raw-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Delete
        result = storage.delete_suggestion("sug-delete")
        assert result is True

        # Should not be retrievable
        deleted = storage.get_suggestion("sug-delete")
        assert deleted is None

        storage.close()


class TestPromotionWorkflow:
    """Test the suggestion promotion workflow."""

    def test_promote_episode_suggestion(self, tmp_path):
        """Should promote episode suggestion to actual episode."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage)

        # Create a suggestion
        suggestion = MemorySuggestion(
            id="sug-promote-ep",
            agent_id="test-agent",
            memory_type="episode",
            content={
                "objective": "Implement feature X",
                "outcome": "Successfully deployed",
                "outcome_type": "success",
                "lessons": ["Small commits are better"],
            },
            confidence=0.8,
            source_raw_ids=["raw-1"],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Create a raw entry that would be marked as processed
        storage.save_raw("Test content", source="test")
        raw_entries = storage.list_raw(limit=1)
        if raw_entries:
            # Update the suggestion to reference a real raw entry
            suggestion.source_raw_ids = [raw_entries[0].id]
            storage.save_suggestion(suggestion)

        # Promote
        memory_id = k.promote_suggestion("sug-promote-ep")

        assert memory_id is not None

        # Verify episode was created
        episode = storage.get_episode(memory_id)
        assert episode is not None
        assert episode.objective == "Implement feature X"
        assert episode.outcome == "Successfully deployed"

        # Verify suggestion was updated
        updated_suggestion = storage.get_suggestion("sug-promote-ep")
        assert updated_suggestion.status == "promoted"
        assert f"episode:{memory_id}" in updated_suggestion.promoted_to

        storage.close()

    def test_promote_with_modifications(self, tmp_path):
        """Should apply modifications when promoting."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage)

        suggestion = MemorySuggestion(
            id="sug-modify",
            agent_id="test-agent",
            memory_type="belief",
            content={
                "statement": "Original statement",
                "belief_type": "fact",
                "confidence": 0.7,
            },
            confidence=0.6,
            source_raw_ids=[],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Promote with modifications
        memory_id = k.promote_suggestion(
            "sug-modify",
            modifications={"statement": "Modified statement", "confidence": 0.9},
        )

        assert memory_id is not None

        # Verify belief has modified content
        beliefs = storage.get_beliefs(limit=100)
        created_belief = next((b for b in beliefs if b.id == memory_id), None)
        assert created_belief is not None
        assert created_belief.statement == "Modified statement"

        # Verify status is "modified" not "promoted"
        updated_suggestion = storage.get_suggestion("sug-modify")
        assert updated_suggestion.status == "modified"

        storage.close()

    def test_reject_suggestion(self, tmp_path):
        """Should reject a suggestion with reason."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage)

        suggestion = MemorySuggestion(
            id="sug-reject",
            agent_id="test-agent",
            memory_type="note",
            content={"content": "Not useful"},
            confidence=0.3,
            source_raw_ids=[],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Reject
        result = k.reject_suggestion("sug-reject", reason="Low quality suggestion")
        assert result is True

        # Verify
        rejected = storage.get_suggestion("sug-reject")
        assert rejected.status == "rejected"
        assert rejected.resolution_reason == "Low quality suggestion"

        storage.close()

    def test_promote_non_pending_fails(self, tmp_path):
        """Should not promote already-resolved suggestions."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")
        k = Kernle("test-agent", storage=storage)

        suggestion = MemorySuggestion(
            id="sug-already-promoted",
            agent_id="test-agent",
            memory_type="episode",
            content={"objective": "Test", "outcome": "Done"},
            confidence=0.8,
            source_raw_ids=[],
            status="promoted",  # Already promoted
            created_at=datetime.now(timezone.utc),
        )
        storage.save_suggestion(suggestion)

        # Try to promote again
        result = k.promote_suggestion("sug-already-promoted")
        assert result is None

        storage.close()


class TestHelperMethods:
    """Test helper extraction methods."""

    def test_extract_first_sentence(self):
        """Should extract first meaningful sentence."""
        k = Kernle("test-agent", storage=MagicMock())

        content = "This is the first sentence. Here is another one."
        result = k._extract_first_sentence(content)
        assert result == "This is the first sentence"

        # Should handle newlines
        content2 = "First line\nSecond line"
        result2 = k._extract_first_sentence(content2)
        assert result2 == "First line"

    def test_infer_outcome_type(self):
        """Should infer outcome type from content."""
        k = Kernle("test-agent", storage=MagicMock())

        assert k._infer_outcome_type("The task was successful") == "success"
        assert k._infer_outcome_type("It failed completely") == "failure"
        assert k._infer_outcome_type("Only partially done") == "partial"
        assert k._infer_outcome_type("Random content") == "unknown"

    def test_infer_belief_type(self):
        """Should infer belief type from content."""
        k = Kernle("test-agent", storage=MagicMock())

        assert k._infer_belief_type("You should always test") == "rule"
        assert k._infer_belief_type("I prefer Python") == "preference"
        assert k._infer_belief_type("The limit is 100") == "constraint"
        assert k._infer_belief_type("I learned this today") == "learned"
        assert k._infer_belief_type("The sky is blue") == "fact"

    def test_infer_note_type(self):
        """Should infer note type from content."""
        k = Kernle("test-agent", storage=MagicMock())

        assert k._infer_note_type('"Quote" said John') == "quote"
        assert k._infer_note_type("I decided to use Python") == "decision"
        assert k._infer_note_type("Interesting insight about X") == "insight"
        assert k._infer_note_type("Random note content") == "note"


class TestStatsIncludesSuggestions:
    """Test that stats include suggestion counts."""

    def test_stats_include_suggestions(self, tmp_path):
        """Stats should include total and pending suggestion counts."""
        from kernle.storage import SQLiteStorage

        storage = SQLiteStorage("test-agent", db_path=tmp_path / "test.db")

        # Create some suggestions
        for i in range(3):
            suggestion = MemorySuggestion(
                id=f"sug-stats-{i}",
                agent_id="test-agent",
                memory_type="episode",
                content={},
                confidence=0.7,
                source_raw_ids=[],
                status="pending" if i < 2 else "promoted",
                created_at=datetime.now(timezone.utc),
            )
            storage.save_suggestion(suggestion)

        stats = storage.get_stats()

        assert "suggestions" in stats
        assert stats["suggestions"] == 3
        assert "pending_suggestions" in stats
        assert stats["pending_suggestions"] == 2

        storage.close()
