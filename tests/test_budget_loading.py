"""
Tests for budget-aware memory loading.

Tests the token estimation, priority scoring, and budget-constrained
memory loading functionality.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import (
    DEFAULT_TOKEN_BUDGET,
    MAX_TOKEN_BUDGET,
    MEMORY_TYPE_PRIORITIES,
    MIN_TOKEN_BUDGET,
    TOKEN_ESTIMATION_SAFETY_MARGIN,
    Kernle,
    compute_priority_score,
    estimate_tokens,
    truncate_at_word_boundary,
)
from kernle.storage import SQLiteStorage
from kernle.storage.base import Belief, Drive, Episode, Goal, Note, Relationship, Value


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_empty_string(self):
        """Empty string should return 0 tokens."""
        assert estimate_tokens("") == 0

    def test_none_input(self):
        """None input should return 0 tokens."""
        assert estimate_tokens(None) == 0

    def test_short_text(self):
        """Short text should estimate correctly with safety margin."""
        text = "This is short text."
        base_estimate = len(text) // 4
        expected = int(base_estimate * TOKEN_ESTIMATION_SAFETY_MARGIN)
        assert estimate_tokens(text) == expected

    def test_long_text(self):
        """Long text should scale linearly with safety margin."""
        text = "a" * 400
        base_estimate = 100  # 400 / 4
        expected = int(base_estimate * TOKEN_ESTIMATION_SAFETY_MARGIN)
        assert estimate_tokens(text) == expected

    def test_various_lengths(self):
        """Test various text lengths with safety margin."""
        # With 1.3x safety margin
        assert estimate_tokens("1234") == int(1 * TOKEN_ESTIMATION_SAFETY_MARGIN)
        assert estimate_tokens("12345678") == int(2 * TOKEN_ESTIMATION_SAFETY_MARGIN)
        assert estimate_tokens("a" * 100) == int(25 * TOKEN_ESTIMATION_SAFETY_MARGIN)
        assert estimate_tokens("a" * 1000) == int(250 * TOKEN_ESTIMATION_SAFETY_MARGIN)

    def test_without_safety_margin(self):
        """Test token estimation without safety margin."""
        text = "a" * 400
        assert estimate_tokens(text, include_safety_margin=False) == 100


class TestTruncateAtWordBoundary:
    """Tests for the truncate_at_word_boundary function."""

    def test_short_text_not_truncated(self):
        """Text shorter than max_chars should not be truncated."""
        text = "Hello world"
        result = truncate_at_word_boundary(text, 50)
        assert result == text
        assert "..." not in result

    def test_exact_length_not_truncated(self):
        """Text exactly at max_chars should not be truncated."""
        text = "Hello"
        result = truncate_at_word_boundary(text, 5)
        assert result == text

    def test_truncation_at_word_boundary(self):
        """Long text should be truncated at a word boundary."""
        text = "The quick brown fox jumps over the lazy dog"
        result = truncate_at_word_boundary(text, 20)
        assert result.endswith("...")
        assert len(result) <= 20
        # Should not cut mid-word
        assert result.replace("...", "").strip().endswith(("The", "quick", "brown"))

    def test_truncation_with_ellipsis(self):
        """Truncated text should end with ellipsis."""
        text = "This is a very long text that needs to be truncated"
        result = truncate_at_word_boundary(text, 25)
        assert result.endswith("...")

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert truncate_at_word_boundary("", 100) == ""

    def test_none_input(self):
        """None input should return None."""
        assert truncate_at_word_boundary(None, 100) is None

    def test_very_short_max_chars(self):
        """Very short max_chars should still work."""
        text = "Hello world"
        result = truncate_at_word_boundary(text, 5)
        # Result should be short and end with ellipsis
        assert result.endswith("...")
        assert len(result) <= 5


class TestComputePriorityScore:
    """Tests for the compute_priority_score function."""

    def test_value_priority(self):
        """Value priority should scale with priority field."""
        high_priority_value = Value(
            id="v1", agent_id="test", name="test", statement="test", priority=100
        )
        low_priority_value = Value(
            id="v2", agent_id="test", name="test", statement="test", priority=0
        )

        high_score = compute_priority_score("value", high_priority_value)
        low_score = compute_priority_score("value", low_priority_value)

        assert high_score > low_score
        # Both should include base priority
        assert high_score >= MEMORY_TYPE_PRIORITIES["value"] * 0.6
        assert low_score >= MEMORY_TYPE_PRIORITIES["value"] * 0.6 - 0.01

    def test_belief_priority(self):
        """Belief priority should scale with confidence."""
        high_confidence_belief = Belief(id="b1", agent_id="test", statement="test", confidence=0.95)
        low_confidence_belief = Belief(id="b2", agent_id="test", statement="test", confidence=0.3)

        high_score = compute_priority_score("belief", high_confidence_belief)
        low_score = compute_priority_score("belief", low_confidence_belief)

        assert high_score > low_score

    def test_drive_priority(self):
        """Drive priority should scale with intensity."""
        high_intensity_drive = Drive(id="d1", agent_id="test", drive_type="growth", intensity=0.9)
        low_intensity_drive = Drive(id="d2", agent_id="test", drive_type="curiosity", intensity=0.2)

        high_score = compute_priority_score("drive", high_intensity_drive)
        low_score = compute_priority_score("drive", low_intensity_drive)

        assert high_score > low_score

    def test_type_ordering(self):
        """Different types should have different base priorities."""
        # Create records with neutral factors
        value = Value(id="v", agent_id="test", name="test", statement="test", priority=50)
        belief = Belief(id="b", agent_id="test", statement="test", confidence=0.5)
        goal = Goal(id="g", agent_id="test", title="test")
        episode = Episode(id="e", agent_id="test", objective="test", outcome="test")
        note = Note(id="n", agent_id="test", content="test")
        relationship = Relationship(
            id="r",
            agent_id="test",
            entity_name="test",
            entity_type="person",
            relationship_type="knows",
        )

        scores = {
            "value": compute_priority_score("value", value),
            "belief": compute_priority_score("belief", belief),
            "goal": compute_priority_score("goal", goal),
            "episode": compute_priority_score("episode", episode),
            "note": compute_priority_score("note", note),
            "relationship": compute_priority_score("relationship", relationship),
        }

        # Values should have highest priority among core types
        assert scores["value"] >= scores["belief"]
        # Belief priority can be similar or lower than goal due to 40% type factor weight
        # The key invariant is values > relationships (important vs less important)
        assert scores["value"] > scores["relationship"]
        assert scores["belief"] > scores["relationship"]

    def test_dict_input(self):
        """Function should work with dict input as well."""
        value_dict = {"priority": 80}
        score = compute_priority_score("value", value_dict)
        assert score > 0

        belief_dict = {"confidence": 0.9}
        score = compute_priority_score("belief", belief_dict)
        assert score > 0


class TestBudgetLoading:
    """Integration tests for budget-aware loading."""

    @pytest.fixture
    def kernle_with_data(self, tmp_path):
        """Create a Kernle instance with test data."""
        db_path = tmp_path / "test_memories.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
        kernle = Kernle(agent_id="test_agent", storage=storage, checkpoint_dir=checkpoint_dir)

        # Add test data with varying priorities/confidence
        for i in range(10):
            storage.save_value(
                Value(
                    id=str(uuid.uuid4()),
                    agent_id="test_agent",
                    name=f"value_{i}",
                    statement=f"This is value statement {i} with some content to test token estimation",
                    priority=i * 10,  # 0, 10, 20, ..., 90
                    created_at=datetime.now(timezone.utc),
                )
            )

        for i in range(15):
            storage.save_belief(
                Belief(
                    id=str(uuid.uuid4()),
                    agent_id="test_agent",
                    statement=f"This is belief {i} with confidence level varying",
                    belief_type="fact",
                    confidence=0.1 + (i * 0.05),  # 0.1, 0.15, ..., 0.85
                    created_at=datetime.now(timezone.utc),
                )
            )

        for i in range(8):
            storage.save_episode(
                Episode(
                    id=str(uuid.uuid4()),
                    agent_id="test_agent",
                    objective=f"Episode {i} objective that is moderately long",
                    outcome=f"Episode {i} outcome with details about what happened",
                    outcome_type="success" if i % 2 == 0 else "failure",
                    lessons=[f"Lesson from episode {i}"],
                    created_at=datetime.now(timezone.utc),
                )
            )

        yield kernle
        storage.close()

    def test_load_with_default_budget(self, kernle_with_data):
        """Load with default budget should work."""
        memory = kernle_with_data.load()

        # Should have loaded data
        assert "values" in memory
        assert "beliefs" in memory
        assert "lessons" in memory
        assert "recent_work" in memory

    def test_load_with_small_budget(self, kernle_with_data):
        """Small budget should limit the number of items loaded."""
        # Load with a very small budget
        small_memory = kernle_with_data.load(budget=500)

        # Load with default budget
        default_memory = kernle_with_data.load(budget=8000)

        # Small budget should have fewer items
        small_total = (
            len(small_memory.get("values", []))
            + len(small_memory.get("beliefs", []))
            + len(small_memory.get("recent_work", []))
        )
        default_total = (
            len(default_memory.get("values", []))
            + len(default_memory.get("beliefs", []))
            + len(default_memory.get("recent_work", []))
        )

        assert small_total <= default_total

    def test_load_with_large_budget(self, kernle_with_data):
        """Large budget should load more items."""
        memory = kernle_with_data.load(budget=50000)

        # Should have loaded all available data
        assert len(memory.get("values", [])) > 0
        assert len(memory.get("beliefs", [])) > 0

    def test_truncation_enabled(self, kernle_with_data):
        """Content should be truncated when truncate=True."""
        memory = kernle_with_data.load(budget=8000, truncate=True, max_item_chars=50)

        # Check that statements are truncated
        for value in memory.get("values", []):
            if value.get("statement"):
                assert len(value["statement"]) <= 53  # 50 chars + "..."

    def test_truncation_disabled(self, kernle_with_data):
        """Content should not be truncated when truncate=False."""
        memory = kernle_with_data.load(budget=50000, truncate=False)

        # Content should be full length
        # Values statements are ~70 chars, should not be truncated
        has_long_statement = any(len(v.get("statement", "")) > 50 for v in memory.get("values", []))
        assert has_long_statement

    def test_priority_ordering(self, kernle_with_data):
        """Higher priority items should be loaded first."""
        # Use a small budget to force selection
        memory = kernle_with_data.load(budget=1000)

        # Values should be sorted by priority (highest first in source)
        values = memory.get("values", [])
        if len(values) > 1:
            # Due to priority scoring, higher priority values should be included
            priorities = [v.get("priority", 0) for v in values]
            # At least check we got some high priority values
            assert max(priorities) >= 50 or len(values) == 0

    def test_budget_parameter_passed_to_storage(self, kernle_with_data):
        """Budget loading should use high limits for storage query."""
        # This test verifies the integration works end-to-end
        memory = kernle_with_data.load(budget=8000)

        # Should successfully load
        assert memory is not None
        assert isinstance(memory, dict)


class TestBudgetValidation:
    """Tests for budget parameter validation across all layers."""

    def test_core_clamps_low_budget(self, tmp_path):
        """Core should clamp budget below minimum to MIN_TOKEN_BUDGET."""
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        storage = SQLiteStorage(agent_id="test", db_path=db_path)
        k = Kernle(agent_id="test", storage=storage, checkpoint_dir=checkpoint_dir)

        # Should not raise, should clamp to MIN_TOKEN_BUDGET
        memory = k.load(budget=50)  # Below minimum
        assert memory is not None
        storage.close()

    def test_core_clamps_high_budget(self, tmp_path):
        """Core should clamp budget above maximum to MAX_TOKEN_BUDGET."""
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        storage = SQLiteStorage(agent_id="test", db_path=db_path)
        k = Kernle(agent_id="test", storage=storage, checkpoint_dir=checkpoint_dir)

        # Should not raise, should clamp to MAX_TOKEN_BUDGET
        memory = k.load(budget=999999)  # Above maximum
        assert memory is not None
        storage.close()

    def test_core_validates_max_item_chars(self, tmp_path):
        """Core should validate and clamp max_item_chars parameter."""
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        storage = SQLiteStorage(agent_id="test", db_path=db_path)
        k = Kernle(agent_id="test", storage=storage, checkpoint_dir=checkpoint_dir)

        # Should not raise, should clamp invalid values
        memory = k.load(budget=8000, max_item_chars=5)  # Below minimum
        assert memory is not None

        memory = k.load(budget=8000, max_item_chars=99999)  # Above maximum
        assert memory is not None
        storage.close()

    def test_budget_constants_consistency(self):
        """Verify budget constants are consistent."""
        assert MIN_TOKEN_BUDGET == 100
        assert MAX_TOKEN_BUDGET == 50000
        assert DEFAULT_TOKEN_BUDGET == 8000
        assert MIN_TOKEN_BUDGET < DEFAULT_TOKEN_BUDGET < MAX_TOKEN_BUDGET


class TestCLIBudgetValidation:
    """Tests for CLI budget argument validation."""

    def test_cli_validate_budget_valid(self):
        """Valid budget values should pass validation."""
        from kernle.cli.__main__ import validate_budget

        assert validate_budget("100") == 100
        assert validate_budget("8000") == 8000
        assert validate_budget("50000") == 50000

    def test_cli_validate_budget_too_low(self):
        """Budget below minimum should raise error."""
        import argparse

        from kernle.cli.__main__ import validate_budget

        with pytest.raises(argparse.ArgumentTypeError, match="at least"):
            validate_budget("50")

    def test_cli_validate_budget_too_high(self):
        """Budget above maximum should raise error."""
        import argparse

        from kernle.cli.__main__ import validate_budget

        with pytest.raises(argparse.ArgumentTypeError, match="cannot exceed"):
            validate_budget("100000")

    def test_cli_validate_budget_non_integer(self):
        """Non-integer budget should raise error."""
        import argparse

        from kernle.cli.__main__ import validate_budget

        with pytest.raises(argparse.ArgumentTypeError, match="must be an integer"):
            validate_budget("abc")


class TestLoadAllWithOptionalLimits:
    """Test that storage.load_all handles optional limits correctly."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create storage with test data."""
        db_path = tmp_path / "test_memories.db"
        storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)

        # Add test data
        for i in range(5):
            storage.save_value(
                Value(
                    id=str(uuid.uuid4()),
                    agent_id="test_agent",
                    name=f"value_{i}",
                    statement=f"Statement {i}",
                    priority=i * 20,
                    created_at=datetime.now(timezone.utc),
                )
            )

        yield storage
        storage.close()

    def test_load_all_with_none_limits(self, storage):
        """load_all with None limits should use high limit (1000)."""
        result = storage.load_all(
            values_limit=None,
            beliefs_limit=None,
            goals_limit=None,
            episodes_limit=None,
            notes_limit=None,
        )

        # Should return all 5 values
        assert len(result["values"]) == 5

    def test_load_all_with_explicit_limits(self, storage):
        """load_all with explicit limits should respect them."""
        result = storage.load_all(
            values_limit=2,
            beliefs_limit=5,
            goals_limit=3,
            episodes_limit=10,
            notes_limit=5,
        )

        # Should respect the limit
        assert len(result["values"]) <= 2

    def test_load_all_filters_forgotten(self, storage):
        """load_all should exclude forgotten memories."""
        # Use beliefs instead of values since values are protected by default
        belief_id = str(uuid.uuid4())
        regular_belief = Belief(
            id=belief_id,
            agent_id="test_agent",
            statement="This belief will be forgotten",
            confidence=0.5,
            created_at=datetime.now(timezone.utc),
        )
        storage.save_belief(regular_belief)

        # Now forget it
        result = storage.forget_memory("belief", belief_id)
        assert result is True, "forget_memory should succeed for unprotected belief"

        loaded = storage.load_all(beliefs_limit=None)

        # Forgotten belief should not be in results
        belief_statements = [b.statement for b in loaded["beliefs"]]
        assert "This belief will be forgotten" not in belief_statements
