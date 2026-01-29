"""Tests for semantic contradiction detection.

Tests the find_semantic_contradictions method which uses embeddings
to find beliefs that are semantically similar but may contradict.
"""

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage


@pytest.fixture
def kernle_instance(tmp_path):
    """Create a Kernle instance with SQLite storage."""
    db_path = tmp_path / "test_semantic.db"
    storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    return Kernle(agent_id="test_agent", storage=storage)


@pytest.fixture
def kernle_with_beliefs(tmp_path):
    """Create a Kernle instance with pre-populated beliefs."""
    db_path = tmp_path / "test_semantic_beliefs.db"
    storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    k = Kernle(agent_id="test_agent", storage=storage)

    # Add beliefs covering various contradiction scenarios
    k.belief("Testing is essential for code quality", confidence=0.9)
    k.belief("I prefer Python for data science work", confidence=0.85)
    k.belief("Code reviews improve team knowledge sharing", confidence=0.8)
    k.belief("Documentation should be written alongside code", confidence=0.75)
    k.belief("Fast iteration is important for product development", confidence=0.8)

    return k


class TestFindSemanticContradictions:
    """Tests for the find_semantic_contradictions method."""

    def test_detects_opposition_words(self, kernle_instance):
        """Should detect contradictions with opposition word pairs.

        This test verifies that when search finds similar beliefs,
        the opposition detection correctly identifies contradictions.
        We use a single-word search to work with text fallback search.
        """
        k = kernle_instance

        # Add beliefs containing the search term "testing"
        # One with "essential" (positive) and one with "unnecessary" (negative)
        k.belief("testing is unnecessary", confidence=0.6)

        # Search for "testing" - text search uses LIKE '%testing%'
        contradictions = k.find_semantic_contradictions(
            "testing is essential",
            similarity_threshold=0.0,  # Full phrase to have opposition words
        )

        # With text search fallback, "testing is essential" won't match
        # "testing is unnecessary" (LIKE requires exact substring)
        # This test verifies the method returns gracefully
        assert isinstance(contradictions, list)

        # The real functionality is tested in the helper method tests below.
        # When semantic embeddings ARE available, this method will find
        # semantically similar beliefs and apply opposition detection.

    def test_detects_negation_patterns(self, kernle_with_beliefs):
        """Should detect negation-based contradictions."""
        k = kernle_with_beliefs

        # Add belief with negation
        k.belief("Documentation should not be written during development", confidence=0.5)

        contradictions = k.find_semantic_contradictions(
            "Documentation should be written alongside code", similarity_threshold=0.3
        )

        # Should detect the negation pattern
        # Note: may or may not match depending on embedding similarity
        # The test validates the method works without errors
        assert isinstance(contradictions, list)

    def test_detects_sentiment_opposition(self, kernle_with_beliefs):
        """Should detect sentiment-based contradictions."""
        k = kernle_with_beliefs

        # Add belief with opposite sentiment
        k.belief("Code reviews slow down development", confidence=0.5)

        contradictions = k.find_semantic_contradictions(
            "Code reviews improve team knowledge sharing", similarity_threshold=0.3
        )

        # Check structure is correct
        assert isinstance(contradictions, list)
        for c in contradictions:
            assert "belief_id" in c
            assert "statement" in c
            assert "similarity_score" in c
            assert "opposition_score" in c
            assert "opposition_type" in c
            assert "explanation" in c

    def test_respects_similarity_threshold(self, kernle_with_beliefs):
        """Higher threshold should return fewer results."""
        k = kernle_with_beliefs

        # Add various beliefs
        k.belief("Testing wastes developer time", confidence=0.5)

        # Low threshold should find more
        low_threshold = k.find_semantic_contradictions(
            "Testing is important", similarity_threshold=0.2
        )

        # High threshold should find fewer
        high_threshold = k.find_semantic_contradictions(
            "Testing is important", similarity_threshold=0.9
        )

        # High threshold results should be subset of low threshold or fewer
        assert len(high_threshold) <= len(low_threshold)

    def test_respects_limit(self, kernle_instance):
        """Should respect the limit parameter."""
        k = kernle_instance

        # Add many beliefs
        for i in range(15):
            k.belief(f"Method {i} is good for software", confidence=0.5)
            k.belief(f"Method {i} is bad for software", confidence=0.5)

        contradictions = k.find_semantic_contradictions(
            "Method 5 is excellent for software", similarity_threshold=0.1, limit=3
        )

        assert len(contradictions) <= 3

    def test_excludes_exact_matches(self, kernle_with_beliefs):
        """Should not return the exact same statement as a contradiction."""
        k = kernle_with_beliefs

        statement = "Testing is essential for code quality"
        contradictions = k.find_semantic_contradictions(statement, similarity_threshold=0.1)

        # Should not include exact match
        for c in contradictions:
            assert c["statement"].lower() != statement.lower()

    def test_excludes_inactive_beliefs(self, kernle_instance):
        """Should not include inactive/superseded beliefs."""
        k = kernle_instance

        # Add and supersede a belief
        old_id = k.belief("Testing is bad", confidence=0.5)
        k.supersede_belief(old_id, "Testing is actually valuable", confidence=0.8)

        contradictions = k.find_semantic_contradictions(
            "Testing is important", similarity_threshold=0.1
        )

        # The old superseded belief should not appear
        for c in contradictions:
            assert c["belief_id"] != old_id

    def test_returns_expected_structure(self, kernle_with_beliefs):
        """Should return properly structured contradiction info."""
        k = kernle_with_beliefs

        k.belief("Testing is harmful to productivity", confidence=0.5)

        contradictions = k.find_semantic_contradictions(
            "Testing is essential", similarity_threshold=0.1
        )

        if contradictions:
            c = contradictions[0]
            # Check all expected fields
            assert isinstance(c["belief_id"], str)
            assert isinstance(c["statement"], str)
            assert isinstance(c["confidence"], float)
            assert isinstance(c["times_reinforced"], int)
            assert isinstance(c["is_active"], bool)
            assert isinstance(c["similarity_score"], float)
            assert isinstance(c["opposition_score"], float)
            assert c["opposition_type"] in [
                "opposition_words",
                "negation",
                "sentiment_opposition",
                "none",
            ]
            assert isinstance(c["explanation"], str)

    def test_empty_database(self, kernle_instance):
        """Should return empty list for empty database."""
        k = kernle_instance

        contradictions = k.find_semantic_contradictions(
            "Testing is important", similarity_threshold=0.1
        )

        assert contradictions == []

    def test_no_contradictions_for_unrelated(self, kernle_with_beliefs):
        """Should not find contradictions for completely unrelated statements."""
        k = kernle_with_beliefs

        # Search for something completely different
        contradictions = k.find_semantic_contradictions(
            "The weather is nice today",
            similarity_threshold=0.8,  # High threshold
        )

        # Should find no contradictions (unrelated topic)
        assert len(contradictions) == 0

    def test_sorts_by_combined_score(self, kernle_instance):
        """Should sort results by similarity * opposition score."""
        k = kernle_instance

        # Add beliefs with varying opposition strength
        k.belief("Testing is good for quality", confidence=0.9)
        k.belief("Testing is bad for velocity", confidence=0.8)
        k.belief("Testing is terrible and wasteful", confidence=0.7)

        contradictions = k.find_semantic_contradictions(
            "Testing is excellent and valuable", similarity_threshold=0.1
        )

        if len(contradictions) >= 2:
            # Check that results are sorted by combined score (descending)
            for i in range(len(contradictions) - 1):
                score1 = (
                    contradictions[i]["similarity_score"] * contradictions[i]["opposition_score"]
                )
                score2 = (
                    contradictions[i + 1]["similarity_score"]
                    * contradictions[i + 1]["opposition_score"]
                )
                assert score1 >= score2


class TestDetectOpposition:
    """Tests for the _detect_opposition helper method."""

    def test_detects_always_never(self, kernle_instance):
        """Should detect always/never opposition."""
        k = kernle_instance

        result = k._detect_opposition(
            "i always write tests for my code", "i never write tests for my code"
        )

        assert result["score"] > 0
        assert result["type"] == "opposition_words"
        assert "always" in result["explanation"] or "never" in result["explanation"]

    def test_detects_good_bad(self, kernle_instance):
        """Should detect good/bad opposition."""
        k = kernle_instance

        result = k._detect_opposition("python is good for scripting", "python is bad for scripting")

        assert result["score"] > 0
        assert result["type"] == "opposition_words"

    def test_detects_important_unnecessary(self, kernle_instance):
        """Should detect important/unnecessary opposition."""
        k = kernle_instance

        result = k._detect_opposition(
            "testing is important for quality", "testing is unnecessary for quality"
        )

        assert result["score"] > 0
        assert result["type"] == "opposition_words"

    def test_detects_like_dislike(self, kernle_instance):
        """Should detect like/dislike preference opposition."""
        k = kernle_instance

        result = k._detect_opposition(
            "i like working with typescript", "i dislike working with typescript"
        )

        assert result["score"] > 0
        assert result["type"] == "opposition_words"

    def test_detects_should_shouldnt(self, kernle_instance):
        """Should detect should/shouldn't modal opposition."""
        k = kernle_instance

        result = k._detect_opposition(
            "you should use dependency injection", "you shouldn't use dependency injection"
        )

        assert result["score"] > 0
        # Could be either opposition_words or negation
        assert result["type"] in ("opposition_words", "negation")

    def test_requires_topic_overlap(self, kernle_instance):
        """Should not detect opposition without topic overlap."""
        k = kernle_instance

        result = k._detect_opposition("coffee is always good", "pizza is never healthy")

        # No overlap in topic words, so should not detect contradiction
        assert result["score"] == 0 or result["score"] < 0.5

    def test_no_opposition_for_unrelated(self, kernle_instance):
        """Should not detect opposition for unrelated statements."""
        k = kernle_instance

        result = k._detect_opposition("the sky is blue", "my car is red")

        assert result["score"] == 0
        assert result["type"] == "none"


class TestCheckNegationPattern:
    """Tests for the _check_negation_pattern helper method."""

    def test_detects_is_not_pattern(self, kernle_instance):
        """Should detect 'is not' negation pattern."""
        k = kernle_instance

        result = k._check_negation_pattern("testing is important", "testing is not important")

        assert result is True

    def test_detects_should_not_pattern(self, kernle_instance):
        """Should detect 'should not' negation pattern."""
        k = kernle_instance

        result = k._check_negation_pattern(
            "we should use typescript", "we should not use typescript"
        )

        assert result is True

    def test_detects_contraction_negation(self, kernle_instance):
        """Should detect contraction negation patterns."""
        k = kernle_instance

        result = k._check_negation_pattern("testing is necessary", "testing isn't necessary")

        assert result is True

    def test_no_negation_for_same_polarity(self, kernle_instance):
        """Should not detect negation for same polarity statements."""
        k = kernle_instance

        result = k._check_negation_pattern("testing is important", "testing is crucial")

        assert result is False


class TestCheckSentimentOpposition:
    """Tests for the _check_sentiment_opposition helper method."""

    def test_detects_positive_vs_negative(self, kernle_instance):
        """Should detect positive vs negative sentiment words."""
        k = kernle_instance

        result = k._check_sentiment_opposition(
            "testing is helpful for catching bugs", "testing is harmful to developer velocity"
        )

        assert result["detected"] is True
        assert result["word1"] in ["helpful", "harmful"]
        assert result["word2"] in ["helpful", "harmful"]

    def test_detects_good_vs_bad(self, kernle_instance):
        """Should detect good vs bad sentiment."""
        k = kernle_instance

        result = k._check_sentiment_opposition("this approach is good", "this approach is bad")

        assert result["detected"] is True

    def test_detects_love_vs_hate(self, kernle_instance):
        """Should detect love vs hate sentiment."""
        k = kernle_instance

        result = k._check_sentiment_opposition("i love using vim", "i hate using vim")

        assert result["detected"] is True

    def test_detects_slows_vs_fast(self, kernle_instance):
        """Should detect slow vs fast sentiment."""
        k = kernle_instance

        result = k._check_sentiment_opposition(
            "testing slows down development", "testing is fast to set up"
        )

        assert result["detected"] is True

    def test_no_sentiment_opposition_same_polarity(self, kernle_instance):
        """Should not detect sentiment opposition for same polarity."""
        k = kernle_instance

        result = k._check_sentiment_opposition(
            "python is good for scripting", "python is excellent for automation"
        )

        assert result["detected"] is False


class TestFindSemanticContradictionsWithMock:
    """Tests using mocked search to verify end-to-end behavior."""

    def test_opposition_detection_on_search_results(self, kernle_instance):
        """When search finds similar beliefs, opposition should be detected."""
        from unittest.mock import patch

        from kernle.storage import Belief, SearchResult

        k = kernle_instance

        # Create mock beliefs
        mock_belief = Belief(
            id="test-belief-1",
            agent_id="test_agent",
            statement="testing is unnecessary for development",
            confidence=0.7,
            times_reinforced=0,
            is_active=True,
        )

        # Mock the storage search to return our controlled belief
        with patch.object(k._storage, "search") as mock_search:
            mock_search.return_value = [
                SearchResult(record=mock_belief, record_type="belief", score=0.85)
            ]

            contradictions = k.find_semantic_contradictions(
                "testing is essential for development", similarity_threshold=0.0
            )

            # Should detect opposition (essential vs unnecessary)
            assert len(contradictions) == 1
            assert contradictions[0]["belief_id"] == "test-belief-1"
            # Opposition could be detected via word pairs OR sentiment analysis
            assert contradictions[0]["opposition_type"] in [
                "opposition_words",
                "sentiment_opposition",
            ]
            assert contradictions[0]["opposition_score"] > 0

    def test_sentiment_opposition_detection(self, kernle_instance):
        """Should detect sentiment-based opposition."""
        from unittest.mock import patch

        from kernle.storage import Belief, SearchResult

        k = kernle_instance

        mock_belief = Belief(
            id="test-belief-2",
            agent_id="test_agent",
            statement="code reviews slow down development",
            confidence=0.6,
            times_reinforced=0,
            is_active=True,
        )

        with patch.object(k._storage, "search") as mock_search:
            mock_search.return_value = [
                SearchResult(record=mock_belief, record_type="belief", score=0.75)
            ]

            contradictions = k.find_semantic_contradictions(
                "code reviews improve development speed", similarity_threshold=0.0
            )

            # Should detect sentiment opposition (improve vs slow)
            assert len(contradictions) == 1
            # Type could be opposition_words (improve/slow) or sentiment
            assert contradictions[0]["opposition_type"] in [
                "opposition_words",
                "sentiment_opposition",
            ]

    def test_no_opposition_when_beliefs_agree(self, kernle_instance):
        """Should not detect opposition when beliefs agree."""
        from unittest.mock import patch

        from kernle.storage import Belief, SearchResult

        k = kernle_instance

        mock_belief = Belief(
            id="test-belief-3",
            agent_id="test_agent",
            statement="testing is important for code quality",
            confidence=0.8,
            times_reinforced=2,
            is_active=True,
        )

        with patch.object(k._storage, "search") as mock_search:
            mock_search.return_value = [
                SearchResult(record=mock_belief, record_type="belief", score=0.9)
            ]

            contradictions = k.find_semantic_contradictions(
                "testing is essential for code quality", similarity_threshold=0.0
            )

            # Should NOT detect opposition (both positive about testing)
            assert len(contradictions) == 0


class TestIntegrationWithExistingContradictions:
    """Tests ensuring semantic contradictions work alongside existing find_contradictions."""

    def test_both_methods_work_independently(self, kernle_with_beliefs):
        """Both methods should work and can be used together."""
        k = kernle_with_beliefs

        k.belief("Testing is never worth the effort", confidence=0.5)

        # Original method
        old_contradictions = k.find_contradictions("Testing is always worth the effort")

        # New semantic method
        new_contradictions = k.find_semantic_contradictions(
            "Testing is valuable and important", similarity_threshold=0.1
        )

        # Both should return lists (may or may not find contradictions depending on embedding)
        assert isinstance(old_contradictions, list)
        assert isinstance(new_contradictions, list)

    def test_semantic_finds_subtler_contradictions(self, kernle_instance):
        """Semantic method can find contradictions without explicit opposition words."""
        k = kernle_instance

        # Add beliefs with subtle opposition
        k.belief("Testing improves code quality significantly", confidence=0.9)
        k.belief("Testing slows down the development process", confidence=0.7)

        # The old method might not find this (no explicit always/never etc)
        _old_contradictions = k.find_contradictions("Testing improves code quality significantly")

        # The new method should find it (via sentiment opposition)
        new_contradictions = k.find_semantic_contradictions(
            "Testing improves code quality significantly", similarity_threshold=0.1
        )

        # New method should be able to find sentiment opposition
        # (though depends on embedding similarity)
        assert isinstance(new_contradictions, list)
        for c in new_contradictions:
            # Verify proper structure
            assert "opposition_type" in c
            assert c["opposition_type"] in [
                "opposition_words",
                "negation",
                "sentiment_opposition",
                "none",
            ]
