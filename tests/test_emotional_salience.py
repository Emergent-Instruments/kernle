"""Tests for emotional salience in priority scoring.

Verifies that compute_priority_score() incorporates emotional valence and
arousal with a 90-day half-life time decay, giving emotionally significant
episodes higher cognitive availability.
"""

from datetime import datetime, timedelta, timezone

from kernle.core import compute_priority_score


class TestEmotionalSalienceInPriority:
    """Test emotional salience factor in compute_priority_score."""

    def test_no_emotion_baseline(self):
        """Records without emotional data should use base scoring."""
        record = {
            "confidence": 0.8,
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
        }
        score = compute_priority_score("belief", record)
        # With zero emotion, emotional_salience = 0
        # score = 0.55 * (0.70 * 0.8) + 0.35 * 0.8 + 0.10 * 0
        # = 0.55 * 0.56 + 0.28 = 0.308 + 0.28 = 0.588
        assert 0.0 <= score <= 1.0

    def test_high_emotion_boosts_score(self):
        """Records with high emotional intensity should score higher."""
        base_record = {
            "confidence": 0.8,
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
        }
        emotional_record = {
            "confidence": 0.8,
            "emotional_valence": 0.9,
            "emotional_arousal": 0.8,
            "created_at": datetime.now(timezone.utc),
        }

        base_score = compute_priority_score("belief", base_record)
        emotional_score = compute_priority_score("belief", emotional_record)

        assert emotional_score > base_score

    def test_negative_valence_also_boosts(self):
        """Negative valence should also boost via abs(valence)."""
        positive_record = {
            "confidence": 0.8,
            "emotional_valence": 0.8,
            "emotional_arousal": 0.7,
            "created_at": datetime.now(timezone.utc),
        }
        negative_record = {
            "confidence": 0.8,
            "emotional_valence": -0.8,
            "emotional_arousal": 0.7,
            "created_at": datetime.now(timezone.utc),
        }

        pos_score = compute_priority_score("belief", positive_record)
        neg_score = compute_priority_score("belief", negative_record)

        # abs(valence) should make them equal
        assert abs(pos_score - neg_score) < 0.01

    def test_time_decay_reduces_salience(self):
        """Older records should have lower emotional salience."""
        now = datetime.now(timezone.utc)
        recent_record = {
            "confidence": 0.8,
            "emotional_valence": 0.9,
            "emotional_arousal": 0.8,
            "created_at": now,
        }
        old_record = {
            "confidence": 0.8,
            "emotional_valence": 0.9,
            "emotional_arousal": 0.8,
            "created_at": now - timedelta(days=180),
        }

        recent_score = compute_priority_score("belief", recent_record)
        old_score = compute_priority_score("belief", old_record)

        assert recent_score > old_score

    def test_half_life_90_days(self):
        """At 90 days, emotional salience should be halved."""
        now = datetime.now(timezone.utc)
        fresh_record = {
            "confidence": 0.8,
            "emotional_valence": 1.0,
            "emotional_arousal": 1.0,
            "created_at": now,
        }
        half_life_record = {
            "confidence": 0.8,
            "emotional_valence": 1.0,
            "emotional_arousal": 1.0,
            "created_at": now - timedelta(days=90),
        }

        fresh_score = compute_priority_score("belief", fresh_record)
        half_life_score = compute_priority_score("belief", half_life_record)

        # At 90 days, time_decay = 90/(90+90) = 0.5
        # So emotional_salience at half_life should be half of fresh
        # The difference in total score should reflect 10% * (1.0 - 0.5) = 0.05
        # Allow some tolerance
        no_emotion_score = compute_priority_score(
            "belief", {"confidence": 0.8, "emotional_valence": 0.0, "emotional_arousal": 0.0}
        )

        fresh_boost = fresh_score - no_emotion_score
        half_boost = half_life_score - no_emotion_score

        # half_boost should be approximately half of fresh_boost
        assert abs(half_boost - fresh_boost / 2) < 0.01

    def test_episode_emotional_salience(self):
        """Episodes with emotional data should get the boost too."""
        base_episode = {
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
        }
        emotional_episode = {
            "emotional_valence": 0.8,
            "emotional_arousal": 0.9,
            "created_at": datetime.now(timezone.utc),
        }

        base_score = compute_priority_score("episode", base_episode)
        emotional_score = compute_priority_score("episode", emotional_episode)

        assert emotional_score > base_score

    def test_zero_arousal_no_boost(self):
        """High valence but zero arousal should produce no emotional boost."""
        record = {
            "confidence": 0.8,
            "emotional_valence": 1.0,
            "emotional_arousal": 0.0,
        }
        no_emotion_record = {
            "confidence": 0.8,
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
        }

        score = compute_priority_score("belief", record)
        no_emotion_score = compute_priority_score("belief", no_emotion_record)

        assert abs(score - no_emotion_score) < 0.001

    def test_belief_scope_boost_still_works(self):
        """Self-belief scope boost should still apply on top of emotional salience."""
        world_belief = {
            "confidence": 0.8,
            "belief_scope": "world",
            "emotional_valence": 0.5,
            "emotional_arousal": 0.5,
            "created_at": datetime.now(timezone.utc),
        }
        self_belief = {
            "confidence": 0.8,
            "belief_scope": "self",
            "emotional_valence": 0.5,
            "emotional_arousal": 0.5,
            "created_at": datetime.now(timezone.utc),
        }

        world_score = compute_priority_score("belief", world_belief)
        self_score = compute_priority_score("belief", self_belief)

        assert self_score > world_score
        assert abs(self_score - world_score - 0.05) < 0.001

    def test_string_created_at(self):
        """String ISO timestamps should be parsed correctly."""
        record = {
            "confidence": 0.8,
            "emotional_valence": 0.9,
            "emotional_arousal": 0.8,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        score = compute_priority_score("belief", record)
        assert 0.0 <= score <= 1.0

    def test_missing_created_at_uses_zero_days(self):
        """Missing created_at should default to days_since=0 (max salience)."""
        with_time = {
            "confidence": 0.8,
            "emotional_valence": 0.9,
            "emotional_arousal": 0.8,
            "created_at": datetime.now(timezone.utc),
        }
        without_time = {
            "confidence": 0.8,
            "emotional_valence": 0.9,
            "emotional_arousal": 0.8,
        }

        score_with = compute_priority_score("belief", with_time)
        score_without = compute_priority_score("belief", without_time)

        # Should be nearly identical (both at ~0 days since)
        assert abs(score_with - score_without) < 0.01
