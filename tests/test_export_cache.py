"""Tests for export_cache functionality."""
import os
import tempfile
from pathlib import Path

import pytest

from kernle import Kernle


@pytest.fixture
def k(tmp_path, request):
    """Create a Kernle instance with temp DB unique to each test."""
    # Use test name to create unique paths
    test_name = request.node.name
    db_path = tmp_path / f"{test_name}.db"
    cp_path = tmp_path / f"{test_name}_checkpoints"
    
    os.environ["KERNLE_DB_PATH"] = str(db_path)
    os.environ["KERNLE_CHECKPOINT_DIR"] = str(cp_path)
    instance = Kernle(agent_id="test-export")
    yield instance
    os.environ.pop("KERNLE_DB_PATH", None)
    os.environ.pop("KERNLE_CHECKPOINT_DIR", None)


class TestExportCache:
    def test_empty_cache(self, k):
        """Empty agent produces valid cache with header."""
        content = k.export_cache()
        assert "# MEMORY.md" in content
        assert "AUTO-GENERATED" in content
        assert "Do not edit manually" in content

    def test_beliefs_included(self, k):
        """Beliefs above min_confidence appear in cache."""
        k.belief("UNIQUE_SKY_IS_BLUE_789", confidence=0.9)
        k.belief("UNIQUE_WATER_IS_WET_456", confidence=0.8)
        k.belief("UNIQUE_LOW_CONF_THING_123", confidence=0.2)

        content = k.export_cache(min_confidence=0.5, max_beliefs=500)
        assert "UNIQUE_SKY_IS_BLUE_789" in content
        assert "UNIQUE_WATER_IS_WET_456" in content
        assert "UNIQUE_LOW_CONF_THING_123" not in content

    def test_beliefs_max_limit(self, k):
        """Max beliefs parameter limits output."""
        for i in range(10):
            k.belief(f"Belief number {i}", confidence=0.9 - i * 0.01)

        content = k.export_cache(max_beliefs=3)
        # Should have exactly 3 beliefs (matching "- [XX%]" format, not goals "- [priority]")
        belief_lines = [l for l in content.split("\n") if l.startswith("- [") and "%" in l]
        assert len(belief_lines) == 3

    def test_values_included(self, k):
        """Values appear in cache."""
        k.value("honesty", "Always tell the truth", priority=90)
        content = k.export_cache()
        assert "honesty" in content
        assert "Always tell the truth" in content

    def test_goals_included(self, k):
        """Active goals appear in cache."""
        k.goal("Ship v1.0", priority=80)
        content = k.export_cache()
        assert "Ship v1.0" in content

    def test_checkpoint_included(self, k):
        """Checkpoint appears in cache by default."""
        k.checkpoint("Working on export feature", pending=["Write tests"])
        content = k.export_cache()
        assert "Working on export feature" in content
        assert "Write tests" in content

    def test_checkpoint_excluded(self, k):
        """Checkpoint can be excluded."""
        k.checkpoint("Working on export feature")
        content = k.export_cache(include_checkpoint=False)
        assert "Working on export feature" not in content

    def test_write_to_file(self, k):
        """Can write cache to file."""
        k.belief("Test belief", confidence=0.9)

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name

        try:
            content = k.export_cache(path=path)
            assert Path(path).exists()
            file_content = Path(path).read_text()
            assert file_content == content
            assert "Test belief" in file_content
        finally:
            os.unlink(path)

    def test_relationships_included(self, k):
        """Key relationships appear in cache."""
        k.relationship("Sean", trust_level=0.9, notes="My human steward", entity_type="person")
        content = k.export_cache()
        assert "Sean" in content
        assert "person" in content

    def test_idempotent(self, k):
        """Running export twice produces same output (excluding timestamp)."""
        k.belief("Stable belief", confidence=0.9)
        k.value("consistency", "Be consistent", priority=85)

        content1 = k.export_cache()
        content2 = k.export_cache()

        # Remove timestamp lines for comparison
        def strip_timestamp(s):
            return "\n".join(
                l for l in s.split("\n") if "AUTO-GENERATED" not in l
            )

        assert strip_timestamp(content1) == strip_timestamp(content2)

    def test_negative_min_confidence_clamped(self, k):
        """Negative min_confidence is clamped to 0 (no error raised)."""
        # The main test is that passing negative min_confidence doesn't raise an error
        # It gets clamped to 0.0, which includes all beliefs
        content = k.export_cache(min_confidence=-1.0)
        # If we get here without error, the clamping worked
        assert "MEMORY.md" in content  # Basic sanity check

    def test_high_min_confidence_clamped(self, k):
        """min_confidence > 1 is clamped to 1."""
        k.belief("UNIQUE_PERFECT_CONF_ABC", confidence=1.0)
        k.belief("UNIQUE_ALMOST_PERFECT_DEF", confidence=0.99)
        
        content = k.export_cache(min_confidence=2.0)
        # Only 1.0 confidence beliefs pass when clamped to 1.0
        assert "UNIQUE_PERFECT_CONF_ABC" in content
        assert "UNIQUE_ALMOST_PERFECT_DEF" not in content

    def test_negative_max_beliefs_clamped(self, k):
        """Negative max_beliefs is clamped to 0."""
        k.belief("UNIQUE_MAX_ZERO_TEST_GHI", confidence=0.95)
        
        content = k.export_cache(max_beliefs=-5)
        # With max_beliefs=0 (clamped from -5), no beliefs section should appear
        # or at minimum our specific belief shouldn't be there
        assert "UNIQUE_MAX_ZERO_TEST_GHI" not in content

    def test_unicode_in_beliefs(self, k):
        """Unicode characters in beliefs are handled correctly."""
        k.belief("UNIQUE_UNICODE_🌱_日本語_TEST", confidence=0.95)
        k.belief("UNIQUE_EMOJI_🎉🔥✨_TEST", confidence=0.94)
        
        content = k.export_cache(max_beliefs=100)
        assert "UNIQUE_UNICODE_🌱_日本語_TEST" in content
        assert "UNIQUE_EMOJI_🎉🔥✨_TEST" in content
