"""Tests for export_cache functionality."""
import os
import tempfile
from pathlib import Path

import pytest

from kernle import Kernle


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance with temp DB."""
    os.environ["KERNLE_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["KERNLE_CHECKPOINT_DIR"] = str(tmp_path / "checkpoints")
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
        k.belief("The sky is blue", confidence=0.9)
        k.belief("Water is wet", confidence=0.8)
        k.belief("Low confidence thing", confidence=0.2)

        content = k.export_cache(min_confidence=0.5)
        assert "sky is blue" in content
        assert "Water is wet" in content
        assert "Low confidence thing" not in content

    def test_beliefs_max_limit(self, k):
        """Max beliefs parameter limits output."""
        for i in range(10):
            k.belief(f"Belief number {i}", confidence=0.9 - i * 0.01)

        content = k.export_cache(max_beliefs=3)
        # Should have exactly 3 beliefs
        belief_lines = [l for l in content.split("\n") if l.startswith("- [")]
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
