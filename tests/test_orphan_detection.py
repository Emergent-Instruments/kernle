"""Tests for orphaned memory detection."""

import pytest


class TestFindOrphanedMemories:
    """Tests for find_orphaned_memories."""

    def test_no_orphans_when_empty(self, kernle_instance):
        """No orphans with empty database."""
        kernle, _ = kernle_instance
        orphans = kernle.find_orphaned_memories()
        assert orphans == []

    def test_detects_unknown_source(self, kernle_instance):
        """Detects memories with source_type='unknown'."""
        kernle, _ = kernle_instance
        # Create a belief with unknown source
        belief_id = kernle.belief("Test belief")
        # Manually set source_type to unknown
        kernle.set_memory_source("belief", belief_id, "unknown")

        orphans = kernle.find_orphaned_memories()
        assert len(orphans) == 1
        assert orphans[0]["source_type"] == "unknown"
        assert orphans[0]["reason"] == "unknown source (legacy memory)"

    def test_detects_consolidation_without_derived_from(self, kernle_instance):
        """Detects consolidation memories without derived_from."""
        kernle, _ = kernle_instance
        # Create a belief sourced from consolidation
        belief_id = kernle.belief("Consolidated insight")
        # Set as consolidation but with no derived_from
        kernle.set_memory_source("belief", belief_id, "consolidation")

        orphans = kernle.find_orphaned_memories()
        assert len(orphans) == 1
        assert orphans[0]["source_type"] == "consolidation"
        assert "derived_from" in orphans[0]["reason"]

    def test_no_orphan_when_properly_sourced(self, kernle_instance):
        """No orphan when memory has proper provenance."""
        kernle, _ = kernle_instance
        # Create properly sourced belief
        belief_id = kernle.belief(
            "Properly sourced belief",
            source="direct observation",
        )

        orphans = kernle.find_orphaned_memories()
        assert len(orphans) == 0

    def test_limit_respected(self, kernle_instance):
        """Limit parameter is respected."""
        kernle, _ = kernle_instance
        # Create multiple orphans
        for i in range(10):
            belief_id = kernle.belief(f"Unknown belief {i}")
            kernle.set_memory_source("belief", belief_id, "unknown")

        orphans = kernle.find_orphaned_memories(limit=5)
        assert len(orphans) == 5

    def test_inference_without_derived_from_is_orphan(self, kernle_instance):
        """Inference source without derived_from is orphan."""
        kernle, _ = kernle_instance
        belief_id = kernle.belief("Inferred conclusion")
        kernle.set_memory_source("belief", belief_id, "inference")

        orphans = kernle.find_orphaned_memories()
        assert len(orphans) == 1
        assert orphans[0]["source_type"] == "inference"

    def test_consolidation_with_derived_from_not_orphan(self, kernle_instance):
        """Consolidation with derived_from is not an orphan."""
        kernle, _ = kernle_instance
        # First create source material
        raw_id = kernle.raw("Source observation")
        belief_id = kernle.belief("Consolidated insight")
        # Set as consolidation WITH derived_from
        kernle.set_memory_source(
            "belief", belief_id, "consolidation",
            derived_from=[f"raw:{raw_id}"]
        )

        orphans = kernle.find_orphaned_memories()
        assert len(orphans) == 0
