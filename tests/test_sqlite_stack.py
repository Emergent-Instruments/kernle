"""Tests for SQLiteStack conforming to StackProtocol.

Tests cover:
- Basic CRUD for each memory type through the stack
- Detached mode operation (no core attached)
- Composition hooks (on_attach, on_detach, on_model_changed)
- Component registry (add/remove)
- Search, load, stats, dump/export
- Meta-memory operations (forget, recover, protect, record_access)
- Trust layer
- Feature methods (consolidate, apply_forgetting)
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kernle.protocols import (
    InferenceService,
    StackComponentProtocol,
)
from kernle.protocols import (
    SearchResult as ProtocolSearchResult,
)
from kernle.protocols import (
    SyncResult as ProtocolSyncResult,
)
from kernle.stack import SQLiteStack
from kernle.types import (
    Belief,
    Drive,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    RawEntry,
    Relationship,
    SelfNarrative,
    Summary,
    TrustAssessment,
    Value,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp database path."""
    return tmp_path / "test_stack.db"


@pytest.fixture
def stack(tmp_db):
    """Create an SQLiteStack instance with a temp database."""
    return SQLiteStack(
        stack_id="test-stack", db_path=tmp_db, components=[], enforce_provenance=False
    )


@pytest.fixture
def stack_id():
    return "test-stack"


def _make_episode(stack_id: str, **overrides) -> Episode:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "objective": "Test objective",
        "outcome": "Test outcome",
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Episode(**defaults)


def _make_belief(stack_id: str, **overrides) -> Belief:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "statement": "Test belief",
        "confidence": 0.8,
    }
    defaults.update(overrides)
    return Belief(**defaults)


def _make_value(stack_id: str, **overrides) -> Value:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "name": "Test Value",
        "statement": "Test value statement",
        "priority": 50,
    }
    defaults.update(overrides)
    return Value(**defaults)


def _make_goal(stack_id: str, **overrides) -> Goal:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "title": "Test goal",
        "status": "active",
    }
    defaults.update(overrides)
    return Goal(**defaults)


def _make_note(stack_id: str, **overrides) -> Note:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "content": "Test note content",
    }
    defaults.update(overrides)
    return Note(**defaults)


def _make_drive(stack_id: str, **overrides) -> Drive:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "drive_type": "curiosity",
        "intensity": 0.7,
    }
    defaults.update(overrides)
    return Drive(**defaults)


def _make_relationship(stack_id: str, **overrides) -> Relationship:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "entity_name": "alice",
        "entity_type": "human",
        "relationship_type": "collaborator",
    }
    defaults.update(overrides)
    return Relationship(**defaults)


# ===========================================================================
# Properties
# ===========================================================================


class TestProperties:
    def test_stack_id(self, stack):
        assert stack.stack_id == "test-stack"

    def test_schema_version(self, stack):
        assert isinstance(stack.schema_version, int)
        assert stack.schema_version > 0


# ===========================================================================
# Write + Read: Episodes
# ===========================================================================


class TestEpisodes:
    def test_save_and_get(self, stack, stack_id):
        ep = _make_episode(stack_id, objective="Learn Python", outcome="Completed tutorial")
        eid = stack.save_episode(ep)
        assert eid == ep.id

        episodes = stack.get_episodes(limit=10)
        assert len(episodes) >= 1
        found = [e for e in episodes if e.id == ep.id]
        assert len(found) == 1
        assert found[0].objective == "Learn Python"

    def test_batch_save(self, stack, stack_id):
        eps = [_make_episode(stack_id, objective=f"Task {i}") for i in range(3)]
        ids = stack.save_episodes_batch(eps)
        assert len(ids) == 3

    def test_get_memory(self, stack, stack_id):
        ep = _make_episode(stack_id)
        stack.save_episode(ep)
        result = stack.get_memory("episode", ep.id)
        assert result is not None
        assert result.id == ep.id


# ===========================================================================
# Write + Read: Beliefs
# ===========================================================================


class TestBeliefs:
    def test_save_and_get(self, stack, stack_id):
        b = _make_belief(stack_id, statement="Python is great")
        bid = stack.save_belief(b)
        assert bid == b.id

        beliefs = stack.get_beliefs(limit=10)
        assert any(bl.id == b.id for bl in beliefs)

    def test_filter_by_type(self, stack, stack_id):
        b1 = _make_belief(stack_id, belief_type="fact", statement="Fact 1")
        b2 = _make_belief(stack_id, belief_type="opinion", statement="Opinion 1")
        stack.save_belief(b1)
        stack.save_belief(b2)

        facts = stack.get_beliefs(belief_type="fact")
        assert all(b.belief_type == "fact" for b in facts)

    def test_filter_by_confidence(self, stack, stack_id):
        b1 = _make_belief(stack_id, confidence=0.9, statement="High conf")
        b2 = _make_belief(stack_id, confidence=0.3, statement="Low conf")
        stack.save_belief(b1)
        stack.save_belief(b2)

        high = stack.get_beliefs(min_confidence=0.5)
        assert all(b.confidence >= 0.5 for b in high)

    def test_batch_save(self, stack, stack_id):
        beliefs = [_make_belief(stack_id, statement=f"Belief {i}") for i in range(3)]
        ids = stack.save_beliefs_batch(beliefs)
        assert len(ids) == 3


# ===========================================================================
# Write + Read: Values
# ===========================================================================


class TestValues:
    def test_save_and_get(self, stack, stack_id):
        v = _make_value(stack_id)
        vid = stack.save_value(v)
        assert vid == v.id

        values = stack.get_values()
        assert any(val.id == v.id for val in values)


# ===========================================================================
# Write + Read: Goals
# ===========================================================================


class TestGoals:
    def test_save_and_get(self, stack, stack_id):
        g = _make_goal(stack_id, title="Ship v1.0")
        gid = stack.save_goal(g)
        assert gid == g.id

        goals = stack.get_goals(status="active")
        assert any(gl.id == g.id for gl in goals)


# ===========================================================================
# Write + Read: Notes
# ===========================================================================


class TestNotes:
    def test_save_and_get(self, stack, stack_id):
        n = _make_note(stack_id)
        nid = stack.save_note(n)
        assert nid == n.id

        notes = stack.get_notes()
        assert any(nt.id == n.id for nt in notes)

    def test_batch_save(self, stack, stack_id):
        notes = [_make_note(stack_id, content=f"Note {i}") for i in range(3)]
        ids = stack.save_notes_batch(notes)
        assert len(ids) == 3


# ===========================================================================
# Write + Read: Drives
# ===========================================================================


class TestDrives:
    def test_save_and_get(self, stack, stack_id):
        d = _make_drive(stack_id)
        did = stack.save_drive(d)
        assert did == d.id

        drives = stack.get_drives()
        assert any(dr.id == d.id for dr in drives)


# ===========================================================================
# Write + Read: Relationships
# ===========================================================================


class TestRelationships:
    def test_save_and_get(self, stack, stack_id):
        r = _make_relationship(stack_id)
        rid = stack.save_relationship(r)
        assert rid == r.id

        rels = stack.get_relationships()
        assert any(rel.id == r.id for rel in rels)

    def test_filter_by_entity_id(self, stack, stack_id):
        r1 = _make_relationship(stack_id, entity_name="alice")
        r2 = _make_relationship(stack_id, entity_name="bob")
        stack.save_relationship(r1)
        stack.save_relationship(r2)

        alice_rels = stack.get_relationships(entity_id="alice")
        assert all(r.entity_name == "alice" for r in alice_rels)


# ===========================================================================
# Write + Read: Playbooks
# ===========================================================================


class TestPlaybooks:
    def test_save_and_retrieve(self, stack, stack_id):
        pb = Playbook(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            name="Deploy",
            description="Deploy procedure",
            trigger_conditions=["release ready"],
            steps=[{"action": "deploy"}],
            failure_modes=["timeout"],
        )
        pid = stack.save_playbook(pb)
        assert pid == pb.id


# ===========================================================================
# Write + Read: Epochs, Summaries, SelfNarratives, Suggestions
# ===========================================================================


class TestAdvancedTypes:
    def test_save_epoch(self, stack, stack_id):
        ep = Epoch(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            epoch_number=1,
            name="Beginning",
        )
        eid = stack.save_epoch(ep)
        assert eid == ep.id

    def test_save_summary(self, stack, stack_id):
        s = Summary(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
            content="Summary of January",
        )
        sid = stack.save_summary(s)
        assert sid == s.id

    def test_save_self_narrative(self, stack, stack_id):
        n = SelfNarrative(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            content="I am a test entity",
            narrative_type="identity",
        )
        nid = stack.save_self_narrative(n)
        assert nid == n.id

    def test_save_suggestion(self, stack, stack_id):
        sg = MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            memory_type="belief",
            content={"statement": "Testing is good"},
            confidence=0.7,
            source_raw_ids=["raw-1"],
        )
        sid = stack.save_suggestion(sg)
        assert sid == sg.id


# ===========================================================================
# Search
# ===========================================================================


class TestSearch:
    def test_search_returns_protocol_results(self, stack, stack_id):
        ep = _make_episode(stack_id, objective="unique searchable objective XYZ123")
        stack.save_episode(ep)

        results = stack.search("XYZ123")
        # Search may or may not find it depending on indexing
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ProtocolSearchResult)
            assert hasattr(r, "memory_type")
            assert hasattr(r, "memory_id")
            assert hasattr(r, "content")
            assert hasattr(r, "score")


# ===========================================================================
# Working Memory (load)
# ===========================================================================


class TestLoad:
    def test_load_returns_dict(self, stack, stack_id):
        # Add some data
        stack.save_value(_make_value(stack_id))
        stack.save_belief(_make_belief(stack_id))
        stack.save_episode(_make_episode(stack_id))

        result = stack.load(token_budget=4000)
        assert isinstance(result, dict)
        assert "values" in result
        assert "beliefs" in result
        assert "_meta" in result
        assert result["_meta"]["budget_total"] == 4000

    def test_load_respects_budget(self, stack, stack_id):
        # Add lots of data
        for i in range(20):
            stack.save_episode(_make_episode(stack_id, objective=f"Episode {i} " * 50))

        result = stack.load(token_budget=MIN_TOKEN_BUDGET)
        used = result["_meta"]["budget_used"]
        assert used <= MIN_TOKEN_BUDGET + 100  # some slack for estimation

    def test_load_empty_stack(self, stack):
        result = stack.load()
        assert isinstance(result, dict)

    def test_load_excludes_weak_and_dormant(self, stack, stack_id):
        """load() should only include Strong (>=0.8) and Fading (0.5-0.8) memories."""
        strong_ep = _make_episode(stack_id, objective="Strong episode", strength=1.0)
        fading_ep = _make_episode(stack_id, objective="Fading episode", strength=0.6)
        weak_ep = _make_episode(stack_id, objective="Weak episode", strength=0.4)
        dormant_ep = _make_episode(stack_id, objective="Dormant episode", strength=0.1)
        stack.save_episode(strong_ep)
        stack.save_episode(fading_ep)
        stack.save_episode(weak_ep)
        stack.save_episode(dormant_ep)

        result = stack.load(token_budget=8000)
        episode_objectives = [e["objective"] for e in result.get("episodes", [])]
        assert "Strong episode" in episode_objectives
        assert "Fading episode" in episode_objectives
        assert "Weak episode" not in episode_objectives
        assert "Dormant episode" not in episode_objectives

    def test_load_excludes_weak_beliefs(self, stack, stack_id):
        """load() should exclude beliefs with strength below 0.5."""
        strong_belief = _make_belief(stack_id, statement="Strong belief", strength=0.9)
        weak_belief = _make_belief(stack_id, statement="Weak belief", strength=0.3)
        stack.save_belief(strong_belief)
        stack.save_belief(weak_belief)

        result = stack.load(token_budget=8000)
        belief_statements = [b["statement"] for b in result.get("beliefs", [])]
        assert "Strong belief" in belief_statements
        assert "Weak belief" not in belief_statements


# Minimum token budget for test reference
MIN_TOKEN_BUDGET = 100


# ===========================================================================
# Detached Mode
# ===========================================================================


class TestDetachedMode:
    """Stack should work without a core attached."""

    def test_works_without_core(self, stack, stack_id):
        assert stack._attached_core_id is None
        assert stack._inference is None

        # All operations should work
        ep = _make_episode(stack_id)
        stack.save_episode(ep)
        episodes = stack.get_episodes()
        assert len(episodes) >= 1

        b = _make_belief(stack_id)
        stack.save_belief(b)
        beliefs = stack.get_beliefs()
        assert len(beliefs) >= 1

        stats = stack.get_stats()
        assert isinstance(stats, dict)

    def test_search_works_detached(self, stack, stack_id):
        stack.save_note(_make_note(stack_id, content="detached search test"))
        results = stack.search("detached")
        assert isinstance(results, list)


# ===========================================================================
# Composition Hooks
# ===========================================================================


class TestCompositionHooks:
    def test_on_attach(self, stack):
        assert stack._attached_core_id is None
        stack.on_attach("core-123")
        assert stack._attached_core_id == "core-123"
        assert stack._inference is None

    def test_on_attach_with_inference(self, stack):
        mock_inference = MagicMock(spec=InferenceService)
        stack.on_attach("core-456", inference=mock_inference)
        assert stack._attached_core_id == "core-456"
        assert stack._inference is mock_inference

    def test_on_detach(self, stack):
        stack.on_attach("core-789")
        assert stack._attached_core_id == "core-789"

        stack.on_detach("core-789")
        assert stack._attached_core_id is None
        assert stack._inference is None

    def test_on_model_changed(self, stack):
        mock_inference = MagicMock(spec=InferenceService)
        stack.on_attach("core-100")

        stack.on_model_changed(mock_inference)
        assert stack._inference is mock_inference

        stack.on_model_changed(None)
        assert stack._inference is None

    def test_hooks_propagate_to_components(self, stack):
        component = MagicMock(spec=StackComponentProtocol)
        component.name = "test-comp"
        component.required = False
        stack.add_component(component)

        mock_inference = MagicMock(spec=InferenceService)
        stack.on_attach("core-200", inference=mock_inference)
        component.set_inference.assert_called_with(mock_inference)

        stack.on_model_changed(None)
        component.set_inference.assert_called_with(None)

        stack.on_detach("core-200")
        component.set_inference.assert_called_with(None)


# ===========================================================================
# Component Registry
# ===========================================================================


class TestComponentRegistry:
    def _make_component(self, name: str = "test-comp", required: bool = False):
        comp = MagicMock(spec=StackComponentProtocol)
        comp.name = name
        comp.required = required
        return comp

    def test_empty_components(self, stack):
        assert stack.components == {}

    def test_add_component(self, stack):
        comp = self._make_component("embedding")
        stack.add_component(comp)

        assert "embedding" in stack.components
        comp.attach.assert_called_once_with(stack.stack_id, None)

    def test_add_duplicate_raises(self, stack):
        comp = self._make_component("embedding")
        stack.add_component(comp)

        with pytest.raises(ValueError, match="already registered"):
            stack.add_component(self._make_component("embedding"))

    def test_remove_component(self, stack):
        comp = self._make_component("forgetting")
        stack.add_component(comp)
        assert "forgetting" in stack.components

        stack.remove_component("forgetting")
        assert "forgetting" not in stack.components
        comp.detach.assert_called_once()

    def test_remove_nonexistent_raises(self, stack):
        with pytest.raises(ValueError, match="not found"):
            stack.remove_component("nonexistent")

    def test_remove_required_raises(self, stack):
        comp = self._make_component("embedding", required=True)
        stack.add_component(comp)

        with pytest.raises(ValueError, match="required"):
            stack.remove_component("embedding")

    def test_get_component(self, stack):
        comp = self._make_component("meta")
        stack.add_component(comp)

        assert stack.get_component("meta") is comp
        assert stack.get_component("nonexistent") is None

    def test_maintenance_calls_components(self, stack):
        comp = self._make_component("sweeper")
        comp.on_maintenance.return_value = {"swept": 5}
        stack.add_component(comp)

        results = stack.maintenance()
        assert "sweeper" in results
        assert results["sweeper"] == {"swept": 5}

    def test_maintenance_handles_errors(self, stack):
        comp = self._make_component("broken")
        comp.on_maintenance.side_effect = RuntimeError("broke")
        stack.add_component(comp)

        results = stack.maintenance()
        assert "error" in results["broken"]

    def test_components_returns_copy(self, stack):
        comp = self._make_component("test")
        stack.add_component(comp)
        components = stack.components
        components["injected"] = MagicMock()
        assert "injected" not in stack.components


# ===========================================================================
# Meta-Memory Operations
# ===========================================================================


class TestMetaMemory:
    def test_record_access(self, stack, stack_id):
        ep = _make_episode(stack_id)
        stack.save_episode(ep)
        result = stack.record_access("episode", ep.id)
        assert result is True

    def test_forget_and_recover(self, stack, stack_id):
        ep = _make_episode(stack_id)
        stack.save_episode(ep)

        forgotten = stack.forget_memory("episode", ep.id, "test reason")
        assert forgotten is True

        # Should be excluded by default
        episodes = stack.get_episodes(include_forgotten=False)
        assert not any(e.id == ep.id for e in episodes)

        # Include forgotten
        all_eps = stack.get_episodes(include_forgotten=True)
        assert any(e.id == ep.id for e in all_eps)

        # Recover
        recovered = stack.recover_memory("episode", ep.id)
        assert recovered is True

        # Should be back (recovered at strength 0.2, in the Weak tier)
        episodes = stack.get_episodes(include_weak=True)
        assert any(e.id == ep.id for e in episodes)

    def test_protect_memory(self, stack, stack_id):
        ep = _make_episode(stack_id)
        stack.save_episode(ep)

        result = stack.protect_memory("episode", ep.id, True)
        assert result is True


# ===========================================================================
# Trust Layer
# ===========================================================================


class TestTrustLayer:
    def test_save_and_get_assessment(self, stack, stack_id):
        ta = TrustAssessment(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            entity="alice",
            dimensions={"general": {"score": 0.8}},
        )
        tid = stack.save_trust_assessment(ta)
        assert tid == ta.id

        assessments = stack.get_trust_assessments(entity_id="alice")
        assert len(assessments) >= 1

    def test_compute_trust_default(self, stack):
        result = stack.compute_trust("unknown-entity")
        assert result["score"] == 0.5
        assert result["source"] == "default"

    def test_compute_trust_with_assessment(self, stack, stack_id):
        ta = TrustAssessment(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            entity="bob",
            dimensions={"general": {"score": 0.9}},
        )
        stack.save_trust_assessment(ta)

        result = stack.compute_trust("bob")
        assert result["score"] == 0.9


# ===========================================================================
# Features
# ===========================================================================


class TestFeatures:
    def test_consolidate_not_enough_episodes(self, stack):
        result = stack.consolidate()
        assert result["consolidated"] == 0

    def test_consolidate_with_episodes(self, stack, stack_id):
        for i in range(5):
            ep = _make_episode(
                stack_id,
                objective=f"Task {i}",
                outcome="done",
                lessons=["lesson A", "lesson B"] if i % 2 == 0 else ["lesson A"],
            )
            stack.save_episode(ep)

        result = stack.consolidate()
        assert result["consolidated"] >= 3

    def test_apply_forgetting(self, stack, stack_id):
        # Create some low-salience episodes
        for i in range(3):
            ep = _make_episode(stack_id, confidence=0.1)
            stack.save_episode(ep)

        result = stack.apply_forgetting()
        assert isinstance(result, dict)
        assert "forgotten" in result


# ===========================================================================
# Stats & Export
# ===========================================================================


class TestStatsExport:
    def test_get_stats(self, stack, stack_id):
        stack.save_episode(_make_episode(stack_id))
        stack.save_belief(_make_belief(stack_id))
        stats = stack.get_stats()
        assert isinstance(stats, dict)
        assert stats.get("episodes", 0) >= 1
        assert stats.get("beliefs", 0) >= 1

    def test_dump_markdown(self, stack, stack_id):
        stack.save_value(_make_value(stack_id))
        content = stack.dump(format="markdown")
        assert "# Memory Dump for test-stack" in content
        assert "## Values" in content

    def test_dump_json(self, stack, stack_id):
        stack.save_belief(_make_belief(stack_id))
        content = stack.dump(format="json")
        data = json.loads(content)
        assert data["stack_id"] == "test-stack"
        assert "beliefs" in data

    def test_export_to_file(self, stack, stack_id, tmp_path):
        stack.save_value(_make_value(stack_id))
        export_path = str(tmp_path / "export.md")
        stack.export(export_path)
        assert Path(export_path).exists()
        content = Path(export_path).read_text()
        assert "Memory Dump" in content

    def test_export_json_by_extension(self, stack, stack_id, tmp_path):
        stack.save_value(_make_value(stack_id))
        export_path = str(tmp_path / "export.json")
        stack.export(export_path, format="markdown")
        content = Path(export_path).read_text()
        data = json.loads(content)
        assert "stack_id" in data


# ===========================================================================
# Sync
# ===========================================================================


class TestSync:
    def test_sync(self, stack):
        result = stack.sync()
        assert isinstance(result, ProtocolSyncResult)

    def test_pull_changes(self, stack):
        result = stack.pull_changes()
        assert isinstance(result, ProtocolSyncResult)

    def test_pending_sync_count(self, stack):
        count = stack.get_pending_sync_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_is_online(self, stack):
        result = stack.is_online()
        assert isinstance(result, bool)


# ===========================================================================
# Raw Entries
# ===========================================================================


class TestRawEntries:
    def test_save_and_get_raw(self, stack, stack_id):
        raw = RawEntry(
            id=str(uuid.uuid4()),
            stack_id=stack_id,
            blob="raw brain dump",
            source="test",
        )
        rid = stack.save_raw(raw)
        assert isinstance(rid, str)

        entries = stack.get_raw()
        assert len(entries) >= 1


# ===========================================================================
# Component Hooks
# ===========================================================================


class TestComponentHooks:
    """Test that save/search/load dispatch to component hooks."""

    def _make_tracking_component(self, name="tracker"):
        """Create a mock component that tracks hook calls."""
        comp = MagicMock(spec=StackComponentProtocol)
        comp.name = name
        comp.required = False
        comp.needs_inference = False
        comp.on_save = MagicMock()
        comp.on_search = MagicMock(return_value=None)  # None = don't modify
        comp.on_load = MagicMock()
        comp.on_maintenance = MagicMock(return_value=None)
        comp.on_model_changed = MagicMock()
        comp.attach = MagicMock()
        comp.detach = MagicMock()
        return comp

    def test_save_episode_calls_on_save(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=stack.stack_id,
            objective="test",
            outcome="done",
        )
        result_id = stack.save_episode(ep)
        comp.on_save.assert_called_once_with("episode", result_id, ep)

    def test_save_belief_calls_on_save(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        b = Belief(
            id=str(uuid.uuid4()),
            stack_id=stack.stack_id,
            statement="test belief",
        )
        result_id = stack.save_belief(b)
        comp.on_save.assert_called_once_with("belief", result_id, b)

    def test_save_note_calls_on_save(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        n = Note(
            id=str(uuid.uuid4()),
            stack_id=stack.stack_id,
            content="test note",
        )
        result_id = stack.save_note(n)
        comp.on_save.assert_called_once_with("note", result_id, n)

    def test_search_calls_on_search(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        # Save something first so search has data
        stack.save_episode(
            Episode(
                id=str(uuid.uuid4()),
                stack_id=stack.stack_id,
                objective="searchable objective",
                outcome="found",
            )
        )
        stack.search("searchable")
        comp.on_search.assert_called_once()
        args = comp.on_search.call_args
        assert args[0][0] == "searchable"

    def test_search_on_search_modifies_results(self, stack):
        modified_result = ProtocolSearchResult(
            memory_type="episode",
            memory_id="modified-id",
            content="modified",
            score=1.0,
        )
        comp = self._make_tracking_component()
        comp.on_search = MagicMock(return_value=[modified_result])
        stack.add_component(comp)
        stack.save_episode(
            Episode(
                id=str(uuid.uuid4()),
                stack_id=stack.stack_id,
                objective="something",
                outcome="done",
            )
        )
        results = stack.search("something")
        assert len(results) == 1
        assert results[0].memory_id == "modified-id"

    def test_load_calls_on_load(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        stack.load()
        comp.on_load.assert_called_once()
        context = comp.on_load.call_args[0][0]
        assert isinstance(context, dict)
        assert "_meta" in context

    def test_on_save_error_isolated(self, stack):
        comp = self._make_tracking_component()
        comp.on_save.side_effect = RuntimeError("boom")
        stack.add_component(comp)
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=stack.stack_id,
            objective="test",
            outcome="done",
        )
        # Should not raise
        result_id = stack.save_episode(ep)
        assert isinstance(result_id, str)

    def test_on_search_error_isolated(self, stack):
        comp = self._make_tracking_component()
        comp.on_search.side_effect = RuntimeError("boom")
        stack.add_component(comp)
        stack.save_episode(
            Episode(
                id=str(uuid.uuid4()),
                stack_id=stack.stack_id,
                objective="test error",
                outcome="done",
            )
        )
        # Should not raise
        results = stack.search("test error")
        assert isinstance(results, list)

    def test_batch_dispatches_per_item(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        episodes = [
            Episode(
                id=str(uuid.uuid4()),
                stack_id=stack.stack_id,
                objective=f"batch {i}",
                outcome="done",
            )
            for i in range(3)
        ]
        ids = stack.save_episodes_batch(episodes)
        assert comp.on_save.call_count == 3
        for i, call in enumerate(comp.on_save.call_args_list):
            assert call[0][0] == "episode"
            assert call[0][1] == ids[i]
            assert call[0][2] == episodes[i]
