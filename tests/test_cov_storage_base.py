"""Coverage tests for kernle/storage/base.py.

Tests default implementations of non-abstract methods on the Storage protocol.
Since Storage is a Protocol with abstract methods, we create a minimal concrete
implementation that only stubs the abstract methods, then exercise the default
method implementations.
"""

from datetime import datetime, timezone
from typing import List

import pytest

from kernle.storage.base import (
    Belief,
    DiagnosticReport,
    DiagnosticSession,
    EntityModel,
    Episode,
    Epoch,
    MemorySuggestion,
    Note,
    RelationshipHistoryEntry,
    SelfNarrative,
    Storage,
    Summary,
    SyncConflict,
    SyncResult,
    TrustAssessment,
)


class MinimalStorage:
    """Minimal concrete implementation that only stubs abstract methods.

    This lets us test the default implementations on the Storage protocol
    by inheriting all non-abstract methods.
    """

    def __init__(self, stack_id: str = "test"):
        self.stack_id = stack_id
        self._episodes: List[Episode] = []
        self._beliefs: List[Belief] = []
        self._notes: List[Note] = []
        self._access_log: List[tuple] = []

    # Abstract method stubs
    def save_episode(self, episode):
        self._episodes.append(episode)
        return episode.id

    def get_episodes(self, limit=100, since=None, tags=None):
        return self._episodes[:limit]

    def get_episode(self, episode_id):
        return next((e for e in self._episodes if e.id == episode_id), None)

    def update_episode_emotion(self, episode_id, valence, arousal, tags=None):
        return False

    def get_emotional_episodes(self, days=7, limit=100):
        return []

    def search_by_emotion(self, valence_range=None, arousal_range=None, tags=None, limit=10):
        return []

    def save_belief(self, belief):
        self._beliefs.append(belief)
        return belief.id

    def get_beliefs(self, limit=100, include_inactive=False):
        return self._beliefs[:limit]

    def find_belief(self, statement):
        return None

    def save_value(self, value):
        return value.id

    def get_values(self, limit=100):
        return []

    def save_goal(self, goal):
        return goal.id

    def get_goals(self, status=None, limit=100):
        return []

    def save_note(self, note):
        self._notes.append(note)
        return note.id

    def get_notes(self, limit=100, since=None, note_type=None):
        return self._notes[:limit]

    def save_drive(self, drive):
        return drive.id

    def get_drives(self):
        return []

    def get_drive(self, drive_type):
        return None

    def save_relationship(self, relationship):
        return relationship.id

    def get_relationships(self, entity_type=None):
        return []

    def get_relationship(self, entity_name):
        return None

    def save_playbook(self, playbook):
        return playbook.id

    def get_playbook(self, playbook_id):
        return None

    def list_playbooks(self, tags=None, limit=100):
        return []

    def search_playbooks(self, query, limit=10):
        return []

    def update_playbook_usage(self, playbook_id, success):
        return False

    def save_raw(self, blob, source="unknown"):
        return "raw-1"

    def get_raw(self, raw_id):
        return None

    def list_raw(self, processed=None, limit=100):
        return []

    def mark_raw_processed(self, raw_id, processed_into):
        return False

    def search(self, query, limit=10, record_types=None, prefer_cloud=True):
        return []

    def get_stats(self):
        return {}

    def sync(self):
        return SyncResult(status="success", pushed=0, pulled=0, conflicts=[])

    def pull_changes(self, since=None):
        return SyncResult(status="success", pushed=0, pulled=0, conflicts=[])

    def get_pending_sync_count(self):
        return 0

    def is_online(self):
        return False

    def get_memory(self, memory_type, memory_id):
        return None

    def memory_exists(self, memory_type, memory_id):
        return False

    def update_strength(self, memory_type, memory_id, strength):
        return False

    def update_memory_meta(self, memory_type, memory_id, **kwargs):
        return False

    def get_memories_by_confidence(self, threshold, below=True, memory_types=None, limit=100):
        return []

    def get_memories_by_source(self, source_type, memory_types=None, limit=100):
        return []

    def record_access(self, memory_type, memory_id):
        self._access_log.append((memory_type, memory_id))
        return True

    def forget_memory(self, memory_type, memory_id, reason=None):
        return False

    def recover_memory(self, memory_type, memory_id):
        return False

    def protect_memory(self, memory_type, memory_id, protected=True):
        return False

    def get_forgetting_candidates(self, memory_types=None, limit=100, threshold=0.5):
        return []

    def get_forgotten_memories(self, memory_types=None, limit=100):
        return []

    # Inherit all default implementations from Storage


# Patch MinimalStorage to include the default methods from Storage protocol.
# Since Storage is a Protocol, its default methods aren't inherited by classes
# that don't explicitly inherit from it. We bind them manually.
_default_method_names = [
    "save_relationship_history",
    "get_relationship_history",
    "save_entity_model",
    "get_entity_models",
    "get_entity_model",
    "save_trust_assessment",
    "get_trust_assessment",
    "get_trust_assessments",
    "delete_trust_assessment",
    "save_diagnostic_session",
    "get_diagnostic_session",
    "get_diagnostic_sessions",
    "complete_diagnostic_session",
    "save_diagnostic_report",
    "get_diagnostic_report",
    "get_diagnostic_reports",
    "get_episodes_by_source_entity",
    "save_suggestion",
    "get_suggestion",
    "get_suggestions",
    "update_suggestion_status",
    "delete_suggestion",
    "save_summary",
    "get_summary",
    "list_summaries",
    "save_self_narrative",
    "get_self_narrative",
    "list_self_narratives",
    "deactivate_self_narratives",
    "save_epoch",
    "get_epoch",
    "get_epochs",
    "get_current_epoch",
    "close_epoch",
    "has_cloud_credentials",
    "cloud_health_check",
    "save_episodes_batch",
    "save_beliefs_batch",
    "save_notes_batch",
    "load_all",
    "record_access_batch",
    "_now",
    "_clear_queued_change",
    "_mark_synced",
    "_set_sync_meta",
    "get_queued_changes",
    "get_last_sync_time",
    "get_sync_conflicts",
    "save_sync_conflict",
    "clear_sync_conflicts",
    "_connect",
    "_get_record_for_push",
]

for _name in _default_method_names:
    if not hasattr(MinimalStorage, _name):
        setattr(MinimalStorage, _name, getattr(Storage, _name))


@pytest.fixture
def storage():
    """Create a MinimalStorage instance."""
    return MinimalStorage("test-agent")


class TestRelationshipHistoryDefaults:
    """Test default relationship history methods — covers lines 242, 260."""

    def test_save_relationship_history_returns_id(self, storage):
        entry = RelationshipHistoryEntry(
            id="rh-1",
            stack_id="test",
            relationship_id="rel-1",
            entity_name="Alice",
            event_type="interaction",
            created_at=datetime.now(timezone.utc),
        )
        result = storage.save_relationship_history(entry)
        assert result == "rh-1"

    def test_get_relationship_history_returns_empty(self, storage):
        result = storage.get_relationship_history("Alice")
        assert result == []


class TestEntityModelDefaults:
    """Test default entity model methods — covers lines 273, 291, 302."""

    def test_save_entity_model_returns_id(self, storage):
        model = EntityModel(
            id="em-1",
            stack_id="test",
            entity_name="Bob",
            model_type="behavioral",
            observation="Bob is helpful",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        result = storage.save_entity_model(model)
        assert result == "em-1"

    def test_get_entity_models_returns_empty(self, storage):
        result = storage.get_entity_models()
        assert result == []

    def test_get_entity_model_returns_none(self, storage):
        result = storage.get_entity_model("em-1")
        assert result is None


class TestTrustAssessmentDefaults:
    """Test default trust assessment methods — covers lines 308, 312, 316, 320."""

    def test_save_trust_assessment_returns_id(self, storage):
        assessment = TrustAssessment(
            id="ta-1",
            stack_id="test",
            entity="Alice",
            dimensions={"competence": 0.8},
        )
        result = storage.save_trust_assessment(assessment)
        assert result == "ta-1"

    def test_get_trust_assessment_returns_none(self, storage):
        result = storage.get_trust_assessment("Alice")
        assert result is None

    def test_get_trust_assessments_returns_empty(self, storage):
        result = storage.get_trust_assessments()
        assert result == []

    def test_delete_trust_assessment_returns_false(self, storage):
        result = storage.delete_trust_assessment("Alice")
        assert result is False


class TestDiagnosticDefaults:
    """Test default diagnostic methods — covers lines 326, 330, 338, 342, 346, 350, 358, 364."""

    def test_save_diagnostic_session_returns_id(self, storage):
        session = DiagnosticSession(
            id="ds-1",
            stack_id="test",
            session_type="routine",
        )
        result = storage.save_diagnostic_session(session)
        assert result == "ds-1"

    def test_get_diagnostic_session_returns_none(self, storage):
        result = storage.get_diagnostic_session("ds-1")
        assert result is None

    def test_get_diagnostic_sessions_returns_empty(self, storage):
        result = storage.get_diagnostic_sessions()
        assert result == []

    def test_complete_diagnostic_session_returns_false(self, storage):
        result = storage.complete_diagnostic_session("ds-1")
        assert result is False

    def test_save_diagnostic_report_returns_id(self, storage):
        report = DiagnosticReport(
            id="dr-1",
            stack_id="test",
            session_id="ds-1",
            findings={"test": "data"},
            created_at=datetime.now(timezone.utc),
        )
        result = storage.save_diagnostic_report(report)
        assert result == "dr-1"

    def test_get_diagnostic_report_returns_none(self, storage):
        result = storage.get_diagnostic_report("dr-1")
        assert result is None

    def test_get_diagnostic_reports_returns_empty(self, storage):
        result = storage.get_diagnostic_reports()
        assert result == []

    def test_get_episodes_by_source_entity_returns_empty(self, storage):
        result = storage.get_episodes_by_source_entity("Alice")
        assert result == []


class TestSuggestionDefaults:
    """Test default suggestion methods — covers lines 450, 461, 479, 499, 510."""

    def test_save_suggestion_returns_id(self, storage):
        suggestion = MemorySuggestion(
            id="ms-1",
            stack_id="test",
            memory_type="belief",
            content="Test suggestion",
            confidence=0.8,
            source_raw_ids=["raw-1"],
        )
        result = storage.save_suggestion(suggestion)
        assert result == "ms-1"

    def test_get_suggestion_returns_none(self, storage):
        result = storage.get_suggestion("ms-1")
        assert result is None

    def test_get_suggestions_returns_empty(self, storage):
        result = storage.get_suggestions()
        assert result == []

    def test_update_suggestion_status_returns_false(self, storage):
        result = storage.update_suggestion_status("ms-1", "promoted")
        assert result is False

    def test_delete_suggestion_returns_false(self, storage):
        result = storage.delete_suggestion("ms-1")
        assert result is False


class TestSummaryDefaults:
    """Test default summary methods — covers lines 516, 520, 524."""

    def test_save_summary_returns_id(self, storage):
        summary = Summary(
            id="sum-1",
            stack_id="test",
            scope="daily",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            content="A summary",
        )
        result = storage.save_summary(summary)
        assert result == "sum-1"

    def test_get_summary_returns_none(self, storage):
        result = storage.get_summary("sum-1")
        assert result is None

    def test_list_summaries_returns_empty(self, storage):
        result = storage.list_summaries("test")
        assert result == []


class TestSelfNarrativeDefaults:
    """Test default self-narrative methods — covers lines 530, 534, 552, 566."""

    def test_save_self_narrative_returns_id(self, storage):
        narrative = SelfNarrative(
            id="sn-1",
            stack_id="test",
            content="I am a test agent",
            narrative_type="identity",
        )
        result = storage.save_self_narrative(narrative)
        assert result == "sn-1"

    def test_get_self_narrative_returns_none(self, storage):
        result = storage.get_self_narrative("sn-1")
        assert result is None

    def test_list_self_narratives_returns_empty(self, storage):
        result = storage.list_self_narratives("test")
        assert result == []

    def test_deactivate_self_narratives_returns_zero(self, storage):
        result = storage.deactivate_self_narratives("test", "identity")
        assert result == 0


class TestEpochDefaults:
    """Test default epoch methods — covers lines 572, 576, 580, 584, 588."""

    def test_save_epoch_returns_id(self, storage):
        epoch = Epoch(
            id="ep-1",
            stack_id="test",
            epoch_number=1,
            name="First epoch",
        )
        result = storage.save_epoch(epoch)
        assert result == "ep-1"

    def test_get_epoch_returns_none(self, storage):
        result = storage.get_epoch("ep-1")
        assert result is None

    def test_get_epochs_returns_empty(self, storage):
        result = storage.get_epochs()
        assert result == []

    def test_get_current_epoch_returns_none(self, storage):
        result = storage.get_current_epoch()
        assert result is None

    def test_close_epoch_returns_false(self, storage):
        result = storage.close_epoch("ep-1")
        assert result is False


class TestCloudDefaults:
    """Test default cloud methods — covers lines 627, 641."""

    def test_has_cloud_credentials_returns_false(self, storage):
        result = storage.has_cloud_credentials()
        assert result is False

    def test_cloud_health_check_returns_not_supported(self, storage):
        result = storage.cloud_health_check()
        assert result["healthy"] is False
        assert "not supported" in result["error"]


class TestBatchDefaults:
    """Test default batch methods — covers lines 670, 687, 704."""

    def test_save_episodes_batch_delegates_to_save_episode(self, storage):
        ep1 = Episode(
            id="e-1",
            stack_id="test",
            objective="Task 1",
            outcome="Done 1",
            created_at=datetime.now(timezone.utc),
        )
        ep2 = Episode(
            id="e-2",
            stack_id="test",
            objective="Task 2",
            outcome="Done 2",
            created_at=datetime.now(timezone.utc),
        )
        result = storage.save_episodes_batch([ep1, ep2])
        assert result == ["e-1", "e-2"]
        assert len(storage._episodes) == 2

    def test_save_beliefs_batch_delegates_to_save_belief(self, storage):
        b1 = Belief(
            id="b-1",
            stack_id="test",
            statement="Belief 1",
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
        )
        b2 = Belief(
            id="b-2",
            stack_id="test",
            statement="Belief 2",
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
        )
        result = storage.save_beliefs_batch([b1, b2])
        assert result == ["b-1", "b-2"]
        assert len(storage._beliefs) == 2

    def test_save_notes_batch_delegates_to_save_note(self, storage):
        n1 = Note(
            id="n-1",
            stack_id="test",
            content="Note 1",
            created_at=datetime.now(timezone.utc),
        )
        n2 = Note(
            id="n-2",
            stack_id="test",
            content="Note 2",
            created_at=datetime.now(timezone.utc),
        )
        result = storage.save_notes_batch([n1, n2])
        assert result == ["n-1", "n-2"]
        assert len(storage._notes) == 2


class TestLoadAllDefault:
    """Test default load_all() — covers line 744."""

    def test_load_all_returns_none(self, storage):
        result = storage.load_all()
        assert result is None


class TestRecordAccessBatch:
    """Test default record_access_batch() — covers lines 926-930."""

    def test_record_access_batch_calls_record_access(self, storage):
        accesses = [
            ("episode", "e-1"),
            ("belief", "b-1"),
            ("note", "n-1"),
        ]
        count = storage.record_access_batch(accesses)
        assert count == 3
        assert len(storage._access_log) == 3
        assert storage._access_log[0] == ("episode", "e-1")
        assert storage._access_log[1] == ("belief", "b-1")
        assert storage._access_log[2] == ("note", "n-1")

    def test_record_access_batch_counts_failures(self):
        """record_access_batch counts only successful accesses."""
        s = MinimalStorage("test")
        call_count = 0

        def flaky_record_access(memory_type, memory_id):
            nonlocal call_count
            call_count += 1
            # Fail every other one
            return call_count % 2 == 1

        s.record_access = flaky_record_access
        # Bind default batch method
        import types

        s.record_access_batch = types.MethodType(Storage.record_access_batch, s)

        accesses = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]
        count = s.record_access_batch(accesses)
        assert count == 2  # Only odds succeed


class TestSyncQueueDefaults:
    """Test default sync queue methods — covers lines 1092-1154."""

    def test_now_returns_iso_string(self, storage):
        result = storage._now()
        assert isinstance(result, str)
        # Should be parseable as ISO datetime
        assert "T" in result or "-" in result

    def test_clear_queued_change_is_noop(self, storage):
        # Should not raise
        storage._clear_queued_change(None, "change-1")

    def test_mark_synced_is_noop(self, storage):
        # Should not raise
        storage._mark_synced(None, "episodes", "e-1")

    def test_set_sync_meta_is_noop(self, storage):
        # Should not raise
        storage._set_sync_meta("last_sync", "2024-01-01")

    def test_get_queued_changes_returns_empty(self, storage):
        result = storage.get_queued_changes()
        assert result == []

    def test_get_last_sync_time_returns_none(self, storage):
        result = storage.get_last_sync_time()
        assert result is None

    def test_get_sync_conflicts_returns_empty(self, storage):
        result = storage.get_sync_conflicts()
        assert result == []

    def test_save_sync_conflict_returns_id(self, storage):
        conflict = SyncConflict(
            id="sc-1",
            table="episodes",
            record_id="e-1",
            local_version="v1",
            cloud_version="v2",
            resolution="local_wins",
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )
        result = storage.save_sync_conflict(conflict)
        assert result == "sc-1"

    def test_clear_sync_conflicts_returns_zero(self, storage):
        result = storage.clear_sync_conflicts()
        assert result == 0

    def test_connect_raises_not_implemented(self, storage):
        with pytest.raises(NotImplementedError, match="Subclass must implement"):
            storage._connect()

    def test_get_record_for_push_returns_none(self, storage):
        result = storage._get_record_for_push("episodes", "e-1")
        assert result is None
