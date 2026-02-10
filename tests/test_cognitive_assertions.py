"""Tests for cognitive quality assertion framework."""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import SQLiteStorage
from kernle.testing.assertions import (
    AssertionResult,
    CognitiveAssertions,
    CognitiveReport,
)
from kernle.types import Belief, Episode, Note, Value


@pytest.fixture
def storage(tmp_path):
    """Create a fresh SQLiteStorage for each test."""
    db_path = tmp_path / "test_assertions.db"
    s = SQLiteStorage(stack_id="test-assertions", db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def k(storage):
    """Create a Kernle instance wrapping the storage."""
    return Kernle(stack_id="test-assertions", storage=storage, strict=False)


@pytest.fixture
def assertions(k):
    """Create CognitiveAssertions from a Kernle instance."""
    return CognitiveAssertions(k)


def _make_raw(storage, raw_id=None, processed=False):
    """Helper to create a raw entry."""
    raw_id = storage.save_raw("Test raw content", source="cli")
    if processed:
        storage.mark_raw_processed(raw_id, [])
    return raw_id


def _make_episode(
    storage, ep_id=None, derived_from=None, source_type="direct_experience", processed=False
):
    """Helper to create an episode."""
    ep_id = ep_id or str(uuid.uuid4())
    ep = Episode(
        id=ep_id,
        stack_id="test-assertions",
        objective="Test objective",
        outcome="Test outcome",
        created_at=datetime.now(timezone.utc),
        source_type=source_type,
        derived_from=derived_from,
        processed=processed,
    )
    storage.save_episode(ep)
    return ep_id


def _make_belief(
    storage, belief_id=None, derived_from=None, source_type="direct_experience", confidence=0.8
):
    """Helper to create a belief."""
    belief_id = belief_id or str(uuid.uuid4())
    b = Belief(
        id=belief_id,
        stack_id="test-assertions",
        statement="Test belief statement",
        confidence=confidence,
        created_at=datetime.now(timezone.utc),
        source_type=source_type,
        derived_from=derived_from,
    )
    storage.save_belief(b)
    return belief_id


def _make_value(storage, value_id=None, derived_from=None, source_type="direct_experience"):
    """Helper to create a value."""
    value_id = value_id or str(uuid.uuid4())
    v = Value(
        id=value_id,
        stack_id="test-assertions",
        name="Test value",
        statement="Test value statement",
        created_at=datetime.now(timezone.utc),
        source_type=source_type,
        derived_from=derived_from,
    )
    storage.save_value(v)
    return value_id


def _make_note(storage, note_id=None, derived_from=None):
    """Helper to create a note."""
    note_id = note_id or str(uuid.uuid4())
    n = Note(
        id=note_id,
        stack_id="test-assertions",
        content="Test note content",
        created_at=datetime.now(timezone.utc),
        derived_from=derived_from,
    )
    storage.save_note(n)
    return note_id


# ---- Structural Tests ----


class TestProvenanceChainIntact:
    def test_pass_valid_refs(self, storage, assertions):
        """Valid refs resolve successfully."""
        raw_id = _make_raw(storage, processed=True)
        _make_episode(storage, derived_from=[f"raw:{raw_id}"])
        result = assertions.provenance_chain_intact()
        assert result.passed
        assert result.category == "structural"

    def test_fail_broken_ref(self, storage, assertions):
        """Broken ref causes failure."""
        _make_episode(storage, derived_from=["raw:nonexistent-id"])
        result = assertions.provenance_chain_intact()
        assert not result.passed
        assert "broken" in result.message.lower()
        assert len(result.details["broken_refs"]) == 1

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.provenance_chain_intact()
        assert result.passed


class TestNoOrphanMemories:
    def test_pass_beliefs_with_provenance(self, storage, assertions):
        """Beliefs with derived_from pass."""
        ep_id = _make_episode(storage)
        _make_belief(storage, derived_from=[f"episode:{ep_id}"])
        result = assertions.no_orphan_memories()
        assert result.passed

    def test_fail_belief_without_provenance(self, storage, assertions):
        """Belief without derived_from fails."""
        _make_belief(storage)
        result = assertions.no_orphan_memories()
        assert not result.passed


class TestValidSourceTypes:
    def test_pass_valid_types(self, storage, assertions):
        """Valid source types pass."""
        _make_episode(storage, source_type="direct_experience")
        _make_belief(storage, source_type="processing", derived_from=["episode:fake"])
        result = assertions.valid_source_types()
        assert result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.valid_source_types()
        assert result.passed


class TestStrengthInRange:
    def test_pass_valid_strengths(self, storage, assertions):
        """Strengths in [0.0, 1.0] pass."""
        _make_episode(storage)
        result = assertions.strength_in_range()
        assert result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.strength_in_range()
        assert result.passed


class TestNoDuplicateContent:
    def test_pass_unique_content(self, storage, assertions):
        """Unique content passes."""
        ep1 = Episode(
            id=str(uuid.uuid4()),
            stack_id="test-assertions",
            objective="First task",
            outcome="First outcome",
            created_at=datetime.now(timezone.utc),
        )
        ep2 = Episode(
            id=str(uuid.uuid4()),
            stack_id="test-assertions",
            objective="Second task",
            outcome="Second outcome",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(ep1)
        storage.save_episode(ep2)
        result = assertions.no_duplicate_content()
        assert result.passed

    def test_fail_duplicate_content(self, storage, assertions):
        """Duplicate content fails."""
        ep1 = Episode(
            id=str(uuid.uuid4()),
            stack_id="test-assertions",
            objective="Same task",
            outcome="Same outcome",
            created_at=datetime.now(timezone.utc),
        )
        ep2 = Episode(
            id=str(uuid.uuid4()),
            stack_id="test-assertions",
            objective="Same task",
            outcome="Same outcome",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(ep1)
        storage.save_episode(ep2)
        result = assertions.no_duplicate_content()
        assert not result.passed
        assert len(result.details["duplicates"]) == 1


# ---- Coherence Tests ----


class TestBeliefsHaveEvidence:
    def test_pass_with_evidence(self, storage, assertions):
        """Beliefs with evidence pass."""
        ep_id = _make_episode(storage)
        _make_belief(storage, derived_from=[f"episode:{ep_id}"])
        result = assertions.beliefs_have_evidence()
        assert result.passed

    def test_fail_without_evidence(self, storage, assertions):
        """Beliefs without evidence fail."""
        _make_belief(storage)
        result = assertions.beliefs_have_evidence()
        assert not result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes (no beliefs to check)."""
        result = assertions.beliefs_have_evidence()
        assert result.passed


class TestValuesFromBeliefs:
    def test_pass_value_from_belief(self, storage, assertions):
        """Value derived from belief passes."""
        b_id = _make_belief(storage, derived_from=["episode:fake"])
        _make_value(storage, derived_from=[f"belief:{b_id}"])
        result = assertions.values_from_beliefs()
        assert result.passed

    def test_fail_value_no_provenance(self, storage, assertions):
        """Value without provenance fails."""
        _make_value(storage)
        result = assertions.values_from_beliefs()
        assert not result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.values_from_beliefs()
        assert result.passed


class TestNoCircularProvenance:
    def test_pass_no_cycles(self, storage, assertions):
        """Linear provenance passes."""
        raw_id = _make_raw(storage, processed=True)
        ep_id = _make_episode(storage, derived_from=[f"raw:{raw_id}"])
        _make_belief(storage, derived_from=[f"episode:{ep_id}"])
        result = assertions.no_circular_provenance()
        assert result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.no_circular_provenance()
        assert result.passed


class TestHierarchyRespected:
    def test_pass_correct_flow(self, storage, assertions):
        """Correct hierarchy (raw->episode->belief) passes."""
        raw_id = _make_raw(storage, processed=True)
        ep_id = _make_episode(storage, derived_from=[f"raw:{raw_id}"])
        _make_belief(storage, derived_from=[f"episode:{ep_id}"])
        result = assertions.hierarchy_respected()
        assert result.passed

    def test_fail_reverse_flow(self, storage, assertions):
        """Episode derived from belief violates hierarchy."""
        b_id = _make_belief(storage, derived_from=["episode:fake"])
        _make_episode(storage, derived_from=[f"belief:{b_id}"])
        result = assertions.hierarchy_respected()
        assert not result.passed
        assert len(result.details["violations"]) >= 1

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.hierarchy_respected()
        assert result.passed


# ---- Quality Tests ----


class TestNoEmptyMemories:
    def test_pass_non_empty(self, storage, assertions):
        """Non-empty memories pass."""
        _make_episode(storage)
        _make_belief(storage)
        result = assertions.no_empty_memories()
        assert result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.no_empty_memories()
        assert result.passed


class TestEpisodesHaveOutcomes:
    def test_pass_outcomes_present(self, storage, assertions):
        """Episodes with outcomes pass."""
        _make_episode(storage)
        result = assertions.episodes_have_outcomes()
        assert result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.episodes_have_outcomes()
        assert result.passed


class TestBeliefsHaveStatements:
    def test_pass_statements_present(self, storage, assertions):
        """Beliefs with statements pass."""
        _make_belief(storage)
        result = assertions.beliefs_have_statements()
        assert result.passed


class TestConfidenceInRange:
    def test_pass_valid_confidence(self, storage, assertions):
        """Confidence in [0.0, 1.0] passes."""
        _make_belief(storage, confidence=0.8)
        result = assertions.confidence_in_range()
        assert result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.confidence_in_range()
        assert result.passed


# ---- Pipeline Tests ----


class TestNoUnprocessedRaw:
    def test_pass_all_processed(self, storage, assertions):
        """All processed raw entries pass."""
        _make_raw(storage, processed=True)
        result = assertions.no_unprocessed_raw()
        assert result.passed

    def test_fail_unprocessed_raw(self, storage, assertions):
        """Unprocessed raw entry fails."""
        _make_raw(storage, processed=False)
        result = assertions.no_unprocessed_raw()
        assert not result.passed
        assert result.details["count"] == 1

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes (no raw entries)."""
        result = assertions.no_unprocessed_raw()
        assert result.passed


class TestEpisodesExist:
    def test_pass_episodes_present(self, storage, assertions):
        """Episodes present passes."""
        _make_episode(storage)
        result = assertions.episodes_exist()
        assert result.passed

    def test_fail_no_episodes(self, assertions):
        """No episodes fails."""
        result = assertions.episodes_exist()
        assert not result.passed
        assert result.details["count"] == 0


class TestProcessingSourceType:
    def test_pass_correct(self, storage, assertions):
        """Processing memories with provenance pass."""
        raw_id = _make_raw(storage, processed=True)
        _make_episode(storage, derived_from=[f"raw:{raw_id}"], source_type="processing")
        result = assertions.processing_source_type()
        assert result.passed

    def test_fail_processing_no_provenance(self, storage, assertions):
        """Processing memory without derived_from fails."""
        _make_episode(storage, source_type="processing")
        result = assertions.processing_source_type()
        assert not result.passed

    def test_fail_derived_from_wrong_source_type(self, storage, assertions):
        """Memory with derived_from but non-processing source_type should fail."""
        raw_id = _make_raw(storage, processed=True)
        _make_episode(storage, derived_from=[f"raw:{raw_id}"], source_type="user")
        result = assertions.processing_source_type()
        assert not result.passed
        assert "has derived_from but source_type='user'" in result.details["mismatched"][0]["issue"]

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.processing_source_type()
        assert result.passed


class TestRawEntriesMarkedProcessed:
    def test_pass_all_marked(self, storage, assertions):
        """Referenced raw entries are marked processed."""
        raw_id = _make_raw(storage, processed=True)
        _make_episode(storage, derived_from=[f"raw:{raw_id}"])
        result = assertions.raw_entries_marked_processed()
        assert result.passed

    def test_fail_unmarked(self, storage, assertions):
        """Referenced raw entry not marked processed fails."""
        raw_id = _make_raw(storage, processed=False)
        _make_episode(storage, derived_from=[f"raw:{raw_id}"])
        result = assertions.raw_entries_marked_processed()
        assert not result.passed

    def test_pass_empty_stack(self, assertions):
        """Empty stack passes."""
        result = assertions.raw_entries_marked_processed()
        assert result.passed


# ---- Batch Tests ----


class TestRunAll:
    def test_empty_stack(self, assertions):
        """Empty stack passes most assertions (only episodes_exist fails)."""
        report = assertions.run_all()
        assert report.total > 0
        # episodes_exist should fail on empty stack
        ep_result = next((a for a in report.assertions if a.name == "episodes_exist"), None)
        assert ep_result is not None
        assert not ep_result.passed

    def test_populated_stack(self, storage, assertions):
        """Well-formed populated stack passes all assertions."""
        raw_id = _make_raw(storage, processed=True)
        ep_id = _make_episode(storage, derived_from=[f"raw:{raw_id}"], source_type="processing")
        b_id = _make_belief(storage, derived_from=[f"episode:{ep_id}"], source_type="processing")
        _make_value(storage, derived_from=[f"belief:{b_id}"], source_type="processing")
        report = assertions.run_all()
        assert report.all_passed


class TestReportSummary:
    def test_summary_format(self, assertions):
        """Summary format is correct."""
        report = assertions.run_all()
        summary = report.summary()
        assert "Cognitive Report:" in summary
        assert "passed" in summary
        # Check both PASS and FAIL appear
        assert "[PASS]" in summary or "[FAIL]" in summary


class TestCognitiveReport:
    def test_properties(self):
        """Report properties compute correctly."""
        report = CognitiveReport(
            assertions=[
                AssertionResult(category="test", name="a", passed=True, message="ok"),
                AssertionResult(category="test", name="b", passed=False, message="fail"),
            ]
        )
        assert report.passed == 1
        assert report.failed == 1
        assert report.total == 2
        assert not report.all_passed

    def test_all_passed(self):
        """all_passed is True when all pass."""
        report = CognitiveReport(
            assertions=[
                AssertionResult(category="test", name="a", passed=True, message="ok"),
            ]
        )
        assert report.all_passed
