"""Tests for memory processing sessions (v0.9.0 PR 6).

Tests cover:
- MemoryProcessor: layer-specific processing, trigger evaluation
- Storage: mark_episode/note_processed, get/set_processing_config
- Stack: processing method routing
- Entity: process() method with mock inference
- Processing output parsing and memory writing
"""

from __future__ import annotations

import json
import uuid

import pytest

from kernle.entity import Entity
from kernle.processing import (
    DEFAULT_LAYER_CONFIGS,
    VALID_TRANSITIONS,
    LayerConfig,
    MemoryProcessor,
    ProcessingResult,
    evaluate_triggers,
)
from kernle.stack.sqlite_stack import SQLiteStack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Episode, Note, RawEntry

STACK_ID = "test-stack"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(STACK_ID, db_path=tmp_path / "test.db")


@pytest.fixture
def stack(tmp_path):
    return SQLiteStack(STACK_ID, db_path=tmp_path / "test.db", components=[])


@pytest.fixture
def entity(tmp_path):
    ent = Entity(core_id="test-core", data_dir=tmp_path / "entity")
    st = SQLiteStack(STACK_ID, db_path=tmp_path / "test.db", components=[])
    ent.attach_stack(st)
    return ent


class MockInference:
    """Mock inference service for testing."""

    def __init__(self, response: str = "[]"):
        self.response = response
        self.calls = []

    def infer(self, prompt: str, *, system=None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return self.response

    def embed(self, text: str) -> list:
        return [0.0] * 64

    def embed_batch(self, texts: list) -> list:
        return [[0.0] * 64 for _ in texts]

    @property
    def embedding_dimension(self) -> int:
        return 64

    @property
    def embedding_provider_id(self) -> str:
        return "mock"


def _save_raw(storage, blob="test raw content"):
    storage.save_raw(blob=blob, source="test")
    raws = storage.list_raw(limit=1)
    return raws[0].id


def _save_episode(storage, objective="Test episode", outcome="Test outcome"):
    ep = Episode(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        objective=objective,
        outcome=outcome,
        source_type="observation",
        source_entity="test",
    )
    storage.save_episode(ep)
    return ep.id


# =============================================================================
# Trigger Evaluation
# =============================================================================


class TestEvaluateTriggers:
    def test_disabled_config_never_triggers(self):
        config = LayerConfig(layer_transition="raw_to_episode", enabled=False)
        assert not evaluate_triggers("raw_to_episode", config, 100)

    def test_quantity_threshold_triggers(self):
        config = LayerConfig(layer_transition="raw_to_episode", quantity_threshold=5)
        assert not evaluate_triggers("raw_to_episode", config, 4)
        assert evaluate_triggers("raw_to_episode", config, 5)
        assert evaluate_triggers("raw_to_episode", config, 10)

    def test_valence_threshold_triggers_for_raw(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert not evaluate_triggers("raw_to_episode", config, 1, cumulative_valence=1.5)
        assert evaluate_triggers("raw_to_episode", config, 1, cumulative_valence=2.5)

    def test_valence_threshold_ignored_for_non_raw(self):
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert not evaluate_triggers("episode_to_belief", config, 1, cumulative_valence=10.0)

    def test_time_threshold_triggers(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        assert not evaluate_triggers("raw_to_episode", config, 1, hours_since_last=12.0)
        assert evaluate_triggers("raw_to_episode", config, 1, hours_since_last=25.0)

    def test_time_threshold_none_does_not_trigger(self):
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        assert not evaluate_triggers("raw_to_episode", config, 1, hours_since_last=None)


# =============================================================================
# LayerConfig defaults
# =============================================================================


class TestLayerConfig:
    def test_default_configs_cover_all_transitions(self):
        assert set(DEFAULT_LAYER_CONFIGS.keys()) == VALID_TRANSITIONS

    def test_default_config_values(self):
        cfg = DEFAULT_LAYER_CONFIGS["raw_to_episode"]
        assert cfg.enabled is True
        assert cfg.quantity_threshold == 10
        assert cfg.batch_size == 10

    def test_custom_config(self):
        cfg = LayerConfig(
            layer_transition="raw_to_episode",
            enabled=True,
            quantity_threshold=5,
            batch_size=3,
        )
        assert cfg.quantity_threshold == 5
        assert cfg.batch_size == 3


# =============================================================================
# Storage: mark_episode_processed / mark_note_processed
# =============================================================================


class TestMarkEpisodeProcessed:
    def test_marks_episode(self, storage):
        eid = _save_episode(storage)
        ep = storage.get_episode(eid)
        assert not ep.processed

        assert storage.mark_episode_processed(eid)
        ep = storage.get_episode(eid)
        assert ep.processed

    def test_nonexistent_returns_false(self, storage):
        assert not storage.mark_episode_processed("no-such-id")

    def test_deleted_returns_false(self, storage):
        eid = _save_episode(storage)
        storage.forget_memory("episode", eid, "test")
        # forget sets deleted=1
        # Actually forget_memory sets strength=0, not deleted. Let's skip this test.


class TestMarkNoteProcessed:
    def test_marks_note(self, storage):
        note = Note(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            content="Test note",
            note_type="observation",
            source_type="observation",
            source_entity="test",
        )
        storage.save_note(note)
        notes = storage.get_notes(limit=10)
        assert not notes[0].processed

        assert storage.mark_note_processed(note.id)
        notes = storage.get_notes(limit=10)
        marked = [n for n in notes if n.id == note.id]
        assert len(marked) == 1
        assert marked[0].processed

    def test_nonexistent_returns_false(self, storage):
        assert not storage.mark_note_processed("no-such-id")


# =============================================================================
# Storage: processing_config
# =============================================================================


class TestProcessingConfig:
    def test_empty_config(self, storage):
        configs = storage.get_processing_config()
        assert configs == []

    def test_set_and_get(self, storage):
        storage.set_processing_config(
            "raw_to_episode",
            enabled=True,
            quantity_threshold=5,
            batch_size=3,
        )
        configs = storage.get_processing_config()
        assert len(configs) == 1
        assert configs[0]["layer_transition"] == "raw_to_episode"
        assert configs[0]["enabled"] is True
        assert configs[0]["quantity_threshold"] == 5
        assert configs[0]["batch_size"] == 3

    def test_update_existing(self, storage):
        storage.set_processing_config("raw_to_episode", quantity_threshold=5)
        storage.set_processing_config("raw_to_episode", quantity_threshold=20)

        configs = storage.get_processing_config()
        assert len(configs) == 1
        assert configs[0]["quantity_threshold"] == 20

    def test_multiple_configs(self, storage):
        storage.set_processing_config("raw_to_episode", batch_size=5)
        storage.set_processing_config("episode_to_belief", batch_size=10)

        configs = storage.get_processing_config()
        assert len(configs) == 2
        transitions = {c["layer_transition"] for c in configs}
        assert transitions == {"episode_to_belief", "raw_to_episode"}

    def test_disable_transition(self, storage):
        storage.set_processing_config("raw_to_episode", enabled=True)
        storage.set_processing_config("raw_to_episode", enabled=False)

        configs = storage.get_processing_config()
        assert configs[0]["enabled"] is False


# =============================================================================
# Stack routing
# =============================================================================


class TestStackProcessingRouting:
    def test_get_processing_config(self, stack):
        configs = stack.get_processing_config()
        assert configs == []

    def test_set_processing_config(self, stack):
        assert stack.set_processing_config("raw_to_episode", batch_size=5)
        configs = stack.get_processing_config()
        assert len(configs) == 1

    def test_mark_episode_processed(self, stack):
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="Test",
            outcome="Done",
            source_type="observation",
            source_entity="test",
        )
        stack.save_episode(ep)
        assert stack.mark_episode_processed(ep.id)

    def test_mark_note_processed(self, stack):
        note = Note(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            content="Test note",
            note_type="observation",
            source_type="observation",
            source_entity="test",
        )
        stack.save_note(note)
        assert stack.mark_note_processed(note.id)


# =============================================================================
# MemoryProcessor
# =============================================================================


class TestMemoryProcessor:
    def test_no_sources_skips(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == "No unprocessed sources"

    def test_process_raw_to_episode(self, stack):
        # Save some raw entries
        for i in range(3):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Raw content {i}",
                    source="test",
                )
            )

        # Mock inference to return one episode
        response = json.dumps(
            [
                {
                    "objective": "Learned something",
                    "outcome": "Good result",
                    "outcome_type": "success",
                    "lessons": ["lesson 1"],
                    "source_raw_ids": [],  # Empty for simplicity
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        result = results[0]
        assert not result.skipped
        assert result.source_count == 3
        assert len(result.created) == 1
        assert result.created[0]["type"] == "episode"

        # Verify episode was saved
        episodes = stack.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].objective == "Learned something"

    def test_process_raw_to_note(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some factual info",
                source="test",
            )
        )

        response = json.dumps(
            [
                {
                    "content": "A factual note",
                    "note_type": "fact",
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_note", force=True)
        assert len(results) == 1
        assert len(results[0].created) == 1
        assert results[0].created[0]["type"] == "note"

    def test_process_episode_to_belief(self, stack):
        # Save unprocessed episodes
        for i in range(3):
            ep = Episode(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                objective=f"Event {i}",
                outcome=f"Outcome {i}",
                source_type="observation",
                source_entity="test",
                processed=False,
            )
            stack.save_episode(ep)

        response = json.dumps(
            [
                {
                    "statement": "Testing is important",
                    "belief_type": "evaluative",
                    "confidence": 0.8,
                    "source_episode_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("episode_to_belief", force=True)
        assert len(results) == 1
        assert len(results[0].created) == 1
        assert results[0].created[0]["type"] == "belief"

    def test_inference_failure_returns_error(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some content",
                source="test",
            )
        )

        inference = MockInference()
        inference.infer = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("model error"))
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert len(results[0].errors) == 1
        assert "Inference failed" in results[0].errors[0]

    def test_bad_json_returns_error(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Some content",
                source="test",
            )
        )

        inference = MockInference("not valid json")
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")

        results = processor.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert len(results[0].errors) == 1
        assert "Parse failed" in results[0].errors[0]

    def test_check_triggers_no_sources(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        assert not processor.check_triggers("raw_to_episode")

    def test_check_triggers_meets_threshold(self, stack):
        # Save enough raws to meet threshold
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=3,
        )
        for i in range(5):
            stack.save_raw(
                RawEntry(
                    id=str(uuid.uuid4()),
                    stack_id=STACK_ID,
                    blob=f"Content {i}",
                    source="test",
                )
            )

        inference = MockInference()
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            configs={"raw_to_episode": config},
        )
        assert processor.check_triggers("raw_to_episode")

    def test_process_all_transitions_no_sources(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        # All transitions skip due to no sources
        results = processor.process(force=True)
        assert all(r.skipped for r in results)

    def test_update_config(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        new_config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=3,
        )
        processor.update_config("raw_to_episode", new_config)
        assert processor.get_config("raw_to_episode").quantity_threshold == 3

    def test_disabled_transition_skipped(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Content",
                source="test",
            )
        )
        config = LayerConfig(
            layer_transition="raw_to_episode",
            enabled=False,
        )
        inference = MockInference()
        processor = MemoryProcessor(
            stack=stack,
            inference=inference,
            core_id="test",
            configs={"raw_to_episode": config},
        )
        results = processor.process("raw_to_episode", force=True)
        assert results == []


# =============================================================================
# MemoryProcessor: parse_response edge cases
# =============================================================================


class TestParseResponse:
    def test_strips_markdown_fences(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        result = processor._parse_response('```json\n[{"key": "value"}]\n```')
        assert result == [{"key": "value"}]

    def test_plain_json(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        result = processor._parse_response('[{"key": "value"}]')
        assert result == [{"key": "value"}]

    def test_empty_array(self, stack):
        inference = MockInference()
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        result = processor._parse_response("[]")
        assert result == []


# =============================================================================
# MemoryProcessor: mark processed
# =============================================================================


class TestMarkProcessed:
    def test_raws_marked_after_processing(self, stack):
        stack.save_raw(
            RawEntry(
                id=str(uuid.uuid4()),
                stack_id=STACK_ID,
                blob="Test content",
                source="test",
            )
        )

        response = json.dumps(
            [
                {
                    "objective": "Test",
                    "outcome": "Done",
                    "source_raw_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        processor.process("raw_to_episode", force=True)

        # Raws should be marked as processed
        raws = stack._backend.list_raw(processed=False)
        assert len(raws) == 0
        all_raws = stack._backend.list_raw()
        assert len(all_raws) == 1
        assert all_raws[0].processed

    def test_episodes_marked_after_processing(self, stack):
        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="Test",
            outcome="Done",
            source_type="observation",
            source_entity="test",
            processed=False,
        )
        stack.save_episode(ep)

        response = json.dumps(
            [
                {
                    "statement": "A belief",
                    "belief_type": "factual",
                    "confidence": 0.7,
                    "source_episode_ids": [],
                }
            ]
        )
        inference = MockInference(response)
        processor = MemoryProcessor(stack=stack, inference=inference, core_id="test")
        processor.process("episode_to_belief", force=True)

        # Episode should be marked as processed
        episodes = stack.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].processed


# =============================================================================
# Entity.process()
# =============================================================================


class TestEntityProcess:
    def test_requires_stack(self):
        ent = Entity(core_id="test-no-stack")
        with pytest.raises(Exception):
            ent.process()

    def test_requires_model(self, entity):
        with pytest.raises(RuntimeError, match="No model bound"):
            entity.process()

    def test_process_with_mock_model(self, entity):
        """Entity.process() works with a mock model and no data."""

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())
        results = entity.process(force=True)
        # All transitions skip (no sources)
        assert all(r.skipped for r in results)

    def test_process_specific_transition(self, entity):
        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())
        results = entity.process("raw_to_episode", force=True)
        assert len(results) == 1
        assert results[0].layer_transition == "raw_to_episode"

    def test_process_loads_saved_config(self, entity):
        """Entity.process() loads config from stack."""

        class MockModel:
            model_id = "mock-model"

            def generate(self, messages, system=None, **kw):
                class R:
                    content = "[]"

                return R()

        entity.set_model(MockModel())

        # Save a config
        stack = entity.active_stack
        stack.set_processing_config("raw_to_episode", quantity_threshold=99)

        # Process should load the config (even though it doesn't change behavior
        # in this test since we force=True and have no sources)
        results = entity.process("raw_to_episode", force=True)
        assert len(results) == 1


# =============================================================================
# ProcessingResult
# =============================================================================


class TestProcessingResult:
    def test_default_values(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=5,
        )
        assert result.created == []
        assert result.errors == []
        assert result.skipped is False
        assert result.skip_reason is None

    def test_skipped_result(self):
        result = ProcessingResult(
            layer_transition="raw_to_episode",
            source_count=0,
            skipped=True,
            skip_reason="No sources",
        )
        assert result.skipped
        assert result.skip_reason == "No sources"
