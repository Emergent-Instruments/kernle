"""Protocol contract tests for tightened protocol defaults.

These are lightweight runtime checks that ensure the protocol typing
contracts remain as constrained as expected (mainly literal value sets).
"""

from __future__ import annotations

from typing import Literal, get_args, get_origin, get_type_hints

from kernle.protocols import (
    BeliefType,
    CoreProtocol,
    DumpFormat,
    GoalStatus,
    GoalType,
    InferenceScope,
    MemoryType,
    ModelRole,
    NoteType,
    ProcessingTransition,
    SearchRecordType,
    StackComponentProtocol,
    StackProtocol,
    SuggestionMemoryType,
    SuggestionStatus,
)


def _literal_values(annotation) -> set:
    return set(get_args(annotation))


class TestProtocolTypeConstraints:
    def test_inference_scope_is_literal(self):
        assert get_origin(InferenceScope) is Literal
        assert _literal_values(InferenceScope) == {
            "none",
            "fast",
            "capable",
            "embedding",
        }

    def test_processing_transition_is_literal(self):
        assert get_origin(ProcessingTransition) is Literal
        assert _literal_values(ProcessingTransition) == {
            "raw_to_episode",
            "raw_to_note",
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "belief_to_value",
            "episode_to_drive",
        }

    def test_protocol_annotations_reference_tight_types(self):
        scope_hints = get_type_hints(StackComponentProtocol.__dict__["inference_scope"].fget)
        assert scope_hints["return"] is InferenceScope

        save_hints = get_type_hints(StackComponentProtocol.__dict__["on_save"])
        assert save_hints["memory_type"] is MemoryType

        set_cfg_hints = get_type_hints(CoreProtocol.__dict__["belief"])
        assert set_cfg_hints["type"] is BeliefType

        assert get_type_hints(StackProtocol.__dict__["dump"])["format"] is DumpFormat
        assert (
            get_type_hints(StackProtocol.__dict__["set_processing_config"])["layer_transition"]
            is ProcessingTransition
        )

    def test_belief_and_note_types_remain_literals(self):
        assert get_origin(BeliefType) is Literal
        assert get_origin(NoteType) is Literal
        assert get_origin(GoalType) is Literal
        assert "factual" in _literal_values(BeliefType)
        assert "note" in _literal_values(NoteType)
        assert "task" in _literal_values(GoalType)

    def test_constrained_contract_literals(self):
        assert get_origin(GoalStatus) is Literal
        assert _literal_values(GoalStatus) == {"active", "completed", "paused"}

        assert get_origin(SuggestionStatus) is Literal
        assert _literal_values(SuggestionStatus) == {
            "pending",
            "promoted",
            "modified",
            "rejected",
            "dismissed",
            "expired",
        }

        assert get_origin(SuggestionMemoryType) is Literal
        assert _literal_values(SuggestionMemoryType) == {
            "belief",
            "note",
            "episode",
            "goal",
            "relationship",
            "value",
            "drive",
        }

        assert get_origin(SearchRecordType) is Literal
        assert _literal_values(SearchRecordType) == {"episode", "note", "belief", "value", "goal"}

        assert get_origin(DumpFormat) is Literal
        assert _literal_values(DumpFormat) == {"markdown", "json"}

        assert get_origin(ModelRole) is Literal
        assert _literal_values(ModelRole) == {"system", "user", "assistant", "tool"}
