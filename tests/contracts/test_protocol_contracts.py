"""Protocol contract tests for tightened protocol defaults.

These are lightweight runtime checks that ensure the protocol typing
contracts remain as constrained as expected (mainly literal value sets).
"""

from __future__ import annotations

from typing import Literal, get_args, get_origin, get_type_hints

from kernle.protocols import (
    BeliefType,
    CoreProtocol,
    GoalType,
    InferenceScope,
    MemoryType,
    NoteType,
    ProcessingTransition,
    StackComponentProtocol,
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

    def test_belief_and_note_types_remain_literals(self):
        assert get_origin(BeliefType) is Literal
        assert get_origin(NoteType) is Literal
        assert get_origin(GoalType) is Literal
        assert "factual" in _literal_values(BeliefType)
        assert "note" in _literal_values(NoteType)
        assert "task" in _literal_values(GoalType)
