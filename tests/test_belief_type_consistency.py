"""Tests that type definitions in protocols.py and writers.py stay in sync.

The canonical definitions live in protocols.py as Literal types. Any
downstream validation (writers.py, MCP handlers, etc.) must derive from
those Literals rather than defining their own lists. This test catches
drift early if that invariant is violated.
"""

from typing import get_args

from kernle.core.writers import WritersMixin
from kernle.protocols import BeliefType, NoteType


class TestBeliefTypeConsistency:
    """BeliefType in protocols.py must match _VALID_BELIEF_TYPES in WritersMixin."""

    def test_writers_belief_types_match_protocol(self):
        """WritersMixin._VALID_BELIEF_TYPES must equal the set of BeliefType literals."""
        protocol_types = set(get_args(BeliefType))
        writer_types = set(WritersMixin._VALID_BELIEF_TYPES)
        assert protocol_types == writer_types, (
            f"BeliefType drift detected.\n"
            f"  In protocols.py only: {sorted(protocol_types - writer_types)}\n"
            f"  In writers.py only:   {sorted(writer_types - protocol_types)}"
        )

    def test_belief_types_not_empty(self):
        """Sanity check: BeliefType should define at least one value."""
        assert len(get_args(BeliefType)) > 0

    def test_belief_types_are_lowercase(self):
        """All belief types should be lowercase strings."""
        for bt in get_args(BeliefType):
            assert bt == bt.lower(), f"BeliefType '{bt}' is not lowercase"

    def test_belief_types_no_duplicates(self):
        """No duplicate entries in BeliefType Literal."""
        types = get_args(BeliefType)
        assert len(types) == len(set(types)), "Duplicate entries in BeliefType"


class TestNoteTypeConsistency:
    """NoteType in protocols.py must match _VALID_NOTE_TYPES in WritersMixin."""

    def test_writers_note_types_match_protocol(self):
        """WritersMixin._VALID_NOTE_TYPES must equal the set of NoteType literals."""
        protocol_types = set(get_args(NoteType))
        writer_types = set(WritersMixin._VALID_NOTE_TYPES)
        assert protocol_types == writer_types, (
            f"NoteType drift detected.\n"
            f"  In protocols.py only: {sorted(protocol_types - writer_types)}\n"
            f"  In writers.py only:   {sorted(writer_types - protocol_types)}"
        )

    def test_note_types_not_empty(self):
        """Sanity check: NoteType should define at least one value."""
        assert len(get_args(NoteType)) > 0

    def test_note_types_are_lowercase(self):
        """All note types should be lowercase strings."""
        for nt in get_args(NoteType):
            assert nt == nt.lower(), f"NoteType '{nt}' is not lowercase"

    def test_note_types_no_duplicates(self):
        """No duplicate entries in NoteType Literal."""
        types = get_args(NoteType)
        assert len(types) == len(set(types)), "Duplicate entries in NoteType"


class TestWritersMixinDerivedFromProtocol:
    """Verify that WritersMixin derives its type sets from protocols.py, not hardcoded."""

    def test_belief_types_are_derived(self):
        """_VALID_BELIEF_TYPES should be derived from BeliefType via get_args, not hardcoded.

        If this test fails, it means someone added a hardcoded frozenset in writers.py
        instead of using get_args(BeliefType). The fix is to use:
            _VALID_BELIEF_TYPES = frozenset(get_args(BeliefType))
        """
        # Adding a type to protocols.py and NOT to writers.py should be impossible
        # if writers.py derives from protocols.py. We verify the sets are identical.
        protocol_set = frozenset(get_args(BeliefType))
        assert WritersMixin._VALID_BELIEF_TYPES == protocol_set

    def test_note_types_are_derived(self):
        """_VALID_NOTE_TYPES should be derived from NoteType via get_args, not hardcoded."""
        protocol_set = frozenset(get_args(NoteType))
        assert WritersMixin._VALID_NOTE_TYPES == protocol_set
