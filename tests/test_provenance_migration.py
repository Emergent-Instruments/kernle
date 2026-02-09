"""Tests for provenance migration (#352).

Tests cover:
- backfill-provenance adding kernle:pre-v0.9-migration to episodes/notes without derived_from
- link-raw matching episodes to raw entries by timestamp and content
- get_pre_v09_memories finding annotated memories
- get_ungrounded_memories correctly skipping pre-v0.9 annotated memories
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from kernle.stack.sqlite_stack import SQLiteStack
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Episode, Note

STACK_ID = "test-migration"


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.db"
    return SQLiteStorage(STACK_ID, db_path=db_path)


@pytest.fixture
def stack(tmp_path):
    db_path = tmp_path / "test.db"
    return SQLiteStack(STACK_ID, db_path=db_path, components=[], enforce_provenance=False)


# ==============================================================================
# Helpers
# ==============================================================================


def _save_raw(storage, blob="test raw entry", captured_at=None):
    """Save a raw entry and return its ID."""
    raw_id = storage.save_raw(blob, source="cli")
    if captured_at:
        with storage._connect() as conn:
            conn.execute(
                "UPDATE raw_entries SET captured_at = ? WHERE id = ?",
                (captured_at.isoformat(), raw_id),
            )
            conn.commit()
    return raw_id


def _ep(objective="Test episode", outcome="It happened", created_at=None):
    return Episode(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        objective=objective,
        outcome=outcome,
        source_type="direct_experience",
        created_at=created_at or datetime.now(timezone.utc),
    )


def _note(content="Test note", created_at=None):
    return Note(
        id=str(uuid.uuid4()),
        stack_id=STACK_ID,
        content=content,
        note_type="observation",
        created_at=created_at or datetime.now(timezone.utc),
    )


# ==============================================================================
# get_pre_v09_memories
# ==============================================================================


class TestGetPreV09Memories:
    """Tests for SQLiteStorage.get_pre_v09_memories()."""

    def test_finds_annotated_episodes(self, storage):
        ep = _ep()
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps(["kernle:pre-v0.9-migration"]), ep.id),
            )
            conn.commit()

        results = storage.get_pre_v09_memories(STACK_ID)
        assert len(results) == 1
        assert results[0][0] == "episode"
        assert results[0][1] == ep.id
        assert results[0][2] is False  # no auto-link

    def test_detects_auto_linked(self, storage):
        ep = _ep()
        raw_id = _save_raw(storage)
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (
                    json.dumps(
                        [f"raw:{raw_id}", "kernle:auto-linked", "kernle:pre-v0.9-migration"]
                    ),
                    ep.id,
                ),
            )
            conn.commit()

        results = storage.get_pre_v09_memories(STACK_ID)
        assert len(results) == 1
        assert results[0][2] is True  # has auto-link

    def test_ignores_non_annotated(self, storage):
        ep = _ep()
        raw_id = _save_raw(storage)
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps([f"raw:{raw_id}"]), ep.id),
            )
            conn.commit()

        results = storage.get_pre_v09_memories(STACK_ID)
        assert len(results) == 0

    def test_empty_stack(self, storage):
        results = storage.get_pre_v09_memories(STACK_ID)
        assert results == []


# ==============================================================================
# get_ungrounded_memories skips pre-v0.9 annotations
# ==============================================================================


class TestUngroundedSkipsPreV09:
    """Verify get_ungrounded_memories doesn't flag pre-v0.9 annotated memories."""

    def test_annotation_only_not_ungrounded(self, storage):
        ep = _ep()
        storage.save_episode(ep)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps(["kernle:pre-v0.9-migration"]), ep.id),
            )
            conn.commit()

        results = storage.get_ungrounded_memories(STACK_ID)
        ids = [r[1] for r in results]
        assert ep.id not in ids

    def test_no_derived_from_not_ungrounded(self, storage):
        ep = _ep()
        storage.save_episode(ep)

        results = storage.get_ungrounded_memories(STACK_ID)
        ids = [r[1] for r in results]
        assert ep.id not in ids


# ==============================================================================
# backfill-provenance: adds pre-v0.9 annotation
# ==============================================================================


class TestBackfillProvenance:
    """Tests for _migrate_backfill_provenance adding pre-v0.9 annotations."""

    def test_episodes_get_annotation(self, stack):
        ep = _ep()
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        ep_calls = [c for c in calls if c[0][1] == ep.id]
        assert len(ep_calls) == 1
        # set_memory_source(type, id, source_type, derived_from=[...])
        call_kwargs = ep_calls[0].kwargs
        derived_from = call_kwargs.get("derived_from")
        assert derived_from is not None
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_notes_get_annotation(self, stack):
        note = _note()
        stack.save_note(note)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        note_calls = [c for c in calls if c[0][1] == note.id]
        assert len(note_calls) == 1

    def test_episodes_with_provenance_untouched(self, stack):
        raw_id = _save_raw(stack._backend)
        ep = _ep()
        ep.derived_from = [f"raw:{raw_id}"]
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        ep_calls = [c for c in calls if c[0][1] == ep.id]
        assert len(ep_calls) == 0

    def test_dry_run_no_changes(self, stack):
        ep = _ep()
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_backfill_provenance

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = True
        args.json = True

        _migrate_backfill_provenance(args, mock_k)

        mock_k.set_memory_source.assert_not_called()


# ==============================================================================
# link-raw: matches episodes to raw entries
# ==============================================================================


class TestLinkRaw:
    """Tests for _migrate_link_raw matching by timestamp and content."""

    def test_links_by_timestamp_and_content(self, stack):
        now = datetime.now(timezone.utc)
        raw_id = _save_raw(
            stack._backend,
            blob="deployed api service to production",
            captured_at=now - timedelta(minutes=5),
        )

        ep = _ep(
            objective="deployed api service to production",
            outcome="success",
            created_at=now,
        )
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{raw_id}" in derived_from
        assert "kernle:auto-linked" in derived_from

    def test_no_link_outside_window_no_content(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="completely unrelated cooking entry",
            captured_at=now - timedelta(hours=2),
        )

        ep = _ep(objective="deployed api service", outcome="success", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_skips_already_linked(self, stack):
        now = datetime.now(timezone.utc)
        raw_id = _save_raw(stack._backend, blob="test content", captured_at=now)

        ep = _ep(objective="test content episode", created_at=now)
        ep.derived_from = [f"raw:{raw_id}"]
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_dry_run_no_changes(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="matching content dry run",
            captured_at=now,
        )

        ep = _ep(objective="matching content dry run", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = True
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()

    def test_links_notes_too(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="insight about testing prevents bugs from production",
            captured_at=now,
        )

        note = _note(content="insight about testing prevents bugs from production", created_at=now)
        stack.save_note(note)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1

    def test_content_match_picks_best(self, stack):
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="cooking recipes for dinner tonight",
            captured_at=now - timedelta(minutes=10),
        )
        raw_matching_id = _save_raw(
            stack._backend,
            blob="deployed api service to staging environment successfully",
            captured_at=now - timedelta(minutes=10),
        )

        ep = _ep(
            objective="deployed api service to staging environment",
            outcome="it worked",
            created_at=now,
        )
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{raw_matching_id}" in derived_from

    def test_no_raw_entries(self, stack):
        ep = _ep()
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)  # Should not error

    def test_preserves_existing_annotations(self, stack):
        now = datetime.now(timezone.utc)
        raw_id = _save_raw(
            stack._backend,
            blob="test preserved annotation content here",
            captured_at=now,
        )

        ep = _ep(objective="test preserved annotation content here", created_at=now)
        stack.save_episode(ep)
        # Manually add pre-v0.9 annotation
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE episodes SET derived_from = ? WHERE id = ?",
                (json.dumps(["kernle:pre-v0.9-migration"]), ep.id),
            )
            conn.commit()

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{raw_id}" in derived_from
        assert "kernle:auto-linked" in derived_from
        assert "kernle:pre-v0.9-migration" in derived_from

    def test_all_flag_creates_synthetic_raw(self, stack):
        """--all creates synthetic raw entries for unmatched memories."""
        now = datetime.now(timezone.utc)
        # Raw entry is far away in time and has no content overlap
        _save_raw(
            stack._backend,
            blob="completely unrelated entry about cooking",
            captured_at=now - timedelta(hours=5),
        )

        ep = _ep(objective="deployed api service", outcome="success", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = True

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID
        mock_k.set_memory_source = MagicMock(return_value=True)
        # save_raw returns a synthetic raw id
        synthetic_id = str(uuid.uuid4())
        mock_k._storage.save_raw = MagicMock(return_value=synthetic_id)

        _migrate_link_raw(args, mock_k)

        # Synthetic raw should have been created
        mock_k._storage.save_raw.assert_called_once()
        call_args = mock_k._storage.save_raw.call_args
        assert "[migrated episode]" in call_args[0][0]
        assert "deployed api service" in call_args[0][0]

        # Memory should be linked to the synthetic raw
        calls = mock_k.set_memory_source.call_args_list
        assert len(calls) == 1
        derived_from = calls[0].kwargs.get("derived_from")
        assert f"raw:{synthetic_id}" in derived_from
        assert "kernle:auto-linked" in derived_from
        assert "kernle:synthetic-raw" in derived_from

    def test_all_flag_dry_run_no_synthetic(self, stack):
        """--all --dry-run shows synthetic links but doesn't create them."""
        now = datetime.now(timezone.utc)
        # Need at least one raw entry so the function doesn't bail early
        _save_raw(
            stack._backend,
            blob="unrelated raw entry",
            captured_at=now - timedelta(hours=5),
        )

        ep = _ep(objective="unmatched task", outcome="done", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        args = MagicMock()
        args.dry_run = True
        args.json = True
        args.window = 30
        args.link_all = True

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID

        _migrate_link_raw(args, mock_k)

        # Nothing should have been applied (dry run)
        mock_k.set_memory_source.assert_not_called()

    def test_all_flag_not_set_skips_unmatched(self, stack):
        """Without --all, unmatched memories are left alone."""
        now = datetime.now(timezone.utc)
        _save_raw(
            stack._backend,
            blob="unrelated content",
            captured_at=now - timedelta(hours=5),
        )

        ep = _ep(objective="deployed api service", outcome="success", created_at=now)
        stack.save_episode(ep)

        from kernle.cli.commands.migrate import _migrate_link_raw

        mock_k = MagicMock()
        mock_k._storage = stack._backend
        mock_k.stack_id = STACK_ID

        args = MagicMock()
        args.dry_run = False
        args.json = True
        args.window = 30
        args.link_all = False

        _migrate_link_raw(args, mock_k)

        mock_k.set_memory_source.assert_not_called()


class TestPreV09ProvenanceBypass:
    """Test that pre-v0.9 migrated memories bypass provenance validation."""

    def test_pre_v09_annotation_bypasses_provenance(self, tmp_path):
        """Memories with kernle:pre-v0.9-migration pass provenance validation."""
        stack = SQLiteStack(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=True
        )
        stack._state = stack._state.__class__["ACTIVE"]

        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="legacy episode",
            outcome="success",
            derived_from=["kernle:pre-v0.9-migration"],
            created_at=datetime.now(timezone.utc),
        )
        # Should not raise ProvenanceError
        stack.save_episode(ep)

    def test_annotation_only_still_fails_without_pre_v09(self, tmp_path):
        """Other annotation-only refs still fail provenance validation."""
        from kernle.stack.sqlite_stack import ProvenanceError

        stack = SQLiteStack(
            STACK_ID, db_path=tmp_path / "test.db", components=[], enforce_provenance=True
        )
        stack._state = stack._state.__class__["ACTIVE"]

        ep = Episode(
            id=str(uuid.uuid4()),
            stack_id=STACK_ID,
            objective="episode with only context annotation",
            outcome="result",
            derived_from=["context:cli"],
            created_at=datetime.now(timezone.utc),
        )
        with pytest.raises(ProvenanceError):
            stack.save_episode(ep)
