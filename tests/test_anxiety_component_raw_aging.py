"""Tests for AnxietyComponent raw_aging dimension (#572).

Verifies that the component computes raw_aging from actual storage data
instead of hardcoding to 0.
"""

from datetime import datetime, timedelta, timezone

import pytest

from kernle.anxiety_core import compute_raw_aging_score
from kernle.stack.components.anxiety import AnxietyComponent
from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test_raw_aging.db"
    s = SQLiteStorage(stack_id="test-stack", db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def comp(storage):
    c = AnxietyComponent()
    c.attach("test-stack")
    c.set_storage(storage)
    return c


def _save_raw(storage, blob="test blob"):
    return storage.save_raw(blob, source="cli")


class TestComponentRawAging:
    """AnxietyComponent computes raw_aging from actual data."""

    def test_component_raw_aging_nonzero_with_aging_entries(self, storage, comp):
        """5 entries older than 24h → score > 0."""
        for i in range(5):
            _save_raw(storage, blob=f"old entry {i}")

        # Backdate all entries to 48 hours ago
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        with storage._connect() as conn:
            conn.execute(
                "UPDATE raw_entries SET captured_at = ?, timestamp = ?",
                (old_time.isoformat(), old_time.isoformat()),
            )

        report = comp.get_anxiety_report()
        raw_aging_score = report["dimensions"]["raw_aging"]["score"]
        assert raw_aging_score > 0, f"Expected nonzero raw_aging, got {raw_aging_score}"

    def test_component_raw_aging_zero_when_no_entries(self, comp):
        """No raw entries → score = 0."""
        report = comp.get_anxiety_report()
        assert report["dimensions"]["raw_aging"]["score"] == 0

    def test_component_raw_aging_matches_core_function(self, storage, comp):
        """Component produces same score as compute_raw_aging_score()."""
        for i in range(3):
            _save_raw(storage, blob=f"entry {i}")

        # Backdate first 2 entries to 72 hours ago
        old_time = datetime.now(timezone.utc) - timedelta(hours=72)
        with storage._connect() as conn:
            rows = conn.execute("SELECT id FROM raw_entries ORDER BY id LIMIT 2").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE raw_entries SET captured_at = ?, timestamp = ? WHERE id = ?",
                    (old_time.isoformat(), old_time.isoformat(), row[0]),
                )

        report = comp.get_anxiety_report()
        component_score = report["dimensions"]["raw_aging"]["score"]

        # Compute expected score from core function
        expected = compute_raw_aging_score(
            total_unprocessed=3,
            aging_count=2,
            oldest_hours=72,
        )
        assert component_score == expected

    def test_component_raw_aging_graceful_without_list_raw(self, storage, comp):
        """Storage without list_raw → defaults to 0."""

        # Use a mock storage without list_raw
        class NoListRawStorage:
            """Minimal storage mock without list_raw."""

            def get_episodes(self, limit=100):
                return []

            def get_beliefs(self, limit=100):
                return []

            def get_values(self, limit=10):
                return []

            def get_current_epoch(self):
                return None

        comp.set_storage(NoListRawStorage())
        report = comp.get_anxiety_report()
        assert report["dimensions"]["raw_aging"]["score"] == 0

        # Restore
        comp.set_storage(storage)
