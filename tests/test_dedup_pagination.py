"""Tests for dedup pagination (load_raw_content_hashes with offset)."""

from unittest.mock import MagicMock

import kernle.dedup as dedup_mod
from kernle.dedup import load_raw_content_hashes


def _make_entry(blob_text, source="cli"):
    e = MagicMock()
    e.blob = blob_text
    e.content = None
    e.source = source
    return e


class TestLoadHashesPaginates:
    """load_raw_content_hashes should paginate through multiple batches."""

    def test_multiple_batches_no_gaps_no_dupes(self, monkeypatch):
        """With batch_size=2, 5 entries should require 3 batches with correct offsets."""
        monkeypatch.setattr(dedup_mod, "_DEDUP_BATCH_SIZE", 2)

        all_entries = [_make_entry(f"unique content {i}") for i in range(5)]
        mock_storage = MagicMock()
        mock_storage.list_raw.side_effect = [
            all_entries[0:2],  # batch 1: full (2 == batch_size), offset=0
            all_entries[2:4],  # batch 2: full (2 == batch_size), offset=2
            all_entries[4:5],  # batch 3: partial (1 < 2), offset=4
        ]

        result = load_raw_content_hashes(mock_storage, limit=100)

        assert result.rows_scanned == 5
        assert result.rows_matched == 5
        assert len(result.hashes) == 5

        # Verify 3 calls with correct offsets
        assert mock_storage.list_raw.call_count == 3
        calls = mock_storage.list_raw.call_args_list
        assert calls[0].kwargs["offset"] == 0
        assert calls[1].kwargs["offset"] == 2
        assert calls[2].kwargs["offset"] == 4

    def test_limit_stops_pagination_mid_batch(self, monkeypatch):
        """Pagination should stop when rows_scanned reaches limit, even mid-page."""
        monkeypatch.setattr(dedup_mod, "_DEDUP_BATCH_SIZE", 3)

        all_entries = [_make_entry(f"content {i}") for i in range(10)]
        mock_storage = MagicMock()
        # limit=5, batch_size=3
        # batch 1: 3 entries, rows_scanned=3 < 5, continue
        # batch 2: remaining=2, fetch_size=min(3,2)=2, returns 2, rows_scanned=5
        mock_storage.list_raw.side_effect = [
            all_entries[0:3],  # offset=0, limit=3
            all_entries[3:5],  # offset=3, limit=2 (remaining)
        ]

        result = load_raw_content_hashes(mock_storage, limit=5)

        assert result.rows_scanned == 5
        assert result.rows_matched == 5
        assert mock_storage.list_raw.call_count == 2
        # Second call should request only 2 (remaining)
        assert calls_kwargs(mock_storage, 1)["limit"] == 2

    def test_single_batch_partial(self):
        """When all entries fit in one batch, only one call is made."""
        mock_storage = MagicMock()
        mock_storage.list_raw.side_effect = [
            [_make_entry(f"content {i}") for i in range(3)],
        ]

        result = load_raw_content_hashes(mock_storage, limit=100)

        assert result.rows_scanned == 3
        assert mock_storage.list_raw.call_count == 1

    def test_source_filter_corpus(self, monkeypatch):
        """Source filter 'corpus' should only match entries with [corpus:] prefix."""
        monkeypatch.setattr(dedup_mod, "_DEDUP_BATCH_SIZE", 3)

        mock_storage = MagicMock()
        mock_storage.list_raw.side_effect = [
            [
                _make_entry("[corpus:repo] [file:x.py]\ncontent", source="corpus"),
                _make_entry("plain entry", source="cli"),
            ],
        ]

        result = load_raw_content_hashes(mock_storage, source_filter="corpus")
        assert result.rows_scanned == 2
        assert result.rows_matched == 1
        assert len(result.hashes) == 1

    def test_empty_storage(self):
        """Empty storage should return zero counts."""
        mock_storage = MagicMock()
        mock_storage.list_raw.return_value = []

        result = load_raw_content_hashes(mock_storage)
        assert result.rows_scanned == 0
        assert result.rows_matched == 0
        assert len(result.hashes) == 0

    def test_warns_when_limit_hit(self, monkeypatch, caplog):
        """Should log a warning when scan hits the limit (possible truncation)."""
        monkeypatch.setattr(dedup_mod, "_DEDUP_BATCH_SIZE", 3)

        mock_storage = MagicMock()
        # limit=3, batch_size=3, returns exactly 3 (full batch) => rows_scanned==limit
        mock_storage.list_raw.side_effect = [
            [_make_entry(f"content {i}") for i in range(3)],
        ]

        import logging

        with caplog.at_level(logging.WARNING, logger="kernle.dedup"):
            result = load_raw_content_hashes(mock_storage, limit=3)

        assert result.rows_scanned == 3
        assert "hit limit" in caplog.text

    def test_no_warning_when_under_limit(self, monkeypatch, caplog):
        """Should not warn when all entries are scanned before hitting limit."""
        monkeypatch.setattr(dedup_mod, "_DEDUP_BATCH_SIZE", 10)

        mock_storage = MagicMock()
        mock_storage.list_raw.side_effect = [
            [_make_entry(f"content {i}") for i in range(3)],
        ]

        import logging

        with caplog.at_level(logging.WARNING, logger="kernle.dedup"):
            result = load_raw_content_hashes(mock_storage, limit=100)

        assert result.rows_scanned == 3
        assert "hit limit" not in caplog.text

    def test_duplicate_content_deduped_in_hashes(self, monkeypatch):
        """Entries with identical content should produce the same hash (set dedupes)."""
        monkeypatch.setattr(dedup_mod, "_DEDUP_BATCH_SIZE", 2)

        mock_storage = MagicMock()
        mock_storage.list_raw.side_effect = [
            [_make_entry("same content"), _make_entry("same content")],
            [_make_entry("different content")],
        ]

        result = load_raw_content_hashes(mock_storage, limit=100)
        assert result.rows_scanned == 3
        assert result.rows_matched == 3
        assert len(result.hashes) == 2  # "same content" deduped


def calls_kwargs(mock_storage, call_index):
    """Helper to get kwargs from a specific mock call."""
    return mock_storage.list_raw.call_args_list[call_index].kwargs
