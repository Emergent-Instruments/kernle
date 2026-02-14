"""Tests for MCP sync error categories."""

from kernle.mcp.handlers.sync import (
    SYNC_ERROR_APPLY,
    SYNC_ERROR_CONFLICT,
    SYNC_ERROR_NETWORK,
    SYNC_ERROR_VALIDATION,
    _classify_sync_error,
)


def test_classify_conflict_error():
    assert _classify_sync_error("merge conflict on record X") == SYNC_ERROR_CONFLICT


def test_classify_network_error():
    assert _classify_sync_error("connection timeout") == SYNC_ERROR_NETWORK


def test_classify_validation_error():
    assert _classify_sync_error("schema validation failed") == SYNC_ERROR_VALIDATION


def test_classify_generic_error():
    assert _classify_sync_error("table not found") == SYNC_ERROR_APPLY


def test_classify_none_error():
    assert _classify_sync_error(None) == SYNC_ERROR_APPLY


def test_error_categories_in_sync_output():
    """Verify error category labels appear in sync output lines."""
    from unittest.mock import MagicMock

    from kernle.mcp.handlers.sync import handle_memory_sync

    k = MagicMock()
    k.sync.return_value = {
        "pushed": 5,
        "pulled": 3,
        "conflicts": [],
        "errors": ["connection timeout", "table not found"],
    }
    output = handle_memory_sync({}, k)
    assert "network: connection timeout" in output
    assert "apply_failed: table not found" in output
