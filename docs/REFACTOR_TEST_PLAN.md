# Kernle Storage Abstraction Refactoring Test Plan

**Date:** 2025-01-27  
**Purpose:** Ensure zero behavior regression when refactoring Kernle from direct Supabase calls to Storage abstraction layer.

---

## Overview

### Current State
- `kernle/core.py`: Direct Supabase client calls for all storage operations
- `kernle/storage/base.py`: Storage protocol/interface with dataclasses (defined but not integrated)
- `kernle/storage/sqlite.py`: Complete SQLite implementation of Storage protocol
- `tests/test_core.py`: Core tests using mocked Supabase responses
- `tests/test_mcp.py`: MCP server tests (mock-based, quality issues per TEST_AUDIT.md)

### Target State
- `kernle/core.py`: Accepts any `Storage` implementation via dependency injection
- Both `SQLiteStorage` and `SupabaseStorage` work identically through the same interface
- All existing behavior preserved

### Key Risks
1. **Behavioral regression** in existing functionality
2. **Data format mismatches** between storage implementations
3. **Edge case handling** differences between SQLite and Supabase
4. **Performance regressions** (not covered in this plan, but noted)

---

## Phase 1: Pre-Refactoring Tests (Write BEFORE Changes)

These tests lock in current behavior so regressions are immediately detected.

### 1.1 Behavioral Snapshot Tests

Create `tests/test_behavioral_snapshots.py` to capture exact current behavior:

```python
"""Behavioral snapshot tests - capture current Kernle behavior before refactoring.

These tests should pass BEFORE and AFTER the refactoring.
Any failure indicates a regression.
"""
```

#### Test Cases:

| ID | Test Name | Description | Expected Behavior |
|----|-----------|-------------|-------------------|
| B01 | `test_load_returns_all_memory_types` | Call `load()` and verify structure | Returns dict with keys: checkpoint, values, beliefs, goals, drives, lessons, recent_work, recent_notes, relationships |
| B02 | `test_load_values_ordered_by_priority` | Load values with `limit=10` | Returns list ordered by `priority DESC` |
| B03 | `test_load_beliefs_ordered_by_confidence` | Load beliefs | Returns list ordered by `confidence DESC` |
| B04 | `test_load_goals_filters_active_only` | Load goals | Only returns goals with `status="active"` |
| B05 | `test_load_lessons_extracts_from_reflected_episodes` | Load lessons | Extracts `lessons_learned` only from episodes with `is_reflected=True` |
| B06 | `test_load_lessons_limits_to_2_per_episode` | Load lessons with many per episode | Takes max 2 lessons per episode |
| B07 | `test_load_recent_work_excludes_checkpoints` | Load recent work | Excludes episodes tagged with "checkpoint" |
| B08 | `test_checkpoint_saves_locally_and_remotely` | Create checkpoint | Saves to local JSON file AND creates Supabase episode |
| B09 | `test_checkpoint_history_limited_to_10` | Create 15 checkpoints | Local file contains only last 10 |
| B10 | `test_episode_outcome_type_detection` | Create episodes with various outcomes | "success"/"done"/"completed" → success; "failed"/"failure"/"error" → failure; others → partial |
| B11 | `test_note_formats_decision_with_reason` | Create decision note with reason | Content formatted as "**Decision**: {content}\n**Reason**: {reason}" |
| B12 | `test_note_formats_quote_with_speaker` | Create quote note | Content formatted as '> "{content}"\n> — {speaker}' |
| B13 | `test_note_formats_insight` | Create insight note | Content formatted as "**Insight**: {content}" |
| B14 | `test_search_case_insensitive` | Search with mixed case | Matches regardless of case |
| B15 | `test_search_across_all_types` | Search term present in episode, note, and belief | Returns results from all three types |
| B16 | `test_drive_intensity_clamped` | Create drive with intensity > 1.0 and < 0.0 | Intensity clamped to [0.0, 1.0] |
| B17 | `test_drive_upsert_by_type` | Create same drive_type twice | Second call updates existing, doesn't create duplicate |
| B18 | `test_satisfy_drive_minimum_intensity` | Satisfy drive beyond current intensity | Intensity never goes below 0.1 |
| B19 | `test_consolidate_creates_beliefs_from_repeated_lessons` | Consolidate episodes with repeated lesson | Creates new belief with "learned" type |
| B20 | `test_consolidate_marks_episodes_reflected` | Consolidate episodes | All processed episodes get `is_reflected=True` |
| B21 | `test_signal_detection_weights` | Test each signal pattern | Verify exact weight values per SIGNAL_PATTERNS |
| B22 | `test_auto_capture_decision_type` | Auto-capture with decision keywords | Creates note with type="decision" |
| B23 | `test_auto_capture_lesson_type` | Auto-capture with lesson keywords | Creates note with type="insight" |
| B24 | `test_what_happened_time_ranges` | Test "today", "yesterday", "this week", "last hour" | Correct date ranges returned |
| B25 | `test_status_returns_accurate_counts` | Get status after adding data | Counts match actual records |

### 1.2 Contract Tests for Storage Interface

Create `tests/test_storage_contract.py` - tests that any Storage implementation MUST pass:

```python
"""Storage contract tests - verify Storage protocol compliance.

Any class implementing Storage must pass ALL these tests.
Run against: SQLiteStorage, SupabaseStorage (when implemented)
"""
```

#### Test Cases:

| ID | Test Name | Description | Contract Requirement |
|----|-----------|-------------|----------------------|
| C01 | `test_save_episode_returns_id` | Save an episode | Returns non-empty string ID |
| C02 | `test_save_episode_sets_created_at` | Save episode without created_at | Storage sets created_at automatically |
| C03 | `test_get_episode_by_id` | Save then retrieve episode | Retrieved episode matches saved data |
| C04 | `test_get_episodes_respects_limit` | Save 20, get with limit=5 | Returns exactly 5 |
| C05 | `test_get_episodes_filters_by_since` | Save episodes over time range | Only returns episodes >= since |
| C06 | `test_get_episodes_filters_by_tags` | Save episodes with various tags | Only returns matching tags |
| C07 | `test_save_belief_deduplication` | Call `find_belief` then `save_belief` | Can find existing by statement |
| C08 | `test_get_values_ordered_by_priority` | Save values with different priorities | Returned ordered by priority DESC |
| C09 | `test_get_goals_filters_by_status` | Save active and completed goals | status="active" filter works |
| C10 | `test_save_note_all_types` | Save note, decision, insight, quote | All types persisted correctly |
| C11 | `test_get_notes_filters_by_type` | Save mixed note types | note_type filter works |
| C12 | `test_save_drive_upsert_behavior` | Save same drive_type twice | Updates existing, increments version |
| C13 | `test_get_drive_by_type` | Save and retrieve drive | Returns correct drive |
| C14 | `test_save_relationship_upsert` | Save relationship twice | Updates existing by entity_name |
| C15 | `test_get_relationship_by_entity` | Save and retrieve relationship | Returns correct relationship |
| C16 | `test_search_returns_search_results` | Search for known term | Returns list of SearchResult |
| C17 | `test_search_respects_record_types` | Search with record_types filter | Only returns specified types |
| C18 | `test_get_stats_counts_all_types` | Add data and get stats | Counts match for each type |
| C19 | `test_deleted_records_excluded` | Soft-delete record, query | Deleted records not returned |
| C20 | `test_agent_id_isolation` | Save with agent_id A, query with B | Agent B cannot see A's data |

### 1.3 SQLiteStorage Unit Tests

Create `tests/test_storage_sqlite.py` - SQLite-specific tests:

```python
"""SQLite storage unit tests - implementation-specific behavior."""
```

#### Test Cases:

| ID | Test Name | Description |
|----|-----------|-------------|
| S01 | `test_database_created_on_init` | Initialize SQLiteStorage | DB file created at specified path |
| S02 | `test_schema_version_tracked` | Initialize twice | schema_version table populated correctly |
| S03 | `test_sync_queue_populated` | Save record | Entry added to sync_queue |
| S04 | `test_json_array_fields_roundtrip` | Save record with list fields | Lists stored as JSON, retrieved correctly |
| S05 | `test_datetime_serialization` | Save record with datetime | Stored as ISO string, parsed back |
| S06 | `test_concurrent_writes` | Write from multiple threads | No database locked errors |
| S07 | `test_text_search_like_behavior` | Search with special chars (%, _) | SQL injection prevented, search works |
| S08 | `test_empty_database_queries` | Query empty database | Returns empty lists, not errors |
| S09 | `test_custom_db_path` | Specify non-default path | DB created at custom location |
| S10 | `test_in_memory_database` | Use `:memory:` path | Works for testing |

---

## Phase 2: Integration Tests (Write BEFORE Refactoring)

Create `tests/test_integration.py` - end-to-end tests with real storage:

```python
"""Integration tests using real SQLite storage.

These test the full stack: Kernle -> Storage -> SQLite
No mocks except for Supabase (which is external).
"""
```

### 2.1 Full Workflow Tests

| ID | Test Name | Description |
|----|-----------|-------------|
| I01 | `test_full_session_workflow` | Load memory → record episodes → add notes → checkpoint → load again | All data persisted and retrievable |
| I02 | `test_belief_value_goal_lifecycle` | Add → load → verify presence | Full CRUD cycle |
| I03 | `test_drive_satisfaction_cycle` | Create drive → satisfy repeatedly → verify minimum | Drive intensity management |
| I04 | `test_consolidation_flow` | Add episodes → consolidate → verify beliefs created | End-to-end consolidation |
| I05 | `test_search_relevance` | Add known data → search → verify ordering | Search returns relevant results |
| I06 | `test_temporal_query_accuracy` | Add timestamped data → query ranges → verify | Date filtering works |
| I07 | `test_auto_capture_integration` | Feed significant text → verify captured correctly | Detection and storage work |
| I08 | `test_format_memory_with_real_data` | Add all data types → format → verify output | Formatting includes all sections |

### 2.2 Regression Tests from Existing Behavior

Based on `test_core.py` patterns, ensure these specific behaviors are preserved:

| ID | Current Test (test_core.py) | Preserve Behavior |
|----|----------------------------|-------------------|
| R01 | `test_load_lessons` | Extracts from `is_reflected=True` episodes only |
| R02 | `test_load_recent_work_filters_checkpoints` | Excludes "checkpoint" tag |
| R03 | `test_episode_outcome_type_detection` | success/failure/partial mapping |
| R04 | `test_note_decision` | Markdown formatting with reason |
| R05 | `test_checkpoint_history_limit` | Keeps last 10 only |
| R06 | `test_drive_intensity_bounds` | Clamps to [0.0, 1.0] |
| R07 | `test_satisfy_drive_minimum_intensity` | Never below 0.1 |
| R08 | `test_consolidate_avoids_duplicate_beliefs` | No duplicate beliefs created |

---

## Phase 3: Post-Refactoring Tests (Write AFTER Refactoring)

### 3.1 Storage Injection Tests

Create `tests/test_kernle_storage_injection.py`:

```python
"""Test Kernle with injected Storage implementations."""
```

| ID | Test Name | Description |
|----|-----------|-------------|
| P01 | `test_kernle_accepts_sqlite_storage` | Pass SQLiteStorage to Kernle | Works correctly |
| P02 | `test_kernle_accepts_custom_storage` | Pass mock Storage | Uses injected storage |
| P03 | `test_kernle_backward_compatible` | Don't pass storage | Falls back to Supabase (or configured default) |
| P04 | `test_storage_error_propagation` | Storage raises exception | Kernle handles gracefully |

### 3.2 Behavioral Equivalence Tests

Create `tests/test_storage_equivalence.py`:

```python
"""Verify SQLiteStorage and SupabaseStorage produce identical results.

Parameterized tests that run same operations on both backends.
"""
import pytest

@pytest.fixture(params=["sqlite", "supabase"])
def storage(request, tmp_path):
    if request.param == "sqlite":
        return SQLiteStorage("test_agent", tmp_path / "test.db")
    else:
        # Requires Supabase credentials - skip in CI without them
        pytest.importorskip("supabase_available_marker")
        return SupabaseStorage("test_agent", url=..., key=...)
```

| ID | Test Name | Description |
|----|-----------|-------------|
| E01 | `test_episode_roundtrip_equivalence` | Save and load episode | Both return same structure |
| E02 | `test_belief_roundtrip_equivalence` | Save and load belief | Both return same structure |
| E03 | `test_search_results_format_equivalence` | Search same data | Both return same format |
| E04 | `test_stats_format_equivalence` | Get stats | Both return same keys |
| E05 | `test_empty_result_equivalence` | Query with no matches | Both return empty list |

### 3.3 SupabaseStorage Implementation Tests

Create `tests/test_storage_supabase.py` (when SupabaseStorage is implemented):

```python
"""Supabase storage tests - implementation-specific.

Requires SUPABASE_URL and SUPABASE_KEY environment variables.
Uses a test-specific agent_id to avoid polluting production data.
"""
```

| ID | Test Name | Description |
|----|-----------|-------------|
| SB01 | `test_supabase_connection` | Initialize SupabaseStorage | Connects without error |
| SB02 | `test_supabase_table_mapping` | Save to each table | Correct Supabase tables used |
| SB03 | `test_supabase_filter_translation` | Query with filters | Supabase query filters correct |
| SB04 | `test_supabase_error_handling` | Invalid credentials | Clear error message |
| SB05 | `test_supabase_network_timeout` | Network delay | Handles timeout gracefully |

---

## Phase 4: Edge Case Tests

### 4.1 Data Validation Edge Cases

| ID | Test Name | Description | Expected |
|----|-----------|-------------|----------|
| V01 | `test_empty_agent_id_rejected` | Create Kernle with "" agent_id | ValueError raised |
| V02 | `test_agent_id_sanitization` | Agent ID with special chars | Sanitized to alphanumeric |
| V03 | `test_long_content_handling` | Note with 10,000 char content | Either accepted or clear error |
| V04 | `test_unicode_content_preserved` | Save emoji and CJK text | Retrieved unchanged |
| V05 | `test_null_fields_handling` | Save episode with None fields | Stored as NULL, retrieved as None |
| V06 | `test_empty_list_vs_none` | Save with `[]` vs `None` for arrays | Distinct handling preserved |
| V07 | `test_invalid_note_type_rejected` | Note with type="invalid" | ValueError raised |
| V08 | `test_invalid_drive_type_rejected` | Drive with type="invalid" | ValueError raised |
| V09 | `test_confidence_boundary_values` | Belief with confidence=0.0 and 1.0 | Both accepted |
| V10 | `test_negative_priority_handling` | Goal with priority=-1 | Either accepted or clear error |

### 4.2 Concurrency Edge Cases

| ID | Test Name | Description |
|----|-----------|-------------|
| CO01 | `test_concurrent_checkpoint_writes` | Two checkpoints at same time | Both saved, no corruption |
| CO02 | `test_concurrent_read_write` | Read while writing | Read gets consistent data |
| CO03 | `test_rapid_drive_updates` | Many drive updates in quick succession | All applied correctly |

### 4.3 Error Recovery Edge Cases

| ID | Test Name | Description |
|----|-----------|-------------|
| ER01 | `test_corrupted_checkpoint_file` | Manually corrupt JSON | Returns None, doesn't crash |
| ER02 | `test_missing_checkpoint_directory` | Delete directory | Recreated on next save |
| ER03 | `test_database_locked` | Simulate DB lock | Graceful error or retry |
| ER04 | `test_storage_unavailable` | Network/disk failure | Clear error, no partial state |

---

## Implementation Priority

### Must Complete Before Refactoring:
1. **B01-B25**: Behavioral snapshot tests (locks in current behavior)
2. **C01-C20**: Contract tests (defines Storage requirements)
3. **S01-S10**: SQLiteStorage unit tests (verifies implementation)
4. **I01-I08**: Integration tests (validates full stack)
5. **R01-R08**: Regression tests (explicit behavior preservation)

### Complete During Refactoring:
6. **P01-P04**: Storage injection tests
7. **E01-E05**: Equivalence tests

### Complete After SupabaseStorage Implementation:
8. **SB01-SB05**: Supabase-specific tests

### Complete As Enhancement:
9. **V01-V10**: Validation edge cases
10. **CO01-CO03**: Concurrency tests
11. **ER01-ER04**: Error recovery tests

---

## Test Infrastructure Requirements

### 1. Fixtures Needed

```python
# conftest.py additions

@pytest.fixture
def sqlite_storage(tmp_path):
    """Fresh SQLite storage for each test."""
    db_path = tmp_path / "test.db"
    return SQLiteStorage("test_agent", db_path)

@pytest.fixture
def populated_sqlite_storage(sqlite_storage):
    """SQLite storage with sample data."""
    # Add standard test data
    sqlite_storage.save_episode(Episode(...))
    sqlite_storage.save_belief(Belief(...))
    # etc.
    return sqlite_storage

@pytest.fixture
def kernle_with_sqlite(sqlite_storage):
    """Kernle instance using SQLite storage."""
    return Kernle(agent_id="test_agent", storage=sqlite_storage)

@pytest.fixture
def supabase_storage():
    """SupabaseStorage for integration tests (requires credentials)."""
    url = os.environ.get("TEST_SUPABASE_URL")
    key = os.environ.get("TEST_SUPABASE_KEY")
    if not url or not key:
        pytest.skip("Supabase credentials not configured")
    return SupabaseStorage("test_agent_" + uuid.uuid4().hex[:8], url, key)
```

### 2. Test Markers

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "behavioral: behavioral snapshot tests",
    "contract: storage contract tests",
    "integration: full integration tests",
    "supabase: requires Supabase connection",
    "slow: tests that take > 1 second",
]
```

### 3. CI Configuration

```yaml
# .github/workflows/test.yml (excerpt)
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run unit tests
        run: pytest -m "not integration and not supabase"
      
      - name: Run integration tests
        run: pytest -m "integration and not supabase"
      
      - name: Run Supabase tests
        if: ${{ secrets.TEST_SUPABASE_URL }}
        env:
          TEST_SUPABASE_URL: ${{ secrets.TEST_SUPABASE_URL }}
          TEST_SUPABASE_KEY: ${{ secrets.TEST_SUPABASE_KEY }}
        run: pytest -m "supabase"
```

---

## Success Criteria

The refactoring is complete when:

1. ✅ All behavioral snapshot tests (B01-B25) pass
2. ✅ All contract tests (C01-C20) pass for SQLiteStorage
3. ✅ All integration tests (I01-I08) pass
4. ✅ All regression tests (R01-R08) pass
5. ✅ Storage injection tests (P01-P04) pass
6. ✅ No changes to `tests/test_mcp.py` required (MCP layer unchanged)
7. ✅ Existing `tests/test_core.py` continues to pass
8. ✅ Code coverage does not decrease

---

## Known Issues to Address

From `TEST_AUDIT.md`, these issues in existing tests should be fixed alongside the refactoring:

1. **`test_memory_drive` tests** assert bug behavior - rewrite to test correct behavior
2. **No-op test** `test_memory_drive_validation_bug_documentation` - delete
3. **Mock-only integration tests** - replace with real integration tests
4. **Missing assertions** in edge case tests - strengthen assertions

---

## Appendix: Data Structure Mappings

### Current core.py → Storage Protocol

| core.py Method | Storage Method | Data Structure |
|---------------|----------------|----------------|
| `load_values()` | `get_values()` | `Value` |
| `load_beliefs()` | `get_beliefs()` | `Belief` |
| `load_goals()` | `get_goals()` | `Goal` |
| `load_lessons()` | `get_episodes()` + extract | `Episode.lessons` |
| `load_recent_work()` | `get_episodes()` | `Episode` |
| `load_recent_notes()` | `get_notes()` | `Note` |
| `load_drives()` | `get_drives()` | `Drive` |
| `load_relationships()` | `get_relationships()` | `Relationship` |
| `episode()` | `save_episode()` | `Episode` |
| `note()` | `save_note()` | `Note` |
| `belief()` | `save_belief()` | `Belief` |
| `value()` | `save_value()` | `Value` |
| `goal()` | `save_goal()` | `Goal` |
| `drive()` | `save_drive()` | `Drive` |
| `relationship()` | `save_relationship()` | `Relationship` |
| `search()` | `search()` | `SearchResult` |
| `status()` | `get_stats()` | `Dict[str, int]` |

### Field Mapping Notes

Some fields have different names between `core.py` and `storage/base.py`:

| core.py | Storage Protocol | Notes |
|---------|-----------------|-------|
| `lessons_learned` | `lessons` | Episode lessons field |
| `outcome_description` | `outcome` | Episode outcome text |
| `patterns_to_repeat` | (not in base) | Currently Episode-only |
| `patterns_to_avoid` | (not in base) | Currently Episode-only |
| `is_reflected` | (not in base) | Episode consolidation flag |
| `owner_id` | `agent_id` | Note owner field |

These mappings need to be handled in the adapter layer or by extending the `Episode` dataclass.
