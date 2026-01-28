# Adversarial Test Audit Report: Kernle Test Suite

**Auditor:** Hostile Test Auditor (Subagent)  
**Date:** 2025-01-28  
**Scope:** Full test suite (`~/kernle/tests/`)

---

## Executive Summary

The test suite has **serious coverage gaps** and **quality issues** that undermine confidence in the codebase. While 572 tests pass with 57% overall coverage, critical paths remain untested, and many tests verify mock behavior rather than production code.

**Key Findings:**
- **6 CLI command modules are nearly untested** (2-4% coverage)
- **Integration tests don't test actual CLI integration**
- **Resource leaks detected** (unclosed database connections)
- **Time-dependent tests** without deterministic control
- **No concurrency tests** despite shared database state
- **Security-sensitive code paths untested**

**Total Issues Found:** 32  
**Critical:** 7 | **High:** 11 | **Medium:** 9 | **Low:** 5

---

## Critical Issues

### 1. CLI Command Modules Have Near-Zero Coverage

**Severity:** CRITICAL  
**Files Affected:**
- `kernle/cli/commands/belief.py`: **4% coverage**
- `kernle/cli/commands/emotion.py`: **4% coverage**
- `kernle/cli/commands/forget.py`: **2% coverage**
- `kernle/cli/commands/meta.py`: **2% coverage**
- `kernle/cli/commands/raw.py`: **25% coverage**

**Problem:** These modules contain 13 command functions and hundreds of lines of user-facing code. The CLI is the **primary interface** for Kernle, yet these commands are essentially untested.

**What's missing:**
```python
# kernle/cli/commands/belief.py - Lines 14-173 UNTESTED
def cmd_belief(args, k: "Kernle"):
    # "revise", "contradictions", "history", "reinforce", "supersede", "list"
    # ALL BRANCHES UNTESTED
```

**Example test to add:**
```python
class TestBeliefCLICommands:
    def test_belief_revise_with_valid_episode(self, temp_kernle):
        """Test belief revise actually calls revise_beliefs_from_episode."""
        k = temp_kernle
        ep_id = k.episode("Learn something", "Learned it")
        
        # Actually invoke CLI command handler
        args = argparse.Namespace(
            belief_action="revise",
            episode_id=ep_id,
            json=False
        )
        
        # Should not raise, should produce output
        with patch('sys.stdout', new=StringIO()) as out:
            cmd_belief(args, k)
        
        assert "Belief Revision" in out.getvalue()
    
    def test_belief_supersede_with_nonexistent_old_id(self, temp_kernle):
        """Superseding nonexistent belief should fail gracefully."""
        args = argparse.Namespace(
            belief_action="supersede",
            old_id="nonexistent-id",
            new_statement="New belief",
            confidence=0.8,
            reason=None
        )
        
        with patch('sys.stdout', new=StringIO()) as out:
            cmd_belief(args, temp_kernle)
        
        assert "✗" in out.getvalue()  # Should show error
```

---

### 2. test_cli.py Tests Mock Behavior, Not Production Code

**Severity:** CRITICAL  
**File:** `tests/test_cli.py`

**Problem:** The entire CLI test file uses a `mock_kernle` fixture that returns hardcoded values. Tests verify that mocked methods are called, but **never verify the actual CLI parsing, validation, or output formatting** works correctly.

**Evidence:**
```python
# Line 15-44: Mock returns hardcoded values
@pytest.fixture
def mock_kernle():
    kernle = Mock(spec=Kernle)
    kernle.load.return_value = {...}  # Hardcoded
    kernle.checkpoint.return_value = {...}  # Hardcoded
```

```python
# Line 194-198: Test asserts mock return value
def test_cmd_checkpoint_save(self, mock_kernle):
    cmd_checkpoint(args, mock_kernle)
    mock_kernle.checkpoint.assert_called_once_with(...)  # Just checks mock was called
    assert "✓ Checkpoint saved: Test task" in fake_out.getvalue()  # "Test task" comes from mock!
```

**What SHOULD be tested but isn't:**
1. Argument parsing with edge cases (`--pending ""`, unicode args)
2. Validation errors actually raise/print correctly
3. Real Kernle integration (not mocked)
4. Output formatting with various data shapes

**Example test to add:**
```python
class TestCLIRealIntegration:
    """Tests using real Kernle, not mocks."""
    
    def test_checkpoint_save_and_load_cycle(self, temp_kernle_with_db):
        """End-to-end: CLI save → CLI load should return same data."""
        k = temp_kernle_with_db
        
        # Save via CLI handler
        save_args = argparse.Namespace(
            checkpoint_action="save",
            task="Real integration task",
            pending=["item1", "item2"],
            context="Testing real flow",
            sync=False,
            no_sync=False,
        )
        cmd_checkpoint(save_args, k)
        
        # Load via CLI handler
        load_args = argparse.Namespace(checkpoint_action="load", json=True)
        with patch('sys.stdout', new=StringIO()) as out:
            cmd_checkpoint(load_args, k)
        
        loaded = json.loads(out.getvalue())
        assert loaded["current_task"] == "Real integration task"
        assert "item1" in loaded["pending"]
```

---

### 3. Resource Leaks: Unclosed Database Connections

**Severity:** CRITICAL  
**Evidence:** Test output shows ResourceWarnings:

```
ResourceWarning: unclosed database in <sqlite3.Connection object at 0x108bbe7a0>
ResourceWarning: unclosed database in <sqlite3.Connection object at 0x108bbea70>
ResourceWarning: unclosed database in <sqlite3.Connection object at 0x108bbe2f0>
ResourceWarning: unclosed database in <sqlite3.Connection object at 0x108bbe6b0>
```

**Problem:** Tests are creating database connections that aren't being closed. This can:
1. Cause flaky tests under load
2. Leak file descriptors
3. Leave locked database files

**Root cause analysis needed:** Check `test_utils.py` (lines 552-572 emit these warnings).

**Fix:**
```python
@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    storage = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
    yield storage
    storage.close()  # MUST close explicitly

# Or use context manager pattern:
@pytest.fixture
def storage(temp_db):
    with SQLiteStorage(agent_id="test-agent", db_path=temp_db) as storage:
        yield storage
```

---

### 4. Integration Tests Don't Test CLI → Core → Storage Flow

**Severity:** CRITICAL  
**File:** `tests/test_integration.py`

**Problem:** Despite being called "integration tests," they test `Kernle` methods directly, not CLI commands. The actual CLI-to-storage flow is untested.

**Evidence:**
```python
# Line 35-50: Tests Kernle.episode(), not CLI command
def test_episode_command_persists(self, temp_kernle):
    k = temp_kernle
    episode_id = k.episode(...)  # Direct API call, not CLI
```

**What's actually an integration test:**
```python
def test_cli_episode_to_storage_flow(self, temp_db_path, temp_checkpoint_dir):
    """Test full CLI command → Core → Storage flow."""
    import subprocess
    
    # Invoke via subprocess like a real user
    result = subprocess.run([
        "python", "-m", "kernle", 
        "-a", "test_agent",
        "episode", "Test objective", "Test outcome",
        "--lesson", "Learned something",
    ], capture_output=True, env={
        "KERNLE_DB_PATH": str(temp_db_path),
    })
    
    assert result.returncode == 0
    assert "✓ Episode saved:" in result.stdout.decode()
    
    # Verify in storage
    storage = SQLiteStorage(agent_id="test_agent", db_path=temp_db_path)
    episodes = storage.get_episodes()
    assert len(episodes) == 1
    assert episodes[0].objective == "Test objective"
```

---

### 5. No Tests for SQL Injection Prevention

**Severity:** CRITICAL  
**File:** `kernle/storage/sqlite.py`

**Problem:** While `validate_table_name()` exists (line 46-58), there are **no tests verifying it works**. A malicious agent_id or query could potentially inject SQL.

**Code at risk:**
```python
# sqlite.py line 46-58
def validate_table_name(table: str) -> str:
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    return table
```

**Missing tests:**
```python
class TestSQLInjectionPrevention:
    def test_validate_table_name_rejects_injection(self):
        """Validate_table_name rejects SQL injection attempts."""
        malicious_inputs = [
            "episodes; DROP TABLE episodes;--",
            "episodes' OR '1'='1",
            "../../../etc/passwd",
            "episodes\x00notes",
        ]
        
        for malicious in malicious_inputs:
            with pytest.raises(ValueError):
                validate_table_name(malicious)
    
    def test_agent_id_sanitized_in_queries(self, temp_db):
        """Agent IDs with special chars should be sanitized."""
        # This should not raise or corrupt the database
        storage = SQLiteStorage(
            agent_id="test'; DROP TABLE episodes;--",
            db_path=temp_db
        )
        
        # Should have sanitized the agent_id
        assert "DROP" not in storage.agent_id
        
        # Should still function
        storage.save_note(Note(
            id="n1",
            agent_id=storage.agent_id,
            content="Safe content"
        ))
        
        notes = storage.get_notes()
        assert len(notes) == 1
```

---

### 6. Postgres Storage Has 48% UNCOVERED Code

**Severity:** CRITICAL  
**File:** `kernle/storage/postgres.py` - **52% coverage (48% uncovered)**

**Uncovered lines include:**
- Connection handling (lines 103-107)
- Error recovery (lines 318-342, 350-359)
- Batch operations (lines 734-779, 783-804)
- Sync operations (lines 976-1029)
- Most of the actual database operations

**Problem:** Postgres is likely the production storage backend, yet nearly half of it is untested.

**Missing tests:**
```python
class TestPostgresStorage:
    @pytest.fixture
    def pg_storage(self):
        """Real postgres connection (or mock for CI)."""
        # Need actual postgres tests, not just mocked
        
    def test_connection_retry_on_failure(self, pg_storage):
        """Should retry connection on transient failures."""
        
    def test_batch_upsert_handles_conflicts(self, pg_storage):
        """Batch upsert should handle duplicate key conflicts."""
        
    def test_sync_pull_handles_network_failure(self, pg_storage):
        """Sync should handle network failures gracefully."""
```

---

### 7. Main CLI Entry Point 66% Uncovered

**Severity:** CRITICAL  
**File:** `kernle/cli/__main__.py` - **34% coverage**

**Uncovered lines:** 324-540, 553-644, 649-718, 779-782, 790-1397, 1402-1735...

This is the **main entry point**. Missing coverage includes:
- Most command implementations
- Argument parsing edge cases
- Error handling paths
- Help text generation

---

## High Severity Issues

### 8. Time-Dependent Tests Without Deterministic Control

**Severity:** HIGH  
**Files:** `test_anxiety.py`, `test_forgetting.py`, `test_sync_engine.py`

**Problem:** Tests use `datetime.now()` directly, making them potentially flaky:

```python
# test_forgetting.py line 66-81
def test_salience_old_memory_decays(self, kernle_instance):
    now = datetime.now(timezone.utc)
    old_date = now - timedelta(days=90)  # Depends on current time
    
    episode = Episode(
        ...
        created_at=old_date,
    )
```

**Risk:** Tests may pass/fail depending on time of day, timezone, or daylight saving transitions.

**Fix:**
```python
from freezegun import freeze_time

class TestSalienceCalculation:
    @freeze_time("2024-06-15 12:00:00")
    def test_salience_old_memory_decays(self, kernle_instance):
        """Old memories decay - with frozen time."""
        # Now deterministic
```

---

### 9. No Concurrency Tests for Database Access

**Severity:** HIGH  
**Affected:** All storage tests

**Problem:** SQLite has locking behavior that can cause issues with concurrent access. No tests verify thread safety or locking behavior.

**Missing test:**
```python
import threading
import concurrent.futures

class TestConcurrentAccess:
    def test_concurrent_writes_dont_corrupt(self, temp_db):
        """Multiple threads writing shouldn't corrupt database."""
        storage = SQLiteStorage(agent_id="test", db_path=temp_db)
        errors = []
        
        def write_episode(n):
            try:
                storage.save_episode(Episode(
                    id=f"ep-{n}",
                    agent_id="test",
                    objective=f"Concurrent task {n}",
                    outcome="done"
                ))
            except Exception as e:
                errors.append(e)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(write_episode, range(100))
        
        assert len(errors) == 0
        episodes = storage.get_episodes(limit=200)
        assert len(episodes) == 100
```

---

### 10. No Tests for Empty/None Input Handling

**Severity:** HIGH  
**Affected:** `test_core.py`, `test_cli.py`, `test_mcp.py`

**Problem:** Minimal testing of edge case inputs:

**Missing tests:**
```python
class TestEdgeCaseInputs:
    def test_episode_with_none_lessons(self, kernle_instance):
        """Episode should handle None lessons gracefully."""
        k, _ = kernle_instance
        # Should not raise
        ep_id = k.episode("Task", "Done", lessons=None, tags=None)
        assert ep_id is not None
    
    def test_episode_with_empty_string_objective(self, kernle_instance):
        """Empty objective should be rejected."""
        k, _ = kernle_instance
        with pytest.raises(ValueError):
            k.episode("", "outcome")
    
    def test_search_with_empty_query(self, kernle_instance):
        """Empty search query behavior should be defined."""
        k, _ = kernle_instance
        # Should either raise or return empty, not crash
        results = k.search("")
        assert isinstance(results, list)
    
    def test_note_with_only_whitespace(self, kernle_instance):
        """Whitespace-only content should be rejected."""
        k, _ = kernle_instance
        with pytest.raises(ValueError):
            k.note("   \n\t  ")
```

---

### 11. Unicode Boundary Tests Missing

**Severity:** HIGH  
**Files:** All tests

**Problem:** While `test_integration.py` has one unicode test, there are no tests for:
- Unicode normalization forms (NFC vs NFD)
- Zero-width characters
- Right-to-left text
- Emoji in identifiers
- Very long unicode sequences

**Missing tests:**
```python
class TestUnicodeBoundaries:
    @pytest.mark.parametrize("content", [
        "Normal text",
        "日本語テスト",
        "🔥💯🎉",  # Emoji
        "مرحبا بالعالم",  # RTL
        "Ñoño café",  # Accented
        "a\u0301",  # Combining characters (á as a + combining acute)
        "\u200b\u200b\u200b",  # Zero-width spaces
        "a" * 10000,  # Very long
        "\x00hidden\x00",  # Null bytes (should be sanitized)
    ])
    def test_note_content_unicode_handling(self, kernle_instance, content):
        """Notes should handle various unicode correctly."""
        k, storage = kernle_instance
        
        if "\x00" in content:
            # Should sanitize null bytes
            note_id = k.note(content)
            note = storage.get_note(note_id)
            assert "\x00" not in note.content
        else:
            note_id = k.note(content)
            notes = storage.get_notes()
            assert any(content in n.content for n in notes)
```

---

### 12. Missing Error Path Tests in Core

**Severity:** HIGH  
**File:** `test_core.py`

**Problem:** Tests focus on happy paths. Error handling is poorly tested.

**Uncovered error paths:**
```python
# kernle/core.py lines that need tests:
# Line 190-198: Supabase storage initialization failures
# Line 500-502: Storage operation errors
# Line 668-699: Batch loading failures
# Line 961-964: Sync errors
```

**Missing tests:**
```python
class TestErrorHandling:
    def test_load_handles_storage_error(self, kernle_instance):
        """Load should handle storage errors gracefully."""
        k, storage = kernle_instance
        
        # Simulate storage failure
        with patch.object(storage, 'get_values', side_effect=Exception("DB error")):
            # Should not crash, should return partial or empty
            with pytest.raises(Exception):  # Or handle gracefully
                k.load()
    
    def test_episode_with_storage_write_failure(self, kernle_instance):
        """Episode should report storage write failures."""
        k, storage = kernle_instance
        
        with patch.object(storage, 'save_episode', side_effect=Exception("Write failed")):
            with pytest.raises(Exception):
                k.episode("Test", "Test")
```

---

### 13. test_mcp.py Issues Persist from Previous Audit

**Severity:** HIGH  
**File:** `tests/test_mcp.py`

**Problem:** The previous audit (`TEST_AUDIT.md`) identified 18 issues. While some may have been fixed, the fundamental structure remains:
- Tests mock get_kernle(), so production Kernle code isn't tested
- Tests assert mock return values
- "Integration" scenarios are still mocked

**Example still problematic:**
```python
# Line 186-198: Still testing mock output
async def test_memory_load_text_format(self, patched_get_kernle):
    result = await call_tool("memory_load", {"format": "text"})
    # This just returns what the mock returns
    patched_get_kernle.load.assert_called_once()  # Mock verification
```

---

### 14. Fixture Creates Deprecated Mock Patterns

**Severity:** HIGH  
**File:** `tests/conftest.py`

**Problem:** `mock_supabase_client` fixture (lines 187-300) is marked DEPRECATED but still exists and may be used by some tests. It simulates behavior that may not match real Supabase.

```python
# conftest.py line 187
@pytest.fixture
def mock_supabase_client():
    """...
    DEPRECATED: Use sqlite_storage fixture instead for new tests.
    """
```

**Action:** Either remove deprecated fixtures or ensure they're not used.

---

### 15. Checkpoint Directory Traversal Tests Missing

**Severity:** HIGH  
**File:** `kernle/core.py` (lines 165-186)

**Problem:** `_validate_checkpoint_dir` prevents directory traversal, but there are no explicit tests for this security-critical code.

```python
# core.py - Security-sensitive, untested:
def _validate_checkpoint_dir(self, checkpoint_dir: Path) -> Path:
    # ... prevents directory traversal ...
```

**Missing test:**
```python
def test_checkpoint_dir_traversal_prevention(self, temp_db_path):
    """Should reject checkpoint dirs outside safe locations."""
    with pytest.raises(ValueError):
        Kernle(
            agent_id="test",
            checkpoint_dir=Path("/etc/passwd"),
        )
    
    with pytest.raises(ValueError):
        Kernle(
            agent_id="test",
            checkpoint_dir=Path("../../../etc"),
        )
```

---

### 16. Missing Sync Engine Failure Mode Tests

**Severity:** HIGH  
**File:** `test_sync_engine.py`

**Problem:** Tests cover happy paths but not failure modes:
- Network timeout during sync
- Partial sync failure
- Corrupted sync queue
- Cloud storage returns invalid data

**Missing tests:**
```python
class TestSyncFailureModes:
    def test_sync_handles_timeout(self, storage_with_cloud, mock_cloud_storage):
        """Sync should handle timeouts gracefully."""
        import socket
        mock_cloud_storage.save_episode.side_effect = socket.timeout("Connection timed out")
        
        # Should not crash, should mark as failed
        result = storage_with_cloud.sync()
        assert result.pushed == 0
        assert len(result.errors) > 0
    
    def test_sync_handles_corrupt_queue(self, storage):
        """Should handle corrupted sync queue entries."""
        # Manually corrupt sync queue
        storage._conn.execute(
            "INSERT INTO sync_queue (operation, table_name, record_id, data) VALUES (?, ?, ?, ?)",
            ("upsert", "episodes", "bad-id", "not valid json")
        )
        
        # Should not crash
        changes = storage.get_queued_changes()
        # Should skip or handle corrupted entry
```

---

### 17. No Tests for Large Data Sets

**Severity:** HIGH  
**Affected:** All storage tests

**Problem:** Tests use small data sets (1-10 records). No tests verify behavior with:
- 10,000+ episodes
- Very large individual records
- Memory pressure scenarios

**Missing test:**
```python
class TestLargeDataSets:
    def test_search_performance_with_many_records(self, storage):
        """Search should complete in reasonable time with many records."""
        import time
        
        # Create many records
        for i in range(1000):
            storage.save_episode(Episode(
                id=f"ep-{i}",
                agent_id="test",
                objective=f"Task {i} about machine learning",
                outcome="completed"
            ))
        
        start = time.time()
        results = storage.search("machine learning", limit=10)
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should complete in under 1 second
        assert len(results) == 10
```

---

### 18. Embedding Provider Tests Use Only HashEmbedder

**Severity:** HIGH  
**File:** `test_sqlite_storage.py`

**Problem:** Only `HashEmbedder` is tested. If real embedding providers (OpenAI, etc.) are used in production, they're untested.

**Missing test:**
```python
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No API key")
class TestRealEmbeddingProvider:
    def test_openai_embedder_produces_valid_embeddings(self):
        """Real OpenAI embedder should produce valid embeddings."""
        from kernle.storage.embeddings import OpenAIEmbedder
        
        embedder = OpenAIEmbedder()
        embedding = embedder.embed("test text")
        
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
```

---

## Medium Severity Issues

### 19. Test Names Don't Follow Conventions Consistently

**Severity:** MEDIUM  
**Affected:** Multiple test files

**Problem:** Inconsistent naming makes tests harder to understand:
```python
# Good: Describes what and expected outcome
def test_episode_with_none_lessons_succeeds
def test_search_empty_query_returns_empty_list

# Bad: Vague, doesn't describe expectation  
def test_get_candidates_empty  # Empty what? Returns empty? Input empty?
def test_salience_new_memory  # What about it?
```

---

### 20. No Parametrized Tests for Validators

**Severity:** MEDIUM  
**Files:** `test_core.py`, `test_cli.py`

**Problem:** Input validation is tested with single examples. Parametrized tests would cover more cases:

```python
# Current: Single test
def test_validate_agent_id_special_chars(self):
    kernle = Kernle(agent_id="test@agent!#$%")
    assert kernle.agent_id == "testagent"

# Better: Parametrized
@pytest.mark.parametrize("input_id,expected", [
    ("test@agent", "testagent"),
    ("test!#$%", "test"),
    ("test_agent", "test_agent"),
    ("test-agent", "test-agent"),
    ("test.agent", "test.agent"),
    ("123", "123"),
    ("😀agent", "agent"),
])
def test_validate_agent_id_sanitization(self, input_id, expected):
    kernle = Kernle(agent_id=input_id)
    assert kernle.agent_id == expected
```

---

### 21. Flaky Test Potential: Mock Call Counts

**Severity:** MEDIUM  
**File:** `test_sync_engine.py`

**Problem:** Tests rely on exact mock call counts which can be brittle:

```python
# Line 177: Connectivity cache test
assert mock_cloud_storage.get_stats.call_count == 1  # Brittle
```

If implementation changes to make 2 calls for safety, test breaks unnecessarily.

---

### 22. Missing Boundary Value Tests

**Severity:** MEDIUM  
**Affected:** Multiple tests

**Missing tests for:**
- `confidence=0.0` and `confidence=1.0`
- `priority=0` and `priority=100`
- `intensity=0.0` and `intensity=1.0`
- `limit=0` behavior

```python
class TestBoundaryValues:
    @pytest.mark.parametrize("confidence", [0.0, 0.001, 0.5, 0.999, 1.0])
    def test_belief_confidence_boundaries(self, kernle_instance, confidence):
        k, _ = kernle_instance
        belief_id = k.belief("Test belief", confidence=confidence)
        assert belief_id is not None
    
    def test_search_with_limit_zero(self, kernle_instance):
        k, _ = kernle_instance
        k.episode("Test", "Test")
        results = k.search("Test", limit=0)
        assert results == []  # Or raises? Document behavior!
```

---

### 23. No Tests for Memory Format Versioning

**Severity:** MEDIUM  
**File:** `kernle/storage/sqlite.py`

**Problem:** `SCHEMA_VERSION = 10` exists but no tests verify migration between versions.

```python
class TestSchemaMigrations:
    def test_migration_from_v9_to_v10(self, tmp_path):
        """Should migrate v9 schema to v10."""
        # Create a v9 database
        # Open with v10 storage
        # Verify migration happened correctly
```

---

### 24. Assertion Messages Missing

**Severity:** MEDIUM  
**Affected:** Most tests

**Problem:** Assertions lack messages, making failures hard to diagnose:

```python
# Bad: No context on failure
assert len(results) == 1

# Good: Clear failure message
assert len(results) == 1, f"Expected 1 result, got {len(results)}: {results}"
```

---

### 25. No Tests for Logging Output

**Severity:** MEDIUM  
**Affected:** All modules using logging

**Problem:** Modules use `logger.debug()`, `logger.error()`, etc., but no tests verify correct log output.

```python
def test_storage_logs_on_error(self, storage, caplog):
    """Storage errors should be logged."""
    import logging
    
    with caplog.at_level(logging.ERROR):
        # Trigger an error condition
        storage.get_memory("invalid_type", "id")
    
    assert "error" in caplog.text.lower()
```

---

### 26. Playbook Tests Don't Test Execution

**Severity:** MEDIUM  
**File:** `tests/test_playbooks.py`

**Problem:** Tests verify playbook CRUD but not execution against real scenarios.

---

### 27. No Tests for Context Manager Protocol

**Severity:** MEDIUM  
**File:** `kernle/storage/sqlite.py`

**Problem:** If `SQLiteStorage` supports `__enter__`/`__exit__`, it should be tested.

```python
def test_storage_context_manager(self, temp_db):
    """Storage should work as context manager."""
    with SQLiteStorage(agent_id="test", db_path=temp_db) as storage:
        storage.save_note(Note(id="n1", agent_id="test", content="test"))
    
    # Connection should be closed after with block
    # Verify by trying to use storage (should fail or reconnect)
```

---

## Low Severity Issues

### 28. Hardcoded Magic Numbers

**Severity:** LOW  
**File:** `test_mcp.py`

```python
# Line 36
assert len(tools) == 23  # Hardcoded, breaks when tools added
```

Better: `assert len(tools) == len(EXPECTED_TOOLS)`

---

### 29. Test Data Patterns Could Use Factories

**Severity:** LOW  
**File:** `tests/conftest.py`

**Problem:** `sample_episode`, `sample_note`, etc. fixtures duplicate code. Consider using a factory pattern like `factory_boy`.

---

### 30. No Property-Based Tests

**Severity:** LOW  
**Affected:** All tests

**Problem:** No property-based testing (hypothesis) for discovering edge cases automatically.

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_note_accepts_arbitrary_valid_content(self, content):
    """Any non-empty string under max length should be accepted."""
    # Hypothesis will generate many test cases
    k.note(content)
```

---

### 31. Test File Organization Could Be Improved

**Severity:** LOW  
**Affected:** `tests/`

**Observation:** Tests are organized by module, but some tests span concerns. Consider:
- `tests/unit/` - Unit tests with mocks
- `tests/integration/` - Integration tests with real components
- `tests/e2e/` - End-to-end tests

---

### 32. Some Tests Have Comments Instead of Assertions

**Severity:** LOW  
**File:** `test_mcp.py`

```python
# Line 467
# Should handle Unicode properly   <-- This is not a test!
```

Comments don't verify behavior. Replace with actual assertions.

---

## Coverage Summary

| Module | Coverage | Grade |
|--------|----------|-------|
| `kernle/__init__.py` | 100% | ✅ |
| `kernle/core.py` | 73% | ⚠️ |
| `kernle/cli/__main__.py` | 34% | ❌ |
| `kernle/cli/commands/belief.py` | 4% | ❌ |
| `kernle/cli/commands/emotion.py` | 4% | ❌ |
| `kernle/cli/commands/forget.py` | 2% | ❌ |
| `kernle/cli/commands/meta.py` | 2% | ❌ |
| `kernle/cli/commands/raw.py` | 25% | ❌ |
| `kernle/storage/sqlite.py` | 71% | ⚠️ |
| `kernle/storage/postgres.py` | 52% | ❌ |
| `kernle/features/metamemory.py` | 55% | ❌ |
| `kernle/mcp/server.py` | 92% | ✅ |
| **TOTAL** | **57%** | **❌** |

---

## Recommendations

### Immediate Actions (Critical)
1. **Add CLI command tests** for belief, emotion, forget, meta, raw modules
2. **Create true integration tests** using subprocess or test client
3. **Fix resource leaks** by ensuring all storage connections are closed
4. **Add SQL injection tests** for all user inputs that reach SQL
5. **Add Postgres storage tests** (or mark as needing production testing)

### Short-term (High Priority)
6. **Replace mock-heavy CLI tests** with real Kernle integration
7. **Add time-freezing** to time-dependent tests
8. **Add concurrency tests** for database access
9. **Add error path tests** for all try/except blocks in production code
10. **Add boundary value tests** for all numeric parameters

### Medium-term
11. Adopt property-based testing with Hypothesis
12. Reorganize test directory structure
13. Add performance benchmarks as tests
14. Document expected behavior for edge cases

---

## Conclusion

The test suite provides a **false sense of security**. While 572 tests pass, critical paths remain untested, and many tests verify mock behavior rather than production code. The CLI—the primary user interface—has near-zero test coverage for most commands.

**Recommended target:** 80% coverage with focus on error paths, edge cases, and true integration tests.

---

*Report generated by adversarial test auditor. All findings are actionable and include example fixes.*
