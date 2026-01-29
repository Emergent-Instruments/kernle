# Kernle Storage Refactor - Architectural Review

**Date:** 2025-01-27  
**Reviewer:** Senior Python Architect (subagent)  
**Files Reviewed:** `kernle/core.py`, `kernle/storage/base.py`, `kernle/storage/sqlite.py`

## Executive Summary

The proposed refactoring plan is **sound in principle** but has several gaps that will cause issues during implementation. The main concerns are:

1. **Field naming mismatches** between the Storage dataclasses and Supabase schema
2. **Missing Storage protocol methods** for operations currently in core.py
3. **The `consolidate()` method** correctly identified as needing special handling
4. **Hybrid storage patterns** (checkpoint, notes) that don't fit the clean interface

This document identifies all issues and proposes solutions.

---

## 1. Field Naming Mismatches 🔴 Critical

The dataclasses in `base.py` don't match the Supabase schema in `core.py`:

### Episode
| Dataclass (`base.py`) | Supabase (`core.py`) | Notes |
|----------------------|---------------------|-------|
| `lessons` | `lessons_learned` | **Mismatch** |
| `outcome` | `outcome_description` | **Mismatch** |
| *(missing)* | `is_reflected` | **Missing field** |
| *(missing)* | `confidence` | **Missing field** |
| *(missing)* | `patterns_to_repeat` | **Missing field** |
| *(missing)* | `patterns_to_avoid` | **Missing field** |

### Relationship
| Dataclass (`base.py`) | Supabase (`core.py`) | Notes |
|----------------------|---------------------|-------|
| `entity_name` | `other_agent_id` | **Different concept** |
| `entity_type` | *(missing)* | **Not in Supabase** |
| `relationship_type` | *(missing)* | **Not in Supabase** |
| `sentiment` | *(missing)* | **Not in Supabase** |
| *(missing)* | `trust_level` | **Missing field** |

### Note
| Dataclass (`base.py`) | Supabase (`core.py`) | Notes |
|----------------------|---------------------|-------|
| `agent_id` | `owner_id` | **Different field name** |
| `note_type` | `metadata.note_type` | In metadata JSON |
| *(missing)* | `source` | Always "curated" |
| *(missing)* | `visibility` | Always "private" |
| *(missing)* | `is_curated` | Always True |
| *(missing)* | `is_protected` | **Missing field** |

### Value
| Dataclass (`base.py`) | Supabase (`core.py`) | Notes |
|----------------------|---------------------|-------|
| *(missing)* | `value_type` | **Missing field** |
| *(missing)* | `is_active` | **Missing field** |
| *(missing)* | `is_foundational` | **Missing field** |

### Belief
| Dataclass (`base.py`) | Supabase (`core.py`) | Notes |
|----------------------|---------------------|-------|
| *(missing)* | `is_active` | **Missing field** |
| *(missing)* | `is_foundational` | **Missing field** |

### Drive
| Dataclass (`base.py`) | Supabase (`core.py`) | Notes |
|----------------------|---------------------|-------|
| *(missing)* | `last_satisfied_at` | **Missing field** |
| *(missing)* | `satisfaction_decay_hours` | **Missing field** |

**Recommendation:** Update `base.py` dataclasses to include all fields from the Supabase schema. Use `Optional` for fields that may not be present in all backends.

---

## 2. Missing Storage Protocol Methods 🔴 Critical

The Storage protocol is missing methods that `core.py` currently uses:

### Episode Operations
```python
# Missing in Storage protocol:
def get_unreflected_episodes(self) -> List[Episode]: ...
def mark_episode_reflected(self, episode_id: str) -> bool: ...
def get_lessons(self, limit: int = 20) -> List[str]: ...  # Extracts from episodes
```

### Belief Operations
```python
# Missing:
def find_belief_by_prefix(self, prefix: str) -> Optional[Belief]: ...  # For consolidate() dedup
```

### Drive Operations
```python
# Missing:
def satisfy_drive(self, drive_type: str, amount: float) -> bool: ...
```

### Temporal Queries
```python
# Missing entirely:
def get_episodes_in_range(self, start: datetime, end: datetime, limit: int) -> List[Episode]: ...
def get_notes_in_range(self, start: datetime, end: datetime, limit: int) -> List[Note]: ...
```

### Count Queries (for `status()`)
```python
# get_stats() exists but doesn't match status() needs exactly
```

**Recommendation:** Add these methods to the Storage protocol, or design them as compositions of existing methods in Kernle.

---

## 3. Methods Requiring Special Handling 🟡 Important

### `consolidate()` - AI Logic Mixed with Storage
This method:
1. Gets unreflected episodes from storage
2. Extracts lessons and counts patterns (Python logic)
3. Creates new beliefs based on patterns
4. Marks episodes as reflected

**Recommendation:** Keep `consolidate()` in Kernle. It should call:
- `storage.get_episodes(reflected=False)`
- `storage.find_belief(statement)` 
- `storage.save_belief(belief)`
- New method: `storage.mark_episode_reflected(id)` or use `save_episode()` with updated flag

### `checkpoint()` - Hybrid Storage
This method:
1. Writes to **local filesystem** (always)
2. Writes to Supabase as an episode (optional, can fail)

**Recommendation:** Keep checkpoint logic in Kernle. The local file storage is intentionally separate from the Storage interface. The episode write should use `storage.save_episode()`.

### `note()` - Formatting in Business Logic
The method formats content based on type (decision, quote, insight). This is presentation logic, not storage.

**Recommendation:** Keep formatting in Kernle. Storage should receive the `Note` dataclass with `content` already formatted.

### `search()` - Different Implementations
- `core.py` does **keyword search** across multiple tables
- `sqlite.py` does **keyword search** (with TODO for semantic)
- SupabaseStorage would need pgvector for semantic search

**Recommendation:** The Storage protocol's `search()` is good. Each backend implements its own search strategy. Document that search behavior may vary between backends.

---

## 4. Backward Compatibility Concerns 🟡 Important

### The `client` Property
Current code exposes `self.client` (Supabase client). Any external code using this will break.

**Recommendation:** 
1. Keep `client` property but deprecate it
2. Add `storage` property as the new interface
3. Log warning when `client` is accessed directly

```python
@property
def client(self):
    """DEPRECATED: Access underlying Supabase client. Use storage methods instead."""
    import warnings
    warnings.warn("Direct client access is deprecated. Use storage methods.", DeprecationWarning)
    if self._legacy_client is None:
        # ...create client
    return self._legacy_client
```

### Default Storage Selection
The plan says "default to SQLiteStorage if no Supabase creds."

**Problem:** This silently changes behavior for existing users who forgot to set env vars.

**Recommendation:** 
```python
def __init__(self, storage: Optional[Storage] = None, ...):
    if storage is not None:
        self.storage = storage
    elif supabase_url and supabase_key:
        self.storage = SupabaseStorage(agent_id, url, key)
    else:
        # Explicit: local mode
        logger.info("No Supabase credentials found. Using local SQLite storage.")
        self.storage = SQLiteStorage(agent_id)
```

---

## 5. Data Conversion Strategy 🟢 Recommendation

Currently `core.py` works with dicts everywhere. The plan calls for dataclasses at boundaries.

**Proposed conversion points:**

```
User Code  <-->  Kernle  <-->  Storage  <-->  Database
           ↑           ↑
         dicts    dataclasses
```

### Option A: Dataclasses Everywhere (Clean but Breaking)
- `Kernle.episode()` returns `Episode` not `str`
- `Kernle.load_values()` returns `List[Value]` not `List[dict]`
- **Pro:** Type safety, IDE support
- **Con:** Breaking change for existing users

### Option B: Dicts at Public API (Backward Compatible)
- Public methods keep dict returns
- Internal methods use dataclasses
- Add `.to_dict()` to dataclasses
- **Pro:** No breaking changes
- **Con:** Conversion overhead, less elegant

### Option C: Support Both (Most Flexible)
```python
def load_values(self, as_dataclass: bool = False) -> Union[List[Value], List[dict]]:
    values = self.storage.get_values()
    if as_dataclass:
        return values
    return [v.to_dict() for v in values]
```

**Recommendation:** Option B for v1, migrate to Option A in v2.

---

## 6. SupabaseStorage Extraction Plan

### Step 1: Create `kernle/storage/supabase.py`
```python
class SupabaseStorage:
    def __init__(self, agent_id: str, url: str, key: str):
        self.agent_id = agent_id
        self._client = create_client(url, key)
```

### Step 2: Move table operations from core.py
Each `self.client.table("X").select/insert/update` becomes a method.

### Step 3: Handle field mapping
SupabaseStorage needs to translate between dataclass fields and Supabase column names:

```python
def _episode_to_row(self, episode: Episode) -> dict:
    return {
        "agent_id": episode.agent_id,
        "objective": episode.objective,
        "outcome_description": episode.outcome,  # Note: different field name
        "outcome_type": episode.outcome_type,
        "lessons_learned": episode.lessons,      # Note: different field name
        "tags": episode.tags,
        "is_reflected": False,                   # Default
        # ...
    }

def _row_to_episode(self, row: dict) -> Episode:
    return Episode(
        id=row["id"],
        agent_id=row["agent_id"],
        objective=row["objective"],
        outcome=row["outcome_description"],      # Map back
        lessons=row.get("lessons_learned"),      # Map back
        # ...
    )
```

---

## 7. Refactoring Order (Suggested)

To minimize risk, refactor in this order:

1. **Update dataclasses** in `base.py` to include all Supabase fields
2. **Add missing protocol methods** to Storage interface
3. **Create SupabaseStorage** class (extract from core.py)
4. **Add `storage` parameter** to Kernle.__init__
5. **Refactor one method at a time** (start with simple ones: `load_values`, `load_beliefs`)
6. **Test each method** before moving to the next
7. **Handle complex methods last** (`consolidate`, `checkpoint`, `search`)
8. **Deprecate direct client access**
9. **Update documentation**

---

## 8. Test Strategy

### Unit Tests
- Each Storage implementation should pass the same test suite
- Create `tests/storage/test_storage_protocol.py` with abstract tests

### Integration Tests  
- Test Kernle with SQLiteStorage
- Test Kernle with SupabaseStorage (requires test DB)
- Test storage switching at runtime

### Migration Tests
- Test that existing Supabase data loads correctly through new interface
- Test that SQLite can import from Supabase export

---

## 9. Summary Checklist

Before starting the refactor:

- [ ] Update `Episode` dataclass: add `is_reflected`, `confidence`, `patterns_to_repeat`, `patterns_to_avoid`, rename `lessons` → `lessons_learned`, `outcome` → `outcome_description` (or handle in storage)
- [ ] Update `Value` dataclass: add `value_type`, `is_active`, `is_foundational`
- [ ] Update `Belief` dataclass: add `is_active`, `is_foundational`
- [ ] Update `Note` dataclass: add `is_protected`, decide on `owner_id` vs `agent_id`
- [ ] Update `Drive` dataclass: add `last_satisfied_at`, `satisfaction_decay_hours`
- [ ] Update `Relationship` dataclass: reconcile with Supabase schema
- [ ] Add `get_unreflected_episodes()` or filter param to `get_episodes()`
- [ ] Add `mark_episode_reflected()` or handle via `save_episode()`
- [ ] Add temporal query methods
- [ ] Decide on dict vs dataclass at public API
- [ ] Plan deprecation path for `client` property

---

## Questions for Stakeholder

1. Should we maintain exact backward compatibility at the API level?
2. Is there external code accessing `kernle.client` directly?
3. What's the timeline? Can we do a breaking v2.0 release?
4. Should SQLite be the default, or should we require explicit storage choice?
5. Do we need sync between SQLite and Supabase in v1?
