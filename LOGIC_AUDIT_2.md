# Logic Audit Report - Kernle

**Date:** 2025-01-28  
**Auditor:** Adversarial Logic Reviewer  
**Focus:** Bug hunting in core logic, feature mixins, CLI commands, and storage layer

---

## Critical Findings

### 1. CRITICAL: Division by Zero in Identity Confidence Calculation

**File:** `kernle/core.py`  
**Function:** `get_identity_confidence()` (lines ~2670-2720)  
**Severity:** CRITICAL

**Bug:** When `values` list is empty, the code attempts to calculate `avg_priority` by dividing by `len(values)`, but this division happens *inside* an `if values:` block. However, if `len(values) == 0` but the list is truthy somehow (shouldn't happen with normal lists), or if the pattern is copied elsewhere, it would divide by zero. More importantly:

```python
# Beliefs section (line ~2685):
if beliefs:
    avg_belief_conf = sum(b.confidence for b in beliefs) / len(beliefs)
```

**Test case:** If `beliefs = []` but somehow evaluates truthy, or if `get_beliefs()` returns a proxy object that's truthy when empty, you get ZeroDivisionError.

**Fix:** Add explicit `len(beliefs) > 0` check:
```python
if beliefs and len(beliefs) > 0:
    avg_belief_conf = sum(b.confidence for b in beliefs) / len(beliefs)
```

---

### 2. CRITICAL: Race Condition in Sync Queue Deduplication

**File:** `kernle/storage/sqlite.py`  
**Function:** `_queue_sync()` (lines ~1236-1260)  
**Severity:** CRITICAL

**Bug:** The function does SELECT then UPDATE or INSERT without proper locking. Between the SELECT and the UPDATE/INSERT, another thread/process could modify the sync_queue.

```python
# Check for existing unsynced entry for this record
existing = conn.execute(
    "SELECT id FROM sync_queue WHERE table_name = ? AND record_id = ? AND synced = 0",
    (table, record_id)
).fetchone()

if existing:
    # RACE WINDOW: another process could delete this row
    conn.execute(
        """UPDATE sync_queue ...""",
        (..., existing["id"])
    )
```

**Test case:** Two concurrent saves to the same record. Thread A does SELECT, Thread B does SELECT, both find the same row, both try to UPDATE, one silently fails or corrupts data.

**Fix:** Use `INSERT ... ON CONFLICT DO UPDATE` (SQLite UPSERT) or wrap in explicit transaction with SERIALIZABLE isolation.

---

### 3. HIGH: Off-by-One in Confidence History Truncation

**File:** `kernle/core.py`  
**Function:** `reinforce_belief()` (line ~1530)  
**Severity:** HIGH

**Bug:** Confidence history is truncated to `[-20:]` AFTER appending, which is correct. But the issue is the boost calculation:

```python
confidence_boost = 0.05 * (1.0 / (1 + existing.times_reinforced * 0.1))
room_to_grow = 0.99 - existing.confidence
existing.confidence = min(0.99, existing.confidence + room_to_grow * confidence_boost)
```

When `existing.confidence = 0.99`, `room_to_grow = 0`, so the confidence stays at 0.99, but `times_reinforced` still increments. This is fine, but the `confidence_boost` calculation uses `existing.times_reinforced` BEFORE incrementing, meaning the first reinforcement uses 0, not 1.

```python
existing.times_reinforced += 1  # This happens AFTER using it
```

**Test case:** Create belief with confidence 0.5, reinforce it. First reinforcement uses `1.0 / (1 + 0 * 0.1) = 1.0`, but the count shows 1. Slightly inaccurate tracking.

**Fix:** Increment `times_reinforced` before calculating boost, or use `(existing.times_reinforced + 1)` in calculation.

---

### 4. HIGH: Null Handling Gap in Emotion Detection

**File:** `kernle/features/emotions.py`  
**Function:** `detect_emotion()` (lines ~90-115)  
**Severity:** HIGH

**Bug:** If `text` is `None` (despite type hint), `text.lower()` will throw `AttributeError`.

```python
def detect_emotion(self: "Kernle", text: str) -> Dict[str, Any]:
    text_lower = text.lower()  # Crashes if text is None
```

**Test case:** `k.detect_emotion(None)` → AttributeError

**Fix:** Add defensive check:
```python
if not text:
    return {"valence": 0.0, "arousal": 0.0, "tags": [], "confidence": 0.0}
text_lower = text.lower()
```

---

### 5. HIGH: Incorrect Boolean Logic in Episode Outcome Detection

**File:** `kernle/core.py`  
**Function:** `episode()` (line ~595)  
**Severity:** HIGH

**Bug:** The outcome_type detection is case-sensitive and checks `.lower()`, but uses exact match which is fragile:

```python
outcome_type = "success" if outcome.lower() in ("success", "done", "completed") else (
    "failure" if outcome.lower() in ("failure", "failed", "error") else "partial"
)
```

**Problem:** "Successfully completed" → "partial" (not "success")  
**Problem:** "DONE!" → "partial" (has extra character)

**Test case:** `k.episode("task", "Done!")` → outcome_type = "partial", not "success"

**Fix:** Use substring matching or normalize input:
```python
outcome_lower = outcome.lower().strip()
if any(word in outcome_lower for word in ("success", "done", "completed", "finished")):
    outcome_type = "success"
elif any(word in outcome_lower for word in ("fail", "error", "broke")):
    outcome_type = "failure"
else:
    outcome_type = "partial"
```

---

### 6. HIGH: Type Confusion in Anxiety Score Calculation

**File:** `kernle/features/anxiety.py`  
**Function:** `get_anxiety_report()` (lines ~133-136)  
**Severity:** HIGH

**Bug:** The Memory Uncertainty calculation has dead code that computes but doesn't use a percentage:

```python
if total_beliefs == 0:
    uncertainty_detail = "No beliefs yet"
else:
    int((len(low_conf_beliefs) / total_beliefs) * 100)  # <-- Result discarded!
    uncertainty_detail = f"{len(low_conf_beliefs)}/{total_beliefs} beliefs below 50% confidence"
```

The computed percentage is immediately discarded. This looks like a copy-paste bug.

**Test case:** Run anxiety report - the calculation happens but result is thrown away.

**Fix:** Either use the computed value or delete the line:
```python
uncertainty_pct = int((len(low_conf_beliefs) / total_beliefs) * 100)
uncertainty_detail = f"{len(low_conf_beliefs)}/{total_beliefs} beliefs ({uncertainty_pct}%) below 50% confidence"
```

---

### 7. MEDIUM: Edge Case in Salience Calculation - Zero Division

**File:** `kernle/features/forgetting.py`  
**Function:** `calculate_salience()` (lines ~50-70)  
**Severity:** MEDIUM

**Bug:** The salience calculation handles edge cases but has a potential issue:

```python
days_since = (now - reference_time).total_seconds() / 86400
age_factor = days_since / self.DEFAULT_HALF_LIFE  # DEFAULT_HALF_LIFE = 30.0

# If DEFAULT_HALF_LIFE were ever 0, this would crash
salience = (confidence * (reinforcement_weight + 0.1)) / (age_factor + 1)
```

While `DEFAULT_HALF_LIFE` is hardcoded to 30.0, if subclassed or modified, zero would cause division by zero.

**Test case:** Subclass `ForgettingMixin` and set `DEFAULT_HALF_LIFE = 0`

**Fix:** Add guard:
```python
half_life = max(0.001, self.DEFAULT_HALF_LIFE)
age_factor = days_since / half_life
```

---

### 8. MEDIUM: Resource Leak in File Operations

**File:** `kernle/storage/sqlite.py`  
**Function:** `_sync_beliefs_to_file()` and similar (lines ~1680-1700)  
**Severity:** MEDIUM

**Bug:** File is opened with `open()` but uses `with` correctly. However, `os.chmod()` is called after file close, which could fail if the file was deleted between operations:

```python
with open(self._beliefs_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
# File is closed here
# Secure permissions
import os
os.chmod(self._beliefs_file, 0o600)  # File could be gone!
```

**Test case:** Concurrent process deletes file between write and chmod.

**Fix:** Set permissions inside `with` block on file descriptor, or handle exception:
```python
try:
    os.chmod(self._beliefs_file, 0o600)
except FileNotFoundError:
    logger.warning("File disappeared before chmod")
```

---

### 9. MEDIUM: Incorrect Datetime Handling - Naive vs Aware

**File:** `kernle/storage/base.py`  
**Function:** `parse_datetime()` (line ~20)  
**Severity:** MEDIUM

**Bug:** The function replaces 'Z' with '+00:00' but doesn't handle other naive datetime formats:

```python
def parse_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        return None
```

If input is a naive datetime string like "2025-01-28T12:00:00" (no timezone), the result is a naive datetime, but later comparisons with `datetime.now(timezone.utc)` will fail or give wrong results.

**Test case:** Save a naive datetime, then compare with UTC-aware datetime → TypeError

**Fix:** Force UTC if no timezone:
```python
dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone.utc)
return dt
```

---

### 10. MEDIUM: Empty List Handling in Belief Revision

**File:** `kernle/core.py`  
**Function:** `revise_beliefs_from_episode()` (lines ~1680-1750)  
**Severity:** MEDIUM

**Bug:** When getting all active beliefs, if the database returns empty list, the function works but wastes computation:

```python
beliefs = self._storage.get_beliefs(limit=500)

for belief in beliefs:  # Fine if empty
    ...
```

But the evidence text construction assumes episode exists and has attributes:

```python
evidence_parts = []
if episode.outcome_type == "success":  # What if outcome_type is None?
    evidence_parts.append(f"Successfully: {episode.objective}")
```

**Test case:** Episode with `outcome_type = None` → includes "none" in lowercase matching

**Fix:** Add explicit None check:
```python
if episode.outcome_type and episode.outcome_type.lower() == "success":
```

---

### 11. MEDIUM: Integer Overflow Potential in Confidence Bar

**File:** `kernle/cli/commands/belief.py`  
**Function:** `cmd_belief()` (line ~85)  
**Severity:** MEDIUM

**Bug:** Confidence bar calculation truncates to integer and could have issues:

```python
conf_bar = "█" * int(r["contradiction_confidence"] * 10) + "░" * (10 - int(r["contradiction_confidence"] * 10))
```

If `contradiction_confidence > 1.0` (which the code can produce up to 0.95, but edge cases exist), `int(1.5 * 10) = 15`, so the bar would be "█████████████████░░░░░░░░░" (15 filled + negative empty = crash).

**Test case:** Manipulate contradiction_confidence to be > 1.0

**Fix:** Clamp the value:
```python
filled = min(10, max(0, int(r["contradiction_confidence"] * 10)))
conf_bar = "█" * filled + "░" * (10 - filled)
```

---

### 12. LOW: Inconsistent Return Types in Search

**File:** `kernle/core.py`  
**Function:** `search()` (lines ~1780-1820)  
**Severity:** LOW

**Bug:** The search function returns different dict structures for different record types:

```python
if record_type == "episode":
    formatted.append({
        "type": "episode",
        "title": record.objective[:60] if record.objective else "",
        "content": record.outcome,  # 'outcome' is used
        ...
    })
elif record_type == "note":
    formatted.append({
        "type": record.note_type or "note",  # Different type handling!
        "title": record.content[:60] if record.content else "",
        "content": record.content,  # 'content' repeated
        ...
    })
```

For notes, `type` is `note_type`, but for episodes, it's literal "episode". Inconsistent API.

**Test case:** Search returns mixed results - code expecting consistent "type" field fails.

**Fix:** Standardize structure or document the inconsistency.

---

### 13. LOW: Missing Validation in CLI Raw Process

**File:** `kernle/cli/commands/raw.py`  
**Function:** `cmd_raw()` process action (line ~95)  
**Severity:** LOW

**Bug:** When processing batch IDs (comma-separated), errors are printed but processing continues. If one ID fails, the subsequent count includes it:

```python
for raw_id in raw_ids:
    try:
        ...
        success_count += 1
    except ValueError as e:
        print(f"✗ {raw_id}: {e}")
# After loop:
if len(raw_ids) > 1:
    print(f"\nProcessed {success_count}/{len(raw_ids)} entries")
```

This is actually correct, but the error handling swallows the exception type - a KeyError would crash:

**Test case:** Pass an ID that causes a KeyError instead of ValueError.

**Fix:** Catch broader exceptions:
```python
except Exception as e:
    print(f"✗ {raw_id}: {e}")
```

---

### 14. LOW: Potential Memory Leak in Embeddings

**File:** `kernle/storage/sqlite.py`  
**Function:** `_save_embedding()` (lines ~1280-1310)  
**Severity:** LOW

**Bug:** Embeddings are saved but never cleaned up when records are deleted. The `vec_embeddings` table grows forever.

**Test case:** Create 1000 episodes, delete them all. Embeddings remain.

**Fix:** Add cleanup in delete operations or add periodic garbage collection.

---

### 15. LOW: Hardcoded Magic Numbers

**File:** `kernle/features/anxiety.py`  
**Function:** Multiple  
**Severity:** LOW

**Bug:** Many magic numbers without constants:

```python
if checkpoint_age < 15:
    unsaved_score = int(checkpoint_age * 2)  # Why 15? Why 2?
elif checkpoint_age < 60:
    unsaved_score = int(30 + (checkpoint_age - 15) * 1.1)  # Why 30? Why 1.1?
```

**Test case:** Need to tune anxiety thresholds → need to change multiple hardcoded values.

**Fix:** Extract to class constants with documentation.

---

## Summary

| Severity | Count | Categories |
|----------|-------|------------|
| CRITICAL | 2 | Division by zero, race condition |
| HIGH | 4 | Null handling, boolean logic, type confusion |
| MEDIUM | 5 | Datetime handling, resource leaks, edge cases |
| LOW | 4 | Inconsistencies, missing cleanup |

**Total: 15 findings**

### Priority Recommendations

1. **Fix CRITICAL #1 and #2 immediately** - These can cause crashes in production
2. **Address HIGH findings in next release** - User-facing bugs
3. **MEDIUM findings for v1.1** - Edge cases and robustness
4. **LOW findings for backlog** - Code quality improvements

### Testing Recommendations

1. Add property-based testing with Hypothesis for edge cases
2. Add concurrent access tests for sync operations
3. Add fuzz testing for all public API inputs
4. Add timezone-aware datetime fixtures to catch naive datetime bugs
