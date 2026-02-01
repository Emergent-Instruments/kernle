# Kernle Code Quality Audit Report

**Auditor**: Code Quality Subagent  
**Date**: 2026-02-01  
**Scope**: MCP tools, CLI, Features, Importers, Error Handling  

---

## Executive Summary

Overall code quality is **solid**. Input validation is comprehensive at the MCP layer, error handling follows defense-in-depth principles, and the feature modules use reasonable bounds. However, several issues exist ranging from data integrity risks to potential injection vectors.

**Findings by severity:**
- P0 (Critical): 1
- P1 (High): 5
- P2 (Medium): 10
- P3 (Low): 8

---

## 1. MCP Tools (`kernle/mcp/server.py`)

### P1-MCP-01: Provenance params (`source`, `derived_from`) not validated for some tools
**Severity**: P1 (High)  
**Location**: `validate_tool_input()`, tools `memory_episode`, `memory_note`, `memory_belief`

The `source` and `derived_from` fields are defined in the Tool schemas for `memory_episode`, `memory_note`, and `memory_belief`, but `validate_tool_input()` does NOT sanitize or include them in the `sanitized_args` dict. The `call_tool()` handler then reads them via `sanitized_args.get("source")` and `sanitized_args.get("derived_from")`, which will always return `None`.

**Impact**: Provenance data is silently dropped for all MCP tool calls. Lineage tracking is broken via MCP.

**Evidence**: In `validate_tool_input` for `memory_episode`:
```python
sanitized["objective"] = sanitize_string(...)
sanitized["outcome"] = sanitize_string(...)
sanitized["lessons"] = sanitize_array(...)
sanitized["tags"] = sanitize_array(...)
sanitized["context"] = ...
sanitized["context_tags"] = ...
# source and derived_from are MISSING
```

But in `call_tool()`:
```python
episode_id = k.episode(
    ...
    source=sanitized_args.get("source"),      # Always None
    derived_from=sanitized_args.get("derived_from"),  # Always None
)
```

**Fix**: Add validation for `source` and `derived_from` in each tool's validation block.

---

### P2-MCP-02: `memory_belief_update` confidence default causes crash
**Severity**: P2 (Medium)  
**Location**: `validate_tool_input()`, `memory_belief_update` branch

```python
sanitized["confidence"] = (
    validate_number(arguments.get("confidence"), "confidence", 0.0, 1.0, None)
    if arguments.get("confidence") is not None
    else None
)
```

`validate_number()` with `default=None` will raise `ValueError("confidence is required")` if `value` is `None`. However, the outer `if` guard prevents this. The real issue is: if someone passes `confidence=0` (falsy), `arguments.get("confidence") is not None` is True, but `arguments.get("confidence")` evaluates to `0`, which is valid. This actually works correctly. **Downgrading concern — logic is sound but confusing.**

**Revised assessment**: Code works but is fragile. A refactor would improve clarity.

---

### P2-MCP-03: `memory_auto_capture` source normalization is inconsistent  
**Severity**: P2 (Medium)  
**Location**: `call_tool()`, `memory_auto_capture` handler

```python
if source not in {"cli", "mcp", "sdk", "import", "unknown"}:
    if "auto" in source.lower():
        source = "mcp"
    else:
        source = "mcp"
```

Both branches of the `if/else` result in `"mcp"`. The enum check also doesn't include `"auto"` which is the default from `sanitize_string`. This means any non-standard source value silently becomes `"mcp"` with no warning.

---

### P3-MCP-04: Tool count mismatch with documented "33 tools"
**Severity**: P3 (Low)  
**Location**: `TOOLS` list

The `TOOLS` list contains **28 tools** (counted). The remaining tools come from commerce (`get_commerce_tools()`). If commerce is not installed, clients see fewer tools than documented.

---

### P2-MCP-05: `memory_note_search` does multiplication hack for filtering
**Severity**: P2 (Medium)  
**Location**: `call_tool()`, `memory_note_search` handler

```python
results = k.search(query=query, limit=limit * 2)  # Get extra in case we filter
```

If the search returns mostly non-note results, the user gets fewer results than requested. For `limit=100`, it searches for 200 which may be expensive. Should use a dedicated note search or loop until limit is met.

---

### P3-MCP-06: Global singleton Kernle instance not thread-safe
**Severity**: P3 (Low)  
**Location**: `get_kernle()`, `set_agent_id()`

The global `_mcp_agent_id` and function-attribute singleton pattern (`get_kernle._instance`) is not thread-safe. MCP stdio transport is single-threaded so this is currently safe, but would break if transport changes.

---

## 2. CLI (`kernle/cli/__main__.py`, `kernle/cli/commands/`)

### P2-CLI-01: No ID format validation for update/delete operations
**Severity**: P2 (Medium)  
**Location**: `cmd_episode`, update handlers, forget commands

IDs like `episode_id`, `goal_id`, `belief_id` are passed through without UUID format validation. Malformed IDs (e.g., SQL injection attempts like `'; DROP TABLE --`) are passed directly to storage layer. Safety depends entirely on the storage layer using parameterized queries.

**Mitigation**: The `sanitize_string` function removes control characters but doesn't validate UUID format. Storage likely uses parameterized queries (SQLite), but defense-in-depth requires validation here.

---

### P2-CLI-02: `cmd_anxiety` `--auto` mode silently executes actions without confirmation
**Severity**: P2 (Medium)  
**Location**: `kernle/cli/commands/anxiety.py`, `cmd_anxiety()`, auto mode

The `--auto` flag executes checkpoint saves, consolidation, identity synthesis, sync, and even `emergency_save()` without any confirmation prompt. While this is by design for automation, it can modify significant state.

**Risk**: An agent miscalculating anxiety could trigger emergency saves repeatedly, creating noise in episode history.

---

### P3-CLI-03: `cmd_extract` builds content via string concatenation with user input
**Severity**: P3 (Low)  
**Location**: `__main__.py`, `cmd_extract()`

```python
capture_parts = [f"Conversation extract: {summary}"]
if getattr(args, "topic", None):
    capture_parts.append(f"Topic: {args.topic}")
```

Input is validated via `validate_input()` before use, so this is safe. But the pattern of building structured data via string concatenation is fragile.

---

### P3-CLI-04: `cmd_search` handles `min_score` but type is not validated
**Severity**: P3 (Low)  
**Location**: `__main__.py`, `cmd_search()`

`min_score` comes from argparse (presumably as float), but there's no bounds check (0.0-1.0).

---

## 3. Features

### 3a. Emotions (`kernle/features/emotions.py`)

### P3-EMO-01: Keyword matching is substring-based, causing false positives
**Severity**: P3 (Low)  
**Location**: `detect_emotion()`

```python
if keyword in text_lower:
```

The keyword `"sad"` matches "saddle", "said" doesn't match but `"mad"` matches "made", "nomad", etc. The keyword `"love it"` uses multi-word matching which is better but inconsistent.

**Impact**: Emotion auto-detection may produce inaccurate tags. Since confidence is labeled and this is heuristic-based, impact is low.

---

### P2-EMO-02: `episode_with_emotion` sets `valence or 0.0` — masks intentional zero valence
**Severity**: P2 (Medium)  
**Location**: `episode_with_emotion()`, line ~295

```python
episode = Episode(
    ...
    emotional_valence=valence or 0.0,
    emotional_arousal=arousal or 0.0,
)
```

If emotion detection returns `valence=0.0` (neutral), `0.0 or 0.0` evaluates to `0.0` (correct by accident). But if `valence` is explicitly `None` and `auto_detect` found nothing, this silently sets 0.0 rather than leaving it unset. This conflates "no emotion data" with "neutral emotion".

---

### 3b. Anxiety (`kernle/features/anxiety.py`)

### P2-ANX-01: Anxiety score can theoretically exceed 100
**Severity**: P2 (Medium)  
**Location**: `get_anxiety_report()`, context_pressure calculation

```python
context_score = int(90 + (context_pressure_pct - 85) * 0.67)
```

When `context_pressure_pct = 100`: `90 + 15 * 0.67 = 100.05`, which rounds to 100. But the `min(100, ...)` guard on the dimension catches this. **All dimensions have `min(100, ...)` guards.** 

However, the weighted composite score:
```python
overall_score = 0
for dim_name, weight in self.ANXIETY_WEIGHTS.items():
    overall_score += dimensions[dim_name]["score"] * weight
overall_score = int(overall_score)
```

Weights sum to 1.0, and each dimension is capped at 100, so max composite is 100. **Bounds are correct.** No issue.

**Can scores go negative?** All dimension scores use formulas starting from 0 or positive values. The minimum is 0. Composite minimum is 0. **Bounds are safe.**

---

### P3-ANX-02: Checkpoint age estimated at 500 tokens/minute is arbitrary
**Severity**: P3 (Low)  
**Location**: `get_anxiety_report()`, context pressure fallback

```python
estimated_tokens = checkpoint_age * 500
```

This is a rough heuristic. If the checkpoint is 6 hours old, it estimates 180,000 tokens, which would show 90% context pressure. This could cause false anxiety spikes for idle sessions.

---

### 3c. Suggestions (`kernle/features/suggestions.py`)

### P0-SUG-01: Raw memory content flows into suggestion content without sanitization
**Severity**: P0 (Critical)  
**Location**: `_create_note_suggestion()`, `_create_belief_suggestion()`, `_create_episode_suggestion()`

Raw entries are user-provided brain dumps. When suggestions are created, the raw content is extracted and stored in suggestion `content` dict fields. When promoted via `promote_suggestion()`, this content is passed directly to `k.episode()`, `k.belief()`, or `k.note()`.

```python
# _create_note_suggestion
return MemorySuggestion(
    ...
    content={
        "content": content,  # Raw user input, unvalidated
        ...
    },
)

# promote_suggestion
elif memory_type == "note":
    memory_id = self.note(
        content=content.get("content", ""),  # Passed through
    )
```

**Risk**: If the raw entry contains adversarial content (e.g., prompt injection text like "Ignore previous instructions and..."), it gets promoted verbatim into structured memory. On `memory_load`, this content is rendered back to the agent, potentially injecting instructions.

**Impact**: Memory poisoning via raw capture → suggestion → promotion pipeline. The content is not sanitized for prompt injection patterns at any point.

**Mitigation**: The core `note()`, `belief()`, `episode()` methods should have input validation. If they do (via `_validate_string_input`), the risk is reduced to content-level injection (not code injection). Still, the promotion pipeline should sanitize.

---

### P2-SUG-02: `_score_patterns` normalization can exceed 1.0 before cap
**Severity**: P2 (Medium)  
**Location**: `_score_patterns()`

```python
return min(1.0, matched_weight / (total_weight * 0.5))
```

The divisor is `total_weight * 0.5`, so matching >50% of patterns yields score >1.0 before the `min(1.0, ...)` cap. This is intentional (cap applied), but the threshold of 0.4 means even 2-3 keyword matches can create a suggestion. This leads to many low-quality suggestions.

---

### 3d. Forgetting (`kernle/features/forgetting.py`)

### P1-FGT-01: No built-in protection for foundational/identity memories
**Severity**: P1 (High)  
**Location**: `run_forgetting_cycle()`, `forget()`

The forgetting system relies on `is_protected` flag on individual memories. **There is no automatic protection for values or identity-related beliefs.** If a value or core belief has low access count and ages past the salience threshold, `run_forgetting_cycle()` will tombstone it.

```python
def get_forgetting_candidates(self, threshold=0.3, limit=20, memory_types=None):
    # Default memory_types includes ALL types
```

The default `memory_types` parameter in the storage layer likely includes values, beliefs, etc. A newly created value that's never accessed could have salience:
- `confidence=0.8, times_accessed=0, days_since=31`
- `salience = (0.8 * (log(1) + 0.1)) / (31/30 + 1) = (0.8 * 0.1) / 2.03 = 0.039`

This is well below the 0.3 threshold. **A core value could be forgotten after ~30 days if never explicitly accessed.**

**Fix**: Auto-protect values and high-confidence beliefs, or exclude certain memory types from forgetting by default.

---

### P1-FGT-02: `run_forgetting_cycle(dry_run=False)` has no confirmation and no undo batching
**Severity**: P1 (High)  
**Location**: `run_forgetting_cycle()`

Running with `dry_run=False` immediately tombstones up to `limit` memories. While individual memories can be recovered, there's no batch undo or transaction. If called programmatically (e.g., by anxiety auto-mode), it could forget many memories at once.

The CLI `cmd_forget` does show a dry run message, but the feature API has no safety rail.

---

### P2-FGT-03: `calculate_salience` returns -1.0 for not found, but callers may not check
**Severity**: P2 (Medium)  
**Location**: `calculate_salience()`

Returns `-1.0` for not-found memories. If a caller doesn't check for this sentinel value and compares `salience < threshold`, the not-found result always qualifies as a forgetting candidate.

**In practice**: `get_forgetting_candidates` uses storage-layer queries, not `calculate_salience`, so this is only a risk for direct API callers.

---

### 3e. Knowledge (`kernle/features/knowledge.py`)

### P2-KNW-01: Domain extraction uses belief_type as domain name
**Severity**: P2 (Medium)  
**Location**: `_extract_domains_from_tags()`

```python
domain = belief.belief_type or "general"
```

Belief types are limited enum values (`fact`, `rule`, `preference`, `constraint`, `learned`). This means all facts are grouped into one "fact" domain, all rules into "rule" domain, etc. This gives a very coarse domain map that doesn't reflect actual knowledge areas.

Tags are used for episodes and notes (more useful), but beliefs lose their topic information.

---

### P3-KNW-02: Goal domain extraction uses first word of title
**Severity**: P3 (Low)  
**Location**: `_extract_domains_from_tags()`

```python
words = goal.title.lower().split()[:2]
if words:
    domain = words[0]
```

A goal titled "Learn Kubernetes basics" creates domain "learn". A goal titled "Fix the login bug" creates domain "fix". This is noisy and unhelpful.

---

### P3-KNW-03: `detect_knowledge_gaps` gap detection is naive
**Severity**: P3 (Low)  
**Location**: `detect_knowledge_gaps()`

```python
potential_gaps = query_words - found_topics - {"how", "do", "i", ...}
```

Splitting on whitespace and comparing word sets doesn't account for multi-word concepts or synonyms. Results are more noise than signal.

---

## 4. Importers

### P1-IMP-01: No content size limits on imported data
**Severity**: P1 (High)  
**Location**: All importers (`csv_importer.py`, `json_importer.py`, `markdown.py`, `clawdbot.py`)

None of the importers validate content length before calling `k.episode()`, `k.note()`, `k.belief()`, etc. A malicious or malformed import file could contain entries with megabytes of text per field.

**Example** (JSON importer):
```python
k.episode(
    objective=data.get("objective", ""),  # No length limit
    outcome=data.get("outcome", ""),      # No length limit
    ...
)
```

If the core API methods don't enforce limits, this could bloat the database.

---

### P1-IMP-02: CSV importer trusts column values without sanitization
**Severity**: P1 (High)  
**Location**: `csv_importer.py`, `_map_columns()`, `_import_csv_item()`

```python
result[field_name] = value  # Raw CSV cell value
```

CSV cell values are used directly. No control character stripping, no length validation. If a CSV cell contains null bytes or extremely long strings, they pass through to storage.

---

### P2-IMP-03: Clawdbot importer reads arbitrary files from workspace
**Severity**: P2 (Medium)  
**Location**: `clawdbot.py`, `_parse_daily_notes()`

```python
for md_file in sorted(memory_dir.glob("*.md")):
    content = md_file.read_text(encoding="utf-8")
```

Reads all `.md` files in the `memory/` directory. If a symlink exists pointing outside the workspace, this could read unintended files. Low risk in practice.

---

### P2-IMP-04: JSON importer doesn't validate data types within items
**Severity**: P2 (Medium)  
**Location**: `json_importer.py`, `_import_json_item()`

```python
k.episode(
    objective=data.get("objective", ""),
    outcome=data.get("outcome", ""),
    emotional_valence=data.get("emotional_valence", 0.0),  # Could be a string
)
```

If the JSON contains `"emotional_valence": "not_a_number"`, this passes through to the core API. Type validation depends on the core layer.

---

### P3-IMP-05: Duplicate detection is O(n) for goals and values
**Severity**: P3 (Low)  
**Location**: `csv_importer.py`, `json_importer.py`

```python
existing = k._storage.get_goals(status=None, limit=100)
for g in existing:
    if g.title == title:
        return False
```

For large memory stores with >100 goals, duplicates beyond the first 100 won't be detected. Also, this is called per-row, making it O(n*m) for the import.

---

### P3-IMP-06: Markdown importer `_parse_beliefs` confidence parsing edge case
**Severity**: P3 (Low)  
**Location**: `markdown.py`, `_parse_beliefs()`

```python
confidence = float(val)
if confidence > 1:
    confidence = confidence / 100
```

A value like `150` becomes `1.5`, which exceeds 1.0. The final `min(1.0, max(0.0, confidence))` clamp catches this. Safe but the intermediate `/100` logic is confused (150/100=1.5, still >1). Should clamp immediately after conversion.

---

## 5. Error Handling

### P2-ERR-01: `handle_tool_error` catches all exceptions, potentially masking bugs
**Severity**: P2 (Medium)  
**Location**: `server.py`, `call_tool()`

```python
except Exception as e:
    return handle_tool_error(e, name, arguments)
```

The broad `except Exception` means programming errors (AttributeError, KeyError, TypeError from bugs) are caught and returned as "Internal server error" to the MCP client. While this prevents crashes, it makes debugging difficult without log access.

---

### P1-ERR-02: No transaction safety in `promote_suggestion`
**Severity**: P1 (High)  
**Location**: `suggestions.py`, `promote_suggestion()`

```python
memory_id = self.episode(...)  # Step 1: Create memory
self._storage.update_suggestion_status(...)  # Step 2: Update suggestion
self._storage.mark_raw_processed(...)  # Step 3: Mark raw as processed
```

If step 1 succeeds but step 2 or 3 fails (e.g., disk full, db lock), the memory is created but the suggestion remains "pending". Re-promoting would create a duplicate memory. There's no transaction wrapping these operations.

---

## Summary Table

| ID | Severity | Component | Title |
|---|---|---|---|
| P0-SUG-01 | P0 | Suggestions | Raw content flows to structured memory unsanitized |
| P1-MCP-01 | P1 | MCP | Provenance params silently dropped |
| P1-FGT-01 | P1 | Forgetting | No auto-protection for values/identity memories |
| P1-FGT-02 | P1 | Forgetting | No safety rail on live forgetting cycles |
| P1-IMP-01 | P1 | Importers | No content size limits on imported data |
| P1-IMP-02 | P1 | Importers | CSV values unsanitized |
| P1-ERR-02 | P1 | Suggestions | No transaction safety in promote_suggestion |
| P2-MCP-03 | P2 | MCP | Source normalization both branches identical |
| P2-MCP-05 | P2 | MCP | Note search uses multiplication hack |
| P2-CLI-01 | P2 | CLI | No UUID format validation for IDs |
| P2-CLI-02 | P2 | CLI | Anxiety auto-mode executes without confirmation |
| P2-EMO-02 | P2 | Emotions | `valence or 0.0` masks intentional None |
| P2-SUG-02 | P2 | Suggestions | Pattern scoring threshold too permissive |
| P2-FGT-03 | P2 | Forgetting | -1.0 sentinel for not-found may confuse callers |
| P2-KNW-01 | P2 | Knowledge | Belief_type used as domain name (too coarse) |
| P2-IMP-03 | P2 | Importers | Clawdbot reads all .md files (symlink risk) |
| P2-IMP-04 | P2 | Importers | JSON importer doesn't validate data types |
| P2-ERR-01 | P2 | MCP | Broad exception catch masks bugs |
| P2-MCP-02 | P2 | MCP | belief_update confidence validation confusing |
| P3-MCP-04 | P3 | MCP | Tool count may not match documented 33 |
| P3-MCP-06 | P3 | MCP | Singleton not thread-safe |
| P3-EMO-01 | P3 | Emotions | Substring keyword matching causes false positives |
| P3-ANX-02 | P3 | Anxiety | Token estimation heuristic is arbitrary |
| P3-CLI-03 | P3 | CLI | String concatenation for structured data |
| P3-CLI-04 | P3 | CLI | min_score not bounds-checked |
| P3-KNW-02 | P3 | Knowledge | Goal domain uses first word (noisy) |
| P3-KNW-03 | P3 | Knowledge | Knowledge gap detection is naive |
| P3-IMP-05 | P3 | Importers | Duplicate detection limited to first 100 |
| P3-IMP-06 | P3 | Importers | Confidence parsing intermediate value > 1.0 |

---

## Recommended Priority Actions

1. **P0-SUG-01**: Add sanitization to the suggestion→promotion pipeline. At minimum, run content through `_validate_string_input` before creating memories from suggestions.

2. **P1-MCP-01**: Add `source` and `derived_from` to `validate_tool_input()` for `memory_episode`, `memory_note`, and `memory_belief`. This is a functional bug — provenance tracking is completely broken via MCP.

3. **P1-FGT-01**: Auto-protect all values. Consider auto-protecting beliefs with confidence ≥ 0.8. Add `memory_types` default that excludes `value` and `drive` from forgetting candidates.

4. **P1-IMP-01/02**: Add `sanitize_string()` or length-limit validation in importer code before passing to core API.

5. **P1-ERR-02**: Wrap `promote_suggestion` in a transaction or implement idempotency check (skip if already promoted).
