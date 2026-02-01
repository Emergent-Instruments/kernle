# Adversarial Audit: export-cache (PR #59)

**Scope**: 284 lines across 3 files (kernle/core.py, kernle/cli/__main__.py, tests/test_export_cache.py)
**Auditor**: Ash (self-audit) + 2 sub-agent specialists (security + correctness)
**Date**: 2026-02-01

## Findings

### P2-1: No input validation on `max_beliefs` parameter
**Severity**: P2 (quality)
**Description**: `max_beliefs` is used in `self._storage.get_beliefs(limit=max_beliefs * 2)`. A negative value would pass a negative limit to SQLite (which SQLite handles gracefully as "no limit", but it's unexpected). A very large value (e.g., `max_beliefs=2**62`) would overflow when multiplied by 2.
**Impact**: Unexpected behavior with extreme inputs. Low real-world risk since this is a CLI flag.
**Fix**: Clamp `max_beliefs` to a reasonable range (1-1000) in the CLI parser or method.

### P2-2: No validation on `min_confidence` bounds  
**Severity**: P2 (quality)
**Description**: `min_confidence` accepts any float. Values > 1.0 would filter out everything. Values < 0.0 would include everything. Neither is harmful, but silent.
**Impact**: Confusing behavior if user passes `--min-confidence 5` (everything filtered) or `--min-confidence -1` (nothing filtered).
**Fix**: Clamp to 0.0-1.0 range or warn.

### P2-3: Relationship notes truncation could cut mid-character
**Severity**: P2 (quality)
**Description**: `r.notes[:80]` truncates notes at 80 chars, which could cut a multi-byte UTF-8 character mid-sequence in Python (though Python strings are Unicode, so this actually just truncates at 80 codepoints, which is fine). However, it could cut mid-word without ellipsis.
**Impact**: Aesthetically poor output. No data integrity issue.
**Fix**: Add `"..."` suffix when truncated: `r.notes[:80] + "..." if len(r.notes) > 80 else r.notes`

### P3-1: `hasattr` check on `interaction_count` is defensive but masks bugs
**Severity**: P3 (style)
**Description**: `hasattr(r, 'interaction_count')` suggests the relationship object might not always have this attribute. If this is a data model issue, it should be fixed at the model level rather than silently defaulting to 0.
**Impact**: Could hide a real bug where relationships lack the expected attribute.
**Fix**: Ensure the Relationship model always has `interaction_count` (even if 0).

### P3-2: `mkdir(parents=True, exist_ok=True)` on user-provided path
**Severity**: P3 (style)  
**Description**: `export_path.parent.mkdir(parents=True, exist_ok=True)` will create arbitrary directory trees. This is intentional behavior (user specified the path via --output), but worth noting for audit completeness.
**Impact**: None — this is a local CLI tool writing where the user told it to.
**Fix**: No fix needed. Behavior is correct and expected.

### P3-3: Agent ID in HTML comment could be information leak in shared contexts
**Severity**: P3 (minor)
**Description**: The header includes `kernle -a {self.agent_id} export-cache` which embeds the agent ID in an HTML comment. If MEMORY.md is committed to a shared repo, this leaks the agent ID.
**Impact**: Minimal — agent IDs aren't secrets, and this is workspace-local.
**Fix**: Optional: use `<agent>` placeholder in the regeneration hint instead of actual ID.

## Summary

| Severity | Count | Action Required |
|----------|-------|-----------------|
| P0       | 0     | —               |
| P1       | 0     | —               |
| P2       | 3     | Fix recommended |
| P3       | 3     | Optional        |

**Verdict**: Clean. No security issues, no correctness bugs, no data integrity concerns. The P2s are quality polish — input validation and truncation aesthetics. Safe to ship as-is, with P2 fixes in a follow-up.
