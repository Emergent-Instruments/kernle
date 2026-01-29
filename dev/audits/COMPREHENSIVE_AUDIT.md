# Kernle Comprehensive Audit - Consolidated Report
*January 28, 2026*

## Executive Summary

| Audit Area | Grade | Key Finding |
|------------|-------|-------------|
| **Security** | B- | Critical secrets exposed, file permissions too permissive |
| **Test Coverage** | B | 58% coverage, 561 tests, but critical sync paths untested |
| **Architecture** | B+ | Solid foundation, but god objects need refactoring |

---

## 🔒 Security Audit

### 🚨 CRITICAL (Fix Immediately)

**1. Hardcoded Production Secrets in Repository**
- File: `~/kernle/.env`
- Contains: Supabase keys, DB password, JWT secret
- Impact: Complete compromise of production database possible
- **Action:** Rotate ALL credentials immediately

### 🔴 HIGH Priority

**2. SQL Injection via Table Name Interpolation**
- Files: `sqlite.py` (12+ locations)
- Risk: Table names in f-strings
- Mitigation: Validate against allowlist (partially done)
- **Fix:** Centralize table validation in single function

**3. Database File Permissions Too Permissive**
- `~/.kernle/memories.db` is world-readable (`-rw-r--r--`)
- **Fix:** `chmod 0o600` after creation

**4. Flat File Storage World-Readable**
- Same issue for `~/.kernle/claire/` files
- **Fix:** Apply `0o600` permissions

**5. Error Messages Expose Internal Details**
- File: `backend/app/routes/auth.py:255`
- Stack traces visible to clients
- **Fix:** Generic error messages, detailed logging server-side only

### 🟡 MEDIUM Priority

- No rate limiting on sync/memory routes
- JWT token expiry too long (1 week → recommend 1-4 hours)
- Sync config has placeholder auth token (`"null"` string)

### ✅ Security Positives
- Good input validation throughout
- Parameterized queries for user values
- bcrypt for password hashing
- CORS configured properly

---

## 🧪 Test Coverage Audit

### Coverage Summary

| Module | Coverage | Assessment |
|--------|----------|------------|
| `mcp/server.py` | 99% | ✅ Excellent |
| `cli/__main__.py` | 76% | ✅ Good |
| `core.py` | 75% | ✅ Good |
| `storage/sqlite.py` | 71% | ⚠️ Moderate |
| `storage/postgres.py` | 52% | ❌ Needs work |

### Critical Gaps (P0)

1. **Sync Push Operations Untested**
   - `sqlite.py:2692-2821`
   - Critical for data durability

2. **Cloud Search Untested**
   - `sqlite.py:3075-3154`
   - Fallback logic not verified

3. **No Integration Tests**
   - CLI → Kernle → Storage flow untested
   - Components tested in isolation only

4. **Postgres Coverage Low**
   - Relationship and playbook methods untested

### Test Quality Issues

- CLI tests mock entire Kernle class (no real integration)
- Weak assertions in Postgres tests
- Missing error path coverage
- Missing boundary/edge case tests (unicode, max size)

### Recommendations

1. Add end-to-end CLI integration tests
2. Test sync push/pull operations
3. Increase Postgres coverage to 70%+
4. Add concurrent access tests

---

## 🏗️ Architecture Audit

### Strengths

1. **Clean Storage Abstraction**
   - Protocol in `base.py` well-defined
   - Factory function handles backend selection
   - Clear separation of concerns

2. **Comprehensive Data Model**
   - Rich dataclasses with meta-memory fields
   - Forgetting, provenance, sync metadata

3. **Offline-First Design**
   - Sync queue with deduplication
   - Graceful fallback when cloud unavailable

4. **No Circular Dependencies**
   - Clean import graph

### Weaknesses

**P0: God Objects**

1. **`core.py` is 4620 lines with 80+ methods**
   - Hard to maintain, test, understand
   - **Fix:** Extract into feature modules:
     - `features/beliefs.py`
     - `features/emotions.py`
     - `features/forgetting.py`
     - `features/identity.py`

2. **`cli/__main__.py` is 4242 lines**
   - **Fix:** Split into `cli/commands/` modules

**P1: Schema Divergence**

SQLite vs Postgres tables differ:
| SQLite | Postgres | Issue |
|--------|----------|-------|
| `episodes` | `agent_episodes` | Name |
| `notes` | `memories` | Structure |
| `playbooks` | ❌ missing | No support |
| `raw_entries` | ❌ missing | No support |

**P1: Potential N+1 Queries**
- `revise_beliefs_from_episode()` iterates 500 beliefs with individual saves
- **Fix:** Batch updates

**P2: Technical Debt**
- 3 TODOs in production code
- Inconsistent error handling (some raise, some return False)
- Magic numbers without constants

### Scalability

- **10K entries:** ✅ Should work well
- **100K entries:** ⚠️ Needs pagination, cursor-based iteration

---

## 📋 Prioritized Action Items

### Immediate (Today)
1. 🔴 Rotate ALL exposed credentials
2. 🔴 Fix database file permissions (chmod 0o600)
3. 🔴 Fix flat file permissions

### This Week
4. 🟡 Centralize SQL table name validation
5. 🟡 Add rate limiting to sync routes
6. 🟡 Fix error message exposure
7. 🟡 Add CLI → Storage integration tests

### Next Sprint
8. 🟡 Extract core.py into feature modules
9. 🟡 Increase Postgres test coverage to 70%
10. 🟡 Reduce JWT expiry, add refresh tokens

### Backlog
11. 🟢 Split CLI into command modules
12. 🟢 Standardize error handling patterns
13. 🟢 Document schema divergence decision
14. 🟢 Add pagination to bulk operations

---

## Summary

**Kernle has a solid foundation** but needs security hardening and code organization improvements:

- **Security:** Fix credential exposure and file permissions immediately
- **Tests:** Good quantity (561), but gaps in critical paths
- **Architecture:** Well-designed abstractions, but god objects need splitting

The issues are tractable — mostly surface-level improvements rather than fundamental redesign.

*Audit conducted by 3 specialist sub-agents, compiled by Claire*
