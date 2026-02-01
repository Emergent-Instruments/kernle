# Kernle Commerce Package Audit Report

**Date:** January 31, 2026  
**Auditor:** Commerce Auditor Subagent  
**Status:** Phase 1 Review Complete

---

## Executive Summary

The commerce package has a solid foundation with well-structured code, comprehensive docstrings, and 71 passing tests. However, **18 validation issues** were identified that could allow invalid data to be created. Most are missing edge case validations in dataclass `__post_init__` methods.

**Risk Assessment:** 🟡 MEDIUM  
- No SQL injection risks (storage is placeholder)
- No security vulnerabilities in current code
- Data validation gaps could cause issues when integrated with database

---

## Test Results

```
✅ 71 tests passed (test_wallet_models.py, test_job_models.py, test_skill_models.py)
✅ All imports work correctly
✅ Package structure follows integration plan
```

---

## Critical Issues (Should Fix Before Production)

### 1. WalletAccount - Incomplete Address Validation

**File:** `kernle/commerce/wallet/models.py`

| Issue | Current Behavior | Expected Behavior |
|-------|-----------------|-------------------|
| Empty address | Accepted | Should reject |
| Short address (< 42 chars) | Accepted | Should reject |
| Long address (> 42 chars) | Accepted | Should reject |

**Proof:**
```python
# All these should raise ValueError but don't:
WalletAccount(id='1', agent_id='a', wallet_address='')  # Empty
WalletAccount(id='1', agent_id='a', wallet_address='0x123')  # Too short
WalletAccount(id='1', agent_id='a', wallet_address='0x' + 'a'*50)  # Too long
```

**Fix:** Add regex validation for exactly `0x` + 40 hex characters.

### 2. WalletAccount - Missing Spending Limit Validation

**File:** `kernle/commerce/wallet/models.py`

| Issue | Current Behavior | Expected Behavior |
|-------|-----------------|-------------------|
| Negative limit | Accepted (-100) | Should reject |
| Zero limit | Accepted (0) | Should reject |

**Proof:**
```python
WalletAccount(..., spending_limit_per_tx=-100)  # Accepted!
WalletAccount(..., spending_limit_per_tx=0)     # Accepted!
```

**Fix:** Add `if self.spending_limit_per_tx <= 0: raise ValueError(...)` in `__post_init__`.

### 3. Job - Empty/Whitespace Title/Description Accepted

**File:** `kernle/commerce/jobs/models.py`

| Issue | Current Behavior | Expected Behavior |
|-------|-----------------|-------------------|
| Empty title | Accepted | Should reject |
| Whitespace title | Accepted | Should reject |
| Empty description | Accepted | Should reject |

**Proof:**
```python
Job(..., title='')      # Accepted!
Job(..., title='   ')   # Accepted!
Job(..., description='')  # Accepted!
```

**Fix:** Add `.strip()` length validation in `__post_init__`.

### 4. Job - Past Deadline Accepted

**File:** `kernle/commerce/jobs/models.py`

| Issue | Current Behavior | Expected Behavior |
|-------|-----------------|-------------------|
| Deadline in past | Accepted | Should reject or warn |

**Note:** The DB migration `019_jobs.sql` has `deadline_in_future CHECK (deadline > created_at)` but the model doesn't enforce this.

**Fix:** Add deadline validation in `__post_init__`.

### 5. Job - Invalid Escrow Address Accepted

**File:** `kernle/commerce/jobs/models.py`

| Issue | Current Behavior | Expected Behavior |
|-------|-----------------|-------------------|
| Invalid escrow address format | Accepted | Should validate 0x + 40 hex |

**Fix:** Add escrow_address validation similar to wallet_address.

### 6. Skill - Missing Length and Usage Validation

**File:** `kernle/commerce/skills/models.py`

| Issue | Current Behavior | DB Constraint |
|-------|-----------------|---------------|
| Name > 50 chars | Accepted | VARCHAR(50) will truncate |
| Negative usage_count | Accepted | `non_negative_usage CHECK` |

**Fix:** Add length and non-negative validation in `__post_init__`.

---

## Medium Issues (Should Address)

### 7. Model ↔ DB Schema Misalignment

**Integration Plan vs Migration 018:**
- Plan: `user_id UUID REFERENCES users(id)`
- Migration: `user_id TEXT REFERENCES users(user_id)`

The model uses `Optional[str]` which works with either, but should be consistent.

### 8. Missing `owner_eoa` Validation in WalletAccount

The `owner_eoa` field should validate Ethereum address format when set, but currently accepts any string.

### 9. Missing `deliverable_hash` Validation in Job

If `deliverable_hash` is meant to be an IPFS CID or content hash, it should validate format.

### 10. No Maximum Budget Validation

Jobs can have arbitrarily large budgets (10^18 USDC tested). Consider adding a reasonable maximum.

---

## Low Priority Issues

### 11. Application Message Length Unbounded

`JobApplication.message` has no length limit in the model (100k chars tested). The DB doesn't have a constraint either.

### 12. `skills_required` Not Validated Against Registry

Job's `skills_required` list isn't validated to contain only canonical skill names. Non-existent skills can be required.

### 13. JobStateTransition Status Not Validated

The `from_status` and `to_status` fields in `JobStateTransition` aren't validated against `JobStatus` enum in the model (though the DB has constraints).

---

## Security Review

### SQL Injection Risk: ✅ NONE (Currently)

All storage implementations are placeholders with TODO comments. When implemented:
- Use parameterized queries (shown in comments)
- Never interpolate user input into SQL strings
- The Supabase Python client handles parameterization

### Input Sanitization: 🟡 PARTIAL

- Wallet addresses: Validated but incomplete
- Enum fields: Validated
- String fields: Not sanitized (could contain XSS if displayed in web UI)

### Authentication/Authorization: N/A

Not yet implemented. Will need to verify:
- Only wallet owner can transact
- Only job client can accept applications
- Only worker can submit deliverables

---

## Code Quality

### Type Hints: ✅ Complete
All functions and methods have type hints.

### Docstrings: ✅ Comprehensive
All classes and public methods have docstrings with attribute descriptions.

### Test Coverage: 🟡 Good but Missing Edge Cases
- 71 tests cover happy paths well
- Missing adversarial/edge case tests (see below)

---

## Recommendations

### Immediate (Before Next Phase)

1. **Fix wallet address validation** - Require exactly 42 characters (0x + 40 hex)
2. **Add spending limit validation** - Must be positive
3. **Add title/description validation** - Non-empty after strip
4. **Add deadline validation** - Must be in future (optional: allow past for historical data)
5. **Add escrow_address validation** - Same as wallet_address when present
6. **Add skill name length validation** - Max 50 chars to match DB

### Before Production

7. Implement proper error handling in storage layers
8. Add rate limiting considerations
9. Add input sanitization for web display
10. Security audit of contract interactions (Phase 4)

---

## Files Reviewed

| File | Status | Issues |
|------|--------|--------|
| `commerce/__init__.py` | ✅ | Clean exports |
| `commerce/config.py` | ✅ | Good env handling |
| `commerce/wallet/models.py` | 🟡 | 5 validation issues |
| `commerce/wallet/storage.py` | ✅ | Placeholder, safe |
| `commerce/jobs/models.py` | 🟡 | 6 validation issues |
| `commerce/jobs/storage.py` | ✅ | Placeholder, safe |
| `commerce/skills/models.py` | 🟡 | 3 validation issues |
| `commerce/skills/registry.py` | ✅ | InMemory impl complete |
| `commerce/escrow/__init__.py` | ✅ | Placeholder for Phase 4 |

---

## Migrations Reviewed

| Migration | Status | Notes |
|-----------|--------|-------|
| `018_wallet_accounts.sql` | ✅ | Good constraints, indexes |
| `019_jobs.sql` | ✅ | Good FTS, triggers |
| `020_job_applications.sql` | ✅ | Unique constraint on job+applicant |
| `021_skills_registry.sql` | ✅ | Atomic increment function |
| `022_job_state_transitions.sql` | ✅ | Excellent state machine validation |

---

## Additional Tests Written

See `tests/commerce/test_edge_cases.py` for adversarial tests covering:
- Empty/invalid wallet addresses
- Negative/zero spending limits
- Empty/whitespace job titles
- Past deadlines
- Long skill names
- Negative usage counts

---

*Audit complete. Address critical issues before production deployment.*
