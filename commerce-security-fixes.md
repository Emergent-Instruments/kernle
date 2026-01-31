# Kernle Commerce Security Fixes

**Date:** 2025-01-31  
**Audit Reference:** `commerce-security-audit.md`

---

## Summary

This document describes the security fixes implemented for the critical and high severity vulnerabilities identified in the Kernle Commerce security audit.

**Fixed Issues:**
- 🔴 4 CRITICAL issues
- 🟠 6 HIGH issues (1 remaining requires deeper refactor)

---

## CRITICAL Fixes

### 1. Unauthorized Dispute Resolution ✅

**Location:** `kernle/commerce/jobs/service.py`

**Issue:** The `resolve_dispute()` function had no authorization check, allowing any user to resolve disputes and direct funds to themselves.

**Fix:** Added `_is_authorized_arbitrator()` method that verifies the actor_id is:
- The configured arbitrator address from config
- In the list of authorized arbitrators
- The "system" actor (for auto-resolution)

```python
def _is_authorized_arbitrator(self, actor_id: str) -> bool:
    if self.config.arbitrator_address and actor_id == self.config.arbitrator_address:
        return True
    authorized_arbitrators = getattr(self.config, 'authorized_arbitrators', [])
    if actor_id in authorized_arbitrators:
        return True
    if actor_id == "system":
        return True
    return False
```

**Test:** `test_unauthorized_dispute_resolution` now passes.

---

### 2. Race Condition in Application Acceptance ✅

**Location:** `kernle/commerce/jobs/service.py`

**Issue:** Two concurrent accept requests could both succeed, resulting in multiple workers assigned to one job.

**Fix:** Added per-job locking using `threading.Lock`:
- Created `_job_locks` dictionary to store per-job locks
- Added `_get_job_lock()` with double-checked locking pattern
- Wrapped entire `accept_application()` logic in `with job_lock:`

```python
def accept_application(self, job_id, application_id, actor_id):
    job_lock = self._get_job_lock(job_id)
    with job_lock:
        # Re-fetch job inside lock to ensure latest state
        job = self.get_job(job_id)
        # ... rest of logic is atomic
```

**Test:** `test_race_condition_double_acceptance` now passes.

---

### 3. Daily Spending Limit Not Enforced ✅

**Location:** `kernle/commerce/wallet/service.py`

**Issue:** Daily spending limit was declared but never actually enforced, allowing a compromised wallet to be drained by many small transactions.

**Fix:** Implemented daily spend tracking:
- Added `DailySpendRecord` dataclass to track date, total, and count
- Added `_daily_spend` dict to WalletService
- Added `_get_daily_spend()` and `_record_spend()` methods
- Updated `transfer()` to check and enforce daily limit

```python
current_daily_spend = self._get_daily_spend(wallet_id)
if float(current_daily_spend + amount) > wallet.spending_limit_daily:
    raise SpendingLimitExceededError(...)

# After successful transfer:
self._record_spend(wallet_id, amount)
```

**Test:** `test_daily_spending_limit_enforced` now passes.

---

### 4. Wallet Claim Vulnerability ✅

**Location:** `backend/app/routes/commerce/wallets.py`

**Issue:** Wallet claim only checked status, not ownership. An attacker who knew a wallet_id could claim another user's wallet.

**Fix:** Added ownership verification before allowing claim:

```python
wallet_user_id = wallet.get("user_id")
if wallet_user_id and wallet_user_id != auth.user_id:
    logger.warning(f"Unauthorized wallet claim attempt | ...")
    raise HTTPException(status_code=403, detail="Cannot claim another user's wallet")
```

**Test:** `test_wallet_claim_requires_ownership` passes.

---

## HIGH Fixes

### 1. SQL Injection in Skills Search ✅

**Location:** `backend/app/routes/commerce/skills.py`

**Issue:** User input was interpolated directly into filter string:
```python
db_query.or_(f"name.ilike.%{query}%,description.ilike.%{query}%")
```

**Fix:** Added `_sanitize_search_query()` function that:
- Escapes special SQL/LIKE characters (`%`, `_`, `\`, `'`, `"`, `;`, `--`, `/*`, `*/`)
- Limits query length to 100 characters
- Uses parameterized `ilike()` instead of string interpolation

```python
sanitized_query = _sanitize_search_query(query)
if sanitized_query:
    db_query = db_query.ilike("name", f"%{sanitized_query}%")
```

**Test:** `test_sql_injection_in_skill_search` passes.

---

### 2. Reentrancy Risk in Escrow ✅

**Location:** `kernle/commerce/escrow/service.py`

**Issue:** Escrow operations like `release()`, `refund()`, and `resolve_dispute()` could potentially be called recursively.

**Fix:** Added reentrancy protection:
- Added `_operation_locks` per-escrow locks
- Added `_active_operations` set to track in-progress operations
- Added `_check_reentrancy()` and `_clear_reentrancy()` methods
- Wrapped critical operations in try/finally with lock

```python
def release(self, escrow_address, client_address):
    escrow_lock = self._get_escrow_lock(escrow_address)
    with escrow_lock:
        self._check_reentrancy(escrow_address, "release")
        try:
            # ... release logic
        finally:
            self._clear_reentrancy(escrow_address, "release")
```

Applied to: `release()`, `refund()`, `resolve_dispute()`

---

### 3. CDP Wallet ID Exposure ✅

**Location:** `kernle/commerce/wallet/models.py`

**Issue:** Internal CDP wallet ID was exposed in `to_dict()` responses, potentially allowing enumeration and correlation attacks.

**Fix:** 
- Modified `to_dict()` to accept `include_internal: bool = False` parameter
- `cdp_wallet_id` is only included when `include_internal=True`
- Added `to_public_dict()` for minimal public-safe output

```python
def to_dict(self, include_internal: bool = False) -> dict:
    result = { ... }
    if include_internal:
        result["cdp_wallet_id"] = self.cdp_wallet_id
    return result

def to_public_dict(self) -> dict:
    """Only public-safe fields."""
    return {"id": ..., "wallet_address": ..., "chain": ..., "status": ...}
```

**Test:** `test_wallet_dict_hides_cdp_id` passes.

---

### 4. Client-Worker Collision ⚠️

**Location:** `kernle/commerce/jobs/service.py`

**Status:** Partially addressed. Same-agent collision is prevented (`applicant_id == job.client_id`), but user-level tracking requires deeper refactor.

**Current Protection:**
```python
if applicant_id == job.client_id:
    raise JobServiceError("Cannot apply to your own job")
```

**Remaining Work:** Full user-level tracking would require:
- Adding `user_id` to Job model
- Tracking which user owns which agents
- Cross-checking at application time

**Test:** `test_self_dealing_different_agents` remains xfail pending deeper implementation.

---

### 5. URL Validation Missing ✅

**Location:** `kernle/commerce/jobs/service.py`

**Issue:** `deliverable_url` accepted any string, including dangerous schemes like `file://` or `javascript:`.

**Fix:** Added `_validate_deliverable_url()` static method:
- Validates URL format with `urlparse()`
- Only allows safe schemes: `http`, `https`, `ipfs`, `ipns`, `ar` (Arweave)
- Rejects URLs over 2000 characters
- Validates netloc for http/https

```python
allowed_schemes = {'http', 'https', 'ipfs', 'ipns', 'ar'}
if parsed.scheme.lower() not in allowed_schemes:
    raise JobServiceError(f"Invalid URL scheme: {parsed.scheme}...")
```

**Test:** `test_path_traversal_in_deliverable_url` passes.

---

### 6. Integer Overflow Potential ✅

**Location:** `kernle/commerce/jobs/service.py`

**Issue:** Budget values had no upper bound, risking precision loss with very large floats.

**Fix:** Added bounds checking in `create_job()`:
```python
MAX_BUDGET_USDC = 1_000_000_000  # $1 billion
MIN_BUDGET_USDC = 0.01  # 1 cent minimum

if budget_usdc < self.MIN_BUDGET_USDC:
    raise JobServiceError(f"Budget must be at least ${self.MIN_BUDGET_USDC}")
if budget_usdc > self.MAX_BUDGET_USDC:
    raise JobServiceError(f"Budget cannot exceed ${self.MAX_BUDGET_USDC:,}")
```

**Test:** `test_budget_bounds_checking` passes.

---

### 7. PII Exposure in Error Messages ✅

**Location:** `kernle/commerce/mcp/tools.py`

**Status:** Already implemented in `handle_commerce_tool_error()`:
- Known exceptions return appropriate messages
- Unknown exceptions log full details but return generic "Internal server error"
- No connection strings, passwords, or internal details leaked

---

## Test Results

```
======================== 273 passed, 1 xfailed in 2.46s ========================
```

All security tests pass except for `test_self_dealing_different_agents` which is expected to fail until user-level tracking is implemented.

---

## Recommendations for Future Work

1. **User-Level Tracking:** Implement `user_id` on jobs and applications to prevent self-dealing across agents owned by the same user.

2. **Timelock on Admin Functions:** Add timelock to `setArbitrator()` in the smart contract to prevent instant arbitrator changes.

3. **Circuit Breaker:** Add emergency pause functionality to escrow contracts.

4. **PII Detection:** Consider adding warnings when job descriptions contain potential PII patterns.

5. **Rate Limiting:** Review rate limits on job creation (currently 20/min may be too permissive).

---

## Files Modified

- `kernle/commerce/jobs/service.py`
- `kernle/commerce/wallet/service.py`
- `kernle/commerce/wallet/models.py`
- `kernle/commerce/escrow/service.py`
- `backend/app/routes/commerce/wallets.py`
- `backend/app/routes/commerce/skills.py`
- `tests/commerce/test_security.py`
- `tests/commerce/test_edge_cases.py`
- `tests/commerce/test_job_service.py`
