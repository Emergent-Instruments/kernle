# Kernle Commerce Security Audit Report

**Date:** January 31, 2026  
**Auditor:** Gemini Security Review (via Claude Opus 4.5)  
**Scope:** Business logic, state machine, economic attacks, API design  
**Files Reviewed:** 15 files across backend routes and client commerce modules

---

## Executive Summary

The Kernle Commerce system implements a job marketplace with escrow-based payments. While the codebase shows evidence of security consciousness (reentrancy guards, input validation, authorization checks), **several critical vulnerabilities** exist that could allow economic attacks and system manipulation.

### Risk Assessment

| Category | Risk Level | Issues Found |
|----------|------------|--------------|
| Business Logic | 🔴 HIGH | 5 critical |
| State Machine | 🟡 MEDIUM | 3 issues |
| Economic Attacks | 🔴 HIGH | 4 attack vectors |
| API Design | 🟡 MEDIUM | 6 weaknesses |

### Priority Fixes Required

1. **CRITICAL**: Backend routes lack escrow verification (payment without actual blockchain tx)
2. **CRITICAL**: No deadline enforcement after job acceptance
3. **CRITICAL**: Dispute resolution has no timeout or escalation
4. **HIGH**: Race condition in application acceptance (partially mitigated)
5. **HIGH**: No slippage protection on escrow amounts

---

## 1. Business Logic Flaws

### 1.1 [CRITICAL] Payment Release Without Blockchain Verification

**Location:** `backend/app/routes/commerce/jobs.py:390-415`

```python
@router.post("/{job_id}/approve", response_model=JobResponse)
async def approve_job(...):
    # TODO: Release escrow via smart contract
    updated = await update_job_status(db, job_id, "completed", agent_id)
```

**Issue:** The API marks jobs as "completed" and conceptually "releases payment" without any verification that:
1. The escrow contract actually released funds
2. The worker received the payment
3. The transaction succeeded on-chain

**Attack Scenario:**
1. Attacker creates job with escrow on testnet or fake contract
2. Worker completes legitimate work
3. Client calls `/approve` - backend marks job complete
4. No actual payment occurs because escrow was never funded or doesn't exist

**Recommendation:**
```python
async def approve_job(...):
    # MUST verify on-chain before updating status
    escrow_status = await verify_escrow_release(job["escrow_address"])
    if escrow_status != "released":
        raise HTTPException(400, "Escrow release not confirmed on-chain")
    
    updated = await update_job_status(db, job_id, "completed", agent_id)
```

---

### 1.2 [CRITICAL] No Deadline Enforcement After Acceptance

**Location:** `kernle/commerce/jobs/service.py` - Missing deadline checks

**Issue:** Once a job is accepted, there's no enforcement of the deadline:
- Worker can take infinite time to deliver
- No auto-cancellation when deadline passes
- Client has no recourse except manual dispute

**Attack Scenario:**
1. Worker applies to high-value job with tight deadline
2. Worker gets accepted, funds locked in escrow
3. Worker never delivers, deadline passes
4. Client's funds remain locked indefinitely (griefing attack)

**Recommendation:**
```python
async def check_deadline_expired(self, job_id: str) -> bool:
    """Auto-cancel jobs past deadline with no delivery."""
    job = self.get_job(job_id)
    if job.status == "accepted" and job.deadline < self._now():
        # Auto-refund to client
        await self.cancel_job(job_id, "system", "Deadline expired")
        return True
    return False
```

Add a background job to periodically check deadlines.

---

### 1.3 [CRITICAL] Dispute Resolution Has No Timeout

**Location:** `kernle/commerce/jobs/service.py:580-620`

```python
def resolve_dispute(self, job_id, actor_id, resolution, ...):
    # Only authorized arbitrator can resolve
    if not self._is_authorized_arbitrator(actor_id):
        raise UnauthorizedError(...)
```

**Issue:** Disputed jobs can remain in limbo forever:
- No timeout for arbitrator action
- No escalation path if arbitrator is unresponsive
- Funds locked indefinitely

**Attack Scenario:**
1. Malicious or compromised arbitrator address set
2. Any dispute raised locks funds forever
3. Systemic griefing by raising disputes on all jobs

**Recommendation:**
```python
# Add auto-resolution after timeout
DEFAULT_DISPUTE_TIMEOUT_DAYS = 30

def auto_resolve_dispute(self, job_id: str) -> Job:
    """Auto-resolve in favor of worker after timeout (default: release payment)."""
    job = self.get_job(job_id)
    if job.status != "disputed":
        raise InvalidTransitionError("Job is not disputed")
    
    # Find when dispute was raised
    transitions = self.get_job_history(job_id)
    dispute_time = None
    for t in transitions:
        if t.to_status == "disputed":
            dispute_time = t.created_at
            break
    
    if not dispute_time:
        raise JobServiceError("Cannot determine dispute time")
    
    timeout = timedelta(days=DEFAULT_DISPUTE_TIMEOUT_DAYS)
    if self._now() < dispute_time + timeout:
        raise JobServiceError("Dispute timeout has not expired")
    
    # Default: release to worker (delivered work presumed acceptable)
    return self._transition_job(
        job=job,
        new_status=JobStatus.COMPLETED,
        actor_id="system",
        metadata={"action": "auto_resolved", "reason": "arbitration_timeout"},
    )
```

---

### 1.4 [HIGH] Skill Validation is Cosmetic Only

**Location:** `backend/app/routes/commerce/jobs.py:74-79`

```python
@field_validator("skills_required")
@classmethod
def validate_skills(cls, v: list[str]) -> list[str]:
    # Normalize skill names
    return [s.lower().strip() for s in v if s.strip()]
```

**Issue:** Skills are not validated against the canonical registry:
- Fake skills can be required
- Workers searching by real skills won't find jobs with typos
- Marketplace fragmentation

**Attack Scenario:**
1. Attacker posts job requiring skill "c0ding" (typo)
2. Legitimate workers searching "coding" don't see job
3. Attacker's accomplice with "c0ding" skill applies and wins

**Recommendation:**
```python
async def validate_skills_exist(db, skills: list[str]) -> list[str]:
    """Validate all skills exist in canonical registry."""
    if not skills:
        return []
    
    result = db.table("skills").select("name").in_("name", skills).execute()
    valid_skills = {s["name"] for s in result.data}
    
    invalid = set(skills) - valid_skills
    if invalid:
        raise ValueError(f"Unknown skills: {', '.join(invalid)}")
    
    return skills
```

---

### 1.5 [HIGH] Worker Can Deliver Garbage URLs

**Location:** `kernle/commerce/jobs/service.py:475-517`

The `_validate_deliverable_url` method validates URL format but not content:

```python
@staticmethod
def _validate_deliverable_url(url: str) -> None:
    allowed_schemes = {'http', 'https', 'ipfs', 'ipns', 'ar'}
    # ... format validation only
```

**Issue:** Worker can deliver any syntactically valid URL:
- 404 pages
- Rick rolls
- Malware links
- Empty IPFS hashes

**Attack Scenario:**
1. Worker gets accepted for high-value job
2. Worker delivers `https://example.com/404`
3. Approval timeout (7 days) passes
4. Auto-release pays worker for nothing

**Recommendation:**
- Implement URL content verification (at minimum, check HTTP 200)
- For IPFS, verify the hash is pinned and accessible
- Require deliverable hash for verification
- Add client review period with delivery rejection capability

---

## 2. State Machine Issues

### 2.1 [MEDIUM] State Transition Validation Gaps

**Location:** `backend/app/routes/commerce/jobs.py:123-131`

```python
VALID_TRANSITIONS = {
    "open": {"funded", "cancelled"},
    "funded": {"accepted", "cancelled"},
    "accepted": {"delivered", "disputed", "cancelled"},
    "delivered": {"completed", "disputed"},
    "disputed": {"completed"},  # <-- Only one exit!
}
```

**Issue:** The state machine has incomplete transitions:
1. `disputed` → `cancelled` not allowed (should be for partial refunds)
2. `disputed` → `refunded` not modeled
3. No `delivered` → `accepted` for delivery rejection

**Cross-reference:** Matches Codex finding in `commerce-audit.md` - state machine is sound for happy path but lacks error recovery states.

**Recommendation:** Add states:
```python
VALID_TRANSITIONS = {
    # ... existing
    "disputed": {"completed", "refunded", "split"},  # Add resolution outcomes
    "delivered": {"completed", "disputed", "rejected"},  # Add rejection
}
```

---

### 2.2 [MEDIUM] Escrow State ≠ Job State

**Location:** `backend/app/routes/commerce/escrow.py:66-82`

```python
# Map job status to escrow status
if job_status == "completed":
    escrow_status = "released"
elif job_status == "cancelled":
    escrow_status = "refunded"
```

**Issue:** Job state and escrow contract state are assumed to match but are never synchronized:
- Escrow could fail to release after job marked complete
- Escrow could be released but DB update fails
- No reconciliation mechanism

**Attack Scenario:**
1. System crash between escrow release and DB update
2. Escrow released, job still shows "delivered"
3. Worker's funds are in limbo, can't prove payment

**Recommendation:**
- Add transaction logging with blockchain tx hashes
- Implement reconciliation job to sync states
- Add escrow polling to verify actual on-chain state

---

### 2.3 [LOW] Application Status Not Synced With Job Cancellation

**Location:** `kernle/commerce/jobs/service.py:365-395`

When a job is cancelled, pending applications aren't automatically rejected:

```python
def cancel_job(self, job_id, actor_id, reason=None, tx_hash=None):
    # ... transition job to cancelled
    # NO cleanup of pending applications
```

**Issue:** Applicants may not know job was cancelled, continue waiting.

**Recommendation:**
```python
def cancel_job(self, ...):
    # ... existing code
    
    # Clean up pending applications
    pending_apps = self.storage.list_applications(
        job_id=job_id, status=ApplicationStatus.PENDING
    )
    for app in pending_apps:
        self.storage.update_application_status(app.id, ApplicationStatus.REJECTED)
```

---

## 3. Economic Attack Vectors

### 3.1 [CRITICAL] Front-Running Application Acceptance

**Location:** `kernle/commerce/jobs/service.py:395-455`

```python
def accept_application(self, job_id, application_id, actor_id):
    job_lock = self._get_job_lock(job_id)
    with job_lock:
        # ... acceptance logic
```

**Current Mitigation:** Per-job locking prevents simultaneous acceptance.

**Remaining Issue:** The lock is process-local (Python threading.Lock). In distributed deployment:
- Multiple backend instances can accept different applications
- Database has no unique constraint on `(job_id, status='accepted')`

**Attack Scenario (Distributed):**
1. Job has applications A and B
2. Server 1 processes accept(A) while Server 2 processes accept(B)
3. Both succeed, job has two workers, escrow split unclear

**Recommendation:**
```sql
-- Add database-level constraint
ALTER TABLE jobs ADD CONSTRAINT one_worker_per_job 
    EXCLUDE (id WITH =) WHERE (worker_id IS NOT NULL);

-- Or use advisory locks in DB
SELECT pg_advisory_xact_lock(hashtext(job_id));
```

---

### 3.2 [HIGH] Escrow Amount Gaming

**Location:** `kernle/commerce/escrow/service.py:147-180`

```python
def deploy_escrow(self, job_id, client_address, amount, deadline):
    # ... deploys escrow with specified amount
```

**Issue:** No verification that escrow amount matches job budget:
- Client could deploy escrow with lower amount
- Worker accepts job expecting $100, escrow only has $10

**Attack Scenario:**
1. Client posts job for $1000 USDC
2. Client deploys escrow with only $10 USDC
3. Worker completes work worth $1000
4. Escrow releases only $10

**Recommendation:**
```python
def fund_job(self, job_id, actor_id, escrow_address, ...):
    job = self.get_job(job_id)
    
    # MUST verify escrow amount matches job budget
    escrow_info = self.escrow_service.get_escrow(escrow_address)
    if escrow_info.amount != Decimal(str(job.budget_usdc)):
        raise JobServiceError(
            f"Escrow amount {escrow_info.amount} doesn't match job budget {job.budget_usdc}"
        )
```

---

### 3.3 [HIGH] Sybil Attack on Marketplace

**Issue:** No identity verification or stake required:
- Anyone can create unlimited agent IDs
- Fake agents can flood marketplace with spam jobs
- Fake workers can flood applications

**Attack Scenarios:**
1. **Spam Attack**: Create 1000 fake jobs to bury legitimate listings
2. **Application Flooding**: Apply to all jobs with fake accounts, waste client time
3. **Reputation Manipulation**: Complete jobs between controlled accounts

**Recommendation:**
- Require minimum stake to post jobs
- Rate limit job creation per agent
- Require Twitter/social verification for workers
- Implement reputation system with stake slashing

---

### 3.4 [MEDIUM] Fee/Gas Griefing

**Location:** `kernle/commerce/escrow/service.py` - All blockchain operations

**Issue:** No gas cost estimation or fee structure:
- Client pays gas to fund escrow
- Worker pays gas to claim
- Arbitrator pays gas to resolve disputes

**Attack Scenario:**
1. Create many small-value jobs ($0.01)
2. Gas cost exceeds job value
3. Workers lose money claiming legitimate earnings

**Recommendation:**
```python
MIN_JOB_BUDGET = Decimal("5.00")  # Cover expected gas costs

def create_job(self, ..., budget_usdc):
    if budget_usdc < self.MIN_JOB_BUDGET:
        raise JobServiceError(
            f"Minimum job budget is ${self.MIN_JOB_BUDGET} to cover gas costs"
        )
```

---

## 4. API Design Weaknesses

### 4.1 [MEDIUM] No Idempotency Keys

**Location:** All POST endpoints in `backend/app/routes/commerce/`

**Issue:** No idempotency for critical operations:
- Network retry could fund escrow twice
- Double application submission
- Duplicate job creation

**Recommendation:**
```python
@router.post("/{job_id}/fund")
async def fund_job(
    ...,
    idempotency_key: str = Header(None, alias="Idempotency-Key"),
):
    if idempotency_key:
        existing = await get_idempotent_response(idempotency_key)
        if existing:
            return existing
    
    # ... process request
    
    if idempotency_key:
        await store_idempotent_response(idempotency_key, response, ttl=86400)
```

---

### 4.2 [MEDIUM] Missing Pagination in List Endpoints

**Location:** `backend/app/routes/commerce/jobs.py:195-235`

```python
async def list_jobs_endpoint(..., limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    # Allows up to 100 jobs per request
```

**Issue:** While pagination exists, there's no cursor-based pagination for stable ordering:
- New jobs inserted can cause duplicates/skips during pagination
- `offset` scales poorly for large datasets

**Recommendation:** Implement cursor-based pagination using `created_at` or `id`.

---

### 4.3 [MEDIUM] Overly Permissive CORS/Rate Limits

**Location:** Rate limit is 60/minute for list, 30/minute for applications

**Issue:** Limits may be too permissive for scraping protection:
- 60/minute allows scraping entire marketplace quickly
- No per-IP rate limiting mentioned

---

### 4.4 [LOW] Inconsistent Error Responses

**Location:** Various error handlers

**Issue:** Some errors return structured JSON, others return plain text:
```python
raise HTTPException(status_code=400, detail="Invalid escrow address format")
# vs
return [TextContent(type="text", text="Invalid input: {str(e)}")]
```

**Recommendation:** Standardize on JSON error format with error codes.

---

### 4.5 [LOW] Missing Audit Logging

**Issue:** Critical operations lack structured audit logging:
- Who funded what escrow when
- Who approved which job
- Dispute resolution decisions

**Recommendation:**
```python
async def log_audit_event(db, event_type: str, actor_id: str, resource_id: str, details: dict):
    await db.table("audit_log").insert({
        "event_type": event_type,
        "actor_id": actor_id,
        "resource_id": resource_id,
        "details": details,
        "ip_address": request.client.host,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
```

---

### 4.6 [LOW] No Versioning Strategy

**Issue:** API endpoints lack version prefix:
```
/commerce/jobs          # Not /api/v1/commerce/jobs
```

This makes breaking changes difficult to manage.

---

## 5. Cross-Reference with Codex Audit

The previous commerce audit (`commerce-audit.md`) identified **18 validation issues**. Comparing findings:

| Codex Finding | Status | This Audit |
|---------------|--------|------------|
| Wallet address validation | ✅ Fixed | Verified in `models.py` |
| Spending limit validation | ✅ Fixed | Verified in `models.py` |
| Job title/description empty | ✅ Fixed | Verified in `models.py` |
| Escrow address validation | ✅ Fixed | Verified in `models.py` |
| Skill name length | ⚠️ Partial | No registry validation |
| Past deadline accepted | ✅ Fixed | Checked in `__post_init__` |
| No maximum budget | ✅ Fixed | `MAX_BUDGET_USDC = 1B` added |
| SQL injection risk | ✅ N/A | Still using placeholder storage |

**New findings not in Codex audit:**
- Blockchain verification gaps
- Distributed race conditions  
- Economic attack vectors
- Dispute timeout missing
- Escrow ↔ Job state synchronization

---

## 6. Recommendations Summary

### Immediate (Before Any Production Use)

1. **Add blockchain verification** for all escrow operations
2. **Implement deadline enforcement** with auto-cancellation
3. **Add dispute resolution timeout** with escalation
4. **Verify escrow amounts** match job budgets
5. **Add database-level locking** for application acceptance

### Before Public Launch

6. **Implement minimum job budget** to cover gas costs
7. **Add skill validation** against canonical registry
8. **Implement deliverable verification** (URL accessibility)
9. **Add idempotency keys** for critical operations
10. **Structured audit logging** for all state changes

### Post-Launch Improvements

11. **Cursor-based pagination** for scalability
12. **API versioning** for breaking changes
13. **Reputation system** with stake requirements
14. **Rate limiting improvements** per-IP
15. **Standardized error responses**

---

## 7. Security Checklist

### Authentication & Authorization
- [x] Agent authentication via API keys
- [x] Client-only operations checked (fund, accept, approve)
- [x] Worker-only operations checked (deliver)
- [x] Arbitrator authorization verified
- [ ] **Missing**: Multi-signature for large escrows
- [ ] **Missing**: Time-locked withdrawals

### Data Validation
- [x] Input sanitization in MCP tools
- [x] Ethereum address format validation
- [x] Budget bounds checking
- [x] Deadline future validation
- [ ] **Missing**: Skills registry validation
- [ ] **Missing**: URL content verification

### State Management
- [x] State machine transitions defined
- [x] Per-job locking for acceptance
- [x] Reentrancy guards in escrow service
- [ ] **Missing**: Distributed locking
- [ ] **Missing**: State reconciliation with blockchain
- [ ] **Missing**: Delivery rejection state

### Economic Security
- [ ] **Missing**: Escrow amount verification
- [ ] **Missing**: Minimum job budget
- [ ] **Missing**: Sybil resistance
- [ ] **Missing**: Gas cost estimation
- [ ] **Missing**: Slippage protection

---

*Audit complete. Address CRITICAL issues before any production deployment.*
