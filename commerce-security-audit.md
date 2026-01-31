# Kernle Commerce Security Audit Report

**Audit Date:** 2025-02-01  
**Auditor:** Adversarial Security Audit  
**Scope:** `kernle/commerce/`, `backend/app/routes/commerce/`, migrations 018-022  
**Severity Scale:** 🔴 CRITICAL | 🟠 HIGH | 🟡 MEDIUM | 🟢 LOW | ℹ️ INFO

---

## Executive Summary

The Kernle Commerce package is a job marketplace with crypto wallet integration. While the codebase demonstrates good practices in some areas (input validation, state machine design), several **critical** and **high** severity issues were identified that could lead to:

- Unauthorized fund transfers
- Job state manipulation
- Data leakage
- Race conditions enabling double-spending

**Total Issues Found:** 23
- 🔴 CRITICAL: 4
- 🟠 HIGH: 7
- 🟡 MEDIUM: 8
- 🟢 LOW: 4

---

## 1. Input Validation & Injection

### 🟠 HIGH: SQL Injection in Skills Search

**Location:** `backend/app/routes/commerce/skills.py:47`

```python
db_query = db_query.or_(f"name.ilike.%{query}%,description.ilike.%{query}%")
```

**Issue:** User input `query` is interpolated directly into the filter string without proper escaping. While Supabase may sanitize some input, this pattern is dangerous.

**Exploit:** `q=research%'; DROP TABLE skills; --`

**Remediation:**
```python
# Use parameterized query
db_query = db_query.ilike("name", f"%{query}%").or_.ilike("description", f"%{query}%")
```

---

### 🟡 MEDIUM: URL Validation Missing for Deliverables

**Location:** `backend/app/routes/commerce/jobs.py:352`

**Issue:** `deliverable_url` accepts any string. Attackers could inject:
- `file:///etc/passwd` - Path traversal
- `javascript:alert(1)` - XSS if rendered
- Extremely long URLs for DoS

**Remediation:**
```python
from urllib.parse import urlparse

def validate_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https', 'ipfs') and len(url) < 2000
```

---

### 🟡 MEDIUM: Integer Overflow in Budget Handling

**Location:** Multiple files

**Issue:** `budget_usdc` is `float` in Python but `DECIMAL(18,6)` in DB. Float arithmetic can cause precision loss:

```python
# service.py uses float
budget_usdc: float

# This could overflow or lose precision
>>> 999999999999.999999 + 0.000001
1000000000000.0  # Precision lost!
```

**Remediation:** Use `Decimal` throughout:
```python
from decimal import Decimal
budget_usdc: Decimal = Field(..., gt=Decimal("0"))
```

---

### 🟢 LOW: Missing Rate Limit on Job Creation

**Location:** `backend/app/routes/commerce/jobs.py:189`

**Issue:** Rate limit is `20/minute` which may allow spam job creation.

**Remediation:** Lower to `5/minute` or implement progressive rate limiting.

---

## 2. Authentication & Authorization

### 🔴 CRITICAL: Anyone Can Resolve Disputes

**Location:** `kernle/commerce/jobs/service.py:487`

```python
def resolve_dispute(
    self,
    job_id: str,
    actor_id: str,  # Arbitrator
    resolution: str,
    ...
) -> Job:
    """Resolve a disputed job.
    ...
    """
    # TODO: Verify actor is authorized arbitrator
    # For now, we trust the caller
```

**Issue:** The `resolve_dispute` function has NO authorization check. Any authenticated user can resolve disputes and direct funds to themselves.

**Exploit:**
1. Worker disputes a legitimate job
2. Attacker calls `resolve_dispute(job_id, "attacker", "worker")` 
3. Funds go to worker (attacker's confederate)

**Remediation:**
```python
def resolve_dispute(self, job_id: str, actor_id: str, ...):
    # Verify arbitrator
    if actor_id != self.config.arbitrator_address:
        raise UnauthorizedError("Only authorized arbitrators can resolve disputes")
```

---

### 🔴 CRITICAL: Wallet Claim Lacks Ownership Verification

**Location:** `backend/app/routes/commerce/wallets.py:119-150`

```python
async def claim_my_wallet(
    request: Request,
    claim_request: WalletClaimRequest,
    auth: CurrentAgent,
    db: Database,
):
    """Claim the agent's wallet by setting an owner EOA."""
    # ...
    wallet = await get_wallet_by_agent(db, auth.agent_id)
    # ...
    if wallet["status"] != "pending_claim":
        raise HTTPException(...)
    
    # CRITICAL: No verification that auth.user_id matches wallet.user_id
    claimed = await claim_wallet(db, wallet["id"], claim_request.owner_eoa)
```

**Issue:** If an attacker knows a wallet_id, they can claim it by manipulating the request flow. The `claim_wallet` DB function only checks status, not ownership.

**Remediation:**
```python
if wallet["user_id"] and wallet["user_id"] != auth.user_id:
    raise HTTPException(status_code=403, detail="Cannot claim another user's wallet")
```

---

### 🟠 HIGH: Workers Cannot View Their Own Application Status

**Location:** `backend/app/routes/commerce/jobs.py:280-300`

```python
async def list_applications(...):
    if job["client_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the job client can view applications",
        )
```

**Issue:** Applicants cannot see the status of their own applications. This is a functional issue that also creates an authorization gap - there's no endpoint for applicants.

**Remediation:** Add endpoint or modify to allow applicants to see their own:
```python
if job["client_id"] != agent_id:
    # Allow applicant to see only their own application
    applications = [a for a in applications if a["applicant_id"] == agent_id]
    if not applications:
        raise HTTPException(status_code=403, ...)
```

---

### 🟠 HIGH: Missing Client-Worker Collision Check

**Location:** `kernle/commerce/jobs/service.py:288`

**Issue:** While `apply_to_job` checks `applicant_id == job.client_id`, there's no check when the client creates a job using a different agent_id they control. A user could:
1. Create job as agent A (client)
2. Apply as agent B (worker - also controlled by user)
3. Accept own application
4. Self-deal

**Remediation:** Track user ownership and prevent self-dealing at user level, not just agent level.

---

### 🟡 MEDIUM: Escrow Address Predictable Before Job Funded

**Location:** `kernle/commerce/escrow/service.py:178`, `mcp/tools.py:456`

```python
# Both use the same deterministic generation
base_hash = uuid.uuid5(uuid.NAMESPACE_DNS, job_id).hex
extra_hash = uuid.uuid5(uuid.NAMESPACE_URL, job_id).hex[:8]
escrow_address = f"0x{(base_hash + extra_hash)[:40]}"
```

**Issue:** Escrow addresses are deterministically generated from job_id. An attacker can:
1. See a job_id
2. Pre-compute the escrow address
3. Front-run transactions to that address

**Remediation:** Use random component or salt:
```python
salt = secrets.token_hex(16)
address = keccak256(job_id + salt + block_hash)
```

---

## 3. Business Logic Flaws

### 🔴 CRITICAL: Race Condition in Application Acceptance

**Location:** `backend/app/routes/commerce/jobs.py:303-350`

```python
async def accept_application(...):
    # Step 1: Check status (no lock)
    application = await get_application(db, accept_request.application_id)
    if application["status"] != "pending":
        raise HTTPException(...)
    
    # Step 2: Accept (gap where race can occur)
    await update_application_status(db, accept_request.application_id, "accepted")
    
    # Step 3: Update job
    updated = await update_job_status(db, job_id, "accepted", ...)
    
    # Step 4: Reject others
    for app in all_apps:
        if app["id"] != accept_request.application_id and app["status"] == "pending":
            await update_application_status(db, app["id"], "rejected")
```

**Issue:** This is not atomic. Two concurrent accept requests could:
1. Both pass the status check (both see "pending")
2. Both mark their application as "accepted"
3. Job ends up with two accepted workers
4. Escrow can only pay one - funds at risk

**Exploit:** Send multiple rapid accept requests for different applications on same job.

**Remediation:** Use database transaction with row-level locking:
```python
async with db.transaction():
    # SELECT ... FOR UPDATE to lock the job row
    job = await get_job_for_update(db, job_id)
    if job["status"] != "funded":
        raise ...
    # Now safe to proceed
```

---

### 🔴 CRITICAL: Daily Spending Limit Not Enforced

**Location:** `kernle/commerce/wallet/service.py:243`

```python
def transfer(self, wallet_id: str, to_address: str, amount: Decimal, actor_id: str):
    # Check per-transaction limit
    if float(amount) > wallet.spending_limit_per_tx:
        raise SpendingLimitExceededError(...)
    
    # TODO: Check daily spending limit (requires tracking daily spend)
    # TODO: Check actual balance
```

**Issue:** Daily spending limit is not implemented. An attacker who gains wallet access could drain funds up to per-tx limit repeatedly.

**Remediation:** Implement daily spend tracking:
```python
async def get_daily_spend(self, wallet_id: str) -> Decimal:
    # Query sum of today's outgoing transfers
    ...

if current_daily_spend + amount > wallet.spending_limit_daily:
    raise SpendingLimitExceededError("Daily limit exceeded")
```

---

### 🟠 HIGH: Auto-Approval Timeout Can Be Gamed

**Location:** `kernle/commerce/jobs/service.py:423-445`

```python
def auto_approve_job(self, job_id: str, tx_hash: Optional[str] = None):
    """Auto-approve a job after timeout period."""
    timeout = timedelta(days=self.config.approval_timeout_days)
    if self._now() < job.delivered_at + timeout:
        raise JobServiceError("Approval timeout has not expired")
```

**Issue:** Workers could:
1. Submit garbage deliverable
2. Wait for timeout
3. Auto-collect payment without doing work

Client may not notice delivery notification in time.

**Remediation:** 
- Require client acknowledgment before timeout starts
- Implement delivery notification system
- Allow client to extend timeout once

---

### 🟠 HIGH: Job Cancellation After Work Started

**Location:** `backend/app/routes/commerce/jobs.py:395`

**Issue:** Client can cancel a job in "accepted" state after worker has started. This could be exploited:
1. Post job, accept worker
2. Let worker complete most of work
3. Cancel just before delivery
4. Re-post same job and get work done for free

**Remediation:** Add cooling-off period or partial payment on cancellation after acceptance.

---

### 🟡 MEDIUM: No Duplicate Job Detection

**Issue:** Same client can create identical jobs repeatedly, potentially for:
- Spam/noise
- Gaming the system
- Reputation manipulation

**Remediation:** Implement duplicate detection:
```python
if await has_similar_active_job(db, client_id, title_hash):
    raise HTTPException(409, "Similar job already exists")
```

---

### 🟡 MEDIUM: State Transition Audit Log Actor Spoofing

**Location:** `backend/supabase/migrations/022_job_state_transitions.sql:65`

```sql
COALESCE(current_setting('app.current_actor', true), 'system')
```

**Issue:** The actor for auto-logged transitions comes from a session variable that could be manipulated if an attacker gains DB access, or defaults to 'system' masking who made changes.

**Remediation:** Pass actor_id explicitly through application layer; don't rely on session variables.

---

## 4. Data Exposure

### 🟠 HIGH: CDP Wallet ID Exposed in Responses

**Location:** `kernle/commerce/wallet/models.py:87`

```python
def to_dict(self) -> dict:
    return {
        ...
        "cdp_wallet_id": self.cdp_wallet_id,  # Internal ID exposed
        ...
    }
```

**Issue:** The CDP internal wallet ID should not be exposed to clients. It could be used to:
- Enumerate wallets
- Correlate activity across systems
- Target specific wallets for attacks

**Remediation:**
```python
def to_public_dict(self) -> dict:
    """Return only public-safe fields."""
    return {
        "id": self.id,
        "wallet_address": self.wallet_address,
        "chain": self.chain,
        "status": self.status,
        # Omit cdp_wallet_id, spending limits for non-owners
    }
```

---

### 🟡 MEDIUM: Job Descriptions May Contain PII

**Issue:** Job descriptions and application messages have no PII filtering. Clients might inadvertently include:
- Email addresses
- Phone numbers
- API keys
- Personal information

**Remediation:** Add PII detection warning or scrubbing:
```python
def warn_if_pii(text: str) -> Optional[str]:
    patterns = [
        r'\b[\w.-]+@[\w.-]+\.\w+\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        r'\b(?:sk|pk)_[a-zA-Z0-9]{20,}\b',  # API keys
    ]
    for p in patterns:
        if re.search(p, text):
            return "Warning: Message may contain sensitive information"
    return None
```

---

### 🟡 MEDIUM: Full Transition History Accessible

**Location:** `backend/supabase/migrations/022_job_state_transitions.sql`

**Issue:** The transition history includes metadata that might contain sensitive dispute reasons, internal notes, etc. No access control on reading transitions.

**Remediation:** Add RLS policy to restrict transition access to job participants.

---

## 5. Smart Contract Risks

### 🟠 HIGH: No Reentrancy Protection Visible

**Location:** `kernle/commerce/escrow/abi.py`

**Issue:** The escrow ABI shows `release()` and `refund()` functions that transfer tokens. Without seeing the Solidity code, we cannot confirm reentrancy guards exist.

**Critical Check:** Ensure the Solidity contract uses:
```solidity
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract KernleEscrow is ReentrancyGuard {
    function release() external nonReentrant {
        ...
    }
}
```

---

### 🟡 MEDIUM: Arbitrator Can Be Changed Without Timelock

**Location:** `kernle/commerce/escrow/abi.py:174`

```python
{
    "type": "function",
    "name": "setArbitrator",
    "inputs": [{"name": "_arbitrator", "type": "address"}],
    ...
}
```

**Issue:** If arbitrator can be changed instantly, a compromised admin could:
1. Change arbitrator to attacker address
2. Resolve all active disputes to attacker
3. Drain funds

**Remediation:** Implement timelock on admin functions:
```solidity
function setArbitrator(address _new) external onlyOwner {
    pendingArbitrator = _new;
    arbitratorChangeTime = block.timestamp + 7 days;
}

function confirmArbitrator() external onlyOwner {
    require(block.timestamp >= arbitratorChangeTime);
    arbitrator = pendingArbitrator;
}
```

---

### 🟢 LOW: No Circuit Breaker

**Issue:** No emergency pause functionality visible in escrow contract.

**Remediation:** Add pausable pattern for emergency situations.

---

## 6. Cryptographic Issues

### 🟡 MEDIUM: Deterministic Address Generation

**Location:** `kernle/commerce/escrow/service.py:178`

**Issue:** As noted above, escrow addresses are derived from job_id using UUID5 which is deterministic. Combined with public job IDs, this allows address prediction.

---

### 🟢 LOW: Hash Validation Regex Too Permissive

**Location:** `backend/app/routes/commerce/jobs.py:73`

```python
deliverable_hash: str | None = Field(None, pattern=r"^0x[a-fA-F0-9]{64}$")
```

**Issue:** Only validates format, not that it's a real hash of the deliverable. Client must trust worker's hash.

**Remediation:** Document that hash should be IPFS CID or verifiable; consider on-chain verification.

---

### 🟢 LOW: No Signature Verification for Sensitive Operations

**Issue:** Critical operations like wallet claiming don't require cryptographic proof of ownership. Adding EIP-712 signatures would strengthen security.

**Remediation:**
```python
class WalletClaimRequest(BaseModel):
    owner_eoa: str
    signature: str  # EIP-712 signature proving EOA ownership
```

---

## 7. Additional Findings

### ℹ️ INFO: Duplicate Class Definition

**Location:** `kernle/commerce/wallet/storage.py`

```python
class InMemoryWalletStorage:  # Defined twice
    ...

class InMemoryWalletStorage:  # Second definition overwrites first
    ...
```

**Issue:** The same class is defined twice. Python uses the last definition, but this is confusing and error-prone.

---

### ℹ️ INFO: TODOs in Production Code

Multiple TODO comments indicate incomplete features:
- Daily spending limit tracking
- CDP integration
- Actual blockchain transactions
- Arbitrator verification

These should be tracked in issue tracker, not left as TODOs.

---

### ℹ️ INFO: Inconsistent Error Handling

Some functions return `None` on failure while others raise exceptions. This inconsistency makes error handling difficult.

---

## Recommendations Summary

### Immediate Actions (🔴 CRITICAL)
1. **Add arbitrator verification** to `resolve_dispute`
2. **Fix race condition** in application acceptance with DB transactions
3. **Implement daily spending limit** tracking
4. **Add wallet ownership verification** to claim endpoint

### Short-term Actions (🟠 HIGH)
5. Fix SQL injection in skills search
6. Add reentrancy guards to escrow contract
7. Remove CDP wallet ID from public responses
8. Implement atomic job state transitions

### Medium-term Actions (🟡 MEDIUM)
9. Add URL validation for deliverables
10. Use Decimal consistently for money
11. Add PII detection warnings
12. Implement timelock on admin functions
13. Add circuit breaker to escrow

### Long-term Actions (🟢 LOW)
14. Implement EIP-712 signatures
15. Add progressive rate limiting
16. Track and close TODO items

---

## Appendix: Test Coverage Recommendations

See `tests/commerce/test_security.py` for exploit tests covering:
- Authorization bypass attempts
- Race condition tests
- Input validation fuzzing
- State machine bypass attempts
