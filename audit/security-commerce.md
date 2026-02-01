# Security Audit: Kernle Commerce & Wallet System

**Auditor:** Automated Security Review  
**Date:** 2026-02-01  
**Scope:** `kernle/commerce/` — wallet, escrow, jobs subsystems  
**Codebase State:** Pre-production (many stubs, CDP/Web3 not yet integrated)

---

## Executive Summary

The commerce system is in early development with most blockchain interactions stubbed out. The **design** shows good security awareness — atomic operations for wallet claims, reentrancy guards on escrow operations, per-job locks for application acceptance, authorization checks, and input validation. However, several significant vulnerabilities exist in the current code, and the stubbed nature of the blockchain integration creates a large surface of **deferred risk** that must be addressed before mainnet deployment.

**Finding Count:** 5 P0 (Critical) · 4 P1 (High) · 6 P2 (Medium) · 5 P3 (Low)

---

## P0 — Critical

### P0-1: No On-Chain Verification of Escrow State Before Service-Layer Transitions

**Files:** `escrow/service.py`, `jobs/service.py`  
**Description:** The escrow service methods (`release()`, `refund()`, `fund()`, `assign_worker()`, `mark_delivered()`) do not read the on-chain escrow contract status before executing state-changing transactions. The TODO stubs simply return `success=True` without verification.

When these are implemented, if the service layer doesn't **read and verify** the current on-chain status before sending a transaction, the following attacks become possible:

- Calling `release()` on an already-released escrow (double-spend)
- Calling `refund()` after a worker is assigned (theft from worker)
- Calling `fund()` on an escrow that's already funded (double-deposit, loss of client funds)

The reentrancy guards only protect against concurrent Python-level calls to the same EscrowService instance. They do **not** protect against direct on-chain calls or multi-instance deployments.

**Impact:** Complete loss of escrowed funds (double-release, unauthorized refund).  
**Recommendation:**
1. Every state-changing escrow method must read `escrow.functions.status().call()` and verify the expected state before building the transaction.
2. The Solidity contracts themselves must enforce state guards (they appear to via `InvalidState` error in ABI), but the Python layer should also verify to provide clear error messages and prevent wasted gas.
3. Add receipt verification — after sending a transaction, parse the receipt logs to confirm the expected event was emitted.

---

### P0-2: Transfer Succeeds Without Balance Check or Actual Blockchain Execution

**File:** `wallet/service.py` — `transfer()` method (lines ~280-320)  
**Description:** The transfer method records the daily spend atomically, but then the actual CDP transfer is stubbed as always-successful:

```python
# STUB: Simulate successful transfer
return TransferResult(
    success=True,
    ...
    tx_hash=f"0x{'0' * 64}",  # Stub tx hash
)
```

The daily spend has **already been incremented** before the stub return. There's a TODO comment about rolling back on failure, but the rollback mechanism (`_rollback_spend`) is **never defined**. This means:

1. If the real transfer fails, the daily spend is permanently consumed — the user loses spending capacity but no money moves.
2. No actual balance check occurs before the transfer (the `get_balance()` also returns stub zeros).

**Impact:** In production, failed transfers would silently consume spending limits. With the stub, any actor can appear to transfer unlimited funds.  
**Recommendation:**
1. Implement `_rollback_spend()` method to decrement daily spend on transfer failure.
2. Add balance verification before recording the spend.
3. Only record spend after confirmed on-chain execution (or use a pending/committed two-phase approach).

---

### P0-3: SupabaseWalletStorage.atomic_claim_wallet Always Returns True

**File:** `wallet/storage.py` — `SupabaseWalletStorage.atomic_claim_wallet()`  
**Description:** The Supabase implementation of `atomic_claim_wallet` is stubbed to always return `True`:

```python
def atomic_claim_wallet(self, wallet_id: str, owner_eoa: str) -> bool:
    ...
    return True  # STUB: Always succeeds
```

If the SupabaseWalletStorage is used in any deployment (even testing against Supabase), **any address can claim any wallet**, and already-claimed wallets can be re-claimed by different addresses.

**Impact:** Complete wallet takeover — any user can steal control of any agent's wallet.  
**Recommendation:** Implement the conditional UPDATE query shown in the docstring comment immediately. Until then, add a `raise NotImplementedError()` to prevent accidental use.

---

### P0-4: No Private Key Management or Signing Infrastructure

**Files:** `wallet/service.py`, `escrow/service.py`  
**Description:** Throughout the codebase, transaction building comments reference `private_key` for signing:

```python
# signed_tx = self._web3.eth.account.sign_transaction(tx, private_key)
```

But there is **no key management infrastructure** — no secure storage, no HSM integration, no key derivation, no encryption at rest. The CDP SDK is intended to handle this, but:

1. CDP credentials (`CDP_API_KEY`, `CDP_API_SECRET`) are loaded from plain environment variables with no rotation mechanism.
2. No mention of CDP's server-side signing — if the key is client-side, it would need secure storage.
3. The `cdp_wallet_id` is stored in the database with no encryption.

**Impact:** When real keys are introduced, improper handling could expose all wallet private keys.  
**Recommendation:**
1. Use CDP's server-side signing exclusively (never store private keys locally).
2. Encrypt `cdp_wallet_id` at rest in the database.
3. Implement CDP credential rotation.
4. Add audit logging for all signing operations.

---

### P0-5: No Transaction Receipt Verification

**Files:** `escrow/service.py` (all transaction methods)  
**Description:** The TODO comments show the intended pattern:

```python
# tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
# receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
```

But there's no code to verify:
- `receipt.status == 1` (transaction succeeded on-chain)
- Expected events were emitted in the receipt logs
- The receipt `to` address matches the expected contract

Without receipt verification, the service layer could believe a transaction succeeded when it was reverted on-chain.

**Impact:** Escrow state desync — service marks job as completed/funded/released but the on-chain state is unchanged. Funds stuck or double-spent.  
**Recommendation:** Implement a `_verify_receipt()` helper that checks status, parses expected events, and raises `TransactionFailedError` on any mismatch.

---

## P1 — High

### P1-1: Daily Spend Resets on Process Restart (In-Memory Fallback)

**File:** `wallet/service.py` — `_daily_spend` dict  
**Description:** The `InMemoryWalletStorage` returns `None` for `get_daily_spend()` and `increment_daily_spend()` in its first implementation, causing the service to fall back to its own in-memory `_daily_spend` dict. This dict:

1. Resets to zero on every process restart
2. Is not shared across multiple service instances (horizontal scaling)
3. Uses `threading.Lock` which doesn't protect across processes

An attacker could:
- Wait for a service restart, then immediately spend the full daily limit again
- If multiple instances exist, spend the daily limit once per instance

**Impact:** Daily spending limits can be bypassed, potentially draining wallet of `spending_limit_daily` multiple times per day.  
**Recommendation:**
1. Implement persistent daily spend tracking in Supabase/database as the **only** path (remove in-memory fallback for production).
2. Use database-level `SELECT FOR UPDATE` for atomic increment (the SQL is already drafted in the Supabase storage comments).
3. Add an integration test that verifies limits survive restarts.

---

### P1-2: Escrow-Job State Synchronization Gap

**Files:** `jobs/service.py`, `escrow/service.py`  
**Description:** The job service and escrow service are **not transactional** together. The flow is:

1. `EscrowService.release(escrow_addr)` — releases on-chain funds
2. `JobService.approve_job(job_id)` — marks job completed in database

If step 1 succeeds but step 2 fails (e.g., database error, process crash), the funds are released but the job is never marked completed. There's no reconciliation mechanism.

Similarly, `fund_job()` takes an `escrow_address` parameter with no verification that the address actually corresponds to a funded escrow contract.

**Impact:** Funds released but job stuck in "delivered" state (worker can't prove completion). Or: job marked as funded with a fake/unfunded escrow address.  
**Recommendation:**
1. Implement an event-driven reconciliation: the `EscrowEventMonitor` should listen for `Released` events and auto-complete the corresponding job.
2. In `fund_job()`, verify on-chain that the escrow at `escrow_address` is in FUNDED status and matches the job's budget.
3. Add a periodic background job that compares on-chain escrow states with database job states and flags discrepancies.

---

### P1-3: No Rate Limiting on Wallet/Job Operations

**Files:** `wallet/service.py`, `jobs/service.py`  
**Description:** There is no rate limiting at the service layer for:
- `create_wallet()` — could spam wallet creation (each costs gas and CDP resources)
- `create_job()` — could flood the marketplace with spam jobs
- `apply_to_job()` — one agent per job limit exists, but one agent could apply to thousands of jobs
- `transfer()` — per-tx and daily limits exist, but there's no transactions-per-minute limit

**Impact:** Resource exhaustion, marketplace spam, gas drainage.  
**Recommendation:** Add rate limiting at the API/service layer (e.g., max N wallet creations per user per hour, max N job postings per day).

---

### P1-4: InMemoryWalletStorage.atomic_claim_wallet Not Actually Atomic Under Threading

**File:** `wallet/storage.py` — first `InMemoryWalletStorage` class  
**Description:** The comment says "In-memory is single-threaded, so this is atomic" but the `WalletService` uses `threading.Lock` for daily spend, implying multi-threaded usage is expected. The `atomic_claim_wallet` in `InMemoryWalletStorage` has no lock:

```python
def atomic_claim_wallet(self, wallet_id, owner_eoa):
    wallet = self._wallets.get(wallet_id)
    if wallet.owner_eoa is not None:
        return False  # Already claimed
    # TOCTOU gap here under threading
    wallet.owner_eoa = owner_eoa
```

Under concurrent requests, two threads could both read `owner_eoa is None`, pass the check, and both write — last writer wins.

**Impact:** Wallet ownership can be stolen via race condition in testing/dev (which may mask bugs that reach production).  
**Recommendation:** Add a `threading.Lock` to the in-memory storage's claim operation, or use the second `InMemoryWalletStorage` class (which also lacks the lock but at least has daily spend locking as a pattern).

Note: There are **two** `InMemoryWalletStorage` classes in `storage.py` — this is itself a bug (see P3-1).

---

## P2 — Medium

### P2-1: Spending Limits Use `float` Type — Precision Loss

**Files:** `wallet/models.py`, `wallet/service.py`, `config.py`  
**Description:** Spending limits (`spending_limit_per_tx`, `spending_limit_daily`) are stored as Python `float`, while amounts use `Decimal`. The comparison in `transfer()`:

```python
if float(amount) > wallet.spending_limit_per_tx:
```

This mixes `float` and `Decimal`, which can cause precision errors. For example:
```python
>>> float(Decimal("0.1") + Decimal("0.2")) == 0.3
False  # It's 0.30000000000000004
```

**Impact:** Transactions near the spending limit boundary could be incorrectly accepted or rejected.  
**Recommendation:** Change `spending_limit_per_tx` and `spending_limit_daily` to `Decimal` throughout. Use `Decimal` comparisons exclusively.

---

### P2-2: No Checksum Validation on Ethereum Addresses

**Files:** `wallet/models.py`, `jobs/models.py`, `wallet/service.py`  
**Description:** Address validation checks length and hex format but not EIP-55 checksums. A typo in an address (e.g., `0xdead...` vs `0xDeaD...`) would pass validation. Sending funds to an invalid checksum address is irrecoverable.

**Impact:** Funds sent to mistyped addresses are permanently lost.  
**Recommendation:** Implement EIP-55 checksum validation using `web3.Web3.is_checksum_address()` or equivalent. Accept lowercase/uppercase but warn on non-checksummed.

---

### P2-3: No Tithe/Fee Mechanism — Potential for Manipulation When Added

**Files:** All commerce files  
**Description:** The codebase has **no tithe/reverse-tithe/fee** implementation despite the project docs referencing it. When this is added, common vulnerabilities include:
- Fee calculated client-side and passed as parameter (attacker sets fee to 0)
- Fee calculated on `budget_usdc` (float) instead of actual escrow amount
- Rounding errors that can be exploited over many small transactions
- Fee not enforced at the smart contract level (bypassed by direct contract interaction)

**Impact:** Revenue loss or fee manipulation when feature is implemented.  
**Recommendation:** When implementing tithe:
1. Calculate exclusively server-side (never trust client-provided fee amounts)
2. Use `Decimal` with explicit rounding (`ROUND_HALF_UP` or `ROUND_CEILING`)
3. Enforce in the Solidity contract (immutable split at release time)
4. Add invariant tests: `worker_payout + tithe == escrow_amount`

---

### P2-4: Job `cancel_job` Allows Cancellation After Worker Accepted

**File:** `jobs/service.py`, `jobs/models.py`  
**Description:** The state transition table allows:
```python
JobStatus.ACCEPTED: {JobStatus.DELIVERED, JobStatus.DISPUTED, JobStatus.CANCELLED}
```

A client can cancel a job **after** a worker has been accepted and started working. The service only checks `job.client_id != actor_id` but doesn't verify whether the cancellation is fair to the worker. No partial payment or dispute is forced.

**Impact:** Client can cancel a job after the worker has invested time, getting a full refund and paying the worker nothing.  
**Recommendation:**
1. Remove `CANCELLED` from `ACCEPTED` transitions, or
2. Force the cancellation through the dispute flow when a worker is assigned, or
3. Implement partial payment (e.g., pro-rata based on time elapsed vs deadline)

---

### P2-5: `system` Actor Bypasses All Arbitrator Checks

**File:** `jobs/service.py` — `_is_authorized_arbitrator()`  
**Description:**
```python
if actor_id == "system":
    return True
```

Any caller that can pass `actor_id="system"` can resolve disputes. If API routes don't sanitize `actor_id`, an external user could claim to be `"system"`.

**Impact:** Unauthorized dispute resolution — attacker redirects escrow funds.  
**Recommendation:**
1. Never accept `"system"` as an actor_id from external input.
2. Add an internal-only flag or use a separate code path for system-initiated actions.
3. Validate at the API layer that `actor_id` matches authenticated session.

---

### P2-6: Duplicate Class Definitions Create Import Ambiguity

**File:** `wallet/storage.py`, `jobs/storage.py`  
**Description:** Both files define `InMemoryWalletStorage` / `InMemoryJobStorage` **twice** — a simpler version followed by a more complete version. Python will use the **last** definition, but:
1. The first definition may be imported by tests that specifically reference it
2. IDE analysis and type checkers may flag the first one
3. The two implementations have different behaviors (first InMemoryWalletStorage lacks daily spend tracking, second has it)

**Impact:** Wrong storage implementation used in tests, potentially hiding bugs.  
**Recommendation:** Remove the duplicate class definitions. Keep only the more complete version.

---

## P3 — Low

### P3-1: Stub Transaction Hashes Are Deterministic

**Files:** `escrow/service.py`, `wallet/service.py`  
**Description:** Stub tx hashes are all `0x` + 64 zeros. If any verification code checks for unique tx hashes, or if these stubs leak into a database that later expects real hashes, it creates confusion.

**Impact:** Low — only affects development, but could mask bugs in tx verification logic.  
**Recommendation:** Generate random stub hashes: `f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}"`.

---

### P3-2: Config Secrets Loaded from Environment with No Validation

**File:** `config.py`  
**Description:** `CDP_API_KEY` and `CDP_API_SECRET` are loaded from env vars with no format validation. If a user sets a malformed key, the error won't surface until a CDP API call is made, potentially mid-transaction.

**Impact:** Debugging difficulty, potential mid-operation failures.  
**Recommendation:** Validate credential format on config load. Log warnings for missing credentials at startup.

---

### P3-3: `_job_id_to_bytes32` Uses SHA-256 — Not Reversible

**File:** `escrow/service.py`  
**Description:** Job IDs are hashed to `bytes32` via SHA-256 for use as Solidity `bytes32`. This means you cannot derive the job ID from the on-chain `bytes32` value — you must maintain an off-chain mapping. If the database is lost, the link between on-chain escrows and jobs is severed.

**Impact:** Disaster recovery difficulty.  
**Recommendation:** Store the `bytes32` mapping in the job record. Alternatively, if job IDs are UUIDs (16 bytes), pad to 32 bytes instead of hashing — this preserves reversibility.

---

### P3-4: No Pagination Bounds on Job/Application Listing

**Files:** `jobs/service.py`, `jobs/storage.py`  
**Description:** `list_jobs(limit=100)` and `list_applications(limit=100)` have defaults but no maximum enforcement. A caller can set `limit=1000000` and fetch the entire database.

**Impact:** Memory exhaustion, slow queries.  
**Recommendation:** Enforce `MAX_LIMIT = 100` (or similar) and clamp user-provided values.

---

### P3-5: Wallet Address Generation Is Deterministic from Agent ID

**File:** `wallet/service.py` — `create_wallet()` stub  
**Description:** The stub generates wallet addresses deterministically from `agent_id`:
```python
base_hash = uuid.uuid5(uuid.NAMESPACE_DNS, agent_id).hex
```

If this stub is ever used in a context where the address is treated as a real address, anyone can predict wallet addresses for any agent.

**Impact:** Low (stub only), but if stub addresses leak into production data, could cause confusion.  
**Recommendation:** Use `uuid.uuid4()` for stub addresses, or clearly mark stub addresses with a prefix like `0xDEAD`.

---

## Deferred Risk Summary

The following areas are **entirely stubbed** and represent major security surface that must be audited when implemented:

| Area | Risk | Stub Location |
|------|------|---------------|
| CDP Wallet Creation | Key generation, seed phrase handling | `wallet/service.py:create_wallet()` |
| CDP Transfer Signing | Private key exposure, replay attacks | `wallet/service.py:transfer()` |
| Web3 Contract Calls | ABI encoding errors, gas manipulation | `escrow/service.py:*` |
| On-Chain Balance Check | False balance reporting | `wallet/service.py:get_balance()` |
| Supabase Storage | SQL injection, RLS bypass | `wallet/storage.py`, `jobs/storage.py` |
| Event Parsing | Log spoofing via crafted events | `escrow/events.py:parse_log()` |
| USDC Allowance Check | Insufficient allowance not caught | `escrow/service.py:fund()` |

---

## Recommendations Priority

### Immediate (Before Any Testnet Deployment)
1. **P0-3:** Fix SupabaseWalletStorage stubs to raise `NotImplementedError`
2. **P0-2:** Implement `_rollback_spend()` and balance checking
3. **P2-6:** Remove duplicate class definitions
4. **P2-5:** Harden `"system"` actor_id handling

### Before Testnet with Real Tokens
5. **P0-1:** Implement on-chain state verification in all escrow methods
6. **P0-5:** Implement receipt verification
7. **P1-1:** Implement persistent daily spend tracking
8. **P1-2:** Build escrow↔job reconciliation via event monitoring
9. **P2-1:** Migrate spending limits from `float` to `Decimal`

### Before Mainnet
10. **P0-4:** Full key management audit once CDP is integrated
11. **P2-2:** Add EIP-55 checksum validation
12. **P2-3:** Design and audit tithe mechanism with smart contract enforcement
13. **P2-4:** Redesign cancellation policy for accepted jobs
14. **P1-3:** Implement rate limiting

---

*End of audit report.*
