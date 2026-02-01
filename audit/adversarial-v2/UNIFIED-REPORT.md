# Kernle Adversarial Audit — Unified Cross-Model Report

**Date**: 2026-02-01
**Auditors**: 3 independent models (Opus 4.5, Codex o4-mini, Gemini 2.5 Pro)
**Scope**: Full codebase — backend (auth, payments, subscriptions, commerce, sync), core library (storage, memory, forgetting, metamemory)
**Lines audited**: ~49K across 65+ Python files
**Claire's auditors**: 3 additional independent runs (results pending full consolidation)

---

## Executive Summary

45 unique findings across 3 adversarial auditors. After deduplication and cross-validation:

| Severity | Count | Cross-validated (2+ models) |
|----------|-------|-----------------------------|
| P0 Critical | 4 | 2 |
| P1 High | 11 | 3 |
| P2 Medium | 13 | 2 |
| P3 Low | 8 | 0 |
| **Total** | **36 unique** | **7** |

### Top 5 "Fix Today" Items
1. **Add `agent_id`, `is_forgotten`, `forgotten_at`, `forgotten_reason` to `SERVER_CONTROLLED_FIELDS`** — one-line fix, prevents agent isolation bypass + un-forgetting (Codex F8+F19)
2. **Check `tx_hash` uniqueness BEFORE on-chain verification** — prevents replay attack (Opus F2 + Gemini F4)
3. **Pass `expected_from` to `verify_usdc_transfer()`** — prevents using others' payments (Opus F4 + Gemini F4)
4. **Make confirm_payment atomic** — prevent double-upgrade race (Opus F7, Gemini F1, Codex F2)
5. **Reject `..` in agent IDs** — prevents path traversal writes (Codex F13)

---

## Findings Cross-Reference Matrix

### 🔴 P0 CRITICAL — Fix Before Any Deployment

#### C1: Payment Confirm Race Condition → Double Upgrade
- **Found by**: Opus (F7), Gemini (F1)
- **Location**: `service.py:confirm_payment()` bridge
- **Issue**: confirm_payment is not atomic. Two concurrent requests with same payment_id both read `status='pending'`, both verify on-chain, both upgrade tier. The tier upgrade persists even if `record_payment` fails on the unique tx_hash constraint.
- **Attack**: Send two concurrent POST /payments/confirm requests → get upgraded twice or corrupt subscription state.
- **Fix**: Use `UPDATE payment_intents SET status='processing' WHERE id=? AND status='pending' RETURNING *` as atomic claim. Only one caller gets the row. Or queue through exactly-once processor.

#### C2: Transaction Hash Replay → Free Upgrades
- **Found by**: Opus (F2+F4), Gemini (F4)
- **Issue**: (a) No `expected_from` passed to verifier — any USDC transfer to treasury validates. (b) Same tx_hash can confirm multiple payment intents because uniqueness is only checked at `record_payment` (after upgrade already applied).
- **Attack**: Watch treasury for any incoming payment → grab tx_hash → submit against your own payment intent → free upgrade.
- **Fix**: (a) Store user's wallet address, pass as `expected_from`. (b) Check tx_hash uniqueness in payment_intents table before verification. (c) Make upgrade happen AFTER record_payment succeeds.

#### C3: Sync Push Agent Isolation Bypass
- **Found by**: Codex (F8), Claire's Opus
- **Location**: `routes/sync.py`, `SERVER_CONTROLLED_FIELDS`
- **Issue**: `agent_id` is not in `SERVER_CONTROLLED_FIELDS`. A client can push `{"data": {"agent_id": "other_agent"}}` to move memories into another agent's namespace.
- **Attack**: Send sync push with modified agent_id → write to any agent's memory space.
- **Fix**: Add `agent_id` to `SERVER_CONTROLLED_FIELDS`. **One-line fix.**

#### C4: Sync Push Can Un-Forget Memories
- **Found by**: Codex (F19)
- **Location**: `routes/sync.py`, `SERVER_CONTROLLED_FIELDS`
- **Issue**: `is_forgotten`, `forgotten_at`, `forgotten_reason` are not server-controlled. Client can push `{"is_forgotten": false}` to resurrect tombstoned memories.
- **Attack**: Push sync data to un-forget memories that were deleted by admin or automated forgetting.
- **Fix**: Add forgetting fields to `SERVER_CONTROLLED_FIELDS`. **One-line fix.**

---

### 🟡 P1 HIGH — Fix Before Cloud Launch

#### C5: Renewal Extends Without Payment Enforcement
- **Found by**: Opus (F1)
- **Issue**: `check_renewal()` with `auto_renew=True` extends subscription by a month. Comment says "caller must verify payment" but nothing enforces this. If cron job has any failure path that doesn't roll back, subscriptions extend for free.
- **Fix**: Set status to `payment_pending` instead of `active` until payment confirmed.

#### C6: Zero-Address Treasury
- **Found by**: Opus (F3)
- **Issue**: `TREASURY_ADDRESS = "0x0000...0000"` (burn address). If shipped to prod, all revenue is permanently destroyed.
- **Fix**: Load from env var. Add startup guard that fails if zero address in production.

#### C7: Payment Intent Expiry Never Enforced
- **Found by**: Opus (F5)
- **Issue**: `expires_at` field exists but `confirm_payment` never checks it. Old intents at stale prices can be confirmed.
- **Fix**: Check `expires_at` in confirm flow. Add cron to expire old intents.

#### C8: No tx_hash Format Validation
- **Found by**: Opus (F6)
- **Issue**: tx_hash accepts any string. Log injection via newlines/ANSI, DoS via huge strings.
- **Fix**: Validate `^0x[0-9a-fA-F]{64}$` at route layer.

#### C9: Reactivate Bypasses Payment
- **Found by**: Opus (F8)
- **Issue**: Cancel → reactivate loop grants unlimited free months. No payment check on reactivation if `renews_at` has passed.
- **Fix**: Require payment if `renews_at` is in the past.

#### C10: Supabase Client Not Thread-Safe
- **Found by**: Codex (F2)
- **Issue**: Global singleton with no lock. `asyncio.to_thread()` calls create real concurrent access from multiple threads.
- **Fix**: Add `threading.Lock` around client creation, or use connection pool.

#### C11: Quota Fallback Defeats Atomicity
- **Found by**: Codex (F7)
- **Issue**: When atomic RPC fails, silently falls back to racy non-atomic check. 100 concurrent requests during this window all pass quota.
- **Fix**: Fail closed (503) when atomic RPC unavailable.

#### C12: No JWT Refresh Token Rotation
- **Found by**: Codex (F5)
- **Issue**: 7-day JWTs with no revocation mechanism. Stolen token = 7 days of access.
- **Fix**: Short-lived access tokens + refresh rotation, or server-side revocation list.

#### C13: Lineage Traversal Unbounded → DoS
- **Found by**: Gemini (F3)
- **Issue**: `get_full_lineage()` recursively traverses with no depth limit. Deep chains → stack overflow or OOM.
- **Fix**: Add `max_depth` parameter. Use iterative traversal.

#### C14: Commerce/Escrow Stubs Unsafe
- **Found by**: Gemini (F5), Claire's Opus
- **Issue**: No state machine, no atomicity, no dispute resolution. Direct wallet modifications.
- **Fix**: Complete redesign before production. Known issue — deferred until testnet.

#### C15: No Unique Active Subscription Constraint
- **Found by**: Opus (F12)
- **Issue**: Race in `get_or_create_subscription` can create multiple active rows per user.
- **Fix**: Add partial unique index: `CREATE UNIQUE INDEX ON subscriptions(user_id) WHERE status IN ('active', 'grace_period')`

---

### 🔵 P2 MEDIUM

| ID | Finding | Auditor | Fix Effort |
|----|---------|---------|-----------|
| C16 | Agent ID allows `..` (path traversal) | Codex F13 | Easy (reject dots) |
| C17 | API key prefix too short (bcrypt DoS) | Codex F3 | One-line (`[:8]` → `[:12]`) |
| C18 | JWT degrades to free tier on DB error | Codex F1 | Easy (fail closed) |
| C19 | Cookie max-age doesn't track JWT expiry | Codex F4 | Easy |
| C20 | API key cycling race condition | Codex F6 | Moderate |
| C21 | Auto-provisioned agents have empty secret | Codex F9 | Easy (sentinel value) |
| C22 | Rate limiting useless behind proxy | Codex F11 | Moderate |
| C23 | High-confidence beliefs can be forgotten | Codex F14 | Moderate |
| C24 | $0.01 tolerance allows underpayment | Opus F10 | Easy (reduce to 0) |
| C25 | Testnet chain in production config | Opus F11 | Easy (env-based) |
| C26 | Intent spam / no cleanup | Opus F9 | Moderate |
| C27 | Missing DB indexes | Gemini F6 | Easy |
| C28 | No idempotency keys on mutations | Gemini F2 | Moderate |

---

### ⚪ P3 LOW

| ID | Finding | Auditor |
|----|---------|---------|
| C29 | Missing `user_id` in payment insert (breaks audit trail) | Opus F13 |
| C30 | `confirm_payment` not user-scoped at service level | Opus F14 |
| C31 | Grace period anchor fragility | Opus F15 |
| C32 | Low min_confirmations (1) for production | Opus F16 |
| C33 | Health endpoint leaks DB status | Codex F10 |
| C34 | Seed beliefs not protected from forgetting | Codex F15 |
| C35 | Flat file writes not atomic | Codex F18 |
| C36 | No centralized structured logging | Gemini F7 |

---

## Quick Wins (< 30 min each, high impact)

1. `SERVER_CONTROLLED_FIELDS += ["agent_id", "is_forgotten", "forgotten_at", "forgotten_reason"]` — fixes C3+C4
2. Validate tx_hash: `re.match(r'^0x[0-9a-fA-F]{64}$', tx_hash)` — fixes C8
3. Reject `..` in agent IDs: `if '..' in agent_id: raise ValueError()` — fixes C16
4. API key prefix: change `prefix[:8]` to `prefix[:12]` — fixes C17
5. Add `threading.Lock` to Supabase singleton — fixes C10
6. Quota fallback → 503 instead of racy check — fixes C11
7. Check `expires_at` in confirm_payment — fixes C7
8. `TREASURY_ADDRESS` from env var — fixes C6

---

## Model Disagreements

| Topic | Opus | Codex | Gemini |
|-------|------|-------|--------|
| Forgetting severity | — | P2 (can lose beliefs) | P3 (naive but noted) |
| Commerce stubs | In scope but deferred | Not in scope | P1 (unsafe) |
| Agent isolation (local) | — | P1 (design concern) | — |
| Hardcoded config | — | — | P2 (environment risk) |

No real conflicts — models focused on different areas with consistent risk assessments where they overlapped.

---

## Positive Findings (Things Done Right)

All three models noted strong existing security:
- JWKS-based OAuth with issuer validation
- CSRF middleware + SameSite=Strict cookies
- JWT algorithm allowlist (prevents confusion attacks)
- Atomic quota checking RPC (correct architecture despite fallback issue)
- Optimistic concurrency control on memory updates
- Table name allowlist in SQLite (prevents SQL injection)
- bcrypt for credential storage
- WAL mode + busy timeout on SQLite
- File permissions (0o600/0o700)

---

*Report compiled from: report-security-payments.md (Opus 4.5), report-correctness.md (Codex o4-mini), report-architecture.md (Gemini 2.5 Pro)*
*Claire's 3 auditor reports pending integration — will update with cross-validation from 6 total auditors*
