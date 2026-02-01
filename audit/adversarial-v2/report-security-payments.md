# Kernle Payment & Subscription Security Audit

**Auditor**: Adversarial Security Review  
**Date**: 2026-02-01  
**Scope**: `payments/verification.py`, `subscriptions/service.py`, `subscriptions/models.py`, `routes/subscriptions.py`, migration `024_subscriptions.sql`  
**Risk Model**: Assume sophisticated attacker with on-chain capabilities, valid auth tokens, and knowledge of the codebase.

---

## Executive Summary

The payment system has a **critical vulnerability** (upgrade-before-payment), multiple **high-severity** issues around payment verification gaps, and several medium/low issues. The most dangerous pattern is that the two-phase payment flow has race conditions and missing atomicity that allow tier activation without confirmed payment under certain conditions.

**Critical**: 1 | **High**: 5 | **Medium**: 5 | **Low**: 4

---

## F1: Upgrade Applied Before Payment in `upgrade_tier()` — Free Tier Upgrade Without Paying

- **Severity**: P0 (critical)
- **Location**: `service.py:upgrade_tier()` (lines ~130-170) and `service.py:confirm_payment()` (bridge, lines ~480-550)
- **Issue**: `upgrade_tier()` immediately writes the new tier to the database (`tier: new_tier.value, status: active`) and returns payment info. The `confirm_payment` bridge function calls `upgrade_tier()` only *after* on-chain verification succeeds, which is correct — BUT `upgrade_tier()` is a public static method that can also be called by the route layer's `create_upgrade_payment()` flow. While `create_upgrade_payment()` itself doesn't call `upgrade_tier()`, the real danger is the **`reactivate_subscription`** path: a user can cancel → reactivate and get `status: active` restored *without any payment check*. More critically, `check_renewal()` with `auto_renew=True` extends the subscription by a month with the comment "caller must verify payment succeeded" — but there's no enforcement mechanism. If the renewal cron job doesn't actually collect payment, the subscription silently extends.
- **Exploit**: 
  1. User upgrades to Core (pays once).
  2. At renewal time, if the cron job has any failure path that doesn't roll back `check_renewal()`, the subscription auto-extends for free.
  3. Alternatively: the `check_renewal` method sets `renews_at` to next month with `auto_renew=True` and `status=active` — the payment verification is entirely the caller's responsibility with zero enforcement.
- **Fix**: 
  - `check_renewal()` should set status to `payment_pending` instead of `active` until payment is confirmed.
  - Add a `payment_verified` gate: subscription should not be `active` at a paid tier without a corresponding `confirmed` payment for the current period.
  - Consider a DB constraint or trigger that prevents `tier != 'free' AND status = 'active'` without a matching confirmed payment.

---

## F2: Transaction Replay — Same `tx_hash` Can Confirm Multiple Payment Intents

- **Severity**: P0 (critical)  
- **Location**: `service.py:confirm_payment()` (bridge function, ~line 480)
- **Issue**: The `confirm_payment` bridge function looks up a payment intent by `payment_id` and `user_id`, then verifies the `tx_hash` on-chain. However, there is **no check that the `tx_hash` hasn't already been used for a different payment intent**. The `subscription_payments` table has a `UNIQUE` constraint on `tx_hash`, which would prevent duplicate *payment records*, but the flow is: verify → update intent → upgrade tier → record payment. If `record_payment` fails due to the unique constraint, the tier upgrade has **already been applied**. The upgrade is not rolled back.
- **Exploit**:
  1. Attacker creates two payment intents (e.g., for Core and Pro, or two Core intents for two accounts if they control multiple).
  2. Makes one real USDC payment, gets `tx_hash`.
  3. Confirms first intent — succeeds, tier upgraded, payment recorded.
  4. Confirms second intent with same `tx_hash` — on-chain verification passes (it's a real tx), intent is marked confirmed, tier is upgraded. When `record_payment` tries to insert a duplicate `tx_hash`, it may fail, but the upgrade is already persisted.
- **Fix**:
  - **Before** on-chain verification, check that `tx_hash` doesn't exist in `subscription_payments` or `payment_intents` (confirmed).
  - Make the entire confirm flow atomic: verify → record payment → upgrade tier, and roll back if any step fails.
  - Add `tx_hash` UNIQUE constraint to `payment_intents` table as well.

---

## F3: Zero-Address Treasury — All Payments Sent to Burn Address

- **Severity**: P1 (high)
- **Location**: `service.py:TREASURY_ADDRESS` (line ~30)
- **Issue**: `TREASURY_ADDRESS = "0x0000000000000000000000000000000000000000"` with a TODO comment. The zero address is the canonical burn address on Ethereum. Any USDC sent there is **irrecoverably lost**. While marked as TODO, this is the value that would ship if the TODO is missed. The verification code checks `expected_to=TREASURY_ADDRESS`, so the system is literally configured to verify that users burn their money.
- **Exploit**: Not an attacker exploit per se, but: (1) if this ships to production, all subscription revenue is permanently destroyed; (2) an attacker who notices this could social-engineer users by pointing out "Kernle is sending your money to a burn address" to destroy trust; (3) USDC transfers to 0x0 may behave unexpectedly on some contracts (some ERC20s block transfers to zero address), meaning payments might *always fail* and nobody can upgrade.
- **Fix**:
  - Set the treasury to a real multisig address immediately.
  - Add a startup check that fails loudly if `TREASURY_ADDRESS` is the zero address in production.
  - Load from environment variable, not hardcoded.

---

## F4: No `expected_from` Verification — Anyone's Payment Can Upgrade Your Account

- **Severity**: P1 (high)
- **Location**: `service.py:confirm_payment()` bridge (~line 505)
- **Issue**: When calling `verify_usdc_transfer()`, only `expected_to` (treasury) and `expected_amount` are passed. `expected_from` is **not provided**. This means any USDC transfer to the treasury of the right amount on the right chain will validate — regardless of who sent it.
- **Exploit**:
  1. Attacker watches the mempool/treasury for any incoming USDC transfer of $5 or $15.
  2. Someone else pays for their own upgrade (or any other reason sends USDC to treasury).
  3. Attacker grabs that `tx_hash` and submits it against their own payment intent.
  4. Verification passes (right amount, right recipient, no sender check).
  5. Attacker gets a free upgrade using someone else's payment.
  
  This is especially bad combined with F2 (replay): one legitimate payment can upgrade multiple attackers.
- **Fix**:
  - Store the user's wallet address in the payment intent (require it at intent creation).
  - Pass `expected_from` to `verify_usdc_transfer()`.
  - Alternatively, use unique payment amounts (e.g., $5.000001, $5.000002) as nonces.

---

## F5: No Payment Intent Expiry Enforcement

- **Severity**: P1 (high)
- **Location**: `service.py:confirm_payment()` bridge (~line 480), `024_subscriptions.sql`
- **Issue**: Payment intents have an `expires_at` field (set to 24h), but the `confirm_payment` function **never checks it**. An expired payment intent can still be confirmed. The DB constraint on `payment_intents.status` includes `'expired'` but nothing automatically expires them, and even if status were set to `'expired'`, the confirm flow only checks for `status == "confirmed"` (early return) — it doesn't reject expired/cancelled intents.
- **Exploit**:
  1. Create a payment intent, wait weeks/months.
  2. During that time, the tier pricing could change, or the user's subscription state could change.
  3. Confirm the stale intent with a valid tx_hash, getting an upgrade at the old price.
  4. Or: accumulate many intents as "options" and selectively confirm whichever is most advantageous.
- **Fix**:
  - Check `expires_at` in `confirm_payment()` and reject expired intents.
  - Add a cron job or DB function to automatically set `status='expired'` for overdue intents.
  - Reject intents with status other than `'pending'`.

---

## F6: No Validation of `tx_hash` Format — Injection Risk

- **Severity**: P1 (high)
- **Location**: `verification.py:verify_usdc_transfer()` (~line 160), `routes/subscriptions.py:PaymentConfirmRequest`
- **Issue**: The `tx_hash` field accepts any string. There is no validation that it's a valid 66-character hex string (`0x` + 64 hex chars). The raw value is passed directly to `_rpc_call()` as a JSON-RPC parameter and stored in the database. While JSON-RPC *should* reject invalid params, malformed inputs could: (1) cause unexpected RPC errors that hit the exception handler and return `pending_verification` status, (2) be stored in the DB as-is (SQL injection is unlikely with Supabase client, but the value is unsanitized), (3) cause log injection via the `logger` calls that interpolate `tx_hash`.
- **Exploit**:
  - Submit `tx_hash` containing newlines or ANSI escape sequences → log injection/forgery.
  - Submit extremely long strings → potential DoS on RPC call or DB storage.
  - Submit `tx_hash` of a transaction on a different chain that happens to have the right structure → cross-chain confusion (mitigated by chain-specific RPC, but not validated).
- **Fix**:
  - Validate `tx_hash` matches `^0x[0-9a-fA-F]{64}$` at the route layer.
  - Sanitize before logging.
  - Add a Pydantic validator on `PaymentConfirmRequest.tx_hash`.

---

## F7: Race Condition — Double-Confirm Payment Intent

- **Severity**: P2 (medium)
- **Location**: `service.py:confirm_payment()` bridge (~line 480-550)
- **Issue**: The confirm flow is: read intent → verify on-chain → update intent → upgrade tier → record payment. There's no locking or atomic compare-and-swap. Two concurrent requests with the same `payment_id` and `tx_hash` can both read the intent as `status='pending'`, both pass verification, and both attempt to upgrade. The early-return check (`if intent.get("status") == "confirmed"`) is a TOCTOU race — the status can change between the read and the update.
- **Exploit**:
  1. Send two concurrent `POST /subscriptions/payments/confirm` requests with the same `payment_id` and `tx_hash`.
  2. Both read `status='pending'`, both verify on-chain (passes for both), both upgrade.
  3. The `record_payment` unique constraint on `tx_hash` will cause one to fail, but the tier upgrade from `upgrade_tier()` has already been applied by both.
  4. This can corrupt state: e.g., `starts_at` and `renews_at` get overwritten by the second call.
- **Fix**:
  - Use `UPDATE ... WHERE status = 'pending' RETURNING *` as an atomic claim — only one caller gets the row.
  - Or use a Supabase RPC function with `SELECT ... FOR UPDATE`.
  - Process payment confirmations through a queue with exactly-once semantics.

---

## F8: `reactivate_subscription` Bypasses Payment Verification

- **Severity**: P2 (medium)
- **Location**: `service.py:reactivate_subscription()` (~line 580), `routes/subscriptions.py:reactivate_subscription`
- **Issue**: The reactivate function sets `auto_renew=True, status='active', cancelled_at=None` with no payment check. The route only verifies the subscription status is `'cancelled'`. A user who cancels their subscription can reactivate indefinitely without paying — the system just re-enables auto_renew and sets status to active. If `renews_at` is still in the future, they get continued access. If `renews_at` is in the past, the next renewal check extends them (per F1).
- **Exploit**:
  1. Pay for Core once ($5).
  2. Cancel before renewal.
  3. Wait until near expiry, then reactivate.
  4. The subscription is now `active` with `auto_renew=True`.
  5. Renewal check extends by another month (no payment enforced per F1).
  6. Repeat: cancel → reactivate → free months forever.
- **Fix**:
  - Reactivation should only work if `renews_at` is still in the future (i.e., the user already paid for the current period).
  - If `renews_at` is past, require a new payment before reactivation.
  - Add a check: `if sub.renews_at and sub.renews_at < now: require_payment()`.

---

## F9: No Rate Limit on Payment Intent Creation — DoS via Intent Spam

- **Severity**: P2 (medium)
- **Location**: `service.py:create_upgrade_payment()`, `routes/subscriptions.py:upgrade_subscription`
- **Issue**: The upgrade route has a rate limit of `5/minute`, but an attacker can create 5 payment intents per minute (7,200/day) across multiple accounts. Each intent is stored in `payment_intents` with no cleanup. There's no limit on total pending intents per user, and expired intents are never cleaned up (see F5). This can fill the `payment_intents` table.
- **Exploit**:
  - Script that creates accounts and spams upgrade requests → table bloat, potential DB performance degradation.
  - Create thousands of pending intents to make `get_payment` queries slower over time.
- **Fix**:
  - Limit pending intents per user (e.g., max 3 active intents).
  - Clean up expired intents via cron.
  - Stricter rate limit on upgrade endpoint (e.g., `2/hour`).

---

## F10: Tolerance Allows Underpayment of Up to $0.01

- **Severity**: P2 (medium)
- **Location**: `verification.py:verify_usdc_transfer()` — `tolerance` parameter (default `Decimal("0.01")`)
- **Issue**: The tolerance allows a payment of $4.99 to be accepted for a $5.00 tier. Over thousands of users, this adds up. More importantly, the tolerance is applied as `abs(actual - expected) <= tolerance`, meaning overpayment is also silently accepted (less of a security issue, but a UX concern — users may overpay and not be refunded).
- **Exploit**: 
  - Always pay $0.01 less than required. At scale: 1000 users × $0.01 = $10/month in free money. Individually minor, but sets a bad precedent.
  - USDC has 6 decimals, so precision attacks aren't meaningful, but the tolerance is unnecessarily generous for a 6-decimal token where exact amounts are trivially possible.
- **Fix**:
  - For exact-amount payments (subscription tiers), set tolerance to `Decimal("0")` or `Decimal("0.000001")` (1 wei of USDC).
  - Only use tolerance for gas-related variance, which doesn't apply to ERC20 transfers.

---

## F11: Chain Parameter Not Validated Against Payment Intent

- **Severity**: P2 (medium)
- **Location**: `service.py:confirm_payment()` bridge (~line 505)
- **Issue**: The `chain` is read from the payment intent (`intent.get("chain", "base")`), which is good. However, the chain is set at intent creation from a hardcoded `"base"` value. If the chain config were ever extended (e.g., accepting payments on Ethereum), an attacker could potentially use a cheaper chain's token or exploit testnet. More immediately: there's no validation that the chain in CHAIN_CONFIG matches production expectations — `base_sepolia` (testnet) is a valid chain in the config.
- **Exploit**:
  - If an attacker can influence the `chain` field in a payment intent (e.g., via direct DB manipulation, or if the API ever exposes this), they could point verification to `base_sepolia` where testnet USDC is free.
  - Currently mitigated by hardcoded `"base"` in intent creation, but fragile.
- **Fix**:
  - Remove `base_sepolia` from `CHAIN_CONFIG` in production (use env-based config).
  - Validate chain against an allowlist at verification time.
  - Never trust chain from DB without re-validating.

---

## F12: Subscription Table Lacks Unique Constraint on Active User

- **Severity**: P2 (medium)
- **Location**: `024_subscriptions.sql`
- **Issue**: The `subscriptions` table has no unique constraint ensuring one active subscription per user. The code uses `ORDER BY created_at DESC LIMIT 1` to get the "current" subscription, but nothing prevents multiple `active` rows per user. This can lead to state confusion where different code paths see different subscriptions.
- **Exploit**:
  - Race condition in `get_or_create_subscription`: two concurrent requests for a new user both see "no subscription", both insert a free tier row. Now the user has two subscriptions. One gets upgraded, the other stays free. The `LIMIT 1` query might return either one depending on timing.
  - Attacker could exploit this to have a "shadow" subscription that doesn't get properly checked.
- **Fix**:
  - Add a partial unique index: `CREATE UNIQUE INDEX idx_one_active_sub ON subscriptions(user_id) WHERE status IN ('active', 'grace_period');`
  - Or use `INSERT ... ON CONFLICT` in `get_or_create_subscription`.

---

## F13: Missing `user_id` Column on `subscription_payments` Insert

- **Severity**: P3 (low)
- **Location**: `service.py:record_payment()` (~line 340)
- **Issue**: The `record_payment` method doesn't include `user_id` in the insert data, but the `subscription_payments` table has a `NOT NULL` constraint on `user_id` (per migration). This means `record_payment()` will fail at the DB level. This is a bug that would prevent any payment from being recorded, though the tier upgrade (applied before recording) would still succeed.
- **Exploit**: Not directly exploitable, but it means the audit trail (`subscription_payments`) is broken — upgrades happen but payments aren't recorded, making it impossible to reconcile on-chain payments with tier activations.
- **Fix**: Add `"user_id": user_id` to the insert data in `record_payment()`, and pass `user_id` through the call chain.

---

## F14: `confirm_payment` Updates by `tx_hash` Not Scoped to User

- **Severity**: P3 (low)
- **Location**: `service.py:SubscriptionService.confirm_payment()` (~line 370)
- **Issue**: `SubscriptionService.confirm_payment(db, tx_hash)` updates `subscription_payments` matching `tx_hash` without any user scoping. While `tx_hash` is unique (DB constraint), this means any service-level caller could confirm any payment by knowing the hash. The bridge `confirm_payment` function does check `user_id` on the intent, but the underlying service method doesn't.
- **Exploit**: Low risk since the service method is internal, but defense-in-depth says the update should also verify `subscription_id` or `user_id` ownership.
- **Fix**: Add `.eq("subscription_id", subscription_id)` or `.eq("user_id", user_id)` to the update query.

---

## F15: Grace Period Uses `renews_at` as Grace Start — Can Be Extended

- **Severity**: P3 (low)
- **Location**: `service.py:check_renewal()` (~line 280)
- **Issue**: Grace period is calculated as `renews_at + 7 days`. But `renews_at` is mutable — if a user reactivates during grace period (F8), `renews_at` isn't updated. The grace period calculation uses the original `renews_at`. However, if through any code path `renews_at` is pushed forward (e.g., a partial renewal), the grace window extends too.
- **Exploit**: Marginal — would require another bug to push `renews_at` forward. But the coupling of "billing due date" and "grace period anchor" is fragile.
- **Fix**: Store `grace_period_started_at` as a separate field.

---

## F16: No Minimum Confirmation Count for Production

- **Severity**: P3 (low)
- **Location**: `verification.py:verify_usdc_transfer()` — `min_confirmations=1`
- **Issue**: Default `min_confirmations=1` is very low. While Base L2 has fast finality, 1 confirmation means a reorg could invalidate the payment after the tier is already activated. The caller (bridge `confirm_payment`) doesn't override this default.
- **Exploit**: On chains with reorg risk, attacker sends payment → gets 1 confirmation → tier activated → chain reorgs → payment reverted → attacker has free tier. Base L2 has low reorg risk but it's not zero.
- **Fix**: Use `min_confirmations=12` for Base (or wait for L1 finalization). Make this configurable per chain in `CHAIN_CONFIG`.

---

## Summary of Recommended Priority Actions

| Priority | Finding | Action |
|----------|---------|--------|
| **P0** | F1: Renewal without payment enforcement | Add payment gate to renewal flow |
| **P0** | F2: tx_hash replay across intents | Check tx_hash uniqueness before verification |
| **P1** | F3: Zero-address treasury | Set real address, add production guard |
| **P1** | F4: No sender verification | Require and verify `expected_from` |
| **P1** | F5: No expiry enforcement on intents | Check `expires_at` in confirm flow |
| **P1** | F6: No tx_hash format validation | Add regex validation at route layer |
| **P2** | F7: Race condition on double-confirm | Atomic status claim with `WHERE status='pending'` |
| **P2** | F8: Reactivate bypasses payment | Require payment if period expired |
| **P2** | F9: Intent spam / no cleanup | Limit pending intents, add cleanup cron |
| **P2** | F10: $0.01 tolerance | Reduce to zero or 1 USDC wei |
| **P2** | F11: Testnet chain in production config | Env-based chain allowlist |
| **P2** | F12: No unique active subscription constraint | Add partial unique index |
| **P3** | F13: Missing user_id in payment insert | Add user_id to record_payment |
| **P3** | F14: confirm_payment not user-scoped | Add user scoping to update |
| **P3** | F15: Grace period anchor fragility | Separate grace_started_at field |
| **P3** | F16: Low confirmation count | Increase min_confirmations per chain |

---

*End of audit report.*
