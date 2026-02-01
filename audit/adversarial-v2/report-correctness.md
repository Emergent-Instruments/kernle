# Kernle Code Correctness Audit Report

**Auditor**: Code Correctness Auditor (adversarial)
**Date**: 2026-02-01
**Scope**: Auth + Database layer (backend), Core library (SQLite storage, metamemory, forgetting)
**Risk Rating Scale**: P0 (critical) / P1 (high) / P2 (medium) / P3 (low)

---

## Executive Summary

The codebase demonstrates strong security awareness with many defenses already in place (JWKS verification, CSRF middleware, SameSite cookies, SQL injection prevention via table allowlists, atomic quota checking, optimistic concurrency control). However, I found **19 findings** across both layers, including 2 P0 criticals, 4 P1 highs, 8 P2 mediums, and 5 P3 lows.

The most critical issues are: (1) a JWT-to-cookie auth path that silently degrades to free/non-admin on DB errors, enabling a potential denial-of-tier attack; and (2) a global singleton Supabase client that is not thread-safe across async workers.

---

## AUTH & DATABASE LAYER

### F1: JWT Auth Degrades to Free Tier on DB Error (Soft Privilege Confusion)

- **Severity**: P2 (medium)
- **Location**: backend/app/auth.py:245-255 (`get_current_agent`, JWT path)
- **Issue**: When a JWT-authenticated user's tier/admin lookup fails (DB down, network blip), the code catches all exceptions and silently defaults to `tier="free"` and `is_admin=False`. This means a legitimate admin user could lose admin access during transient DB issues, and a paid-tier user gets rate-limited as free tier.
- **Exploit scenario**: If the database is intermittently unavailable, all JWT-authenticated requests (web UI users) get downgraded to free tier. An attacker who can cause DB latency (e.g., via expensive queries on another endpoint) could force all JWT users into free-tier rate limits.
- **Fix**: Either fail the request with 503 (consistent with how API key auth handles DB errors) or cache the user's tier in the JWT claims so it doesn't require a DB lookup every request. The comment says "don't block auth" but this creates an inconsistency: API key auth fails closed (503), JWT auth fails open (free tier).

### F2: Global Supabase Client Singleton Not Thread-Safe

- **Severity**: P0 (critical)
- **Location**: backend/app/database.py:10-19 (`get_supabase_client`)
- **Issue**: `_supabase_client` is a module-level global with no locking. In a multi-worker async environment (uvicorn with multiple workers), there's a classic check-then-act race condition: two concurrent requests could both see `_supabase_client is None` and both create clients. While Python's GIL mitigates this for CPython, the supabase client itself may not be safe for concurrent use across asyncio tasks, and with `asyncio.to_thread()` calls in `get_changes_since`, actual concurrent access from multiple threads is possible.
- **Exploit scenario**: Under high concurrent load at startup, multiple Supabase clients could be created, leading to connection pool exhaustion or inconsistent state. The `asyncio.to_thread` calls in `get_changes_since` explicitly move queries to threads, making true concurrent access to the shared client object a real concern.
- **Fix**: Use a `threading.Lock` around the client creation, or use `functools.lru_cache` pattern. Better yet, use a connection pool or create clients per-request.

### F3: API Key Prefix Collision Allows Brute-Force Amplification

- **Severity**: P2 (medium)
- **Location**: backend/app/auth.py:70-76 (`get_api_key_prefix`), backend/app/database.py:471-485 (`get_active_api_keys_by_prefix`)
- **Issue**: The API key lookup uses only the first 8 characters after `knl_sk_` (just 1 hex char for discrimination since `knl_sk_` is 7 chars, plus 1 more = `knl_sk_X`). Wait — looking more carefully: `key[:12]` gives `"knl_sk_XXXXX"` which is 12 chars. But `get_active_api_keys_by_prefix` truncates to `prefix[:8]` = `"knl_sk_X"`, using only **1 hex character** for the LIKE query. This means every lookup fans out to ~1/16th of all active keys, and bcrypt verification is expensive (~100ms each). An attacker with any valid-looking prefix could trigger multiple bcrypt comparisons per request.
- **Exploit scenario**: Attacker sends requests with `knl_sk_0AAAAAAAAAAAAAAAAAAAAAA` — the lookup prefix `knl_sk_0` matches ~6.25% of all keys. With 1000 active keys, that's ~62 bcrypt checks per request, taking ~6 seconds. This is a CPU exhaustion DoS vector.
- **Fix**: Use a longer prefix for the LIKE query (the full 12-char prefix `knl_sk_XXXXX` provides 5 hex chars = 1M possibilities). The code already stores 12-char prefixes; the `[:8]` truncation in `get_active_api_keys_by_prefix` is the bug. Change `lookup_prefix = prefix[:8]` to `lookup_prefix = prefix[:12]`.

### F4: Cookie Max-Age Exceeds JWT Expiry

- **Severity**: P2 (medium)
- **Location**: backend/app/routes/auth.py:475-476, backend/app/config.py:27
- **Issue**: `COOKIE_MAX_AGE = 7 * 24 * 60 * 60` (7 days) and `jwt_expire_minutes = 60 * 24 * 7` (also 7 days). These match in the default case, BUT: when `create_access_token` is called with a custom `expires_delta` (e.g., in `login_with_api_key`), the cookie's max_age doesn't change. The cookie will persist in the browser after the JWT inside it expires, causing confusing "authenticated but rejected" states.
- **Exploit scenario**: User logs in, cookie is set for 7 days. If JWT expiry is configured shorter (e.g., admin changes `jwt_expire_minutes` to 60), users get persistent cookies containing expired JWTs. Each request triggers a full JWT decode + failure, wasting server resources and creating a poor UX.
- **Fix**: Derive `COOKIE_MAX_AGE` from `settings.jwt_expire_minutes` dynamically in `set_auth_cookie`, not as a module-level constant.

### F5: No Refresh Token Rotation

- **Severity**: P1 (high)
- **Location**: backend/app/auth.py (entire auth flow)
- **Issue**: The system issues long-lived JWTs (7 days) with no refresh token mechanism. There is no way to revoke a JWT once issued — if a JWT is compromised, it remains valid for up to 7 days. The only "revocation" is deleting the cookie (client-side), which doesn't help if the token was exfiltrated.
- **Exploit scenario**: Attacker steals a JWT from a cookie (via XSS in a third-party script, browser extension, etc.). The token remains valid for 7 days with no server-side revocation capability. The attacker has full access to the user's account for the entire token lifetime.
- **Fix**: Implement short-lived access tokens (15-30 min) with refresh token rotation, or maintain a server-side token blacklist/allowlist. At minimum, add a `jti` (JWT ID) claim and a revocation check.

### F6: Race Condition in API Key Cycling

- **Severity**: P2 (medium)
- **Location**: backend/app/routes/auth.py:422-460 (`cycle_api_key`)
- **Issue**: The cycle operation (create new key, then deactivate old key) is not atomic. If the deactivation fails (and the code explicitly handles this with a warning), the user ends up with two active keys. More critically: between the creation and deactivation, there's a window where both keys are active. If concurrent cycle requests are made, multiple new keys could be created.
- **Exploit scenario**: Attacker with a valid API key sends multiple concurrent cycle requests. Each creates a new key before deactivating the old one. Result: N active keys instead of 1, all valid. This bypasses any per-key rate limiting.
- **Fix**: Wrap the cycle operation in a database transaction (if Supabase supports it), or use a distributed lock on the key_id. At minimum, add a rate limit specific to cycle operations per key_id.

### F7: Quota Check Fallback Defeats Atomicity

- **Severity**: P1 (high)
- **Location**: backend/app/database.py:580-588 (`check_and_increment_quota`)
- **Issue**: When the atomic `check_and_increment_quota` RPC fails, it falls back to the non-atomic `check_quota` function, which the code itself warns has a TOCTOU race condition (comment at line 1499). This fallback silently degrades security guarantees.
- **Exploit scenario**: If the Supabase RPC function is temporarily unavailable (deployment, migration, etc.), all quota checks fall back to the racy path. An attacker sending 100 concurrent requests during this window could exceed their quota significantly — the check reads the current count, all 100 see "below quota", and then 100 increments happen.
- **Fix**: Either fail closed (503) when the atomic RPC is unavailable, or implement the atomic check in application code using a distributed lock. The non-atomic fallback should at minimum log at ERROR level, not WARNING.

### F8: Sync Push Allows Arbitrary Table Data Without Schema Validation

- **Severity**: P1 (high)
- **Location**: backend/app/routes/sync.py:65-120 (`push_changes`)
- **Issue**: While `SERVER_CONTROLLED_FIELDS` strips dangerous fields like `agent_ref`, `deleted`, `version`, and `id`, the sync push accepts arbitrary key-value pairs in `op.data` and passes them directly to `upsert_memory`, which does `db.table(table_name).upsert(record)`. There is no schema validation of the field names or values. An attacker could inject unexpected columns (e.g., `is_admin`, `tier`, `user_id` on memory tables if those columns existed).
- **Exploit scenario**: Client sends a sync push with `{"data": {"agent_id": "other_agents_id", "is_forgotten": false, ...}}` — the `agent_id` field is NOT in `SERVER_CONTROLLED_FIELDS` so it passes through. The upsert overwrites the record's `agent_id`, potentially moving a memory to another agent's namespace. While `agent_ref` is stripped, `agent_id` (the string identifier) is not.
- **Fix**: Add `agent_id` to `SERVER_CONTROLLED_FIELDS`. Better yet, maintain an allowlist of permitted fields per table instead of a denylist of forbidden fields. Denylist approaches are inherently fragile — any new sensitive column added to the schema must also be added to the denylist.

### F9: `ensure_agent_exists` Creates Agents with Empty Secret Hash

- **Severity**: P2 (medium)
- **Location**: backend/app/database.py:149-164 (`ensure_agent_exists`)
- **Issue**: When a user syncs with a new agent name, `ensure_agent_exists` creates the agent with `secret_hash=""`. If there's ever a code path that attempts to authenticate via agent secret (the `get_token` endpoint), bcrypt's `checkpw("", "")` behavior becomes relevant. While `bcrypt.checkpw` should raise an error on invalid hash format (empty string), this is relying on bcrypt's implementation detail.
- **Exploit scenario**: An attacker calls POST `/auth/token` with `agent_id` of an auto-provisioned agent and `secret=""`. If `verify_secret("", "")` doesn't properly reject this (bcrypt should, but it's implementation-dependent), they'd get authenticated.
- **Fix**: Use a sentinel value like `"$DISABLED$"` instead of empty string, and explicitly check for it in `verify_secret`. Or better: make `verify_secret` reject empty hashes explicitly before calling bcrypt.

### F10: Health Check Endpoint Leaks Database Status

- **Severity**: P3 (low)
- **Location**: backend/app/main.py:129-147 (`/health` endpoint)
- **Issue**: The `/health` endpoint is unauthenticated and returns database connection status ("connected", "disconnected", "error"). This gives attackers information about infrastructure state.
- **Exploit scenario**: Attacker monitors `/health` to detect when the database is down, then targets the application during degraded state (knowing quota checks may be failing).
- **Fix**: Return only "healthy"/"unhealthy" without specifying which component failed. Detailed health info should be behind admin auth.

### F11: Rate Limiting Uses `get_remote_address` — Bypassable Behind Proxy

- **Severity**: P2 (medium)
- **Location**: backend/app/rate_limit.py:5-6
- **Issue**: `get_remote_address` uses the raw TCP source IP. If the application is behind a reverse proxy (Railway uses its own routing layer), this will rate-limit by proxy IP, not by client IP. All clients would share the same rate limit bucket.
- **Exploit scenario**: Behind a reverse proxy, all requests appear to come from the proxy's IP. A single user hitting rate limits blocks ALL users from the same endpoint. Conversely, a single attacker is never rate-limited because the limit is shared across all users.
- **Fix**: Use `X-Forwarded-For` header parsing with a trusted proxy configuration. SlowAPI supports this via custom key functions. Ensure the proxy is configured to set a trusted header.

---

## CORE LIBRARY (SQLite Storage, Memory, Forgetting)

### F12: Agent Isolation — Shared SQLite Database, `agent_id` Filter Only

- **Severity**: P1 (high)
- **Location**: kernle/storage/sqlite.py (all query methods)
- **Issue**: All agents on the same machine share a single SQLite database (`~/.kernle/memories.db`). Agent isolation is enforced only by `WHERE agent_id = ?` filters in SQL queries. The `agent_id` is validated via `_validate_agent_id` (alphanumeric + `-_.`), but any process with filesystem access to the database can read ALL agents' memories directly.
- **Exploit scenario**: A malicious Kernle plugin, compromised agent, or any process with read access to `~/.kernle/memories.db` can query memories for any `agent_id`. There's no encryption at rest, no per-agent access control at the database level. On a multi-user system, if file permissions aren't set correctly (the code does `chmod 0o600` on the DB but not all paths check this), another user could read all agents' memories.
- **Fix**: For the local-first design, this is arguably acceptable if file permissions are correctly set (which they are: 0o600). However, document clearly that filesystem access = full access to all agents. For higher isolation, consider per-agent database files or encryption at rest.

### F13: `_validate_agent_id` Allows Path Traversal Characters

- **Severity**: P2 (medium)
- **Location**: kernle/core.py:200-211 (`_validate_agent_id`)
- **Issue**: The validation allows `.` (dots) in agent IDs: `c.isalnum() or c in "-_."`. Since `agent_id` is used to create directories (`self._agent_dir = self.db_path.parent / agent_id`), an agent_id like `..` or `...` would create directories that traverse the path hierarchy.
- **Exploit scenario**: `Kernle(agent_id="..")` creates `~/.kernle/..` as the agent directory, which resolves to `~/`. The flat file writes (`_sync_beliefs_to_file`, etc.) would then write `~/beliefs.md`, `~/values.md`, etc., overwriting arbitrary files in the home directory.
- **Fix**: Strip leading dots, disallow consecutive dots, or reject agent IDs that resolve to a path outside the expected parent directory. Add: `if '..' in sanitized or sanitized.startswith('.'): raise ValueError(...)`.

### F14: Forgetting Can Delete High-Confidence Beliefs

- **Severity**: P2 (medium)
- **Location**: kernle/features/forgetting.py (via `ForgettingMixin.run_forgetting_cycle`)
- **Issue**: The forgetting system's salience formula is `(confidence × (log(times_accessed + 1) + 0.1)) / (age_factor + 1)`. A belief with confidence=0.95 that was created 90 days ago and accessed only once has salience ≈ `(0.95 × (0 + 0.1)) / (3 + 1)` = 0.024 — well below the default threshold of 0.3. This means a high-confidence belief that hasn't been retrieved recently WILL be forgotten.
- **Exploit scenario**: An agent has a critical belief with confidence=0.95 (e.g., "I should never share private keys"). If the agent doesn't load this belief for 30+ days (because the budget-aware loading might prioritize other memories), it becomes a forgetting candidate. A `run_forgetting_cycle(dry_run=False)` call would tombstone it. While it can be recovered, the agent wouldn't know to recover it since it's been forgotten.
- **Fix**: Either: (a) make high-confidence beliefs (>0.8) automatically protected, (b) add a minimum salience floor based on confidence (e.g., `salience = max(salience, confidence * 0.5)`), or (c) factor confidence more heavily into the salience formula. Note: Values and drives are already protected by default, but beliefs are not.

### F15: Seed Beliefs Not Protected from Forgetting

- **Severity**: P3 (low)
- **Location**: backend/app/database.py:265-330 (`create_seed_beliefs`), related to F14
- **Issue**: Seed beliefs (the carefully curated foundational beliefs from the 11-model roundtable) are created with `is_foundational=True` but NOT with `is_protected=True`. The `is_foundational` field is not checked by the forgetting system. This means the meta-belief ("These starting beliefs are scaffolding...") at confidence=0.95 could be forgotten if not accessed frequently.
- **Exploit scenario**: An agent that doesn't regularly load its seed beliefs (e.g., uses context-scoped loads that filter them out) will see these beliefs decay below the forgetting threshold over time. The entire philosophical scaffolding could be erased by routine forgetting cycles.
- **Fix**: Set `is_protected=True` for seed beliefs (at minimum for Tier 1 and Meta beliefs), or have the forgetting system check `is_foundational` in addition to `is_protected`.

### F16: `record_access_batch` Uses Dynamic Table Names in SQL (Validated)

- **Severity**: P3 (low — mitigated)
- **Location**: kernle/storage/sqlite.py:9296-9354 (`record_access_batch`)
- **Issue**: The function uses f-string SQL: `f"UPDATE {table} SET ..."`. However, this IS validated against `ALLOWED_TABLES` via `validate_table_name(table)` called within the function. The table name also comes from a hardcoded `table_map` dict, not from user input.
- **Exploit scenario**: None currently — the validation is present. But if `table_map` ever includes a key that's not in `ALLOWED_TABLES`, or if `validate_table_name` is bypassed, SQL injection becomes possible.
- **Fix**: No immediate fix needed. This is noted as a defense-in-depth observation. Consider using a mapping to actual table references rather than string interpolation.

### F17: `update_memory_meta` Dynamic SQL Construction

- **Severity**: P3 (low — mitigated)
- **Location**: kernle/storage/sqlite.py:9060-9127 (`update_memory_meta`)
- **Issue**: Builds SQL dynamically: `f"UPDATE {table} SET {', '.join(updates)} WHERE ..."`. Table is validated via `validate_table_name(table)`, and updates are built from hardcoded field names (not user input). Column names in `updates` are statically defined strings.
- **Exploit scenario**: None currently. This follows the same pattern as F16.
- **Fix**: No immediate fix needed.

### F18: Flat File Writes Not Atomic

- **Severity**: P3 (low)
- **Location**: kernle/storage/sqlite.py:6800-6820 (`_sync_beliefs_to_file`), and similar `_sync_*_to_file` methods
- **Issue**: Flat file writes (beliefs.md, values.md, etc.) use direct `open(file, "w")` which truncates the file before writing. If the process crashes mid-write, the file is left empty or partially written. On next startup, `_init_flat_files` checks if files are empty and rebuilds from DB, so data isn't lost — but during the write, another reader could see partial data.
- **Exploit scenario**: Process crash during flat file write leaves an empty beliefs.md. If another tool reads this file before Kernle reinitializes it, it sees no beliefs. Low impact since these are convenience files, not the source of truth (SQLite DB is).
- **Fix**: Use atomic write pattern: write to a temp file, then `os.rename()` to the target. This ensures the file is always either the old version or the new version, never partial.

### F19: `get_changes_since` in Backend Doesn't Filter `is_forgotten`

- **Severity**: P1 (medium-high, reclassified from P1 since it's intentional but leaky)
- **Location**: backend/app/database.py:380-420 (`get_changes_since`)
- **Issue**: Wait — actually, it DOES filter: `query = query.or_("is_forgotten.is.null,is_forgotten.eq.false")`. This is correct. Let me re-examine... Actually the issue is subtler: the `get_changes_since` function correctly filters forgotten memories during `pull`, but the `full_sync` endpoint at `routes/sync.py` uses `get_changes_since` with `limit=100000`. This means a full sync correctly excludes forgotten memories. However, the **push** path doesn't validate that the client isn't trying to un-forget memories — `is_forgotten` is NOT in `SERVER_CONTROLLED_FIELDS`.
- **Revised Issue**: A malicious client could push `{"is_forgotten": false}` on a memory that was forgotten server-side, effectively recovering it without going through the proper recovery flow.
- **Exploit scenario**: Agent A's memory was forgotten (e.g., by admin or automated cycle). Attacker uses the sync push endpoint to send `{"table": "beliefs", "record_id": "xxx", "data": {"is_forgotten": false, "forgotten_at": null}}` to un-forget it.
- **Fix**: Add `is_forgotten`, `forgotten_at`, and `forgotten_reason` to `SERVER_CONTROLLED_FIELDS`, or add them to a separate "forgetting-controlled fields" set that's also stripped during push.

---

## Summary Table

| ID | Title | Severity | Category |
|----|-------|----------|----------|
| F1 | JWT auth degrades to free tier on DB error | P2 | Auth bypass |
| F2 | Global Supabase client not thread-safe | P0 | Race condition |
| F3 | API key prefix collision amplifies bcrypt DoS | P2 | Rate limiting |
| F4 | Cookie max-age exceeds JWT expiry | P2 | Session management |
| F5 | No refresh token rotation | P1 | Session management |
| F6 | Race condition in API key cycling | P2 | Race condition |
| F7 | Quota check fallback defeats atomicity | P1 | Race condition |
| F8 | Sync push allows agent_id override via data | P1 | Privilege escalation |
| F9 | Auto-provisioned agents have empty secret hash | P2 | Auth bypass |
| F10 | Health endpoint leaks DB status | P3 | Configuration |
| F11 | Rate limiting bypassable behind proxy | P2 | Rate limiting |
| F12 | Agent isolation via agent_id filter only | P1 | Agent isolation |
| F13 | Agent ID validation allows path traversal | P2 | Data validation |
| F14 | Forgetting can delete high-confidence beliefs | P2 | Forgetting correctness |
| F15 | Seed beliefs not protected from forgetting | P3 | Forgetting correctness |
| F16 | Dynamic table names in SQL (validated) | P3 | SQL injection |
| F17 | Dynamic SQL construction (validated) | P3 | SQL injection |
| F18 | Flat file writes not atomic | P3 | Data integrity |
| F19 | Sync push can un-forget memories | P1 | Privilege escalation |

---

## Positive Findings (Things Done Right)

1. **JWKS-based OAuth verification** — Correctly uses public key verification from Supabase's JWKS endpoint rather than sharing secrets. Issuer validation uses strict equality (not prefix match), preventing SSRF.
2. **CSRF middleware** — Origin header validation for cookie-based auth, SameSite=Strict cookies, defense-in-depth approach.
3. **JWT algorithm allowlist** — `ALLOWED_JWT_ALGORITHMS` prevents algorithm confusion attacks.
4. **Atomic quota checking** — The `check_and_increment_quota` RPC approach is the right architecture (though the fallback defeats it per F7).
5. **Optimistic concurrency control** — `update_episode_atomic` and `update_belief_atomic` use version-based OCC to prevent lost updates.
6. **Table name allowlist** — `validate_table_name()` in SQLite storage prevents SQL injection via table names.
7. **Server-controlled fields stripping** — The sync push correctly strips sensitive fields (though needs `agent_id` added per F8).
8. **Fail-closed quota behavior** — API key quota check returns 503 on DB errors rather than allowing unlimited access.
9. **Secure cookie settings** — HttpOnly, Secure, SameSite=Strict, proper path scoping.
10. **bcrypt for API keys and secrets** — Proper use of bcrypt with generated salts for credential storage.
11. **WAL mode and busy timeout** — SQLite is configured with WAL mode and 5s busy timeout for concurrent access.
12. **File permissions** — Database and agent directories get 0o600/0o700 permissions.

---

## Recommended Priority Order

1. **F8** (P1) — Add `agent_id` to `SERVER_CONTROLLED_FIELDS` — trivial fix, high impact
2. **F19** (P1) — Add forgetting fields to `SERVER_CONTROLLED_FIELDS` — trivial fix
3. **F2** (P0) — Add threading lock to Supabase client singleton — easy fix
4. **F7** (P1) — Change fallback to fail-closed (503) — easy fix
5. **F13** (P2) — Reject `..` in agent IDs — easy fix
6. **F3** (P2) — Change `prefix[:8]` to `prefix[:12]` — one-line fix
7. **F5** (P1) — Implement refresh token rotation — larger effort
8. **F11** (P2) — Configure proxy-aware rate limiting — moderate effort
9. **F14** (P2) — Protect high-confidence beliefs from forgetting — moderate effort
10. **F15** (P3) — Set `is_protected=True` on seed beliefs — trivial fix
