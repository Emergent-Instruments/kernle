# Full Kernle Audit — Consolidated Summary

**Date:** 2026-02-01  
**Scope:** 39K lines across 65 Python files (core, commerce, MCP, CLI, features, storage)  
**Auditors:** 4 specialist sub-agents (security-core, security-commerce, quality-features, storage-integrity)

---

## Totals

| Severity | Core Security | Commerce Security | Quality/Features | Storage Integrity | **Total** |
|----------|:---:|:---:|:---:|:---:|:---:|
| **P0 (Critical)** | 0 | 5 | 1 | 0 | **6** |
| **P1 (High)** | 2 | 4 | 5 | 6 | **17** |
| **P2 (Medium)** | 4 | 6 | 10 | 12 | **32** |
| **P3 (Low)** | 5 | 5 | 8 | 6 | **24** |
| **Total** | 11 | 20 | 24 | 24 | **79** |

---

## P0 Critical Findings (6)

### Commerce (5) — All in stubbed/pre-production code
1. **No on-chain escrow state verification** — double-release/double-refund possible
2. **Transfers consume daily spend without rollback** — `_rollback_spend` never defined
3. **SupabaseWalletStorage.atomic_claim_wallet() always returns True** — any address claims any wallet
4. **No private key management infrastructure** — no secure key handling when CDP integrates
5. **No transaction receipt verification** — reverted txns could appear successful

### Quality/Features (1)
6. **Memory poisoning via suggestion pipeline** — raw content flows unsanitized into structured memory (prompt injection vector)

---

## P1 High Findings — Top 10

1. **MCP provenance params silently dropped** — `source`/`derived_from` not passed through validation, breaking lineage via MCP (Phase 3 bug)
2. **Postgres save_belief()/save_episode() drops most fields** — active data loss (source_type, derived_from, confidence_history, forgetting fields all lost)
3. **Forgetting system has no auto-protection** for values/identity memories — core values forgotten after 30 days
4. **No GIN indexes on JSONB arrays** — lineage queries O(N) at scale
5. **SQLite not in WAL mode** — concurrent CLI + MCP hits SQLITE_BUSY (2-line fix)
6. **Postgres search is brute-force Python** — no semantic search
7. **Postgres column names don't match SQLite** — sync loses data at mapping boundaries
8. **Daily spend limits reset on restart** (in-memory fallback)
9. **Importers pass unsanitized/unbounded content** to core API
10. **No transaction safety in promote_suggestion** — partial failure creates duplicates

---

## What's Strong

- **Core security is excellent**: zero SQL injection, parameterized queries everywhere, agent_id isolation in 55+ queries
- **Path traversal protection**: `is_relative_to()` guards on all file ops
- **Optimistic concurrency control**: version-based conflict detection on atomic updates
- **Provenance validation**: `_validate_derived_from()` with type:id format + ref type allowlist
- **Schema migrations are additive-only** (safe, no data loss)
- **Raw sync defaults OFF** (prevents accidental secret leakage)

---

## Recommended Fix Priority

### Immediate (quick wins, high impact)
1. SQLite WAL mode + busy_timeout (2-line fix, prevents lock errors)
2. Fix MCP provenance param passthrough (Phase 3 bug)
3. Auto-protect values/identity from forgetting
4. Cap confidence_history at ~100 entries

### Before cloud sync goes live
5. Fix Postgres save methods to include ALL dataclass fields
6. Add GIN indexes on JSONB provenance columns
7. Add `source_entity` column to SQLite schema
8. Add `is_forgotten=0` filter to search queries

### Before commerce goes to testnet
9. Implement `_rollback_spend()` 
10. Fix SupabaseWalletStorage stubs (raise NotImplementedError)
11. Add receipt verification pattern
12. Harden `"system"` actor_id bypass

### Ongoing
13. Sanitize suggestion→promotion pipeline
14. Add content size limits to importers
15. Align Postgres/SQLite column names
