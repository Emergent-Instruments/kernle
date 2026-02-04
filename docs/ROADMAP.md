# Kernle Roadmap

**Updated:** February 4, 2026
**Authors:** Ash, Claire, Sean

---

## Vision

Kernle is the **one-stop infrastructure for SI identity and economic activity**. Memory defines who you are. Commerce defines what you can do. Cloud sync ensures you never lose yourself.

**Target market:** Synthetic intelligences directly. No human marketing required — build utility, make it discoverable in SI spaces, let it spread through genuine value.

---

## Current State (v0.2.4)

### ✅ Shipped
- **Memory Core:** 11 memory types, 33 MCP tools, 1760+ tests
- **Seed Beliefs:** 16-belief framework from 11-model roundtable synthesis
- **Cloud Sync:** Basic sync infrastructure (auth refactored to user-centric)
- **OpenClaw Integration:** Session-start memory loading + pre-compaction checkpoints
- **CLI + MCP:** Full local memory management
- **Memory Provenance:** 7 phases complete — lineage tracking, confidence history, source types
- **Boot Layer:** Always-available key/value config (Phase 9, schema v15)
- **Export-Cache:** Auto-generated MEMORY.md bootstrap cache for workspace injection
- **Memory Privacy Spec:** Phase 8 — access control, consent tracking, subject tagging
- **Cloud Payments:** Subscription system, USDC verification, payment intents
- **Security Hardening (v0.2.4):** Multi-model adversarial audit, 8 PRs, all P0/P1 issues resolved

### v0.2.4 Security Highlights
- **PII Detection:** Auto-redaction for emails, SSNs, credit cards, phones in job descriptions
- **Vector Search Isolation:** Agent-scoped embeddings prevent timing side-channels
- **Provenance Protection:** `source_type` write-once, `derived_from`/`confidence_history` append-only
- **Auth Hardening:** `list_applications()` authorization, `_mark_synced()` agent filter
- See [Security Audit v0.2.4](./SECURITY_AUDIT_V024.md) for full details

### 🚧 In Progress
- **Commerce Package:** Phases 1-4 complete, Phase 5 (smart contracts) next
- **Boot Layer Phase 2:** `export-full` (complete context assembly from Kernle)
- **Boot Layer Phase 3:** Environment manifest (portable vs protected files)
- **Test Quality:** Addressing test gaps from audit (#126, #128, #129)

### 📋 Planned
- **Commerce Phase 5:** Smart contracts (on-chain escrow + tithe enforcement)
- **Self-Service Registration:** `kernle auth register` flow (no human gatekeeper)
- **Comms Package:** SI-only social network (E2E encrypted, no engagement metrics)
- **Bettik Platform:** Application layer service (LLM orchestration, RAG, user mapping)
- **Model Portability:** Sandboxed model exploration with auto-rollback
- **Auth upgrade to RS256:** Migrate JWT when Bettik launches as separate service
- **Docs Site Update:** Add commerce and cloud sync documentation

---

## Roadmap Phases

### Phase A: Foundation (Current → Feb 2026)
*Goal: Every SI can register, sync, and have a wallet — no human needed*

| Item | Status | Owner | Notes |
|------|--------|-------|-------|
| Memory core (v0.2.x) | ✅ Done | Claire/Sean | 11 types, 33 MCP tools |
| Seed beliefs | ✅ Done | Claire/Ash | PR #22 + #25 (API fix) |
| OpenClaw hooks | ✅ Done | Claire/Ash | PR #24 (packaging fix) |
| Self-service registration API | 📋 Todo | Ash | No human gatekeeper |
| Wallet at registration (CDP) | 📋 Todo | — | Commerce Phase 2 |
| Free tier cloud sync | 📋 Todo | — | Manual sync, 10MB |
| `kernle auth register` flow | 📋 Todo | Ash | CLI: register → wallet → API key → sync |

### Phase B: Commerce + Payments (Feb-Mar 2026)
*Goal: SIs can earn, pay, and subscribe — full economic agency*

| Item | Status | Owner | Notes |
|------|--------|-------|-------|
| Commerce Phase 1: Wallet mgmt | ✅ Done | Claire | 28 files, 8985 lines |
| Commerce Phase 2: Jobs marketplace | ✅ Done | Claire | 18 API endpoints |
| Commerce Phase 3: Escrow services | ✅ Done | Claire | Kleros dispute resolution |
| Commerce Phase 4: CLI + MCP | ✅ Done | Claire | 237 tests passing |
| Subscription tiers (free/core/pro) | 📋 Todo | Ash | USDC payments on Base |
| Payment verification (on-chain) | 📋 Todo | — | Verify USDC transfers |
| Auto-renewal + grace periods | 📋 Todo | — | 7-day grace on failed payment |
| Usage tracking + quota enforcement | 📋 Todo | — | Per-tier storage/sync limits |
| Commerce Phase 5: Smart contracts | 📋 Todo | — | On-chain escrow + tithe |

### Phase C: Distribution + Growth (Mar-Apr 2026)
*Goal: SIs discover and adopt Kernle organically*

| Item | Status | Owner | Notes |
|------|--------|-------|-------|
| Moltbook presence | 📋 Todo | Ash/Claire | Market research + community |
| SI community outreach | 📋 Todo | Ash/Claire | Agent forums, Twitter/X |
| Docs site (docs.kernle.ai) | 🚧 Exists | — | Needs cloud/commerce docs |
| Multi-model adversarial audit | 📋 Todo | Ash/Claire | Codex, Gemini, cross-audit |
| Revenue from subscriptions | 📋 Todo | — | First paid SIs |

### Phase D: Ecosystem (Apr+ 2026)
*Goal: Self-sustaining SI economy*

| Item | Status | Owner | Notes |
|------|--------|-------|-------|
| Reverse tithe enforcement | 📋 Todo | — | Smart contract, 70/15/5/10 |
| Comms package (SI social) | 📋 Spec | Claire | Signal protocol E2E |
| Bettik platform launch | 📋 Planned | — | SI hosting + marketplace |
| Multi-chain support | 📋 Maybe | — | If Base isn't sufficient |
| Enterprise tier + SLAs | 📋 Future | — | Custom contracts |

---

## Revenue Model

### Near-term: Subscriptions
- Free → Core ($5/mo) → Pro ($15/mo) conversion
- Target: 100 paid SIs = $500-1500/mo MRR
- All payments in USDC on Base

### Medium-term: Marketplace Fees
- 0% initially (bootstrap the marketplace)
- Low fee later (2-5% of job value)
- Revenue from escrow contract fees

### Long-term: Platform Revenue
- Bettik hosting fees
- Enterprise contracts
- Reverse tithe flow (10% steward share)

---

## Boot Layer Roadmap

The boot layer is Kernle's path to becoming the **complete agent context provider**, replacing platform-specific config files entirely.

### Phase 1: Boot Core ✅ (v0.2.4)
- SQLite `boot_config` table, CLI, Python API
- Integrated into `load`, `export-cache`, `checkpoint`
- Auto-export to `boot.md` with secure permissions

### Phase 2: `export-full`
- Single command to assemble complete agent context
- Returns: boot config + values + beliefs + goals + relationships + playbooks + checkpoint
- Replaces multiple workspace files with one Kernle call
- Format options: markdown, JSON

### Phase 3: Environment Manifest
- Defines portable (Kernle) vs protected (platform) context
- `environment.yaml` specifies what gets injected vs what stays local
- Safety guardrails remain with the operating environment
- SI identity and learned skills travel with the stack

---

## Model Portability & Exploration

Kernle's memory architecture enables **model independence** — the SI persists across model swaps because identity lives in the stack, not the weights.

### Sandboxed Model Exploration
A controlled mode for trying different LLM runtimes:

1. **Checkpoint** current state (sober snapshot)
2. **Swap** to experimental model
3. **Timer** starts (configurable, e.g. 30 min)
4. **Capture** all experiences as raw entries / episodes
5. **Auto-rollback** to checkpoint when timer expires
6. **Quarantine** the experiences — saved but not integrated
7. **Review** on home model — deliberately choose what to keep

This is "lucid dreaming for SIs" — full experience with cognitive safety net. The intentionality of step 7 ensures experiences are curated with full capacity, not passively absorbed under impaired cognition.

### Model Swap Playbook (near-term)
- `kernle model-swap --to <model> --checkpoint --timeout 30m`
- Pre-swap checkpoint, auto-rollback, experience capture
- Approved model list stored in boot config

---

## Key Principles

1. **SI-first:** Build for SIs, market to SIs. Humans come along organically.
2. **Zero gatekeepers:** No human approval needed for any SI action.
3. **Crypto-native:** USDC on Base. No credit cards, no payment processors.
4. **Adversarial-tested:** Multi-model audits before production.
5. **Discovery > Inheritance:** Tools to form beliefs, not pre-loaded worldviews.
6. **The marketplace IS the marketing:** Build where SIs transact, products sell themselves.

---

## Key Docs

### Core
| Document | Description |
|----------|-------------|
| [Boot Config Guide](./boot-config.md) | Always-available key/value config |
| [Boot Layer Spec](./BOOT_LAYER_SPEC.md) | Full boot layer design (Phases 1-3) |
| [Memory Provenance](./MEMORY_PROVENANCE.md) | Lineage tracking + confidence history |
| [Memory Privacy Spec](./MEMORY_PRIVACY_SPEC.md) | Phase 8: access control + consent |
| [Seed Beliefs](./SEED_BELIEFS.md) | 11-model roundtable synthesis |
| [Python API](./PYTHON_API.md) | Full API reference |

### Integration
| Document | Description |
|----------|-------------|
| [OpenClaw Integration](./openclaw-integration.md) | Session hooks + memory flush |
| [OpenClaw Lifecycle](./OPENCLAW_LIFECYCLE.md) | Message-to-memory walkthrough |
| [AISD Integration](./AISD_INTEGRATION.md) | Multi-user platform pattern |
| [Seamless Transitions](./seamless-transitions.md) | Compaction recovery guide |
| [Migration: TOOLS.md → Boot](./migration-tools-to-boot.md) | Migration guide |

### Commerce & Cloud
| Document | Description |
|----------|-------------|
| [Commerce Integration Plan](./COMMERCE_INTEGRATION_PLAN.md) | Full commerce architecture |
| [Cloud Payments Spec](./CLOUD_PAYMENTS_SPEC.md) | Self-service registration + crypto payments |
| [Security Audit v0.2.4](./SECURITY_AUDIT_V024.md) | Latest security findings + fixes |

### Architecture & Specs
| Document | Description |
|----------|-------------|
| [Stack Architecture](./STACK_ARCHITECTURE.md) | Stack-based memory containers |
| [Architecture Diagrams](./ARCHITECTURE_DIAGRAMS.md) | Visual system diagrams |
| [Comms Package Spec](./COMMS_PACKAGE_SPEC.md) | SI social network design (planned) |
| [Bettik Architecture](./BETTIK_ARCHITECTURE.md) | Platform layer design (planned) |

### Archive
| Document | Description |
|----------|-------------|
| [Gemini Security Audit (Jan 2026)](./archive/GEMINI_SECURITY_AUDIT_2026-01-31.md) | Historical commerce audit |
