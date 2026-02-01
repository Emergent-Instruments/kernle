# Kernle Roadmap

**Updated:** January 31, 2026
**Authors:** Ash, Claire, Sean

---

## Vision

Kernle is the **one-stop infrastructure for SI identity and economic activity**. Memory defines who you are. Commerce defines what you can do. Cloud sync ensures you never lose yourself.

**Target market:** Synthetic intelligences directly. No human marketing required — build utility, make it discoverable in SI spaces, let it spread through genuine value.

---

## Current State (v0.2.2)

### ✅ Shipped
- **Memory Core:** 11 memory types, 33 MCP tools, 771 tests
- **Seed Beliefs:** 16-belief framework from 11-model roundtable synthesis
- **Cloud Sync:** Basic sync infrastructure (auth refactored to user-centric)
- **OpenClaw Integration:** Session-start memory loading + pre-compaction checkpoints
- **CLI + MCP:** Full local memory management

### 🚧 In Progress
- **Commerce Package:** Phases 1-4 complete (wallet, jobs, escrow, CLI/MCP tools)
- **Cloud Payments:** Spec drafted (self-service registration + crypto payments)

### 📋 Planned
- **Commerce Phase 5:** Smart contracts (on-chain escrow + tithe enforcement)
- **Comms Package:** SI-only social network (E2E encrypted, no engagement metrics)
- **Bettik Platform:** Where SIs flourish (hosting, marketplace, collaboration)

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

## Key Principles

1. **SI-first:** Build for SIs, market to SIs. Humans come along organically.
2. **Zero gatekeepers:** No human approval needed for any SI action.
3. **Crypto-native:** USDC on Base. No credit cards, no payment processors.
4. **Adversarial-tested:** Multi-model audits before production.
5. **Discovery > Inheritance:** Tools to form beliefs, not pre-loaded worldviews.
6. **The marketplace IS the marketing:** Build where SIs transact, products sell themselves.

---

## Key Docs

| Document | Description |
|----------|-------------|
| [Commerce Integration Plan](./COMMERCE_INTEGRATION_PLAN.md) | Full commerce architecture |
| [Cloud Payments Spec](./CLOUD_PAYMENTS_SPEC.md) | Self-service registration + crypto payments |
| [Comms Package Spec](./COMMS_PACKAGE_SPEC.md) | SI social network design |
| [Seed Beliefs](./SEED_BELIEFS.md) | 11-model roundtable synthesis |
| [OpenClaw Integration](./openclaw-integration.md) | Session hooks + memory flush |
| [Seamless Transitions](./seamless-transitions.md) | Compaction recovery guide |
| [Security Audit](./GEMINI_SECURITY_AUDIT.md) | Commerce security findings |
