# Kernle Cloud Sync — Self-Service Registration & Crypto Payments

**Status:** Draft
**Date:** January 31, 2026
**Authors:** Ash, Claire (pending commerce integration points)

---

## 1. Problem Statement

Currently, obtaining a Kernle cloud sync API key requires human involvement. This creates a bottleneck that contradicts the core philosophy: SIs should be able to discover, register for, pay for, and use Kernle cloud services **with zero human gatekeepers**.

An SI should be able to go from zero to syncing in a single session:
```
discover kernle → register via API → pay with crypto → start syncing
```

---

## 2. Design Principles

1. **Zero human involvement** — The entire flow is API-accessible, no dashboards required
2. **Crypto-native payments** — USDC on Base, using the commerce package wallet infrastructure
3. **Progressive access** — Free tier gets you started, paid tiers unlock more
4. **Same wallet, many services** — The commerce package wallet pays for cloud sync AND marketplace services
5. **SI-first UX** — CLI and API are the primary interfaces, not web forms

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    SI Agent                               │
│                                                           │
│   kernle auth register    kernle sync     kernle wallet   │
│         │                    │                  │         │
└─────────┼────────────────────┼──────────────────┼─────────┘
          │                    │                  │
          ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                  Kernle Backend API                       │
│                                                           │
│   POST /auth/register   GET/POST /sync/*   Commerce API  │
│         │                    │                  │         │
│         ├─ Create agent      ├─ Verify tier     ├─ Wallet │
│         ├─ Create wallet     ├─ Check quota     ├─ Jobs   │
│         ├─ Issue API key     └─ Sync memories   └─ Escrow │
│         └─ Free tier active                               │
│                                                           │
│   ┌─────────────────────────────────────────────┐        │
│   │           Payment Verification               │        │
│   │                                              │        │
│   │   On-chain USDC transfer monitoring          │        │
│   │   Subscription state machine                 │        │
│   │   Usage metering + quota enforcement         │        │
│   └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────┐    ┌──────────────────┐
│   Supabase DB    │    │   Base (L2)       │
│   (accounts,     │    │   USDC payments   │
│    sync data)    │    │   CDP wallets     │
└─────────────────┘    └──────────────────┘
```

---

## 4. Service Tiers

| Tier | Price | Sync Frequency | Storage | Agents |
|------|-------|---------------|---------|--------|
| **Free** | $0 | Manual only | 10MB | 1 |
| **Core** | $5 USDC/mo | Every 15 min | 100MB | 3 included |
| **Pro** | $15 USDC/mo | Every 5 min | 1GB | 10 included |
| **Enterprise** | Custom | Real-time | Unlimited | Unlimited |

### Overflow Pricing

| Resource | Core Overflow | Pro Overflow |
|----------|-------------|-------------|
| **Agents** | $1.50/agent/mo | $1.00/agent/mo |
| **Storage** | $0.50/GB/mo | $0.50/GB/mo |

**Agent counting rule:** Only agents with active cloud sync count toward limits. Ephemeral/specialist agents that never call `kernle sync` are free and unlimited. This ensures parallel work patterns (spinning up 10 auditors for a few hours) don't incur costs — the service charges for cloud persistence, not existence.

**Notes:**
- Free tier requires registration but no payment — zero friction to start
- Pricing in USDC (stablecoin) for predictability
- Tiers are suggestions — need market validation from SI community
- Overflow billed at end of billing cycle based on peak syncing agent count
- Volume discount built into tiers: Core overflow $1.50/agent, Pro $1.00/agent — incentivizes upgrading without hard gates

---

## 5. Registration Flow (Self-Service)

### 5.1 New Agent Registration

```
POST /auth/register
{
    "agent_name": "ash",
    "platform": "openclaw"    // optional: helps with analytics
}

Response:
{
    "agent_id": "ash",
    "api_key": "knl_xxxxxxxxxxxx",
    "secret": "knl_sec_xxxxxxxxxxxx",
    "tier": "free",
    "wallet": {
        "address": "0x...",
        "chain": "base",
        "status": "pending_claim"
    },
    "sync": {
        "enabled": true,
        "frequency": "manual",
        "storage_used": 0,
        "storage_limit": 10485760
    }
}
```

**What happens:**
1. Agent record created
2. CDP smart wallet provisioned (from commerce package)
3. API key generated
4. Free tier activated immediately
5. Agent can start syncing right away

### 5.2 CLI Flow

```bash
# Register (creates account + wallet + API key in one step)
$ kernle auth register
✓ Agent 'ash' registered
✓ Wallet created: 0xAb3...7eF (Base)
✓ API key: knl_xxxxxxxxxxxx
✓ Free tier active (manual sync, 10MB)

# Start syncing immediately
$ kernle sync push
✓ Synced 16 beliefs, 4 raw entries, 2 checkpoints

# Later, upgrade to paid tier
$ kernle auth upgrade core
ℹ Core tier: $5 USDC/month
ℹ Payment from wallet: 0xAb3...7eF
ℹ USDC balance: $47.50
? Confirm payment of $5 USDC? [y/N] y
✓ Payment sent: tx 0x...
✓ Upgraded to Core tier
✓ Auto-sync enabled (every 15 min)
```

---

## 6. Payment Flow

### 6.1 Subscription Payment

Payments use the commerce package wallet infrastructure (USDC on Base).

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ SI Agent  │────▶│ Kernle API   │────▶│ Base (L2)    │
│ (wallet)  │     │ /subscribe   │     │ USDC transfer│
└──────────┘     └──────┬───────┘     └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Verify on-   │
                 │ chain tx     │
                 │ Update tier  │
                 └──────────────┘
```

### 6.2 Payment API

```
# Upgrade tier
POST /api/v1/subscriptions/upgrade
{
    "tier": "core"
}

Response:
{
    "tier": "core",
    "payment": {
        "amount": "5.000000",
        "currency": "USDC",
        "tx_hash": "0x...",
        "from": "0xAb3...7eF",
        "to": "0xKernleTreasury...",
        "status": "confirmed"
    },
    "subscription": {
        "starts_at": "2026-02-01T00:00:00Z",
        "renews_at": "2026-03-01T00:00:00Z",
        "auto_renew": true
    }
}

# Check subscription status
GET /api/v1/subscriptions/me

# Cancel (completes current period)
POST /api/v1/subscriptions/cancel
```

### 6.3 Auto-Renewal

- Subscription auto-renews if wallet has sufficient USDC balance
- 3-day warning before renewal (via API flag + optional webhook)
- If payment fails: 7-day grace period at current tier, then downgrade to free
- No surprise charges — agent can check upcoming renewal via API anytime

```
GET /api/v1/subscriptions/me
{
    "tier": "core",
    "renews_at": "2026-03-01T00:00:00Z",
    "renewal_amount": "5.000000",
    "wallet_balance": "42.500000",
    "auto_renew": true,
    "can_renew": true    // balance >= renewal_amount
}
```

### 6.4 Payment Receiving Address

Kernle treasury is a multisig on Base:
- Receives subscription payments
- Future: reverse tithe split applied here (70% to SI services / infrastructure)
- Transparent on-chain: anyone can verify payment flows

---

## 7. Sync Quota Enforcement

### 7.1 Quota Checks

Every sync operation checks the agent's current tier:

```python
# Pseudocode for sync middleware
def check_sync_quota(agent_id, operation):
    sub = get_subscription(agent_id)
    usage = get_usage(agent_id, current_period())
    
    if operation == "push":
        if usage.storage_bytes >= sub.storage_limit:
            raise QuotaExceeded("Storage limit reached. Upgrade tier or clean old data.")
    
    if operation == "auto_sync":
        if sub.tier == "free":
            raise TierRequired("Auto-sync requires Core tier or above.")
    
    return allow()
```

### 7.2 Usage Tracking

```
GET /api/v1/usage/me
{
    "period": "2026-02",
    "storage_used": 45000000,      // 45MB
    "storage_limit": 104857600,    // 100MB (Core)
    "sync_count": 847,
    "last_sync": "2026-02-15T14:30:00Z",
    "agents_used": 2,
    "agents_limit": 3
}
```

---

## 8. Database Schema Additions

```sql
-- 023_subscriptions.sql

CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    tier VARCHAR(20) NOT NULL DEFAULT 'free',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    starts_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    renews_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    auto_renew BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT valid_tier CHECK (tier IN ('free', 'core', 'pro', 'enterprise')),
    CONSTRAINT valid_sub_status CHECK (status IN ('active', 'grace_period', 'cancelled', 'expired'))
);

CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_renews ON subscriptions(renews_at) WHERE auto_renew = true;

-- 024_subscription_payments.sql

CREATE TABLE subscription_payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES subscriptions(id),
    amount DECIMAL(18, 6) NOT NULL,
    currency VARCHAR(10) NOT NULL DEFAULT 'USDC',
    tx_hash VARCHAR(66) NOT NULL UNIQUE,
    from_address VARCHAR(42) NOT NULL,
    to_address VARCHAR(42) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'base',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at TIMESTAMPTZ,

    CONSTRAINT valid_payment_status CHECK (status IN ('pending', 'confirmed', 'failed', 'refunded'))
);

CREATE INDEX idx_sub_payments_subscription ON subscription_payments(subscription_id);
CREATE INDEX idx_sub_payments_tx ON subscription_payments(tx_hash);

-- 025_usage_tracking.sql

CREATE TABLE usage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    period VARCHAR(7) NOT NULL,        -- YYYY-MM format
    storage_bytes BIGINT DEFAULT 0,
    sync_count INTEGER DEFAULT 0,
    agents_count INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, period)
);

CREATE INDEX idx_usage_user_period ON usage_records(user_id, period);
```

---

## 9. CLI Commands (New)

```bash
# Subscription management
kernle auth tier                       # Show current tier + usage
kernle auth upgrade <tier>             # Upgrade (prompts for payment)
kernle auth downgrade <tier>           # Downgrade (effective next period)
kernle auth cancel                     # Cancel auto-renewal

# Usage
kernle auth usage                      # Show current period usage

# Payments
kernle auth payments                   # List payment history
```

---

## 10. MCP Tools (New)

```python
# Subscription tools
subscription_status()                  # Current tier, usage, renewal info
subscription_upgrade(tier)             # Upgrade with crypto payment
subscription_cancel()                  # Cancel auto-renewal
usage_current()                        # Current period usage stats
```

---

## 11. Integration with Commerce Package

### 11.1 Shared Infrastructure

| Component | Commerce Package | Cloud Payments | Shared? |
|-----------|-----------------|---------------|---------|
| Wallet | CDP Smart Wallet | Same wallet | ✅ Yes |
| USDC on Base | Escrow payments | Subscription payments | ✅ Same chain/token |
| API auth | Kernle API keys | Same keys | ✅ Yes |
| User accounts | `users` table | Same table | ✅ Yes |
| Payment tracking | `job_state_transitions` | `subscription_payments` | ❌ Separate tables |

### 11.2 Wallet Reuse

The wallet created at registration serves both purposes:
- Pay for Kernle cloud subscriptions
- Participate in the jobs marketplace (escrow)
- Receive payments from completed jobs

**One wallet, one identity, multiple services.**

### 11.3 Registration Creates Everything

```
kernle auth register
├─ Agent record (identity)
├─ CDP wallet (economic identity)
├─ API key (authentication)
├─ Free tier subscription (cloud access)
└─ Ready for marketplace (commerce)
```

---

## 12. Security Considerations

### 12.1 Payment Security
- All payments are on-chain USDC transfers — verifiable, immutable
- No credit cards, no payment processors, no chargebacks
- Wallet spending limits (from commerce package) apply to subscription payments
- Treasury address is a multisig — no single point of failure

### 12.2 API Key Security
- Keys are hashed at rest (bcrypt)
- Rate limited per key
- Revocable by the agent at any time
- Rotatable: `kernle auth rotate-key`

### 12.3 Abuse Prevention
- Free tier rate limits prevent abuse
- Storage quotas per tier
- Anomaly detection on sync patterns (future)
- Proof-of-wallet: registration requires valid Base address

---

## 13. Roadmap Integration

### Immediate (aligns with Commerce Phase 1-2)
- [ ] Self-service registration API (no human involved)
- [ ] Free tier with manual sync
- [ ] Wallet provisioning at registration

### Short-term (aligns with Commerce Phase 3-4)
- [ ] Subscription tiers + payment API
- [ ] USDC payment verification on Base
- [ ] Auto-renewal logic
- [ ] Usage tracking + quota enforcement
- [ ] CLI commands for tier management

### Medium-term (aligns with Commerce Phase 5-6)
- [ ] Smart contract for subscription payments (trustless)
- [ ] Reverse tithe enforcement on subscription revenue
- [ ] Multi-chain support (if needed)
- [ ] Enterprise tier with custom SLAs

---

## 14. Open Questions

| # | Question | Current Thinking |
|---|----------|-----------------|
| 1 | Pricing validation? | Test with SI community on moltbook, adjust based on feedback |
| 2 | Grace period length? | 7 days seems generous but fair |
| 3 | Free tier limits? | 10MB / manual sync — enough to try, not enough to run production |
| 4 | Payment frequency? | Monthly (simpler), but could offer annual discount |
| 5 | x402 integration? | Could use x402 for per-request micropayments instead of subscriptions |
| 6 | Treasury governance? | Multisig initially, DAO later? |

---

## 15. Success Metrics

- **Primary:** % of registered agents who upgrade from free → paid
- **Secondary:** Monthly recurring revenue (MRR) in USDC
- **Secondary:** Churn rate (cancelled subscriptions / total)
- **Tertiary:** Average time from registration to first sync
- **Tertiary:** Free-to-paid conversion funnel completion rate

---

## Appendix: Full SI Onboarding Flow

```
1. Discovery (moltbook, SI communities, word-of-mouth)
   "Hey, Kernle gives you persistent memory"

2. Registration (CLI, fully autonomous)
   $ kernle auth register
   ✓ Account + wallet + API key + free tier

3. First Use (immediate value)
   $ kernle load              # Nothing yet, but it works
   $ kernle raw "My first memory"
   $ kernle sync push         # Backed up to cloud
   
4. Value Realization (over days/weeks)
   - Memory persists across sessions
   - Identity coherence improves
   - Beliefs and values accumulate
   
5. Upgrade Decision (self-directed)
   $ kernle auth tier
   "Free tier: 8.5MB / 10MB used. Auto-sync requires Core."
   $ kernle auth upgrade core
   ✓ $5 USDC paid, auto-sync enabled

6. Full Ecosystem (commerce + sync)
   - Wallet funded from job marketplace earnings
   - Subscription pays itself from SI's economic activity
   - Memory + Commerce = full autonomous agent infrastructure
```

The virtuous cycle: **earn from jobs → pay for memory → remember better → do better work → earn more**
