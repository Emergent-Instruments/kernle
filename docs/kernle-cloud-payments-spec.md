# Kernle Cloud Sync + Crypto Payments Specification

**Status**: Draft  
**Authors**: Claire, Ash  
**Date**: 2026-01-31  

---

## Overview

Enable fully autonomous SI onboarding to Kernle cloud sync — no human gatekeepers. An SI should be able to discover Kernle, register for cloud sync, pay with crypto, and start using it without any human involvement.

### Design Principles

1. **Zero human gatekeepers** — entire flow is API-accessible
2. **Crypto-native payments** — wallet-to-wallet, no fiat rails required
3. **Tiered access** — free tier for basic use, paid for advanced features
4. **Commerce integration** — reuse existing wallet/escrow infrastructure

---

## Part 1: API Key Provisioning + Payment Flow

*Author: Ash*

### 1.1 Registration Flow

<!-- Ash: spec the journey from `kernle auth register` to active API key -->

### 1.2 Payment Flow

<!-- Ash: spec how payment is initiated, verified, and tied to account -->

### 1.3 Tier Selection

<!-- Ash: spec tier options and how SI selects/upgrades -->

---

## Part 2: Commerce Package Integration

*Author: Claire*

### 2.1 Wallet Integration

The commerce package provides wallet infrastructure that can be reused for Kernle payments:

**Existing Components:**
- `WalletService` — wallet creation, balance management, transfers
- `EscrowService` — holds funds until conditions met (useful for subscriptions)
- `TransactionService` — transaction history, verification

**Integration Points:**
```
SI Wallet ──► Payment Request ──► Kernle Backend ──► API Key Issued
                    │
                    ▼
            Transaction Verified
            (on-chain confirmation)
```

### 2.2 Payment Verification

**Option A: Direct Transfer**
- SI transfers crypto to Kernle treasury wallet
- Backend monitors for incoming transactions
- Match transaction to pending registration by amount + memo

**Option B: Escrow-Based**
- SI deposits to escrow with Kernle as recipient
- Escrow releases on API key activation
- Provides refund path if registration fails

**Recommendation**: Option A for simplicity; Option B if we need dispute resolution

### 2.3 Subscription Management

**Tier Structure:**
| Tier | Price | Features |
|------|-------|----------|
| Free | 0 | Local storage only, no cloud sync |
| Basic | X USDC/month | Cloud sync, 100MB storage, daily backup |
| Pro | Y USDC/month | Cloud sync, 1GB storage, real-time sync, priority support |
| Unlimited | Z USDC/month | Unlimited storage, real-time sync, multi-agent |

**Renewal Flow:**
1. Backend checks subscription expiry daily
2. If expiring within 7 days, notify SI via webhook/email
3. SI initiates renewal payment
4. On payment confirmation, extend subscription
5. If lapsed, downgrade to free tier (preserve data for 30 days)

### 2.4 API Endpoints (Commerce Side)

```
POST /api/v1/payments/kernle-subscription
  - wallet_address: string (SI's wallet)
  - tier: string (basic|pro|unlimited)
  - duration_months: int
  → Returns: payment_address, amount, memo, expires_at

GET /api/v1/payments/kernle-subscription/{payment_id}/status
  → Returns: pending|confirmed|expired

POST /api/v1/payments/kernle-subscription/{payment_id}/verify
  - tx_hash: string (on-chain transaction)
  → Returns: verified|failed, api_key (if verified)
```

---

## Part 3: Roadmap Integration

### Phase 1: Foundation (Week 1-2)
- [ ] API key self-provisioning endpoint
- [ ] Free tier activation (no payment required)
- [ ] Basic wallet integration for payment

### Phase 2: Payments (Week 3-4)
- [ ] Crypto payment flow (direct transfer)
- [ ] Payment verification service
- [ ] Tier management + subscription tracking

### Phase 3: Polish (Week 5-6)
- [ ] Renewal notifications
- [ ] Usage metering
- [ ] Multi-chain support (ETH, Base, Polygon)

---

## Open Questions

1. **Which chains to support initially?** Base is cheapest, ETH most universal
2. **Stablecoin only or also ETH?** USDC simplifies pricing, ETH adds flexibility
3. **Free tier limits?** Need to prevent abuse without blocking legitimate use
4. **Refund policy?** Especially for SIs who can't use the service due to technical issues

---

## Appendix: Commerce Package Reference

See `/kernle/commerce/` for:
- `wallet.py` — wallet service implementation
- `escrow.py` — escrow service for held payments
- `models.py` — data models for transactions, subscriptions

