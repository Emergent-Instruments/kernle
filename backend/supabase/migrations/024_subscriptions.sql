-- =============================================================================
-- Migration 024: Subscription management tables
-- =============================================================================
-- Implements the cloud payments spec (docs/CLOUD_PAYMENTS_SPEC.md)
-- Supports tiered subscriptions with crypto (USDC) payments on Base L2

-- =============================================================================
-- Update users table tier constraint for new tiers
-- =============================================================================

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_tier_check;
ALTER TABLE users ADD CONSTRAINT users_tier_check
    CHECK (tier IN ('free', 'core', 'pro', 'enterprise', 'paid', 'unlimited'));

-- =============================================================================
-- Subscriptions table
-- =============================================================================

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    tier TEXT NOT NULL DEFAULT 'free',
    status TEXT NOT NULL DEFAULT 'active',
    starts_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    renews_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    auto_renew BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT sub_valid_tier CHECK (tier IN ('free', 'core', 'pro', 'enterprise')),
    CONSTRAINT sub_valid_status CHECK (status IN ('active', 'grace_period', 'cancelled', 'expired'))
);

CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_renews ON subscriptions(renews_at) WHERE auto_renew = true AND status = 'active';
CREATE INDEX idx_subscriptions_status ON subscriptions(status);

-- =============================================================================
-- Subscription payments table
-- =============================================================================

CREATE TABLE IF NOT EXISTS subscription_payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    amount DECIMAL(18, 6) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USDC',
    tx_hash TEXT UNIQUE,
    from_address TEXT,
    to_address TEXT,
    chain TEXT NOT NULL DEFAULT 'base',
    status TEXT NOT NULL DEFAULT 'pending',
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at TIMESTAMPTZ,

    CONSTRAINT pay_valid_status CHECK (status IN ('pending', 'confirmed', 'failed', 'refunded')),
    CONSTRAINT pay_valid_currency CHECK (currency IN ('USDC'))
);

CREATE INDEX idx_sub_payments_subscription ON subscription_payments(subscription_id);
CREATE INDEX idx_sub_payments_user ON subscription_payments(user_id);
CREATE INDEX idx_sub_payments_tx ON subscription_payments(tx_hash);
CREATE INDEX idx_sub_payments_status ON subscription_payments(status);

-- =============================================================================
-- Storage usage tracking table
-- =============================================================================

CREATE TABLE IF NOT EXISTS usage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    period TEXT NOT NULL,                    -- YYYY-MM format
    storage_bytes BIGINT DEFAULT 0,
    sync_count INTEGER DEFAULT 0,
    stacks_syncing INTEGER DEFAULT 0,
    peak_stacks INTEGER DEFAULT 0,          -- Peak syncing stacks for overflow billing
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, period)
);

CREATE INDEX idx_usage_records_user_period ON usage_records(user_id, period);

-- =============================================================================
-- Payment intents (for the two-phase upgrade flow)
-- =============================================================================

CREATE TABLE IF NOT EXISTS payment_intents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    subscription_id UUID REFERENCES subscriptions(id),
    tier TEXT NOT NULL,
    amount DECIMAL(18, 6) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USDC',
    treasury_address TEXT NOT NULL,
    chain TEXT NOT NULL DEFAULT 'base',
    status TEXT NOT NULL DEFAULT 'pending',
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at TIMESTAMPTZ,

    CONSTRAINT intent_valid_status CHECK (status IN ('pending', 'confirmed', 'expired', 'cancelled'))
);

CREATE INDEX idx_payment_intents_user ON payment_intents(user_id);
CREATE INDEX idx_payment_intents_status ON payment_intents(status) WHERE status = 'pending';

-- =============================================================================
-- Row Level Security
-- =============================================================================

ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscription_payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_intents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access" ON subscriptions FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON subscription_payments FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON usage_records FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON payment_intents FOR ALL TO service_role USING (true);

-- =============================================================================
-- Initialize free subscriptions for existing users
-- =============================================================================

INSERT INTO subscriptions (user_id, tier, status, starts_at, auto_renew)
SELECT user_id, 'free', 'active', created_at, false
FROM users
WHERE NOT EXISTS (
    SELECT 1 FROM subscriptions s WHERE s.user_id = users.user_id
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE subscriptions IS 'User subscription state. One active subscription per user.';
COMMENT ON TABLE subscription_payments IS 'On-chain USDC payment records for tier upgrades/renewals.';
COMMENT ON TABLE usage_records IS 'Monthly storage and sync usage tracking per user.';
COMMENT ON TABLE payment_intents IS 'Two-phase payment: intent created → user pays on-chain → confirmed.';
COMMENT ON COLUMN usage_records.peak_stacks IS 'Peak syncing stacks in period, used for overflow billing.';
COMMENT ON COLUMN payment_intents.treasury_address IS 'Address user should send USDC to for this payment.';
