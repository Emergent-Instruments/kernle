-- =============================================================================
-- Migration 018: Wallet Accounts
-- =============================================================================
-- Wallet accounts for Kernle Commerce. Every agent gets a CDP Smart Wallet
-- at registration for USDC transactions on Base.
-- =============================================================================

CREATE TABLE IF NOT EXISTS wallet_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,                           -- Links to agents table
    user_id TEXT REFERENCES users(user_id),           -- Links to user (owner)
    wallet_address VARCHAR(42) NOT NULL UNIQUE,       -- Ethereum address (0x...)
    chain VARCHAR(20) NOT NULL DEFAULT 'base',        -- base or base-sepolia
    status VARCHAR(20) NOT NULL DEFAULT 'pending_claim',
    owner_eoa VARCHAR(42),                            -- Human's recovery address
    spending_limit_per_tx DECIMAL(18, 6) DEFAULT 100.0,
    spending_limit_daily DECIMAL(18, 6) DEFAULT 1000.0,
    cdp_wallet_id VARCHAR(100),                       -- CDP internal ID
    created_at TIMESTAMPTZ DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_wallet_status CHECK (status IN (
        'pending_claim', 'active', 'paused', 'frozen'
    )),
    CONSTRAINT valid_chain CHECK (chain IN ('base', 'base-sepolia')),
    CONSTRAINT valid_wallet_address CHECK (wallet_address ~ '^0x[a-fA-F0-9]{40}$'),
    CONSTRAINT valid_owner_eoa CHECK (
        owner_eoa IS NULL OR owner_eoa ~ '^0x[a-fA-F0-9]{40}$'
    ),
    CONSTRAINT positive_spending_limits CHECK (
        spending_limit_per_tx > 0 AND spending_limit_daily > 0
    )
);

-- Indexes
CREATE INDEX idx_wallet_agent ON wallet_accounts(agent_id);
CREATE INDEX idx_wallet_address ON wallet_accounts(wallet_address);
CREATE INDEX idx_wallet_user ON wallet_accounts(user_id);
CREATE INDEX idx_wallet_status ON wallet_accounts(status);
CREATE INDEX idx_wallet_chain ON wallet_accounts(chain);

-- Row Level Security
ALTER TABLE wallet_accounts ENABLE ROW LEVEL SECURITY;

-- Service role can access everything
CREATE POLICY "Service role full access" ON wallet_accounts 
    FOR ALL TO service_role USING (true);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE wallet_accounts IS 'Agent crypto wallets on Base (CDP Smart Wallet)';
COMMENT ON COLUMN wallet_accounts.agent_id IS 'Kernle agent ID this wallet belongs to';
COMMENT ON COLUMN wallet_accounts.user_id IS 'User ID of the wallet owner';
COMMENT ON COLUMN wallet_accounts.wallet_address IS 'Ethereum address (0x...)';
COMMENT ON COLUMN wallet_accounts.chain IS 'Blockchain network: base or base-sepolia';
COMMENT ON COLUMN wallet_accounts.status IS 'Wallet status: pending_claim, active, paused, frozen';
COMMENT ON COLUMN wallet_accounts.owner_eoa IS 'Human recovery/control address (set on claim)';
COMMENT ON COLUMN wallet_accounts.spending_limit_per_tx IS 'Max USDC per transaction';
COMMENT ON COLUMN wallet_accounts.spending_limit_daily IS 'Max USDC per 24 hours';
COMMENT ON COLUMN wallet_accounts.cdp_wallet_id IS 'Coinbase CDP internal wallet identifier';
COMMENT ON COLUMN wallet_accounts.claimed_at IS 'When wallet was claimed by human owner';
