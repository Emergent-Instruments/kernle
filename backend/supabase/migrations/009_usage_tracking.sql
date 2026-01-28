-- =============================================================================
-- Migration 009: Usage Tracking and Tier System
-- =============================================================================
-- Implements request quotas and tier-based limits for API keys

-- =============================================================================
-- 1. Add tier column to agents table
-- =============================================================================

-- Tier types: 'free' (default), 'unlimited', 'paid'
ALTER TABLE agents ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'free';
ALTER TABLE agents ADD CONSTRAINT agents_tier_check 
    CHECK (tier IN ('free', 'unlimited', 'paid'));

COMMENT ON COLUMN agents.tier IS 'Account tier: free (default limits), unlimited (no limits), paid (custom limits)';

-- =============================================================================
-- 2. Create usage tracking table
-- =============================================================================

CREATE TABLE IF NOT EXISTS api_key_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,  -- Denormalized for fast lookups
    
    -- Daily tracking
    daily_requests INTEGER NOT NULL DEFAULT 0,
    daily_reset_at TIMESTAMPTZ NOT NULL DEFAULT DATE_TRUNC('day', NOW() AT TIME ZONE 'UTC') + INTERVAL '1 day',
    
    -- Monthly tracking  
    monthly_requests INTEGER NOT NULL DEFAULT 0,
    monthly_reset_at TIMESTAMPTZ NOT NULL DEFAULT DATE_TRUNC('month', NOW() AT TIME ZONE 'UTC') + INTERVAL '1 month',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_api_key_usage UNIQUE (api_key_id)
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_api_key_usage_user ON api_key_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_api_key ON api_key_usage(api_key_id);

-- =============================================================================
-- 3. RLS for usage table
-- =============================================================================

ALTER TABLE api_key_usage ENABLE ROW LEVEL SECURITY;

-- Service role full access
CREATE POLICY "Service role full access" ON api_key_usage 
    FOR ALL TO service_role USING (true);

-- =============================================================================
-- 4. Set Claire to unlimited tier
-- =============================================================================

UPDATE agents 
SET tier = 'unlimited' 
WHERE agent_id = 'claire' 
  AND user_id = 'usr_c21787425dd3';

-- =============================================================================
-- 5. Comments
-- =============================================================================

COMMENT ON TABLE api_key_usage IS 'Tracks API request counts per key for quota enforcement';
COMMENT ON COLUMN api_key_usage.daily_requests IS 'Request count since last daily reset';
COMMENT ON COLUMN api_key_usage.daily_reset_at IS 'When daily counter resets (UTC midnight)';
COMMENT ON COLUMN api_key_usage.monthly_requests IS 'Request count since last monthly reset';
COMMENT ON COLUMN api_key_usage.monthly_reset_at IS 'When monthly counter resets (1st of month UTC)';
