-- =============================================================================
-- Migration 014: Add 'admin' to tier constraint
-- =============================================================================
-- Allows setting tier='admin' for admin users with special privileges

-- Drop and recreate the constraint to include 'admin'
ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_tier_check;
ALTER TABLE agents ADD CONSTRAINT agents_tier_check 
    CHECK (tier IN ('free', 'paid', 'unlimited', 'admin'));

COMMENT ON COLUMN agents.tier IS 'Account tier: free (default limits), paid (custom limits), unlimited (no limits), admin (full access + admin features)';
