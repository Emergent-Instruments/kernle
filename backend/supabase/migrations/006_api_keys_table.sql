-- =============================================================================
-- API Keys Table
-- =============================================================================
-- Allows users to create multiple API keys for programmatic access
-- Keys are stored as bcrypt hashes, never in plaintext

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,  -- Links to agents.user_id (stable user identifier)
    key_hash TEXT NOT NULL,  -- bcrypt hash of the API key
    key_prefix TEXT NOT NULL,  -- First 8 chars of key for identification (e.g., "knl_sk_a")
    name TEXT NOT NULL DEFAULT 'Default',  -- User-provided name for the key
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT name_not_empty CHECK (name <> ''),
    CONSTRAINT key_prefix_format CHECK (key_prefix ~ '^knl_sk_[a-f0-9]+$')
);

-- Index for fast lookups by user
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);

-- Index for active keys only (most common query)
CREATE INDEX idx_api_keys_user_active ON api_keys(user_id, is_active) WHERE is_active = TRUE;

-- Index for prefix lookups (used during auth to narrow down candidates)
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);

-- =============================================================================
-- Row Level Security (RLS)
-- =============================================================================

ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Service role can access everything (for backend)
CREATE POLICY "Service role full access" ON api_keys FOR ALL TO service_role USING (true);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE api_keys IS 'API keys for programmatic access. Keys are bcrypt hashed.';
COMMENT ON COLUMN api_keys.key_hash IS 'bcrypt hash of the full API key';
COMMENT ON COLUMN api_keys.key_prefix IS 'First 8 chars of key (knl_sk_ + 1 hex char) for display/identification';
COMMENT ON COLUMN api_keys.name IS 'User-friendly name for the key';
COMMENT ON COLUMN api_keys.last_used_at IS 'Updated on each successful authentication';
COMMENT ON COLUMN api_keys.is_active IS 'FALSE = revoked/cycled key';
