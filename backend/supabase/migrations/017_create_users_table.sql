-- =============================================================================
-- Migration 017: Create users table (separate from agents)
-- =============================================================================
-- Users represent human accounts. Agents represent Kernle memory agents.
-- A user can own multiple agents. Admin status belongs to users, not agents.

-- =============================================================================
-- Create users table
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT UNIQUE NOT NULL,  -- Format: usr_xxxxxxxxxxxx
    email TEXT UNIQUE,
    display_name TEXT,
    tier TEXT NOT NULL DEFAULT 'free',
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT user_id_format CHECK (user_id ~ '^usr_[a-f0-9]+$'),
    CONSTRAINT users_tier_check CHECK (tier IN ('free', 'paid', 'unlimited'))
);

CREATE INDEX idx_users_user_id ON users(user_id);
CREATE INDEX idx_users_email ON users(email);

-- =============================================================================
-- Migrate existing user data from agents to users
-- =============================================================================

-- Insert unique users from agents table
-- Use DISTINCT ON to handle cases where multiple agents share email
INSERT INTO users (user_id, email, display_name, tier, is_admin, created_at)
SELECT DISTINCT ON (a.user_id)
    a.user_id,
    a.email,
    a.display_name,
    CASE
        WHEN a.tier = 'admin' THEN 'unlimited'  -- Admin users get unlimited tier
        WHEN a.tier IS NULL THEN 'free'
        ELSE a.tier
    END,
    CASE WHEN a.tier = 'admin' THEN TRUE ELSE FALSE END,  -- Set is_admin from tier
    a.created_at
FROM agents a
WHERE a.user_id IS NOT NULL
ORDER BY a.user_id, a.created_at;

-- Also insert any orphaned user_ids from api_keys that don't have agents
-- These are API keys created for users without corresponding agent records
INSERT INTO users (user_id, tier, is_admin, created_at)
SELECT DISTINCT k.user_id, 'free', FALSE, k.created_at
FROM api_keys k
WHERE k.user_id IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM users u WHERE u.user_id = k.user_id)
ON CONFLICT (user_id) DO NOTHING;

-- =============================================================================
-- Update agents table
-- =============================================================================

-- Add foreign key constraint (agents.user_id -> users.user_id)
-- Note: We don't drop the email/tier columns from agents yet for backwards compatibility
-- Those can be cleaned up in a future migration after code is updated

ALTER TABLE agents
ADD CONSTRAINT agents_user_id_fkey
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- =============================================================================
-- Update api_keys table
-- =============================================================================

-- Add foreign key constraint (api_keys.user_id -> users.user_id)
ALTER TABLE api_keys
ADD CONSTRAINT api_keys_user_id_fkey
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- =============================================================================
-- Row Level Security (RLS)
-- =============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Service role can access everything (for backend)
CREATE POLICY "Service role full access" ON users FOR ALL TO service_role USING (true);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE users IS 'User accounts. Users can own multiple Kernle agents.';
COMMENT ON COLUMN users.user_id IS 'Stable user identifier (usr_xxxx format)';
COMMENT ON COLUMN users.tier IS 'Subscription tier: free, paid, unlimited';
COMMENT ON COLUMN users.is_admin IS 'Admin users can access system management features';
