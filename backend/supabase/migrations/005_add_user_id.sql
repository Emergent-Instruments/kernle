-- =============================================================================
-- Add user_id column for namespacing
-- =============================================================================
-- user_id is a stable identifier (usr_xxxx) generated on first registration
-- Used to namespace agent_ids: full agent_id = {user_id}/{project_name}

ALTER TABLE agents ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Generate user_ids for existing agents (use first 8 chars of UUID)
UPDATE agents 
SET user_id = 'usr_' || SUBSTRING(REPLACE(gen_random_uuid()::TEXT, '-', ''), 1, 12)
WHERE user_id IS NULL;

-- Make it non-null after backfill
ALTER TABLE agents ALTER COLUMN user_id SET NOT NULL;

-- Index for lookups
CREATE INDEX IF NOT EXISTS idx_agents_user_id ON agents(user_id);
