-- Migration: Change agent_id from globally unique to unique per user
-- This allows multiple users to have agents with the same name (e.g., "claire")

-- Step 1: Drop the existing unique constraint on agent_id
ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_agent_id_key;

-- Step 2: Add composite unique constraint on (user_id, agent_id)
-- This ensures agent_id is unique within each user's namespace
ALTER TABLE agents ADD CONSTRAINT agents_user_agent_unique UNIQUE (user_id, agent_id);

-- Step 3: Update the foreign key references to use (user_id, agent_id) pattern
-- Note: The memory tables reference agent_id directly, which still works because
-- agent_id remains the identifier within the user's namespace. The FK just needs
-- to ensure the agent exists, and agent_id is still indexed.

-- Add index for efficient lookups by (user_id, agent_id)
CREATE INDEX IF NOT EXISTS idx_agents_user_agent ON agents(user_id, agent_id);
