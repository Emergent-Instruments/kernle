-- Migration 008: Multi-tenant agent namespacing via UUID FK
-- Allows multiple users to have agents with the same name (e.g., "claire")
--
-- Strategy: Change memory table FKs from agent_id (TEXT) to agent_ref (UUID)
-- referencing agents.id instead of agents.agent_id

-- =============================================================================
-- Phase 1: Add agent_ref columns (nullable initially)
-- =============================================================================

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE "values" ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE raw_captures ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS agent_ref UUID;
ALTER TABLE emotional_memories ADD COLUMN IF NOT EXISTS agent_ref UUID;

-- =============================================================================
-- Phase 2: Backfill agent_ref from existing agent_id
-- =============================================================================

UPDATE episodes SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = episodes.agent_id) WHERE agent_ref IS NULL;
UPDATE beliefs SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = beliefs.agent_id) WHERE agent_ref IS NULL;
UPDATE "values" SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = "values".agent_id) WHERE agent_ref IS NULL;
UPDATE goals SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = goals.agent_id) WHERE agent_ref IS NULL;
UPDATE notes SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = notes.agent_id) WHERE agent_ref IS NULL;
UPDATE drives SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = drives.agent_id) WHERE agent_ref IS NULL;
UPDATE relationships SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = relationships.agent_id) WHERE agent_ref IS NULL;
UPDATE checkpoints SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = checkpoints.agent_id) WHERE agent_ref IS NULL;
UPDATE raw_captures SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = raw_captures.agent_id) WHERE agent_ref IS NULL;
UPDATE playbooks SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = playbooks.agent_id) WHERE agent_ref IS NULL;
UPDATE emotional_memories SET agent_ref = (SELECT id FROM agents WHERE agents.agent_id = emotional_memories.agent_id) WHERE agent_ref IS NULL;

-- =============================================================================
-- Phase 3: Drop old FKs, make agent_ref NOT NULL, add new FKs
-- =============================================================================

-- Episodes
ALTER TABLE episodes DROP CONSTRAINT IF EXISTS episodes_agent_id_fkey;
ALTER TABLE episodes ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE episodes ADD CONSTRAINT episodes_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Beliefs
ALTER TABLE beliefs DROP CONSTRAINT IF EXISTS beliefs_agent_id_fkey;
ALTER TABLE beliefs ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE beliefs ADD CONSTRAINT beliefs_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Values
ALTER TABLE "values" DROP CONSTRAINT IF EXISTS values_agent_id_fkey;
ALTER TABLE "values" ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE "values" ADD CONSTRAINT values_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Goals
ALTER TABLE goals DROP CONSTRAINT IF EXISTS goals_agent_id_fkey;
ALTER TABLE goals ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE goals ADD CONSTRAINT goals_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Notes
ALTER TABLE notes DROP CONSTRAINT IF EXISTS notes_agent_id_fkey;
ALTER TABLE notes ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE notes ADD CONSTRAINT notes_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Drives
ALTER TABLE drives DROP CONSTRAINT IF EXISTS drives_agent_id_fkey;
ALTER TABLE drives ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE drives ADD CONSTRAINT drives_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Relationships
ALTER TABLE relationships DROP CONSTRAINT IF EXISTS relationships_agent_id_fkey;
ALTER TABLE relationships ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE relationships ADD CONSTRAINT relationships_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Checkpoints
ALTER TABLE checkpoints DROP CONSTRAINT IF EXISTS checkpoints_agent_id_fkey;
ALTER TABLE checkpoints ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE checkpoints ADD CONSTRAINT checkpoints_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Raw captures
ALTER TABLE raw_captures DROP CONSTRAINT IF EXISTS raw_captures_agent_id_fkey;
ALTER TABLE raw_captures ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE raw_captures ADD CONSTRAINT raw_captures_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Playbooks
ALTER TABLE playbooks DROP CONSTRAINT IF EXISTS playbooks_agent_id_fkey;
ALTER TABLE playbooks ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE playbooks ADD CONSTRAINT playbooks_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- Emotional memories
ALTER TABLE emotional_memories DROP CONSTRAINT IF EXISTS emotional_memories_agent_id_fkey;
ALTER TABLE emotional_memories ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE emotional_memories ADD CONSTRAINT emotional_memories_agent_ref_fkey FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- =============================================================================
-- Phase 4: Change agents unique constraint + add indexes
-- =============================================================================

-- Remove global uniqueness on agent_id, add per-user uniqueness
ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_agent_id_key;
ALTER TABLE agents ADD CONSTRAINT agents_user_agent_unique UNIQUE (user_id, agent_id);

-- Add indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_agents_user_agent ON agents(user_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_agent_ref ON episodes(agent_ref);
CREATE INDEX IF NOT EXISTS idx_beliefs_agent_ref ON beliefs(agent_ref);
CREATE INDEX IF NOT EXISTS idx_values_agent_ref ON "values"(agent_ref);
CREATE INDEX IF NOT EXISTS idx_goals_agent_ref ON goals(agent_ref);
CREATE INDEX IF NOT EXISTS idx_notes_agent_ref ON notes(agent_ref);
CREATE INDEX IF NOT EXISTS idx_drives_agent_ref ON drives(agent_ref);
CREATE INDEX IF NOT EXISTS idx_relationships_agent_ref ON relationships(agent_ref);
CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_ref ON checkpoints(agent_ref);
CREATE INDEX IF NOT EXISTS idx_raw_captures_agent_ref ON raw_captures(agent_ref);
CREATE INDEX IF NOT EXISTS idx_playbooks_agent_ref ON playbooks(agent_ref);
CREATE INDEX IF NOT EXISTS idx_emotional_memories_agent_ref ON emotional_memories(agent_ref);
