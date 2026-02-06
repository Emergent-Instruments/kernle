-- Migration 002: Add GIN indexes for JSONB provenance fields
--
-- derived_from and source_episodes are JSONB arrays used for reverse lookups
-- (e.g., "find all memories derived from X"). Without indexes, these queries
-- require full table scans at scale. GIN indexes with jsonb_path_ops provide
-- efficient containment (@>) queries on these columns.
--
-- Uses jsonb_path_ops operator class for better performance on containment
-- queries (smaller index, faster lookups) vs the default jsonb_ops class.
--
-- Covers all memory tables that have provenance fields.

-- Episodes
CREATE INDEX IF NOT EXISTS idx_episodes_derived_from_gin
    ON episodes USING GIN (derived_from jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_episodes_source_episodes_gin
    ON episodes USING GIN (source_episodes jsonb_path_ops);

-- Beliefs
CREATE INDEX IF NOT EXISTS idx_beliefs_derived_from_gin
    ON beliefs USING GIN (derived_from jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_beliefs_source_episodes_gin
    ON beliefs USING GIN (source_episodes jsonb_path_ops);

-- Notes
CREATE INDEX IF NOT EXISTS idx_notes_derived_from_gin
    ON notes USING GIN (derived_from jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_notes_source_episodes_gin
    ON notes USING GIN (source_episodes jsonb_path_ops);

-- Drives
CREATE INDEX IF NOT EXISTS idx_drives_derived_from_gin
    ON drives USING GIN (derived_from jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_drives_source_episodes_gin
    ON drives USING GIN (source_episodes jsonb_path_ops);

-- Values (agent_values)
CREATE INDEX IF NOT EXISTS idx_agent_values_derived_from_gin
    ON agent_values USING GIN (derived_from jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_agent_values_source_episodes_gin
    ON agent_values USING GIN (source_episodes jsonb_path_ops);

-- Relationships
CREATE INDEX IF NOT EXISTS idx_relationships_derived_from_gin
    ON relationships USING GIN (derived_from jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_relationships_source_episodes_gin
    ON relationships USING GIN (source_episodes jsonb_path_ops);

-- Goals
CREATE INDEX IF NOT EXISTS idx_goals_derived_from_gin
    ON goals USING GIN (derived_from jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_goals_source_episodes_gin
    ON goals USING GIN (source_episodes jsonb_path_ops);
