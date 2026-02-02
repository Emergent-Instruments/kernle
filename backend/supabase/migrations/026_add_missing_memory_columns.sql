-- Migration 026: Align Supabase schema with local SQLite schema
--
-- The local SQLite schema evolved through Phases 1-8, adding provenance,
-- forgetting, emotional, and context fields. The Supabase schema only had
-- partial coverage, causing sync push failures ("Database error: operation
-- failed") when clients send records containing these fields.
--
-- This migration adds ALL missing columns to bring Supabase in sync with
-- the local SQLite dataclass definitions.

-- ============================================================================
-- Episodes: source_entity, context, context_tags
-- (Most fields already added in migrations 001 + 003)
-- ============================================================================
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS source_entity TEXT;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';

-- ============================================================================
-- Beliefs: source_entity, context, context_tags, supersedes/superseded_by,
--          times_reinforced
-- (Most provenance + forgetting fields already in migration 003)
-- ============================================================================
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS source_entity TEXT;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS supersedes TEXT;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS superseded_by TEXT;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS times_reinforced INTEGER DEFAULT 0;

-- ============================================================================
-- Values: many fields missing — provenance, forgetting, context, plus
--         statement/priority which exist in dataclass but not Supabase
-- ============================================================================
ALTER TABLE values ADD COLUMN IF NOT EXISTS statement TEXT;
ALTER TABLE values ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 50;
ALTER TABLE values ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.9;
ALTER TABLE values ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE values ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE values ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE values ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE values ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE values ADD COLUMN IF NOT EXISTS confidence_history JSONB;
ALTER TABLE values ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE values ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE values ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE values ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE values ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE values ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;
ALTER TABLE values ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE values ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';

-- ============================================================================
-- Goals: title, provenance, forgetting, context fields
-- ============================================================================
ALTER TABLE goals ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE goals ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS confidence_history JSONB;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';

-- ============================================================================
-- Drives: intensity, focus_areas, provenance, forgetting, context
-- Note: Supabase has 'strength' while SQLite uses 'intensity' — keep both
-- ============================================================================
ALTER TABLE drives ADD COLUMN IF NOT EXISTS intensity REAL DEFAULT 0.5;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS focus_areas JSONB DEFAULT '[]';
ALTER TABLE drives ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE drives ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS confidence_history JSONB;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';

-- ============================================================================
-- Relationships: entity_name, entity_type, notes, interaction_count,
--                last_interaction, confidence, provenance, forgetting, context
-- Note: Supabase has 'entity' + 'trust'; SQLite uses 'entity_name' +
--       'sentiment'. Keep both column names for compatibility.
-- ============================================================================
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS entity_name TEXT;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS entity_type TEXT DEFAULT 'person';
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS notes TEXT;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS interaction_count INTEGER DEFAULT 0;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS last_interaction TIMESTAMPTZ;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS confidence_history JSONB;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';

-- ============================================================================
-- Notes: source_entity, tags, provenance, forgetting, context
-- ============================================================================
ALTER TABLE notes ADD COLUMN IF NOT EXISTS source_entity TEXT;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]';
ALTER TABLE notes ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE notes ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS confidence_history JSONB;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';

-- ============================================================================
-- Playbooks: additional fields from local schema
-- ============================================================================
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]';
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS mastery_level REAL DEFAULT 0.0;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS success_rate REAL DEFAULT 0.0;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS times_used INTEGER DEFAULT 0;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS last_used TIMESTAMPTZ;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS failure_modes JSONB DEFAULT '[]';
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS recovery_steps JSONB DEFAULT '[]';
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS context_tags JSONB DEFAULT '[]';

-- ============================================================================
-- Indexes for commonly queried new columns
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_values_is_forgotten ON values(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_goals_is_forgotten ON goals(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_drives_is_forgotten ON drives(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_relationships_is_forgotten ON relationships(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_notes_is_forgotten ON notes(is_forgotten);
