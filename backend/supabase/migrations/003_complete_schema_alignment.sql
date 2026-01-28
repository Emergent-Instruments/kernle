-- =============================================================================
-- Migration 003: Complete Schema Alignment with SQLite
-- =============================================================================
-- This migration adds all missing columns to align Supabase with the local SQLite schema.
-- The SQLite schema has many meta-memory and forgetting fields that weren't in the initial migration.

-- =============================================================================
-- EPISODES TABLE
-- =============================================================================

-- Core field
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS outcome_type TEXT;

-- Emotional fields (rename to match SQLite naming)
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS emotional_valence REAL DEFAULT 0.0;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS emotional_arousal REAL DEFAULT 0.0;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS emotional_tags JSONB DEFAULT '[]';

-- Meta-memory fields
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS confidence_history JSONB;

-- Forgetting fields
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;

-- Create indexes for emotional search
CREATE INDEX IF NOT EXISTS idx_episodes_emotional_valence ON episodes(emotional_valence);
CREATE INDEX IF NOT EXISTS idx_episodes_emotional_arousal ON episodes(emotional_arousal);
CREATE INDEX IF NOT EXISTS idx_episodes_confidence ON episodes(confidence);
CREATE INDEX IF NOT EXISTS idx_episodes_is_forgotten ON episodes(is_forgotten);

-- =============================================================================
-- BELIEFS TABLE
-- =============================================================================

-- Core fields
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS belief_type TEXT DEFAULT 'fact';

-- Meta-memory fields
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS confidence_history JSONB;

-- Belief revision fields
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS supersedes TEXT;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS superseded_by TEXT;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS times_reinforced INTEGER DEFAULT 0;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- Forgetting fields
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);
CREATE INDEX IF NOT EXISTS idx_beliefs_is_active ON beliefs(is_active);
CREATE INDEX IF NOT EXISTS idx_beliefs_is_forgotten ON beliefs(is_forgotten);

-- =============================================================================
-- VALUES TABLE
-- =============================================================================

-- Core fields (SQLite has 'statement', Supabase has 'description')
ALTER TABLE values ADD COLUMN IF NOT EXISTS statement TEXT;
ALTER TABLE values ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 50;

-- Meta-memory fields
ALTER TABLE values ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.9;
ALTER TABLE values ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE values ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE values ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE values ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE values ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE values ADD COLUMN IF NOT EXISTS confidence_history JSONB;

-- Forgetting fields (values protected by default)
ALTER TABLE values ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE values ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE values ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT TRUE;
ALTER TABLE values ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE values ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE values ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_values_confidence ON values(confidence);
CREATE INDEX IF NOT EXISTS idx_values_is_forgotten ON values(is_forgotten);

-- =============================================================================
-- GOALS TABLE
-- =============================================================================

-- Core fields (SQLite has 'title', Supabase has 'description' - add title for compatibility)
ALTER TABLE goals ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS target_date TIMESTAMPTZ;

-- Meta-memory fields
ALTER TABLE goals ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE goals ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS confidence_history JSONB;

-- Forgetting fields
ALTER TABLE goals ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_goals_confidence ON goals(confidence);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_is_forgotten ON goals(is_forgotten);

-- =============================================================================
-- NOTES TABLE
-- =============================================================================

-- Core fields
ALTER TABLE notes ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]';

-- Meta-memory fields
ALTER TABLE notes ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE notes ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS confidence_history JSONB;

-- Forgetting fields
ALTER TABLE notes ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_notes_confidence ON notes(confidence);
CREATE INDEX IF NOT EXISTS idx_notes_is_forgotten ON notes(is_forgotten);

-- =============================================================================
-- DRIVES TABLE
-- =============================================================================

-- Core fields (SQLite uses 'intensity', Supabase uses 'strength')
ALTER TABLE drives ADD COLUMN IF NOT EXISTS intensity REAL DEFAULT 0.5;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS focus_areas JSONB DEFAULT '[]';
ALTER TABLE drives ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Meta-memory fields
ALTER TABLE drives ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE drives ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS confidence_history JSONB;

-- Forgetting fields (drives protected by default)
ALTER TABLE drives ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT TRUE;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE drives ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_drives_confidence ON drives(confidence);
CREATE INDEX IF NOT EXISTS idx_drives_is_forgotten ON drives(is_forgotten);

-- =============================================================================
-- RELATIONSHIPS TABLE
-- =============================================================================

-- Core fields (SQLite uses entity_name/entity_type, Supabase uses entity)
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS entity_name TEXT;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS entity_type TEXT;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS notes TEXT;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS trust_level REAL DEFAULT 0.5;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS interaction_count INTEGER DEFAULT 0;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS last_interaction TIMESTAMPTZ;

-- Meta-memory fields
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS source_episodes JSONB;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS derived_from JSONB;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS last_verified TIMESTAMPTZ;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS verification_count INTEGER DEFAULT 0;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS confidence_history JSONB;

-- Forgetting fields
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS times_accessed INTEGER DEFAULT 0;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS is_protected BOOLEAN DEFAULT FALSE;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS is_forgotten BOOLEAN DEFAULT FALSE;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS forgotten_at TIMESTAMPTZ;
ALTER TABLE relationships ADD COLUMN IF NOT EXISTS forgotten_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON relationships(confidence);
CREATE INDEX IF NOT EXISTS idx_relationships_is_forgotten ON relationships(is_forgotten);

-- =============================================================================
-- PLAYBOOKS TABLE
-- =============================================================================

-- Core fields
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS failure_modes JSONB DEFAULT '[]';
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS recovery_steps JSONB;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS mastery_level TEXT DEFAULT 'novice';
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS times_used INTEGER DEFAULT 0;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS success_rate REAL DEFAULT 0.0;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]';
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS last_used TIMESTAMPTZ;

-- Meta-memory fields
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
ALTER TABLE playbooks ADD COLUMN IF NOT EXISTS source_episodes JSONB;

CREATE INDEX IF NOT EXISTS idx_playbooks_mastery ON playbooks(mastery_level);
CREATE INDEX IF NOT EXISTS idx_playbooks_times_used ON playbooks(times_used);
CREATE INDEX IF NOT EXISTS idx_playbooks_confidence ON playbooks(confidence);

-- =============================================================================
-- RAW_CAPTURES TABLE (SQLite calls it raw_entries)
-- =============================================================================

-- Core fields
ALTER TABLE raw_captures ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE raw_captures ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'manual';
ALTER TABLE raw_captures ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]';

-- Meta-memory fields
ALTER TABLE raw_captures ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 1.0;
ALTER TABLE raw_captures ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'direct_experience';

CREATE INDEX IF NOT EXISTS idx_raw_captures_processed ON raw_captures(processed);
CREATE INDEX IF NOT EXISTS idx_raw_captures_timestamp ON raw_captures(timestamp);

-- =============================================================================
-- Migrate existing data (map old columns to new where applicable)
-- =============================================================================

-- Copy valence/arousal to emotional_valence/emotional_arousal if they exist
UPDATE episodes SET emotional_valence = valence WHERE valence IS NOT NULL AND emotional_valence = 0.0;
UPDATE episodes SET emotional_arousal = arousal WHERE arousal IS NOT NULL AND emotional_arousal = 0.0;

-- Copy statement to description in values if needed (when statement is set but description isn't)
UPDATE values SET description = statement WHERE statement IS NOT NULL AND description IS NULL;

-- Copy entity to entity_name in relationships if needed
UPDATE relationships SET entity_name = entity WHERE entity IS NOT NULL AND entity_name IS NULL;

-- Copy title to description in goals if needed (for backwards compatibility)
UPDATE goals SET description = title WHERE title IS NOT NULL AND description IS NULL;
