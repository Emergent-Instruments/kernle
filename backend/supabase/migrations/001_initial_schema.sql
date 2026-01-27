-- =============================================================================
-- Kernle Backend Schema
-- =============================================================================

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Agents Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT UNIQUE NOT NULL,
    secret_hash TEXT NOT NULL,
    display_name TEXT,
    email TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_sync_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT agent_id_format CHECK (agent_id ~ '^[a-z0-9_-]+$')
);

CREATE INDEX idx_agents_agent_id ON agents(agent_id);

-- =============================================================================
-- Memory Tables (match local SQLite schema)
-- =============================================================================

-- Episodes
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    objective TEXT NOT NULL,
    outcome TEXT NOT NULL,
    lessons JSONB DEFAULT '[]',
    tags JSONB DEFAULT '[]',
    emotions JSONB DEFAULT '[]',
    valence REAL,
    arousal REAL,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_episodes_agent ON episodes(agent_id);
CREATE INDEX idx_episodes_deleted ON episodes(deleted);

-- Beliefs
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source TEXT,
    evidence JSONB DEFAULT '[]',
    contradicts JSONB DEFAULT '[]',
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_beliefs_agent ON beliefs(agent_id);

-- Values
CREATE TABLE IF NOT EXISTS values (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    importance REAL DEFAULT 0.5,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_values_agent ON values(agent_id);

-- Goals
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'active',
    deadline TIMESTAMPTZ,
    progress REAL DEFAULT 0.0,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_goals_agent ON goals(agent_id);

-- Notes
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    note_type TEXT DEFAULT 'note',
    speaker TEXT,
    reason TEXT,
    protected BOOLEAN DEFAULT FALSE,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_notes_agent ON notes(agent_id);

-- Drives
CREATE TABLE IF NOT EXISTS drives (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    drive_type TEXT NOT NULL,
    description TEXT,
    strength REAL DEFAULT 0.5,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_drives_agent ON drives(agent_id);

-- Relationships
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    entity TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    description TEXT,
    sentiment REAL DEFAULT 0.0,
    trust REAL DEFAULT 0.5,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_relationships_agent ON relationships(agent_id);

-- Checkpoints
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    current_task TEXT,
    pending JSONB DEFAULT '[]',
    context TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_checkpoints_agent ON checkpoints(agent_id);

-- Raw Captures
CREATE TABLE IF NOT EXISTS raw_captures (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    processed_into TEXT,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_raw_captures_agent ON raw_captures(agent_id);

-- Playbooks
CREATE TABLE IF NOT EXISTS playbooks (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    trigger_conditions JSONB DEFAULT '[]',
    steps JSONB DEFAULT '[]',
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_playbooks_agent ON playbooks(agent_id);

-- Emotional Memories
CREATE TABLE IF NOT EXISTS emotional_memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    trigger_event TEXT NOT NULL,
    emotional_response TEXT NOT NULL,
    valence REAL,
    arousal REAL,
    intensity REAL,
    coping_strategy TEXT,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_emotional_memories_agent ON emotional_memories(agent_id);

-- =============================================================================
-- Row Level Security (RLS)
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE beliefs ENABLE ROW LEVEL SECURITY;
ALTER TABLE values ENABLE ROW LEVEL SECURITY;
ALTER TABLE goals ENABLE ROW LEVEL SECURITY;
ALTER TABLE notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE drives ENABLE ROW LEVEL SECURITY;
ALTER TABLE relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_captures ENABLE ROW LEVEL SECURITY;
ALTER TABLE playbooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE emotional_memories ENABLE ROW LEVEL SECURITY;

-- Service role can access everything (for backend)
CREATE POLICY "Service role full access" ON agents FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON episodes FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON beliefs FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON values FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON goals FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON notes FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON drives FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON relationships FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON checkpoints FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON raw_captures FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON playbooks FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON emotional_memories FOR ALL TO service_role USING (true);
