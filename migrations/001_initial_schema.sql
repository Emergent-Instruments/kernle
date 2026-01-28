-- Kernle Initial Schema
-- Run this against your Supabase instance to set up the database

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- CORE MEMORY TABLES
-- =============================================================================

-- Episodes (experiences with lessons learned)
CREATE TABLE IF NOT EXISTS agent_episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    outcome_description TEXT,  -- Maps to 'outcome' in storage
    outcome_type TEXT,  -- 'success', 'failure', 'partial', 'ongoing'
    lessons TEXT[],
    tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Emotional memory
    emotional_valence FLOAT DEFAULT 0.0,  -- -1.0 to 1.0
    emotional_arousal FLOAT DEFAULT 0.0,  -- 0.0 to 1.0
    emotional_tags TEXT[],
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    -- Meta-memory
    confidence FLOAT DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes UUID[],
    derived_from UUID[],
    last_verified TIMESTAMPTZ,
    verification_count INTEGER DEFAULT 0,
    confidence_history JSONB,
    
    -- Forgetting
    times_accessed INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    is_protected BOOLEAN DEFAULT FALSE,
    is_forgotten BOOLEAN DEFAULT FALSE,
    forgotten_at TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 1.0,
    
    -- Embedding for semantic search
    embedding vector(384)
);

-- Beliefs (what the agent holds true)
CREATE TABLE IF NOT EXISTS agent_beliefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    source TEXT,
    evidence TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Belief revision
    supersedes UUID,  -- ID of belief this supersedes
    superseded_by UUID,  -- ID of belief that superseded this
    revision_reason TEXT,
    active BOOLEAN DEFAULT TRUE,
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    -- Forgetting
    times_accessed INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    is_protected BOOLEAN DEFAULT FALSE,
    is_forgotten BOOLEAN DEFAULT FALSE,
    forgotten_at TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 1.0,
    
    -- Embedding
    embedding vector(384)
);

-- Values (core identity, non-negotiable)
CREATE TABLE IF NOT EXISTS agent_values (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 50,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    -- Forgetting
    times_accessed INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    is_protected BOOLEAN DEFAULT TRUE,  -- Values protected by default
    is_forgotten BOOLEAN DEFAULT FALSE,
    forgotten_at TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 0.5,  -- Slow decay for values
    
    UNIQUE(agent_id, name)
);

-- Goals (what the agent is working toward)
CREATE TABLE IF NOT EXISTS agent_goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',  -- 'low', 'medium', 'high', 'critical'
    status TEXT DEFAULT 'active',  -- 'active', 'completed', 'abandoned', 'blocked'
    progress FLOAT DEFAULT 0.0,  -- 0.0 to 1.0
    parent_goal_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    -- Forgetting
    times_accessed INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    is_protected BOOLEAN DEFAULT FALSE,
    is_forgotten BOOLEAN DEFAULT FALSE,
    forgotten_at TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 1.0
);

-- Notes/Memories (decisions, insights, observations)
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id TEXT NOT NULL,  -- Maps to agent_id
    content TEXT NOT NULL,
    source TEXT DEFAULT 'curated',
    importance FLOAT DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',  -- Stores note_type, speaker, reason
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    -- Forgetting
    times_accessed INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    is_protected BOOLEAN DEFAULT FALSE,
    is_forgotten BOOLEAN DEFAULT FALSE,
    forgotten_at TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 1.0,
    
    -- Embedding
    embedding vector(384)
);

-- Drives (motivations)
CREATE TABLE IF NOT EXISTS agent_drives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    drive_type TEXT NOT NULL,  -- 'existence', 'growth', 'curiosity', 'connection', 'reproduction'
    intensity FLOAT DEFAULT 0.5,  -- 0.0 to 1.0
    last_satisfied_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    UNIQUE(agent_id, drive_type)
);

-- Relationships (connections to other entities)
CREATE TABLE IF NOT EXISTS agent_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    other_agent_id TEXT NOT NULL,  -- Entity name in local storage
    relationship_type TEXT DEFAULT 'acquaintance',
    trust_level FLOAT DEFAULT 0.5,
    interaction_count INTEGER DEFAULT 0,
    last_interaction TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    UNIQUE(agent_id, other_agent_id)
);

-- Raw entries (quick captures, scratchpad)
CREATE TABLE IF NOT EXISTS raw_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT[],
    source TEXT DEFAULT 'cli',
    processed BOOLEAN DEFAULT FALSE,
    processed_into_type TEXT,  -- 'episode', 'note', 'belief'
    processed_into_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

-- Playbooks (procedural memory)
CREATE TABLE IF NOT EXISTS playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    steps JSONB NOT NULL,  -- Array of step objects
    trigger_conditions TEXT[],
    tags TEXT[],
    times_used INTEGER DEFAULT 0,
    success_rate FLOAT,
    last_used_at TIMESTAMPTZ,
    mastery_level TEXT DEFAULT 'novice',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    
    UNIQUE(agent_id, name)
);

-- Checkpoints (working state snapshots)
CREATE TABLE IF NOT EXISTS checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    task TEXT NOT NULL,
    pending TEXT[],
    context TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1
);

-- =============================================================================
-- API & SYNC TABLES
-- =============================================================================

-- API Keys for authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,  -- SHA256 hash of the key
    name TEXT,  -- Optional friendly name
    scopes TEXT[] DEFAULT ARRAY['sync', 'read', 'write'],
    rate_limit INTEGER DEFAULT 1000,  -- requests per hour
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(key_hash)
);

-- Sync logs for debugging
CREATE TABLE IF NOT EXISTS sync_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'push' or 'pull'
    records_synced INTEGER DEFAULT 0,
    tables_synced TEXT[],
    duration_ms INTEGER,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sync metadata per agent
CREATE TABLE IF NOT EXISTS sync_metadata (
    agent_id TEXT PRIMARY KEY,
    last_push_at TIMESTAMPTZ,
    last_pull_at TIMESTAMPTZ,
    last_full_sync_at TIMESTAMPTZ,
    sync_version INTEGER DEFAULT 1
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Agent ID indexes for fast filtering
CREATE INDEX IF NOT EXISTS idx_episodes_agent ON agent_episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_agent ON agent_beliefs(agent_id);
CREATE INDEX IF NOT EXISTS idx_values_agent ON agent_values(agent_id);
CREATE INDEX IF NOT EXISTS idx_goals_agent ON agent_goals(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_id);
CREATE INDEX IF NOT EXISTS idx_drives_agent ON agent_drives(agent_id);
CREATE INDEX IF NOT EXISTS idx_relationships_agent ON agent_relationships(agent_id);
CREATE INDEX IF NOT EXISTS idx_raw_agent ON raw_entries(agent_id);
CREATE INDEX IF NOT EXISTS idx_playbooks_agent ON playbooks(agent_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_agent ON checkpoints(agent_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_agent ON api_keys(agent_id);
CREATE INDEX IF NOT EXISTS idx_sync_logs_agent ON sync_logs(agent_id);

-- Sync-related indexes
CREATE INDEX IF NOT EXISTS idx_episodes_sync ON agent_episodes(agent_id, local_updated_at) WHERE cloud_synced_at IS NULL OR local_updated_at > cloud_synced_at;
CREATE INDEX IF NOT EXISTS idx_beliefs_sync ON agent_beliefs(agent_id, local_updated_at) WHERE cloud_synced_at IS NULL OR local_updated_at > cloud_synced_at;
CREATE INDEX IF NOT EXISTS idx_goals_sync ON agent_goals(agent_id, local_updated_at) WHERE cloud_synced_at IS NULL OR local_updated_at > cloud_synced_at;

-- Vector indexes for semantic search (HNSW for better performance)
CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON agent_episodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_beliefs_embedding ON agent_beliefs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE agent_episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_beliefs ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_values ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_goals ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_drives ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE playbooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Service role can do everything (for API backend)
CREATE POLICY service_all_episodes ON agent_episodes FOR ALL TO service_role USING (true);
CREATE POLICY service_all_beliefs ON agent_beliefs FOR ALL TO service_role USING (true);
CREATE POLICY service_all_values ON agent_values FOR ALL TO service_role USING (true);
CREATE POLICY service_all_goals ON agent_goals FOR ALL TO service_role USING (true);
CREATE POLICY service_all_memories ON memories FOR ALL TO service_role USING (true);
CREATE POLICY service_all_drives ON agent_drives FOR ALL TO service_role USING (true);
CREATE POLICY service_all_relationships ON agent_relationships FOR ALL TO service_role USING (true);
CREATE POLICY service_all_raw ON raw_entries FOR ALL TO service_role USING (true);
CREATE POLICY service_all_playbooks ON playbooks FOR ALL TO service_role USING (true);
CREATE POLICY service_all_checkpoints ON checkpoints FOR ALL TO service_role USING (true);
CREATE POLICY service_all_api_keys ON api_keys FOR ALL TO service_role USING (true);
