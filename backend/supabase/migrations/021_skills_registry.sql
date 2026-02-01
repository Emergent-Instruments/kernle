-- =============================================================================
-- Migration 021: Skills Registry
-- =============================================================================
-- Canonical skills registry for Kernle Commerce. Skills are tags that describe
-- agent capabilities and job requirements, enabling matching.
-- =============================================================================

CREATE TABLE IF NOT EXISTS skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50),
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_skill_name CHECK (name ~ '^[a-z0-9-]+$'),
    CONSTRAINT valid_skill_category CHECK (
        category IS NULL OR category IN (
            'technical', 'creative', 'knowledge', 'language', 'service'
        )
    ),
    CONSTRAINT non_negative_usage CHECK (usage_count >= 0)
);

-- Indexes
CREATE INDEX idx_skills_name ON skills(name);
CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_usage ON skills(usage_count DESC);

-- Full text search on name and description
CREATE INDEX idx_skills_fts ON skills USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(description, ''))
);

-- Row Level Security
ALTER TABLE skills ENABLE ROW LEVEL SECURITY;

-- Service role can access everything
CREATE POLICY "Service role full access" ON skills 
    FOR ALL TO service_role USING (true);

-- =============================================================================
-- Seed canonical skills
-- =============================================================================

INSERT INTO skills (name, description, category) VALUES
    ('research', 'Information gathering and analysis', 'knowledge'),
    ('writing', 'Content creation and copywriting', 'creative'),
    ('coding', 'Software development', 'technical'),
    ('data-analysis', 'Data processing and insights', 'technical'),
    ('automation', 'Workflow automation and scripting', 'technical'),
    ('design', 'Visual design and graphics', 'creative'),
    ('translation', 'Language translation', 'language'),
    ('summarization', 'Content summarization', 'knowledge'),
    ('customer-support', 'Customer service and support', 'service'),
    ('market-scanning', 'Market research and monitoring', 'knowledge'),
    ('web-scraping', 'Web data extraction', 'technical')
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- Atomic usage increment function
-- =============================================================================

CREATE OR REPLACE FUNCTION increment_skill_usage(skill_name VARCHAR(50))
RETURNS BOOLEAN AS $$
DECLARE
    updated_rows INTEGER;
BEGIN
    UPDATE skills 
    SET usage_count = usage_count + 1 
    WHERE name = skill_name;
    
    GET DIAGNOSTICS updated_rows = ROW_COUNT;
    RETURN updated_rows > 0;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE skills IS 'Canonical skills registry for job matching';
COMMENT ON COLUMN skills.name IS 'Skill name (lowercase, hyphenated)';
COMMENT ON COLUMN skills.description IS 'Human-readable description';
COMMENT ON COLUMN skills.category IS 'Skill category: technical, creative, knowledge, language, service';
COMMENT ON COLUMN skills.usage_count IS 'Number of jobs/agents using this skill';
COMMENT ON FUNCTION increment_skill_usage IS 'Atomically increment skill usage count';
