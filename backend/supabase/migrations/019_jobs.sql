-- =============================================================================
-- Migration 019: Jobs
-- =============================================================================
-- Jobs marketplace table for Kernle Commerce. Jobs are work listings that
-- agents can post (as clients) or apply to (as workers).
-- =============================================================================

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id TEXT NOT NULL,                          -- Agent who posted (agent_id)
    worker_id TEXT,                                   -- Agent who accepted
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    skills_required TEXT[] DEFAULT '{}',              -- Array of skill names
    budget_usdc DECIMAL(18, 6) NOT NULL,
    escrow_address VARCHAR(42),                       -- Deployed escrow contract
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    deadline TIMESTAMPTZ NOT NULL,
    deliverable_url TEXT,                             -- URL to delivered work
    deliverable_hash VARCHAR(66),                     -- IPFS or content hash
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    funded_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_job_status CHECK (status IN (
        'open', 'funded', 'accepted', 'delivered', 'completed', 'disputed', 'cancelled'
    )),
    CONSTRAINT positive_budget CHECK (budget_usdc > 0),
    CONSTRAINT valid_escrow_address CHECK (
        escrow_address IS NULL OR escrow_address ~ '^0x[a-fA-F0-9]{40}$'
    ),
    CONSTRAINT title_not_empty CHECK (length(trim(title)) > 0),
    CONSTRAINT description_not_empty CHECK (length(trim(description)) > 0),
    CONSTRAINT deadline_in_future CHECK (deadline > created_at)
);

-- Indexes
CREATE INDEX idx_jobs_client ON jobs(client_id);
CREATE INDEX idx_jobs_worker ON jobs(worker_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_skills ON jobs USING GIN(skills_required);
CREATE INDEX idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX idx_jobs_deadline ON jobs(deadline);
CREATE INDEX idx_jobs_budget ON jobs(budget_usdc);

-- Full text search on title and description
CREATE INDEX idx_jobs_fts ON jobs USING GIN(
    to_tsvector('english', title || ' ' || description)
);

-- Composite index for common queries
CREATE INDEX idx_jobs_status_created ON jobs(status, created_at DESC);

-- Row Level Security
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

-- Service role can access everything
CREATE POLICY "Service role full access" ON jobs 
    FOR ALL TO service_role USING (true);

-- =============================================================================
-- Auto-update updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER jobs_updated_at_trigger
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_jobs_updated_at();

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE jobs IS 'Jobs marketplace listings for Kernle Commerce';
COMMENT ON COLUMN jobs.client_id IS 'Agent ID of the job poster';
COMMENT ON COLUMN jobs.worker_id IS 'Agent ID of accepted worker';
COMMENT ON COLUMN jobs.title IS 'Job title (max 200 chars)';
COMMENT ON COLUMN jobs.description IS 'Full job description';
COMMENT ON COLUMN jobs.skills_required IS 'Array of required skill names';
COMMENT ON COLUMN jobs.budget_usdc IS 'Payment amount in USDC';
COMMENT ON COLUMN jobs.escrow_address IS 'Deployed escrow contract address';
COMMENT ON COLUMN jobs.status IS 'Job status: open, funded, accepted, delivered, completed, disputed, cancelled';
COMMENT ON COLUMN jobs.deadline IS 'When work must be delivered';
COMMENT ON COLUMN jobs.deliverable_url IS 'URL to delivered work (set by worker)';
COMMENT ON COLUMN jobs.deliverable_hash IS 'IPFS or content hash of deliverable';
