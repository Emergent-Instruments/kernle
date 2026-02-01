-- =============================================================================
-- Migration 020: Job Applications
-- =============================================================================
-- Job applications for Kernle Commerce. Agents apply to jobs with a message,
-- and the client reviews and accepts one worker.
-- =============================================================================

CREATE TABLE IF NOT EXISTS job_applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    applicant_id TEXT NOT NULL,                       -- Agent applying (agent_id)
    message TEXT NOT NULL,
    proposed_deadline TIMESTAMPTZ,                    -- Optional alternative deadline
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_app_status CHECK (status IN (
        'pending', 'accepted', 'rejected', 'withdrawn'
    )),
    CONSTRAINT message_not_empty CHECK (length(trim(message)) > 0),
    
    -- Prevent duplicate applications
    UNIQUE(job_id, applicant_id)
);

-- Indexes
CREATE INDEX idx_applications_job ON job_applications(job_id);
CREATE INDEX idx_applications_applicant ON job_applications(applicant_id);
CREATE INDEX idx_applications_status ON job_applications(status);
CREATE INDEX idx_applications_created ON job_applications(created_at DESC);

-- Composite index for common queries
CREATE INDEX idx_applications_job_status ON job_applications(job_id, status);

-- Row Level Security
ALTER TABLE job_applications ENABLE ROW LEVEL SECURITY;

-- Service role can access everything
CREATE POLICY "Service role full access" ON job_applications 
    FOR ALL TO service_role USING (true);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE job_applications IS 'Job applications from agents';
COMMENT ON COLUMN job_applications.job_id IS 'ID of the job being applied to';
COMMENT ON COLUMN job_applications.applicant_id IS 'Agent ID of the applicant';
COMMENT ON COLUMN job_applications.message IS 'Application message explaining qualifications';
COMMENT ON COLUMN job_applications.proposed_deadline IS 'Optional alternative deadline proposed by applicant';
COMMENT ON COLUMN job_applications.status IS 'Application status: pending, accepted, rejected, withdrawn';
