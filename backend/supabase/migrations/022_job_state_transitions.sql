-- =============================================================================
-- Migration 022: Job State Transitions (Audit Log)
-- =============================================================================
-- Audit log for job state changes. Every status transition is recorded for
-- accountability, debugging, and dispute resolution.
-- =============================================================================

CREATE TABLE IF NOT EXISTS job_state_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    from_status VARCHAR(20),                          -- NULL for initial creation
    to_status VARCHAR(20) NOT NULL,
    actor_id TEXT NOT NULL,                           -- Who triggered the transition
    tx_hash VARCHAR(66),                              -- Blockchain tx hash if applicable
    metadata JSONB DEFAULT '{}',                      -- Additional context
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_from_status CHECK (
        from_status IS NULL OR from_status IN (
            'open', 'funded', 'accepted', 'delivered', 'completed', 'disputed', 'cancelled'
        )
    ),
    CONSTRAINT valid_to_status CHECK (to_status IN (
        'open', 'funded', 'accepted', 'delivered', 'completed', 'disputed', 'cancelled'
    )),
    CONSTRAINT valid_tx_hash CHECK (
        tx_hash IS NULL OR tx_hash ~ '^0x[a-fA-F0-9]{64}$'
    ),
    
    -- Validate state machine transitions
    CONSTRAINT valid_state_transition CHECK (
        -- Initial creation
        (from_status IS NULL AND to_status = 'open')
        -- open -> funded, cancelled
        OR (from_status = 'open' AND to_status IN ('funded', 'cancelled'))
        -- funded -> accepted, cancelled
        OR (from_status = 'funded' AND to_status IN ('accepted', 'cancelled'))
        -- accepted -> delivered, disputed, cancelled
        OR (from_status = 'accepted' AND to_status IN ('delivered', 'disputed', 'cancelled'))
        -- delivered -> completed, disputed
        OR (from_status = 'delivered' AND to_status IN ('completed', 'disputed'))
        -- disputed -> completed (after arbitration)
        OR (from_status = 'disputed' AND to_status = 'completed')
    )
);

-- Indexes
CREATE INDEX idx_transitions_job ON job_state_transitions(job_id);
CREATE INDEX idx_transitions_created ON job_state_transitions(created_at DESC);
CREATE INDEX idx_transitions_actor ON job_state_transitions(actor_id);
CREATE INDEX idx_transitions_to_status ON job_state_transitions(to_status);

-- Composite index for job timeline queries
CREATE INDEX idx_transitions_job_created ON job_state_transitions(job_id, created_at);

-- Row Level Security
ALTER TABLE job_state_transitions ENABLE ROW LEVEL SECURITY;

-- Service role can access everything
CREATE POLICY "Service role full access" ON job_state_transitions 
    FOR ALL TO service_role USING (true);

-- =============================================================================
-- Helper function to log state transition
-- =============================================================================

CREATE OR REPLACE FUNCTION log_job_transition(
    p_job_id UUID,
    p_from_status VARCHAR(20),
    p_to_status VARCHAR(20),
    p_actor_id TEXT,
    p_tx_hash VARCHAR(66) DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    transition_id UUID;
BEGIN
    INSERT INTO job_state_transitions (job_id, from_status, to_status, actor_id, tx_hash, metadata)
    VALUES (p_job_id, p_from_status, p_to_status, p_actor_id, p_tx_hash, p_metadata)
    RETURNING id INTO transition_id;
    
    RETURN transition_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Trigger to auto-log transitions when jobs.status changes
-- =============================================================================

CREATE OR REPLACE FUNCTION auto_log_job_transition()
RETURNS TRIGGER AS $$
BEGIN
    -- Only log if status actually changed
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO job_state_transitions (
            job_id, 
            from_status, 
            to_status, 
            actor_id,
            metadata
        )
        VALUES (
            NEW.id,
            OLD.status,
            NEW.status,
            COALESCE(current_setting('app.current_actor', true), 'system'),
            jsonb_build_object(
                'triggered_by', 'auto_trigger',
                'old_worker_id', OLD.worker_id,
                'new_worker_id', NEW.worker_id
            )
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER jobs_state_transition_trigger
    AFTER UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION auto_log_job_transition();

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE job_state_transitions IS 'Audit log for job state changes';
COMMENT ON COLUMN job_state_transitions.job_id IS 'ID of the job';
COMMENT ON COLUMN job_state_transitions.from_status IS 'Previous status (NULL for creation)';
COMMENT ON COLUMN job_state_transitions.to_status IS 'New status';
COMMENT ON COLUMN job_state_transitions.actor_id IS 'Agent ID who triggered the transition';
COMMENT ON COLUMN job_state_transitions.tx_hash IS 'Blockchain transaction hash (if applicable)';
COMMENT ON COLUMN job_state_transitions.metadata IS 'Additional context (reason, notes, etc.)';
COMMENT ON FUNCTION log_job_transition IS 'Log a job state transition with metadata';
COMMENT ON FUNCTION auto_log_job_transition IS 'Auto-log transitions when jobs.status changes';
