-- =============================================================================
-- Migration 023: Timeout Enforcement for Jobs
-- =============================================================================
-- Adds fields and constraints needed for timeout enforcement:
-- - disputed_at: When a dispute was raised (for timeout calculation)
-- - delivery_deadline: The actual delivery deadline (may differ from original deadline)
-- - Updates state machine to allow disputed -> cancelled transition
-- =============================================================================

-- Add new timestamp fields for timeout tracking
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS disputed_at TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ;

-- Update the status constraint to allow the disputed -> cancelled transition
-- First drop the old constraint
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS valid_job_status;

-- Re-add with same valid statuses (the constraint itself doesn't control transitions)
ALTER TABLE jobs ADD CONSTRAINT valid_job_status CHECK (status IN (
    'open', 'funded', 'accepted', 'delivered', 'completed', 'disputed', 'cancelled'
));

-- Update the state transition constraint to allow disputed -> cancelled
ALTER TABLE job_state_transitions DROP CONSTRAINT IF EXISTS valid_state_transition;

ALTER TABLE job_state_transitions ADD CONSTRAINT valid_state_transition CHECK (
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
    -- disputed -> completed (after arbitration), cancelled (timeout/escalation)
    OR (from_status = 'disputed' AND to_status IN ('completed', 'cancelled'))
);

-- Index for finding jobs past deadline
CREATE INDEX IF NOT EXISTS idx_jobs_deadline_status ON jobs(deadline, status)
    WHERE status IN ('accepted');

-- Index for finding disputed jobs by dispute time
CREATE INDEX IF NOT EXISTS idx_jobs_disputed_at ON jobs(disputed_at)
    WHERE status = 'disputed';

-- =============================================================================
-- Function to check and auto-cancel overdue jobs
-- =============================================================================

CREATE OR REPLACE FUNCTION check_job_timeouts(
    deadline_grace_hours INTEGER DEFAULT 24,
    dispute_timeout_days INTEGER DEFAULT 14
)
RETURNS TABLE(
    job_id UUID,
    action_taken TEXT,
    reason TEXT
) AS $$
DECLARE
    job_record RECORD;
    deadline_cutoff TIMESTAMPTZ;
    dispute_cutoff TIMESTAMPTZ;
BEGIN
    deadline_cutoff := NOW() - (deadline_grace_hours || ' hours')::INTERVAL;
    dispute_cutoff := NOW() - (dispute_timeout_days || ' days')::INTERVAL;

    -- Auto-cancel jobs past deadline without delivery
    FOR job_record IN
        SELECT j.id, j.deadline, j.status
        FROM jobs j
        WHERE j.status = 'accepted'
          AND j.deadline < deadline_cutoff
          AND j.delivered_at IS NULL
    LOOP
        -- Update job status to cancelled
        UPDATE jobs
        SET status = 'cancelled',
            cancelled_at = NOW()
        WHERE id = job_record.id;

        -- Log the transition
        INSERT INTO job_state_transitions (
            job_id, from_status, to_status, actor_id, metadata
        ) VALUES (
            job_record.id,
            'accepted',
            'cancelled',
            'system:timeout',
            jsonb_build_object(
                'reason', 'deadline_exceeded',
                'deadline', job_record.deadline,
                'grace_hours', deadline_grace_hours
            )
        );

        job_id := job_record.id;
        action_taken := 'cancelled';
        reason := 'Deadline exceeded without delivery';
        RETURN NEXT;
    END LOOP;

    -- Auto-cancel (escalate) disputes past timeout
    FOR job_record IN
        SELECT j.id, j.disputed_at, j.status
        FROM jobs j
        WHERE j.status = 'disputed'
          AND j.disputed_at IS NOT NULL
          AND j.disputed_at < dispute_cutoff
    LOOP
        -- For now, auto-cancel disputes that haven't been resolved
        -- In production, this might trigger an escalation workflow instead
        UPDATE jobs
        SET status = 'cancelled',
            cancelled_at = NOW()
        WHERE id = job_record.id;

        -- Log the transition
        INSERT INTO job_state_transitions (
            job_id, from_status, to_status, actor_id, metadata
        ) VALUES (
            job_record.id,
            'disputed',
            'cancelled',
            'system:timeout',
            jsonb_build_object(
                'reason', 'dispute_timeout',
                'disputed_at', job_record.disputed_at,
                'timeout_days', dispute_timeout_days
            )
        );

        job_id := job_record.id;
        action_taken := 'cancelled';
        reason := 'Dispute timeout exceeded - escalated to cancellation';
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN jobs.disputed_at IS 'Timestamp when job entered disputed status';
COMMENT ON COLUMN jobs.cancelled_at IS 'Timestamp when job was cancelled';
COMMENT ON FUNCTION check_job_timeouts IS 'Check and auto-cancel overdue jobs and stale disputes. Call periodically via cron.';
