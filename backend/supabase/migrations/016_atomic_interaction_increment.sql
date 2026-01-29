-- =============================================================================
-- Migration 015: Atomic Interaction Count Increment
-- =============================================================================
-- Fixes race condition in relationship interaction_count tracking by using 
-- atomic database operations instead of read-then-write in application code.

CREATE OR REPLACE FUNCTION increment_interaction_count(
    p_agent_id TEXT,
    p_other_agent_id TEXT,
    p_trust_level FLOAT DEFAULT NULL,
    p_notes TEXT DEFAULT NULL,
    p_last_interaction TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    interaction_count INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_now TIMESTAMPTZ := COALESCE(p_last_interaction, NOW() AT TIME ZONE 'UTC');
    v_id UUID;
    v_count INTEGER;
BEGIN
    -- Try to update existing relationship atomically
    UPDATE agent_relationships
    SET 
        interaction_count = agent_relationships.interaction_count + 1,
        last_interaction = v_now,
        trust_level = COALESCE(p_trust_level, trust_level),
        notes = COALESCE(p_notes, notes),
        local_updated_at = v_now,
        cloud_synced_at = v_now
    WHERE agent_id = p_agent_id 
      AND other_agent_id = p_other_agent_id
    RETURNING agent_relationships.id, agent_relationships.interaction_count
    INTO v_id, v_count;
    
    -- If no row was updated, insert a new one
    IF NOT FOUND THEN
        INSERT INTO agent_relationships (
            id,
            agent_id,
            other_agent_id,
            trust_level,
            interaction_count,
            last_interaction,
            notes,
            local_updated_at,
            cloud_synced_at,
            version
        )
        VALUES (
            gen_random_uuid(),
            p_agent_id,
            p_other_agent_id,
            COALESCE(p_trust_level, 0.5),
            1,
            v_now,
            p_notes,
            v_now,
            v_now,
            1
        )
        RETURNING agent_relationships.id, agent_relationships.interaction_count
        INTO v_id, v_count;
    END IF;
    
    RETURN QUERY SELECT v_id, v_count;
END;
$$;

COMMENT ON FUNCTION increment_interaction_count IS 'Atomically increments interaction_count for a relationship. Fixes race condition in relationship tracking.';
