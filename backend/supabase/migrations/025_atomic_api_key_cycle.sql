-- =============================================================================
-- Migration 025: Atomic API Key Cycle
-- =============================================================================
-- Deactivate an existing API key and create a replacement in a single transaction.

CREATE OR REPLACE FUNCTION cycle_api_key(
    p_key_id UUID,
    p_user_id TEXT,
    p_key_hash TEXT,
    p_key_prefix TEXT,
    p_name TEXT
)
RETURNS TABLE (
    id UUID,
    user_id TEXT,
    key_prefix TEXT,
    name TEXT,
    created_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Lock the existing key row to prevent concurrent cycles
    PERFORM 1
    FROM api_keys
    WHERE id = p_key_id
      AND user_id = p_user_id
      AND is_active = TRUE
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'API key not found or inactive';
    END IF;

    UPDATE api_keys
    SET is_active = FALSE
    WHERE id = p_key_id
      AND user_id = p_user_id
      AND is_active = TRUE;

    RETURN QUERY
    INSERT INTO api_keys (
        user_id,
        key_hash,
        key_prefix,
        name,
        is_active
    )
    VALUES (
        p_user_id,
        p_key_hash,
        p_key_prefix,
        p_name,
        TRUE
    )
    RETURNING
        api_keys.id,
        api_keys.user_id,
        api_keys.key_prefix,
        api_keys.name,
        api_keys.created_at,
        api_keys.last_used_at,
        api_keys.is_active;
END;
$$;

COMMENT ON FUNCTION cycle_api_key IS 'Atomically deactivates an API key and creates a replacement.';
