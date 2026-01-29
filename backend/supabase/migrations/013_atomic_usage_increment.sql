-- =============================================================================
-- Migration 013: Atomic Usage Increment
-- =============================================================================
-- Fixes race condition in usage tracking by using atomic database operations
-- instead of read-then-write in application code.

CREATE OR REPLACE FUNCTION increment_api_usage(
    p_api_key_id UUID,
    p_user_id TEXT
)
RETURNS TABLE (
    daily_requests INTEGER,
    monthly_requests INTEGER,
    daily_reset_at TIMESTAMPTZ,
    monthly_reset_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_now TIMESTAMPTZ := NOW() AT TIME ZONE 'UTC';
    v_next_daily TIMESTAMPTZ := DATE_TRUNC('day', v_now) + INTERVAL '1 day';
    v_next_monthly TIMESTAMPTZ := DATE_TRUNC('month', v_now) + INTERVAL '1 month';
BEGIN
    -- Upsert with atomic increment, handling resets inline
    -- Uses ON CONFLICT to handle both insert and update atomically
    INSERT INTO api_key_usage (
        api_key_id,
        user_id,
        daily_requests,
        monthly_requests,
        daily_reset_at,
        monthly_reset_at,
        updated_at
    )
    VALUES (
        p_api_key_id,
        p_user_id,
        1,  -- First request
        1,
        v_next_daily,
        v_next_monthly,
        v_now
    )
    ON CONFLICT (api_key_id) DO UPDATE SET
        -- Reset daily if past reset time, then increment
        daily_requests = CASE 
            WHEN api_key_usage.daily_reset_at <= v_now THEN 1
            ELSE api_key_usage.daily_requests + 1
        END,
        daily_reset_at = CASE 
            WHEN api_key_usage.daily_reset_at <= v_now THEN v_next_daily
            ELSE api_key_usage.daily_reset_at
        END,
        -- Reset monthly if past reset time, then increment
        monthly_requests = CASE 
            WHEN api_key_usage.monthly_reset_at <= v_now THEN 1
            ELSE api_key_usage.monthly_requests + 1
        END,
        monthly_reset_at = CASE 
            WHEN api_key_usage.monthly_reset_at <= v_now THEN v_next_monthly
            ELSE api_key_usage.monthly_reset_at
        END,
        updated_at = v_now;

    -- Return the updated values
    RETURN QUERY
    SELECT 
        u.daily_requests,
        u.monthly_requests,
        u.daily_reset_at,
        u.monthly_reset_at
    FROM api_key_usage u
    WHERE u.api_key_id = p_api_key_id;
END;
$$;

COMMENT ON FUNCTION increment_api_usage IS 'Atomically increments usage counters, handling period resets. Fixes race condition in quota tracking.';
