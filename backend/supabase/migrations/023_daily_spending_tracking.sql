-- =============================================================================
-- Migration 023: Daily Spending Tracking
-- =============================================================================
-- Moves daily spending tracking from in-memory to database with atomic operations.
-- Fixes: restart data loss, race conditions, and midnight UTC reset.
-- =============================================================================

-- Add columns for persistent daily spending tracking
ALTER TABLE wallet_accounts 
ADD COLUMN IF NOT EXISTS daily_spent DECIMAL(18, 6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS daily_spent_date DATE DEFAULT CURRENT_DATE;

-- Add constraint for non-negative spending
ALTER TABLE wallet_accounts 
ADD CONSTRAINT non_negative_daily_spent CHECK (daily_spent >= 0);

-- Index for efficient date-based queries
CREATE INDEX IF NOT EXISTS idx_wallet_daily_spent_date ON wallet_accounts(daily_spent_date);

-- =============================================================================
-- Atomic Spend Increment Function
-- =============================================================================
-- Atomically increments daily spending with limit checking.
-- Resets at midnight UTC. Returns NULL if limit would be exceeded.
-- =============================================================================

CREATE OR REPLACE FUNCTION increment_wallet_daily_spend(
    p_wallet_id UUID,
    p_amount DECIMAL(18, 6)
)
RETURNS TABLE (
    wallet_id UUID,
    daily_spent DECIMAL(18, 6),
    daily_limit DECIMAL(18, 6),
    daily_spent_date DATE,
    remaining DECIMAL(18, 6)
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_today DATE := (NOW() AT TIME ZONE 'UTC')::DATE;
    v_current_spent DECIMAL(18, 6);
    v_daily_limit DECIMAL(18, 6);
    v_new_spent DECIMAL(18, 6);
BEGIN
    -- Reject non-positive amounts
    IF p_amount <= 0 THEN
        RAISE EXCEPTION 'Amount must be positive, got %', p_amount;
    END IF;

    -- Lock the row and get current state
    SELECT 
        w.id,
        CASE 
            WHEN w.daily_spent_date = v_today THEN w.daily_spent 
            ELSE 0 
        END,
        w.spending_limit_daily
    INTO wallet_id, v_current_spent, v_daily_limit
    FROM wallet_accounts w
    WHERE w.id = p_wallet_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Wallet not found: %', p_wallet_id;
    END IF;

    v_new_spent := v_current_spent + p_amount;

    -- Check if new amount would exceed limit
    IF v_new_spent > v_daily_limit THEN
        -- Return NULL to indicate limit exceeded (caller checks for empty result)
        RETURN;
    END IF;

    -- Perform atomic update with date reset if needed
    UPDATE wallet_accounts w
    SET 
        daily_spent = CASE 
            WHEN w.daily_spent_date = v_today THEN w.daily_spent + p_amount
            ELSE p_amount  -- New day, start fresh with this amount
        END,
        daily_spent_date = v_today
    WHERE w.id = p_wallet_id;

    -- Return updated state
    RETURN QUERY
    SELECT 
        w.id,
        w.daily_spent,
        w.spending_limit_daily,
        w.daily_spent_date,
        w.spending_limit_daily - w.daily_spent
    FROM wallet_accounts w
    WHERE w.id = p_wallet_id;
END;
$$;

COMMENT ON FUNCTION increment_wallet_daily_spend IS 
    'Atomically increments daily spending with limit check. Returns empty if limit exceeded. Resets at midnight UTC.';

-- =============================================================================
-- Get Daily Spend Function
-- =============================================================================
-- Returns current daily spending, handling date reset.
-- =============================================================================

CREATE OR REPLACE FUNCTION get_wallet_daily_spend(
    p_wallet_id UUID
)
RETURNS TABLE (
    wallet_id UUID,
    daily_spent DECIMAL(18, 6),
    daily_limit DECIMAL(18, 6),
    daily_spent_date DATE,
    remaining DECIMAL(18, 6)
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_today DATE := (NOW() AT TIME ZONE 'UTC')::DATE;
BEGIN
    RETURN QUERY
    SELECT 
        w.id,
        CASE 
            WHEN w.daily_spent_date = v_today THEN w.daily_spent 
            ELSE 0::DECIMAL(18, 6)
        END AS daily_spent,
        w.spending_limit_daily,
        v_today,
        w.spending_limit_daily - CASE 
            WHEN w.daily_spent_date = v_today THEN w.daily_spent 
            ELSE 0::DECIMAL(18, 6)
        END AS remaining
    FROM wallet_accounts w
    WHERE w.id = p_wallet_id;
END;
$$;

COMMENT ON FUNCTION get_wallet_daily_spend IS 
    'Returns current daily spending for a wallet, handling midnight UTC reset.';

-- =============================================================================
-- Reset Daily Spend Function (for manual/admin use)
-- =============================================================================

CREATE OR REPLACE FUNCTION reset_wallet_daily_spend(
    p_wallet_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE wallet_accounts
    SET 
        daily_spent = 0,
        daily_spent_date = (NOW() AT TIME ZONE 'UTC')::DATE
    WHERE id = p_wallet_id;
    
    RETURN FOUND;
END;
$$;

COMMENT ON FUNCTION reset_wallet_daily_spend IS 
    'Manually resets daily spending for a wallet (admin use).';

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN wallet_accounts.daily_spent IS 'Amount spent today in USDC (resets at midnight UTC)';
COMMENT ON COLUMN wallet_accounts.daily_spent_date IS 'Date of last spending update (UTC)';
