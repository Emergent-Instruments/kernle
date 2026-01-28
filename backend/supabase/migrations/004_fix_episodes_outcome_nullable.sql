-- Fix episodes.outcome to allow NULL values
-- Some episodes don't have an outcome yet (in progress, or just objectives)
ALTER TABLE episodes ALTER COLUMN outcome DROP NOT NULL;
