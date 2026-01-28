-- Migration: Add confidence columns to goals and episodes tables
-- These columns were missing from the initial Supabase schema but exist in local SQLite

-- Add confidence to goals table
ALTER TABLE goals ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;

-- Add confidence to episodes table  
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS confidence REAL DEFAULT 0.8;
