-- Migration 003: Rename agent_id → stack_id and drop agent_ table prefixes
-- This migration renames the "agent_id" column to "stack_id" across all tables
-- and renames tables that had the "agent_" prefix.

BEGIN;

-- 1. Rename agent_id column to stack_id in all tables
ALTER TABLE episodes RENAME COLUMN agent_id TO stack_id;
ALTER TABLE beliefs RENAME COLUMN agent_id TO stack_id;
ALTER TABLE notes RENAME COLUMN agent_id TO stack_id;
ALTER TABLE goals RENAME COLUMN agent_id TO stack_id;
ALTER TABLE drives RENAME COLUMN agent_id TO stack_id;
ALTER TABLE relationships RENAME COLUMN agent_id TO stack_id;
ALTER TABLE playbooks RENAME COLUMN agent_id TO stack_id;
ALTER TABLE raw_entries RENAME COLUMN agent_id TO stack_id;
ALTER TABLE sync_queue RENAME COLUMN agent_id TO stack_id;
ALTER TABLE checkpoints RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_values RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_trust_assessments RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_epochs RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_registry RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_diagnostic_sessions RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_diagnostic_reports RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_summaries RENAME COLUMN agent_id TO stack_id;
ALTER TABLE agent_self_narrative RENAME COLUMN agent_id TO stack_id;

-- 2. Rename tables that had agent_ prefix
-- Note: agent_values is NOT renamed (values is a SQL reserved word in some contexts)
ALTER TABLE agent_trust_assessments RENAME TO trust_assessments;
ALTER TABLE agent_epochs RENAME TO epochs;
ALTER TABLE agent_registry RENAME TO stack_registry;
ALTER TABLE agent_diagnostic_sessions RENAME TO diagnostic_sessions;
ALTER TABLE agent_diagnostic_reports RENAME TO diagnostic_reports;
ALTER TABLE agent_summaries RENAME TO summaries;
ALTER TABLE agent_self_narrative RENAME TO self_narratives;

-- 3. Update entity_type enum values: 'agent' → 'si'
UPDATE relationships SET entity_type = 'si' WHERE entity_type = 'agent';

COMMIT;
