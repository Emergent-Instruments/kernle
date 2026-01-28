-- Migration: Update embedding dimensions from 384 to 1536
-- Reason: Switching from all-MiniLM-L6-v2 (384 dims) to OpenAI text-embedding-3-small (1536 dims)
-- Note: This will clear existing embeddings - they need to be regenerated with the new model

-- Update episodes embedding column
ALTER TABLE episodes
    ALTER COLUMN embedding TYPE vector(1536);

-- Update beliefs embedding column
ALTER TABLE beliefs
    ALTER COLUMN embedding TYPE vector(1536);

-- Update values embedding column
ALTER TABLE values
    ALTER COLUMN embedding TYPE vector(1536);

-- Update goals embedding column
ALTER TABLE goals
    ALTER COLUMN embedding TYPE vector(1536);

-- Update notes embedding column
ALTER TABLE notes
    ALTER COLUMN embedding TYPE vector(1536);

-- Update drives embedding column
ALTER TABLE drives
    ALTER COLUMN embedding TYPE vector(1536);

-- Update relationships embedding column
ALTER TABLE relationships
    ALTER COLUMN embedding TYPE vector(1536);

-- Update playbooks embedding column
ALTER TABLE playbooks
    ALTER COLUMN embedding TYPE vector(1536);

-- Update raw_captures embedding column
ALTER TABLE raw_captures
    ALTER COLUMN embedding TYPE vector(1536);

-- Update emotional_memories embedding column
ALTER TABLE emotional_memories
    ALTER COLUMN embedding TYPE vector(1536);
