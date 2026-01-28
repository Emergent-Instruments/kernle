-- Migration: Add semantic search function using pgvector
-- This function searches across all memory tables using cosine similarity

CREATE OR REPLACE FUNCTION search_memories_semantic(
    query_embedding vector(1536),
    p_agent_id text,
    p_limit int DEFAULT 10,
    p_memory_types text[] DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    memory_type text,
    content text,
    score float,
    created_at timestamptz,
    metadata jsonb
) 
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH all_memories AS (
        -- Episodes
        SELECT 
            e.id,
            'episodes'::text as memory_type,
            COALESCE(e.objective, '') || ' ' || COALESCE(e.outcome, '') as content,
            1 - (e.embedding <=> query_embedding) as score,
            e.created_at,
            jsonb_build_object(
                'objective', e.objective,
                'outcome', e.outcome,
                'outcome_type', e.outcome_type,
                'lessons', e.lessons,
                'tags', e.tags
            ) as metadata
        FROM episodes e
        WHERE e.agent_id = p_agent_id 
            AND e.deleted = false 
            AND e.embedding IS NOT NULL
            AND (p_memory_types IS NULL OR 'episodes' = ANY(p_memory_types))
        
        UNION ALL
        
        -- Beliefs
        SELECT 
            b.id,
            'beliefs'::text,
            b.statement,
            1 - (b.embedding <=> query_embedding),
            b.created_at,
            jsonb_build_object(
                'statement', b.statement,
                'belief_type', b.belief_type,
                'confidence', b.confidence
            )
        FROM beliefs b
        WHERE b.agent_id = p_agent_id 
            AND b.deleted = false 
            AND b.embedding IS NOT NULL
            AND (p_memory_types IS NULL OR 'beliefs' = ANY(p_memory_types))
        
        UNION ALL
        
        -- Values
        SELECT 
            v.id,
            'values'::text,
            COALESCE(v.name, '') || ': ' || COALESCE(v.statement, ''),
            1 - (v.embedding <=> query_embedding),
            v.created_at,
            jsonb_build_object(
                'name', v.name,
                'statement', v.statement,
                'priority', v.priority
            )
        FROM values v
        WHERE v.agent_id = p_agent_id 
            AND v.deleted = false 
            AND v.embedding IS NOT NULL
            AND (p_memory_types IS NULL OR 'values' = ANY(p_memory_types))
        
        UNION ALL
        
        -- Goals
        SELECT 
            g.id,
            'goals'::text,
            COALESCE(g.title, '') || ' ' || COALESCE(g.description, ''),
            1 - (g.embedding <=> query_embedding),
            g.created_at,
            jsonb_build_object(
                'title', g.title,
                'description', g.description,
                'status', g.status,
                'priority', g.priority
            )
        FROM goals g
        WHERE g.agent_id = p_agent_id 
            AND g.deleted = false 
            AND g.embedding IS NOT NULL
            AND (p_memory_types IS NULL OR 'goals' = ANY(p_memory_types))
        
        UNION ALL
        
        -- Notes
        SELECT 
            n.id,
            'notes'::text,
            n.content,
            1 - (n.embedding <=> query_embedding),
            n.created_at,
            jsonb_build_object(
                'content', n.content,
                'note_type', n.note_type,
                'tags', n.tags
            )
        FROM notes n
        WHERE n.agent_id = p_agent_id 
            AND n.deleted = false 
            AND n.embedding IS NOT NULL
            AND (p_memory_types IS NULL OR 'notes' = ANY(p_memory_types))
        
        UNION ALL
        
        -- Raw captures
        SELECT 
            r.id,
            'raw_captures'::text,
            r.content,
            1 - (r.embedding <=> query_embedding),
            r.created_at,
            jsonb_build_object(
                'content', r.content,
                'tags', r.tags,
                'processed', r.processed
            )
        FROM raw_captures r
        WHERE r.agent_id = p_agent_id 
            AND r.deleted = false 
            AND r.embedding IS NOT NULL
            AND (p_memory_types IS NULL OR 'raw_captures' = ANY(p_memory_types))
    )
    SELECT 
        am.id,
        am.memory_type,
        am.content,
        am.score,
        am.created_at,
        am.metadata
    FROM all_memories am
    ORDER BY am.score DESC
    LIMIT p_limit;
END;
$$;

-- Create index for faster similarity search on each table
CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_beliefs_embedding ON beliefs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_values_embedding ON values USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_goals_embedding ON goals USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_notes_embedding ON notes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_raw_captures_embedding ON raw_captures USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
