-- Migration: Fix semantic search function return types
-- The id column needs to be explicitly cast to uuid

DROP FUNCTION IF EXISTS search_memories_semantic(vector(1536), text, int, text[]);

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
            e.id::uuid as id,
            'episodes'::text as memory_type,
            COALESCE(e.objective, '') || ' ' || COALESCE(e.outcome, '') as content,
            (1 - (e.embedding <=> query_embedding))::float as score,
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
            b.id::uuid,
            'beliefs'::text,
            b.statement,
            (1 - (b.embedding <=> query_embedding))::float,
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
            v.id::uuid,
            'values'::text,
            COALESCE(v.name, '') || ': ' || COALESCE(v.statement, ''),
            (1 - (v.embedding <=> query_embedding))::float,
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
            g.id::uuid,
            'goals'::text,
            COALESCE(g.title, '') || ' ' || COALESCE(g.description, ''),
            (1 - (g.embedding <=> query_embedding))::float,
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
            n.id::uuid,
            'notes'::text,
            n.content,
            (1 - (n.embedding <=> query_embedding))::float,
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
            r.id::uuid,
            'raw_captures'::text,
            r.content,
            (1 - (r.embedding <=> query_embedding))::float,
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
