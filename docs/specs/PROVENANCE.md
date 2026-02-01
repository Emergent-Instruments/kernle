# Memory Provenance Specification

## Overview

Memory provenance tracks the lineage of memories as they flow through Kernle's stratified memory system. This enables agents to understand *where* their knowledge came from, *how* it was derived, and *what evidence* supports it.

### Core Problem

Currently, when raw entries are promoted to episodes/notes/beliefs, the lineage is not tracked. When consolidation produces beliefs from multiple episodes, there's no record of which episodes contributed. This makes it impossible to:

- Trace a belief back to its supporting evidence
- Understand why confidence changed
- Audit the reasoning chain behind conclusions
- Identify which raw captures contributed to higher-level memories

### Design Goals

1. **Full Lineage**: Every memory should be traceable to its sources
2. **Minimal Overhead**: Provenance capture should be automatic, not require extra effort
3. **Query-Friendly**: Lineage should be easily retrievable via CLI and API
4. **Bidirectional**: Support both "what did this come from?" and "what was derived from this?"

---

## Data Model

### Existing Fields (Already in Place)

The data model already has provenance fields on all memory types:

```python
# On Episode, Belief, Note, Value, Goal, Drive, Relationship:
source_type: str = "direct_experience"        # How this memory was created
source_episodes: Optional[List[str]] = None   # Episode IDs that support this
derived_from: Optional[List[str]] = None      # Memory refs this was derived from (format: type:id)
confidence_history: Optional[List[Dict]] = None  # Confidence change audit trail

# On RawEntry:
processed_into: Optional[List[str]] = None    # Audit trail: ["episode:abc", "note:xyz"]
```

### Enhanced Provenance Model

Add a new `provenance` field that consolidates lineage info into a single queryable structure:

```python
@dataclass
class Provenance:
    """Provenance chain for a memory."""
    
    # Creation context
    created_via: str                          # "promote" | "consolidate" | "direct" | "inference"
    created_at: datetime                      # When this memory was created
    
    # Source references (backward links)
    source_raw_ids: Optional[List[str]] = None     # Raw entries this was promoted from
    source_memory_refs: Optional[List[str]] = None  # Memory refs (format: type:id) this derived from
    
    # Derivation metadata
    consolidation_id: Optional[str] = None    # If from consolidation, which batch
    promotion_context: Optional[str] = None   # CLI command or operation that created this
    
    # Forward links (populated lazily/on-demand)
    derived_into: Optional[List[str]] = None  # Memories that were derived from this (type:id)
```

### Memory Reference Format

All memory references use the format `type:id`:
- `raw:abc123` - Raw entry
- `episode:def456` - Episode
- `note:ghi789` - Note
- `belief:jkl012` - Belief
- `value:mno345` - Value
- `goal:pqr678` - Goal

This allows uniform reference across the memory graph.

### Source Type Values

```python
class SourceType(Enum):
    DIRECT_EXPERIENCE = "direct_experience"  # Directly observed/experienced
    PROMOTE = "promote"                      # Promoted from raw entry
    INFERENCE = "inference"                  # Inferred from other memories
    TOLD_BY_AGENT = "told_by_agent"         # Told by another agent/user
    CONSOLIDATION = "consolidation"          # Created during consolidation
    REVISION = "revision"                    # Revised from another belief
    UNKNOWN = "unknown"                      # Legacy or untracked
```

---

## Use Cases

### 1. Raw → Episode Promotion

When a raw entry is promoted to an episode:

```python
# Current: raw.processed_into gets updated, but episode doesn't link back

# New behavior:
episode = Episode(
    id=episode_id,
    source_type="promote",
    derived_from=[f"raw:{raw_id}"],           # NEW: link back to source
    source_episodes=None,                      # No supporting episodes for promoted content
    # ... other fields
)

# Raw entry update remains the same:
raw.processed_into = [f"episode:{episode_id}"]
```

### 2. Raw → Belief Promotion

When a raw entry becomes a belief directly:

```python
belief = Belief(
    id=belief_id,
    source_type="promote",
    derived_from=[f"raw:{raw_id}"],
    source_episodes=None,                      # Could add episode support later
    # ... other fields
)
```

### 3. Episode → Belief Consolidation

When analyzing episodes produces a new belief:

```python
belief = Belief(
    id=belief_id,
    source_type="consolidation",
    source_episodes=[ep1_id, ep2_id, ep3_id],  # Episodes that support this belief
    derived_from=[
        f"episode:{ep1_id}",
        f"episode:{ep2_id}",
        f"episode:{ep3_id}"
    ],
    # ... other fields
)
```

### 4. Belief Revision Chain

When a belief is revised:

```python
# Old belief
old_belief.superseded_by = new_belief_id
old_belief.is_active = False

# New belief
new_belief = Belief(
    id=new_belief_id,
    source_type="revision",
    derived_from=[f"belief:{old_belief_id}"],
    supersedes=old_belief_id,
    # ... other fields
)
```

### 5. Multi-Level Derivation

When a belief is derived from notes that were promoted from raw:

```
raw:abc → note:def → belief:ghi
```

The belief's `derived_from` is `["note:def"]`, and we can traverse to find the ultimate source.

---

## CLI Interface Design

### Show Memory with Provenance

Add `--trace` flag to show lineage:

```bash
# Show belief with its provenance
kernle belief show <belief_id> --trace

# Output:
Belief: "API endpoints should be RESTful"
Confidence: 0.85
Type: principle

📍 Provenance:
  Source: consolidation (2024-01-15)
  Derived from:
    ├─ episode:abc123 "Implemented REST API for users"
    │   └─ raw:111aaa "Finished user endpoints today..."
    ├─ episode:def456 "Refactored payments to REST"
    │   └─ raw:222bbb "Rewrote payments module..."
    └─ episode:ghi789 "REST conversion complete"
        └─ raw:333ccc "All services now REST..."

  Confidence history:
    2024-01-15: 0.85 (created via consolidation)
```

### Trace Command

Add dedicated `trace` command for lineage exploration:

```bash
# Trace a memory's full lineage
kernle trace belief:abc123

# Output:
📍 Lineage for belief:abc123 "API endpoints should be RESTful"

BACKWARD (where it came from):
  belief:abc123 [consolidation, 2024-01-15]
    ├─ episode:def456 [direct_experience, 2024-01-10]
    │   └─ raw:aaa111 [cli, 2024-01-10]
    ├─ episode:ghi789 [direct_experience, 2024-01-12]
    │   └─ raw:bbb222 [cli, 2024-01-12]
    └─ episode:jkl012 [direct_experience, 2024-01-14]
        └─ raw:ccc333 [cli, 2024-01-14]

FORWARD (what derived from it):
  belief:abc123
    └─ note:xyz789 [inference, 2024-01-20] "REST API design decision"
```

### List with Provenance Info

Add `--provenance` flag to list commands:

```bash
# List beliefs with source info
kernle belief list --provenance

# Output:
ID        Statement                           Confidence  Source
────────────────────────────────────────────────────────────────
abc123    API endpoints should be RESTful     0.85        3 episodes
def456    Tests should be fast                0.90        2 episodes
ghi789    Documentation is important          0.75        1 note
jkl012    User feedback is valuable           0.80        direct
```

### Search with Provenance

Optionally include provenance in search results:

```bash
kernle search "REST API" --include-provenance

# Output:
1. [belief] "API endpoints should be RESTful" (0.85)
   📍 From: 3 episodes via consolidation

2. [episode] "Implemented REST API for users"
   📍 From: raw:aaa111

3. [note] "REST API design decision"
   📍 From: belief:abc123 (inference)
```

### Raw Entry Trace

Show what a raw entry became:

```bash
kernle raw show <raw_id> --trace

# Output:
Raw Entry: abc123def456...
Captured: 2024-01-10 14:30
Content: "Finished implementing user endpoints today. REST feels clean..."

📍 Promoted to:
  └─ episode:xyz789 "Implemented REST API for users"
      └─ belief:abc123 "API endpoints should be RESTful" (via consolidation)
```

---

## API Additions

### Storage Protocol Extensions

```python
class Storage(Protocol):
    # ... existing methods ...
    
    # === Provenance Methods ===
    
    @abstractmethod
    def get_memory_provenance(
        self,
        memory_type: str,
        memory_id: str,
        depth: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Get provenance chain for a memory.
        
        Args:
            memory_type: Type of memory (episode, belief, note, etc.)
            memory_id: ID of the memory
            depth: How many levels deep to traverse (default 3)
            
        Returns:
            Dict with backward/forward lineage, or None if not found
        """
        ...
    
    @abstractmethod
    def get_derived_memories(
        self,
        memory_type: str,
        memory_id: str,
    ) -> List[Dict[str, Any]]:
        """Get memories that were derived from this one (forward links).
        
        Returns list of {type, id, created_at, relationship} dicts.
        """
        ...
    
    @abstractmethod
    def get_source_memories(
        self,
        memory_type: str,
        memory_id: str,
    ) -> List[Dict[str, Any]]:
        """Get source memories this was derived from (backward links).
        
        Returns list of {type, id, created_at, relationship} dicts.
        """
        ...
    
    @abstractmethod
    def get_memories_from_raw(
        self,
        raw_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all memories that trace back to a raw entry.
        
        Includes direct promotions and transitive derivations.
        """
        ...
    
    @abstractmethod
    def get_raw_sources(
        self,
        memory_type: str,
        memory_id: str,
    ) -> List[str]:
        """Get the ultimate raw entry IDs that a memory traces back to.
        
        Traverses the full derivation chain to find raw sources.
        """
        ...
```

### Core API Extensions

```python
class Kernle:
    # ... existing methods ...
    
    def trace(
        self,
        memory_ref: str,
        direction: str = "both",
        depth: int = 3,
    ) -> Dict[str, Any]:
        """Trace a memory's lineage.
        
        Args:
            memory_ref: Memory reference (format: type:id)
            direction: "backward", "forward", or "both"
            depth: Max traversal depth
            
        Returns:
            Lineage tree with source and derived memories
        """
        ...
    
    def get_provenance(
        self,
        memory_ref: str,
    ) -> Dict[str, Any]:
        """Get provenance info for a memory.
        
        Returns:
            Dict with source_type, derived_from, source_episodes, etc.
        """
        ...
    
    def get_evidence(
        self,
        belief_id: str,
    ) -> Dict[str, Any]:
        """Get supporting evidence for a belief.
        
        Traverses backward to find episodes and raw entries
        that support this belief.
        
        Returns:
            Dict with episodes, notes, and raw entries
        """
        ...
```

### MCP Tool Extensions

```python
# In kernle.mcp.tools:

@mcp_tool
def memory_trace(
    memory_ref: str,
    direction: str = "both",
    depth: int = 3,
) -> Dict[str, Any]:
    """Trace a memory's provenance chain.
    
    Args:
        memory_ref: Memory reference (format: type:id, e.g., "belief:abc123")
        direction: "backward" (sources), "forward" (derived), or "both"
        depth: Maximum traversal depth (default 3)
    
    Returns:
        Lineage tree showing where memory came from and what it became
    """
    ...

@mcp_tool
def get_belief_evidence(
    belief_id: str,
) -> Dict[str, Any]:
    """Get supporting evidence for a belief.
    
    Returns the episodes, notes, and raw entries that support this belief.
    """
    ...
```

---

## Example Outputs

### Full Trace Output (JSON)

```json
{
  "memory": {
    "type": "belief",
    "id": "abc123",
    "statement": "API endpoints should be RESTful",
    "confidence": 0.85
  },
  "backward": {
    "source_type": "consolidation",
    "created_at": "2024-01-15T10:30:00Z",
    "sources": [
      {
        "type": "episode",
        "id": "def456",
        "title": "Implemented REST API for users",
        "source_type": "promote",
        "sources": [
          {
            "type": "raw",
            "id": "aaa111",
            "blob": "Finished implementing user endpoints today...",
            "captured_at": "2024-01-10T14:30:00Z"
          }
        ]
      },
      {
        "type": "episode",
        "id": "ghi789",
        "title": "Refactored payments to REST",
        "source_type": "promote",
        "sources": [
          {
            "type": "raw",
            "id": "bbb222",
            "blob": "Rewrote payments module to be RESTful...",
            "captured_at": "2024-01-12T09:15:00Z"
          }
        ]
      }
    ]
  },
  "forward": {
    "derived": [
      {
        "type": "note",
        "id": "xyz789",
        "content": "REST API design decision for new service",
        "source_type": "inference",
        "created_at": "2024-01-20T11:00:00Z"
      }
    ]
  }
}
```

### Evidence for Belief Output

```json
{
  "belief": {
    "id": "abc123",
    "statement": "API endpoints should be RESTful",
    "confidence": 0.85
  },
  "evidence": {
    "episodes": [
      {
        "id": "def456",
        "objective": "Implemented REST API for users",
        "outcome": "success",
        "lessons": ["REST patterns make the API intuitive"]
      },
      {
        "id": "ghi789",
        "objective": "Refactored payments to REST",
        "outcome": "success",
        "lessons": ["Standard HTTP methods reduce documentation needs"]
      }
    ],
    "notes": [],
    "raw_entries": [
      {
        "id": "aaa111",
        "blob": "Finished implementing user endpoints today...",
        "captured_at": "2024-01-10T14:30:00Z"
      },
      {
        "id": "bbb222",
        "blob": "Rewrote payments module to be RESTful...",
        "captured_at": "2024-01-12T09:15:00Z"
      }
    ]
  },
  "total_evidence_count": 4,
  "direct_episodes": 2,
  "source_raw_entries": 2
}
```

---

## Database Schema Changes

### SQLite

```sql
-- Add provenance columns to existing tables (if not present)
-- Most columns already exist; ensure derived_from is populated

-- Index for efficient provenance queries
CREATE INDEX IF NOT EXISTS idx_episodes_derived_from ON episodes(derived_from);
CREATE INDEX IF NOT EXISTS idx_beliefs_derived_from ON beliefs(derived_from);
CREATE INDEX IF NOT EXISTS idx_notes_derived_from ON notes(derived_from);
CREATE INDEX IF NOT EXISTS idx_beliefs_source_episodes ON beliefs(source_episodes);

-- View for provenance graph traversal
CREATE VIEW IF NOT EXISTS memory_provenance AS
SELECT 
    'episode' as memory_type,
    id,
    source_type,
    derived_from,
    source_episodes,
    created_at
FROM episodes WHERE deleted = 0
UNION ALL
SELECT 
    'belief' as memory_type,
    id,
    source_type,
    derived_from,
    source_episodes,
    created_at
FROM beliefs WHERE deleted = 0
UNION ALL
SELECT 
    'note' as memory_type,
    id,
    source_type,
    derived_from,
    source_episodes,
    created_at
FROM notes WHERE deleted = 0;
```

### PostgreSQL (Supabase)

```sql
-- Same indexes for Postgres
CREATE INDEX IF NOT EXISTS idx_episodes_derived_from 
ON episodes USING GIN (derived_from);

CREATE INDEX IF NOT EXISTS idx_beliefs_source_episodes 
ON beliefs USING GIN (source_episodes);

-- Function for recursive provenance traversal
CREATE OR REPLACE FUNCTION get_provenance_chain(
    p_memory_type TEXT,
    p_memory_id UUID,
    p_depth INT DEFAULT 3
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    -- Implementation would use recursive CTE
    -- to traverse derived_from links
    ...
END;
$$ LANGUAGE plpgsql;
```

---

## Implementation Plan

### Phase 1: Populate Provenance on Creation (Week 1)

1. Update `process_raw()` to set `derived_from` on created memory
2. Update `promote_suggestion()` to set `derived_from` from `source_raw_ids`
3. Update `consolidate()` to set `derived_from` and `source_episodes` on new beliefs
4. Update `revise_belief()` to set `derived_from` to superseded belief

### Phase 2: Query Methods (Week 2)

1. Implement `get_memory_provenance()` in storage backends
2. Implement `get_derived_memories()` and `get_source_memories()`
3. Implement `trace()` method in core Kernle class
4. Add recursive traversal for `get_raw_sources()`

### Phase 3: CLI Integration (Week 3)

1. Add `--trace` flag to `belief show`, `episode show`, `note show`
2. Add `kernle trace <memory_ref>` command
3. Add `--provenance` flag to list commands
4. Update `raw show --trace` for forward links

### Phase 4: Search & MCP (Week 4)

1. Add `--include-provenance` to search results
2. Add `memory_trace` and `get_belief_evidence` MCP tools
3. Update `memory_search` to optionally include provenance
4. Add provenance to `load` output when verbose

---

## Migration Strategy

For existing memories without provenance:

1. **Don't backfill raw→memory links**: Too error-prone to guess
2. **Mark legacy memories**: Set `source_type = "unknown"` for unmigrated
3. **Forward-only**: New memories get full provenance from here on
4. **Optional inference**: Provide `kernle migrate --infer-provenance` that attempts to link based on timestamps and content similarity (experimental, agent-approved)

---

## Open Questions

1. **Storage overhead**: Should `derived_into` (forward links) be computed lazily or stored?
   - *Recommendation*: Compute lazily to avoid write amplification

2. **Depth limits**: How deep should automatic trace go?
   - *Recommendation*: Default depth=3, configurable via `--depth` flag

3. **Circular references**: How to handle (rare) cycles in derivation?
   - *Recommendation*: Track visited IDs during traversal, skip cycles

4. **Performance**: Index strategy for JSONB array columns?
   - *Recommendation*: GIN indexes on derived_from and source_episodes

---

## Summary

This specification enables full memory provenance tracking in Kernle by:

1. **Leveraging existing fields**: `derived_from`, `source_episodes`, `source_type` already exist
2. **Consistent population**: Update all creation paths to set provenance
3. **CLI exposure**: `--trace` flags and dedicated `trace` command
4. **API methods**: `trace()`, `get_provenance()`, `get_evidence()`
5. **Forward compatibility**: Design supports future graph-based memory exploration
