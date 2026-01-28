# Raw Memory Layer

> **Status**: ✅ IMPLEMENTED (v0.4.0+)

The raw memory layer provides zero-friction capture for later processing.

## Problem Solved

Before the raw layer, Kernle required structured input:
- Episodes need title, outcome, lessons
- Notes need type classification
- Beliefs need confidence scores

But sometimes I just need to **dump thoughts quickly** without structure. Like writing in a journal vs filling out a form.

## Solution: Raw Memory Layer

Add a base layer of unstructured memory that everything else derives from.

### Data Model

```python
@dataclass
class RawEntry:
    id: str
    agent_id: str
    content: str                    # Free-form text
    timestamp: datetime
    source: str = "manual"          # manual, auto_capture, voice, etc.
    processed: bool = False         # Has this been consolidated?
    processed_into: List[str] = []  # IDs of memories derived from this
    tags: List[str] = []            # Optional quick tags
    
    # Meta-memory fields
    confidence: float = 1.0
    source_type: str = "direct_experience"
```

### Commands

```bash
# Quick capture - zero friction
kernle raw "Just realized the sync queue needs deduplication"
kernle raw "Feeling anxious about context limits"
kernle raw "Sean suggested raw dump layer - good idea"

# With tags for easier processing later
kernle raw "MCP server needs error handling for edge cases" --tags dev,kernle

# Batch capture (pipe in)
echo "Stream of consciousness here" | kernle raw -

# View unprocessed raw entries
kernle raw list --unprocessed

# Process raw into structured memories
kernle raw process          # Interactive: review each, convert to episode/note/belief
kernle raw process --auto   # Auto-classify and convert
```

### Processing Flow

```
┌─────────────────────────────────────────────────────────┐
│                     Raw Dump                             │
│  "Just realized sync queue needs deduplication"         │
│  timestamp: 2026-01-27 09:30                            │
│  processed: false                                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼ kernle raw process
┌─────────────────────────────────────────────────────────┐
│  Agent reviews and classifies:                          │
│  - Is this an insight? → Episode with lesson            │
│  - Is this a decision? → Note (type: decision)          │
│  - Is this a new belief? → Belief with confidence       │
│  - Is this just context? → Note (type: context)         │
│  - Is this nothing? → Mark processed, no output         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Episode: "Discovered sync queue deduplication need"    │
│  lessons: ["Check for duplicate queue entries"]         │
│  source_raw_id: "raw_abc123"                           │
└─────────────────────────────────────────────────────────┘
```

### Auto-Processing Heuristics

When `kernle raw process --auto`:

| Pattern | Classification |
|---------|---------------|
| "I decided..." / "Decision:" | Note (decision) |
| "I learned..." / "Realized..." | Episode with lesson |
| "I believe..." / "I think..." | Belief (needs confidence) |
| "TODO:" / "Need to..." | Goal or Note (task) |
| "Feeling..." / emotional words | Note with emotional tagging |
| Question marks | Note (question) - maybe review later |
| Short/context-only | Note (context) |

### Readability / Trust

The raw layer also solves the trust problem:

```bash
# See everything in human-readable form
kernle dump --include-raw

# Export just raw entries (like a journal)
kernle raw export ~/memory-journal.md

# The raw layer IS the readable backup
```

### Integration with Existing Layers

```
Raw Entries (new base layer)
    ↓ process
Episodes + Notes
    ↓ consolidate  
Beliefs + Values + Goals
    ↓ synthesize
Identity
```

The consolidation flow already exists. We're just adding a layer below Episodes.

### Benefits

1. **Zero-friction capture**: Just type, worry about structure later
2. **Trust through visibility**: Raw entries are readable text
3. **Nothing lost**: Even unprocessed thoughts are saved
4. **Deferred processing**: Batch-process when you have time
5. **Provenance**: Can trace any belief back to raw experience
6. **Mirrors human memory**: Raw experience → processed memory

### Schema Addition

```sql
CREATE TABLE raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now(),
    source TEXT DEFAULT 'manual',
    processed BOOLEAN DEFAULT FALSE,
    processed_into JSONB DEFAULT '[]',
    tags JSONB DEFAULT '[]',
    -- Meta-memory
    confidence REAL DEFAULT 1.0,
    source_type TEXT DEFAULT 'direct_experience',
    -- Sync
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_raw_agent ON raw_entries(agent_id);
CREATE INDEX idx_raw_processed ON raw_entries(agent_id, processed);
CREATE INDEX idx_raw_timestamp ON raw_entries(agent_id, timestamp);
```

### Example Session

```bash
# Throughout the day, quick dumps
$ kernle raw "Storage abstraction working now"
$ kernle raw "Sean wants raw memory layer"
$ kernle raw "Anxiety tracking shows 11/100 - feeling calm"
$ kernle raw "Should add dump command for trust"

# At end of day or pre-compaction
$ kernle raw list --unprocessed
4 unprocessed raw entries

$ kernle raw process
[1/4] "Storage abstraction working now" (09:14)
  → [e]pisode [n]ote [b]elief [s]kip [q]uit: n
  Note type? [context/decision/insight/task]: insight
  ✓ Saved as note

[2/4] "Sean wants raw memory layer" (09:33)
  → [e]pisode [n]ote [b]elief [s]kip [q]uit: e
  Episode title: Designing raw memory layer
  Outcome: Architectured new base layer for zero-friction capture
  Lessons (comma-sep): Trust comes from readability, structure can be deferred
  ✓ Saved as episode

...
```

## Implementation Status

✅ **Fully Implemented** in `kernle/storage/sqlite.py` and `kernle/core.py`:

### Core API (Kernle class)
- `raw(content, tags, source)` - Quick capture
- `list_raw(processed, limit)` - List entries with optional filter
- `get_raw(raw_id)` - Get specific entry
- `process_raw(raw_id, as_type, **kwargs)` - Convert to episode/note/belief

### CLI Commands
```bash
kernle -a <agent> raw "quick thought"           # Capture
kernle -a <agent> raw list                       # List all
kernle -a <agent> raw list --unprocessed         # Unprocessed only
kernle -a <agent> raw process <id> --as episode  # Convert to episode
kernle -a <agent> raw process <id> --as note     # Convert to note
kernle -a <agent> raw process <id> --as belief   # Convert to belief
```

### Storage Layer
- Table: `raw_entries` with sync support
- Indexes on `agent_id`, `processed`, `timestamp`
- Meta-memory fields: `confidence`, `source_type`
- Included in `kernle dump` output
