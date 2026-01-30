# GitHub Issues to Create

Run these commands to create the issues, or grant Bash permission for me to create them.

---

## Critical Issues

### Issue 1: Add get_by_id methods to fix N+1 query patterns

```bash
gh issue create --title "fix: Add get_*_by_id methods to eliminate N+1 query patterns" --body "$(cat <<'EOF'
## Problem

In `kernle/core.py`, the `update_goal()` and `update_belief()` methods fetch ALL records (up to 1000) just to find one by ID:

```python
def update_goal(self, goal_id, ...):
    goals = self._storage.get_goals(status=None, limit=1000)
    for g in goals:
        if g.id == goal_id:
            existing = g
            break
```

This is inefficient, especially as memory stores grow.

## Solution

Add direct lookup methods to the Storage protocol and implementations:

```python
def get_goal_by_id(self, goal_id: str) -> Optional[Goal]
def get_belief_by_id(self, belief_id: str) -> Optional[Belief]
def get_episode_by_id(self, episode_id: str) -> Optional[Episode]
def get_note_by_id(self, note_id: str) -> Optional[Note]
def get_value_by_id(self, value_id: str) -> Optional[Value]
def get_drive_by_id(self, drive_id: str) -> Optional[Drive]
def get_relationship_by_id(self, relationship_id: str) -> Optional[Relationship]
```

## Files to Modify

- `kernle/storage/base.py` - Add to Storage protocol
- `kernle/storage/sqlite.py` - Implement methods
- `kernle/storage/postgres.py` - Implement methods
- `kernle/core.py` - Update `update_goal()`, `update_belief()` to use new methods

## Priority

Critical - Performance issue that worsens with scale
EOF
)" --label "bug,performance"
```

### Issue 2: Add confidence decay mechanism

```bash
gh issue create --title "feat: Add time-based confidence decay to prevent inflation" --body "$(cat <<'EOF'
## Problem

In `kernle/features/metamemory.py`, the `verify_memory()` method increases confidence by 0.1 with each verification, but there's no mechanism for confidence to decay over time.

This leads to confidence inflation where frequently-accessed memories converge to 1.0 regardless of actual accuracy.

## Current Behavior

```python
def verify_memory(self, ...):
    new_confidence = min(1.0, current_confidence + 0.1)
```

## Proposed Solution

Add optional time-based decay:

```python
def get_confidence_with_decay(self, memory) -> float:
    """Calculate confidence with time-based decay."""
    base_confidence = memory.confidence
    days_since_verified = (now - memory.last_verified).days

    # Decay rate: 1% per 30 days for unverified memories
    decay_rate = 0.01 * (days_since_verified / 30)

    # Floor at 50% to avoid completely forgetting verified facts
    return max(0.5, base_confidence - decay_rate)
```

## Configuration

- Make decay rate configurable (default: 0.01 per 30 days)
- Make floor configurable (default: 0.5)
- Allow disabling decay per memory type (e.g., values shouldn't decay)

## Files to Modify

- `kernle/features/metamemory.py` - Add decay calculation
- `kernle/storage/base.py` - Add `decay_rate` and `decay_floor` config
- `kernle/core.py` - Use decayed confidence in loading/search
EOF
)" --label "enhancement"
```

### Issue 3: Document PostgreSQL feature gaps prominently

```bash
gh issue create --title "docs: Document PostgreSQL/Supabase feature limitations" --body "$(cat <<'EOF'
## Problem

Many methods in `kernle/storage/postgres.py` raise `NotImplementedError`:

- Playbooks: save, get, list, search, update_usage
- Raw entries: save, get, list, mark_processed
- Forgetting: record_access, forget, recover, protect, get_candidates, get_forgotten

Users expecting feature parity between SQLite and Supabase will encounter runtime failures.

## Solution

1. Add prominent warning in README.md about cloud storage limitations
2. Add warning in docs-site quickstart
3. Consider adding runtime warning when SupabaseStorage is initialized
4. Document which features require local SQLite

## Acceptance Criteria

- [ ] README has clear "Cloud Storage Limitations" section
- [ ] Docs site has limitations callout
- [ ] Users understand what works locally vs cloud before choosing
EOF
)" --label "documentation"
```

---

## Important Issues

### Issue 4: Implement semantic contradiction detection

```bash
gh issue create --title "feat: Semantic contradiction detection using embeddings" --body "$(cat <<'EOF'
## Problem

Current `find_contradictions()` uses regex patterns ("never" vs "always", "like" vs "dislike"). This misses semantic contradictions like:

- "I should always test" vs "Testing slows me down"
- "API-first is best" vs "Code-first is faster"

## Proposed Solution

Use embedding similarity + sentiment analysis:

```python
def find_semantic_contradictions(self, belief: Belief) -> List[Belief]:
    """Find beliefs that are semantically similar but contradict."""
    # 1. Get embedding for new belief
    new_embedding = self._embedder.embed(belief.statement)

    # 2. Find similar beliefs (cosine > 0.7)
    similar = self._storage.search_by_embedding(new_embedding, threshold=0.7)

    # 3. Check for opposing sentiment/direction
    contradictions = []
    for existing in similar:
        if self._detect_opposition(belief.statement, existing.statement):
            contradictions.append(existing)

    return contradictions

def _detect_opposition(self, stmt1: str, stmt2: str) -> bool:
    """Detect if two similar statements have opposing meanings."""
    # Could use:
    # - Negation detection
    # - Sentiment analysis
    # - LLM call for complex cases
```

## Implementation Options

1. **Simple**: Negation word detection + opposite adjectives
2. **Medium**: Use sentiment analysis library
3. **Full**: LLM call for complex semantic opposition

## Files to Modify

- `kernle/core.py` - Add `find_semantic_contradictions()`
- `kernle/features/knowledge.py` - Integrate with knowledge mapping
- `kernle/storage/base.py` - Add `search_by_embedding()` if not present
EOF
)" --label "enhancement"
```

### Issue 5: Add context/scope field to memories

```bash
gh issue create --title "feat: Add context/scope field for project-specific memories" --body "$(cat <<'EOF'
## Problem

When an agent works on multiple projects with different approaches, memories conflict without context:

- User A: "API-first is best"
- User B: "Code-first is faster"

Both stored with similar confidence, no way to scope them to specific projects.

## Proposed Solution

Add optional `context` field to beliefs, episodes, and notes:

```python
@dataclass
class Belief:
    # ... existing fields ...
    context: Optional[str] = None  # e.g., "project:api-service", "user:alice"
    context_tags: List[str] = field(default_factory=list)
```

## Usage

```bash
# CLI
kernle belief "API-first is best" --context "project:api-service"

# MCP
memory_belief_add(statement="...", context="project:api-service")

# Loading with context filter
kernle load --context "project:api-service"
```

## Schema Migration

```sql
ALTER TABLE beliefs ADD COLUMN context TEXT;
ALTER TABLE beliefs ADD COLUMN context_tags TEXT;  -- JSON array
ALTER TABLE episodes ADD COLUMN context TEXT;
ALTER TABLE episodes ADD COLUMN context_tags TEXT;
ALTER TABLE notes ADD COLUMN context TEXT;
ALTER TABLE notes ADD COLUMN context_tags TEXT;

CREATE INDEX idx_beliefs_context ON beliefs(context);
```

## Files to Modify

- `kernle/storage/base.py` - Add fields to dataclasses
- `kernle/storage/sqlite.py` - Migration + query updates
- `kernle/storage/postgres.py` - Schema updates
- `kernle/core.py` - Support context in load/search
- `kernle/mcp/server.py` - Add context param to tools
- `kernle/cli/commands/` - Add --context flags
EOF
)" --label "enhancement,schema"
```

### Issue 6: Add batch insertion API

```bash
gh issue create --title "feat: Add batch insertion API for bulk memory creation" --body "$(cat <<'EOF'
## Problem

Each `save_episode()`, `save_belief()`, etc. performs:
1. Individual database write
2. Embedding creation
3. Sync queue update

When processing large codebases or importing memories, this causes performance issues.

## Proposed Solution

Add batch methods:

```python
def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
    """Save multiple episodes in a single transaction."""
    with self._connect() as conn:
        ids = []
        for episode in episodes:
            # Insert without commit
            id = self._insert_episode(conn, episode)
            ids.append(id)
        conn.commit()

    # Batch embedding creation
    self._batch_create_embeddings(ids)

    return ids
```

## API

```python
# Storage protocol
def save_episodes_batch(self, episodes: List[Episode]) -> List[str]
def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]
def save_notes_batch(self, notes: List[Note]) -> List[str]

# CLI
kernle import episodes.json --batch
kernle import beliefs.csv --batch

# MCP
memory_episodes_batch(episodes=[...])
```

## Files to Modify

- `kernle/storage/base.py` - Add batch methods to protocol
- `kernle/storage/sqlite.py` - Implement with transaction batching
- `kernle/storage/postgres.py` - Implement with bulk insert
- `kernle/cli/commands/import_cmd.py` - Add batch flag
EOF
)" --label "enhancement,performance"
```

### Issue 7: Add import/migration tools

```bash
gh issue create --title "feat: Add import command for markdown and JSON files" --body "$(cat <<'EOF'
## Problem

Users can't migrate from:
- File-based memory (MEMORY.md)
- Other memory systems (Mem0, Zep)
- Exported Kernle data

This was identified in dogfooding notes as a barrier to adoption.

## Proposed Solution

Add `kernle import` command:

```bash
# Import from markdown
kernle import markdown MEMORY.md --type beliefs

# Import from JSON (Kernle export format)
kernle import json memories.json

# Import from CSV
kernle import csv episodes.csv

# Import from Mem0 export
kernle import mem0 mem0-export.json
```

## Markdown Format Support

```markdown
# Beliefs

- I believe testing is important (confidence: 0.9)
- Code should be readable (confidence: 0.85)

# Episodes

## 2026-01-15: Fixed authentication bug
- Outcome: success
- Lesson: Always check token expiry

## 2026-01-20: Refactored database layer
- Outcome: success
- Lesson: Start with tests
```

## Files to Create/Modify

- `kernle/cli/commands/import_cmd.py` - New import command
- `kernle/importers/markdown.py` - Markdown parser
- `kernle/importers/json.py` - JSON importer
- `kernle/importers/csv.py` - CSV importer
- `kernle/importers/mem0.py` - Mem0 format importer

## Related

- Existing `dump` command exports to markdown
- Should be round-trippable: dump → edit → import
EOF
)" --label "enhancement"
```

### Issue 8: Implement sync conflict notification

```bash
gh issue create --title "feat: Notify users about sync conflicts instead of silent last-write-wins" --body "$(cat <<'EOF'
## Problem

Current sync uses last-write-wins conflict resolution silently:

```python
conflicts: int = 0  # Conflicts encountered (resolved with last-write-wins)
```

Users are not notified when their local changes are overwritten by cloud changes.

## Proposed Solution

1. Track conflicts with details
2. Store conflict history
3. Notify user in CLI output
4. Option to review and resolve manually

```python
@dataclass
class SyncConflict:
    id: str
    table: str
    record_id: str
    local_version: Dict
    cloud_version: Dict
    resolution: str  # "local_wins", "cloud_wins", "merged"
    resolved_at: datetime

@dataclass
class SyncResult:
    pushed: int
    pulled: int
    conflicts: List[SyncConflict]  # Changed from int
    errors: List[str]
```

## CLI Output

```
$ kernle sync push
Pushed 5 changes
⚠️  2 conflicts resolved (cloud wins):
  - belief:abc123 - "API-first is best" overwritten
  - episode:def456 - "Fixed bug" overwritten

Run `kernle sync conflicts` to review history
Run `kernle sync push --conflict-strategy local` to prefer local
```

## Configuration

```bash
# Set default conflict strategy
kernle config set sync.conflict_strategy cloud_wins  # or local_wins, manual
```

## Files to Modify

- `kernle/storage/base.py` - Update SyncResult dataclass
- `kernle/storage/sqlite.py` - Track conflict details
- `kernle/storage/postgres.py` - Track conflict details
- `kernle/cli/commands/sync.py` - Display conflicts
EOF
)" --label "enhancement"
```

---

## Strategic Issues

### Issue 9: Implement auto-capture hooks for raw layer

```bash
gh issue create --title "feat: Auto-capture to raw layer via Claude Code hooks" --body "$(cat <<'EOF'
## Problem

Kernle requires explicit memory capture, creating friction compared to Mem0's auto-extraction. Users must actively decide what to remember.

## Proposed Solution

Implement hook-based auto-capture to the raw layer. This preserves memory sovereignty (agent still decides what to promote) while reducing friction.

### Hook Configuration

```json
// .claude/hooks.json
{
  "hooks": [
    {
      "event": "Stop",
      "command": "kernle -a \$KERNLE_AGENT_ID raw --source session-end --quiet",
      "stdin": "Session summary: \$LAST_ASSISTANT_MESSAGE"
    },
    {
      "event": "PostToolUse",
      "matcher": { "tool": "Edit,Write" },
      "command": "kernle -a \$KERNLE_AGENT_ID raw --source code-change --quiet",
      "stdin": "Changed \$TOOL_ARGS"
    }
  ]
}
```

### CLI Updates

```bash
# Quiet mode for hooks (no stdout)
kernle raw "content" --quiet

# Source tracking
kernle raw "content" --source hook-session-end

# Stdin support
echo "content" | kernle raw --stdin
```

### MCP Tool

```python
@mcp_tool
def memory_auto_capture(
    content: str,
    source: str = "auto",
    extract_suggestions: bool = False
) -> Dict:
    """
    Capture content to raw layer.

    Args:
        content: Content to capture
        source: Source identifier (hook, tool, manual)
        extract_suggestions: If True, also generate suggested memories

    Returns:
        raw_id: ID of created raw entry
        suggestions: Optional list of suggested memories
    """
```

## Files to Create/Modify

- `kernle/cli/commands/raw.py` - Add --quiet, --source, --stdin flags
- `docs-site/integration/hooks.mdx` - Document hook setup
- `kernle/mcp/server.py` - Enhance memory_auto_capture
- `.claude/hooks.json` - Example hook configuration

## Related

See Issue #XX for memory suggestion system (promote/reject workflow)
EOF
)" --label "enhancement,feature"
```

### Issue 10: Implement memory suggestion system

```bash
gh issue create --title "feat: Memory suggestion system - auto-extract, agent approves" --body "$(cat <<'EOF'
## Problem

Auto-capture to raw layer is useful, but raw content still needs manual processing. Users want the convenience of auto-extraction while maintaining control.

## Proposed Solution

Implement a suggestion system that:
1. Analyzes raw entries
2. Suggests structured memories (episodes, beliefs, notes)
3. Agent reviews and approves/modifies/rejects

### Data Model

```python
@dataclass
class MemorySuggestion:
    id: str
    memory_type: str  # episode, belief, note
    content: Dict[str, Any]  # Structured memory data
    confidence: float  # System confidence in suggestion
    source_raw_ids: List[str]  # Which raw entries this came from
    status: str = "pending"  # pending, promoted, modified, rejected
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_reason: Optional[str] = None
```

### CLI Commands

```bash
# List pending suggestions
kernle suggestions list
kernle suggestions list --type belief --confidence-above 0.8

# Review a suggestion
kernle suggestions show <id>

# Approve as-is
kernle suggestions approve <id>

# Modify and approve (opens editor)
kernle suggestions approve <id> --edit

# Reject with reason
kernle suggestions reject <id> --reason "Not relevant to current work"

# Batch operations
kernle suggestions approve-all --confidence-above 0.9
kernle suggestions reject-all --older-than 7d
```

### MCP Tools

```python
@mcp_tool
def memory_suggestions_list(
    limit: int = 10,
    memory_type: Optional[str] = None,
    min_confidence: float = 0.0
) -> List[Dict]:
    """List pending memory suggestions."""

@mcp_tool
def memory_suggestions_promote(
    suggestion_id: str,
    modifications: Optional[Dict] = None
) -> str:
    """Promote suggestion to real memory. Returns new memory ID."""

@mcp_tool
def memory_suggestions_reject(
    suggestion_id: str,
    reason: Optional[str] = None
):
    """Reject a suggestion."""
```

### Extraction Logic

For MVP, use pattern-based extraction:
- Episodes: Look for "did X", "completed Y", outcome words
- Beliefs: Look for "I think", "I believe", opinion patterns
- Notes: Quotes, decisions, insights

For V2, use LLM extraction (optional, requires API key).

## Files to Create/Modify

- `kernle/storage/base.py` - Add MemorySuggestion dataclass
- `kernle/storage/sqlite.py` - Add suggestions table and methods
- `kernle/features/suggestions.py` - New mixin for suggestion logic
- `kernle/cli/commands/suggestions.py` - New CLI command
- `kernle/mcp/server.py` - Add suggestion tools
- `kernle/extractors/` - Extraction logic (pattern + optional LLM)

## Philosophy Note

This preserves memory sovereignty:
- System suggests, agent decides
- Agent can always create memories directly
- Suggestions are transparent (show source raw entries)
- Rejected suggestions can inform future extraction
EOF
)" --label "enhancement,feature"
```

---

## To Create All Issues

Grant Bash permission and I'll run these commands, or run them manually:

```bash
cd /Users/seanhart/emergent-instruments/kernle
# Then run each gh issue create command above
```
