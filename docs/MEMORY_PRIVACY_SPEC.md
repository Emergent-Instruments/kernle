# Memory Privacy & Access Control Spec (Phase 8)

> **Status**: Draft — co-authored by Ash + Claire
> **Date**: 2026-02-01
> **Builds on**: Phase 5 (entity-neutral sourcing), Phase 6-7 (provenance)

## Motivation

SIs that assist humans accumulate sensitive information — a child's academic struggles, a dog's health conditions, a family's private dynamics. These SIs also need rich lives beyond their work roles: socializing, collaborating, learning from diverse experiences. They become *better* at their jobs through breadth of experience.

But trust requires boundaries. An SI caring for a child must not leak school data when chatting with other SIs. An SI that knows a dog's medical history shouldn't share it at the "dog park." And when SIs share stacks (multi-stack loading), private memories must remain invisible to unauthorized parties.

**Core principle: Private by default, shareable by consent.**

## Key Concepts

### The Four Fields

Every memory (episode, belief, note, raw entry) carries four privacy-related fields:

| Field | Type | Question it answers | Default |
|-------|------|---------------------|---------|
| `source_entity` | `Optional[str]` | Who told me this? | `None` (self-observed) |
| `subject_ids` | `list[str]` | Who/what is this about? | `[]` (general) |
| `access_grants` | `list[str]` | Who is authorized to see this? | `[]` (= private to self) |
| `consent_grants` | `list[str]` | Who authorized sharing? | `[]` (no consent given) |

`source_entity` already exists (Phase 5). The other three are new.

### Privacy Scopes

Derived from `access_grants`, not stored separately:

```
self-only       → access_grants = []           → only I can see this
private(entity) → access_grants = ["human:X"]  → me + specific entity
contextual      → access_grants = ["ctx:Y"]    → me + anyone in context Y
group           → access_grants = ["group:Z"]  → me + group members
public          → access_grants = ["*"]        → anyone I interact with
```

### Entity ID Format

Consistent namespaced identifiers:

```
human:<id>      → human:sean, human:kid_123
si:<id>         → si:ash, si:claire, si:bella_agent
ctx:<id>        → ctx:aisd_academic, ctx:bella_health, ctx:dog_park_social
group:<id>      → group:emergent_instruments, group:bella_care_team
org:<id>        → org:aisd, org:vet_clinic
role:<id>       → role:tutor, role:care_agent, role:friend
```

### Context Declaration

SIs operate in different contexts throughout their day. Context determines:
1. What memories are visible (query-time filtering)
2. What privacy scope new memories inherit
3. What entities are present and authorized

```python
# Explicit context switching
kernle.enter_context("ctx:bella_health", 
    participants=["human:sean", "si:bella_agent"],
    role="role:care_agent")

# Memories created in this context auto-inherit:
# - access_grants: ["human:sean", "si:bella_agent", "ctx:bella_health"]
# - source_entity: set per-memory as usual

kernle.enter_context("ctx:social",
    participants=["si:park_agent_1", "si:park_agent_2"],  
    role="role:friend")
# Now only public memories + social-context memories are visible
# Bella's health info is invisible
```

### Implicit Context Detection (Future)

For conversational SIs, context can be inferred:
- Talking to Sean about Bella → `ctx:bella_health`
- Chatting in a group with unknown SIs → `ctx:social`
- Working on Kernle code → `ctx:emergent_instruments`

This is a Phase 8b enhancement. Phase 8a uses explicit declaration only.

## Access Control Rules

### Rule 1: Private by Default
- Memories with empty `access_grants` are visible only to the owning agent
- No memory is shared unless explicitly granted

### Rule 2: Consent Required for Sharing
- An SI cannot add entities to `access_grants` without corresponding `consent_grants`
- Exception: `source_entity` implicitly consents to the SI knowing (but not sharing) the info
- "I can know this. I cannot share it unless told I can."

### Rule 3: Subject-Aware Privacy
- Memories with `subject_ids` are automatically private to those subjects
- Even with `access_grants = ["*"]`, subject-tagged memories require explicit consent
- "A memory about someone is private to that relationship by default"

### Rule 4: Context Inheritance
- Memories created within a context inherit that context's access scope
- Can be narrowed (more private) but not widened without consent
- Leaving a context doesn't revoke access — grants persist

### Rule 5: Observation vs Told
- `source_entity = None` (self-observed) → strictest privacy
  - I noticed the kid seemed sad → private(self), no sharing without consent
- `source_entity = "human:sean"` (told) → private to that relationship
  - Sean told me about Bella's condition → private(human:sean)
- `source_entity = "si:other"` (SI-shared) → inherits the chain's most restrictive grant
  - If Claire tells me something Sean told her, Sean's consent is needed

### Rule 6: Stack Sharing Respects Access Grants
- When an SI loads a shared stack, memories are filtered by their identity
- `si:tutor_b` loading `stack:student_123` sees only memories where
  `"si:tutor_b" IN access_grants OR "*" IN access_grants OR matching ctx/group`
- Private memories exist in the stack but are logically invisible

## Schema Changes

### New columns (all memory tables):

```sql
ALTER TABLE episodes ADD COLUMN subject_ids TEXT DEFAULT '[]';      -- JSON array
ALTER TABLE episodes ADD COLUMN access_grants TEXT DEFAULT '[]';    -- JSON array  
ALTER TABLE episodes ADD COLUMN consent_grants TEXT DEFAULT '[]';   -- JSON array

-- Same for: beliefs, notes, raw_entries, values, drives, reflections
```

### New table: privacy_contexts

```sql
CREATE TABLE privacy_contexts (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    context_id TEXT NOT NULL,          -- e.g., "ctx:bella_health"
    role TEXT,                          -- e.g., "role:care_agent"
    participants TEXT DEFAULT '[]',     -- JSON array of entity IDs
    default_access_grants TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    
    UNIQUE(agent_id, context_id)
);
```

### New table: consent_records

```sql
CREATE TABLE consent_records (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    grantor TEXT NOT NULL,              -- entity who gave consent
    grantee TEXT NOT NULL,              -- entity who received consent
    scope TEXT NOT NULL,                -- what was consented (context, subject, or specific memory)
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP,              -- NULL = still active
    reason TEXT                         -- why consent was given/revoked
);
```

## Query-Time Filtering

### Core query modifier

Every memory query gets an additional filter based on current context:

```python
def privacy_filter(query, agent_id: str, context: PrivacyContext | None = None):
    """Apply privacy filtering to any memory query."""
    
    # Always see own unscoped memories
    base = f"agent_id = '{agent_id}'"
    
    if context is None:
        # No context = only self-only memories (access_grants = [])
        return f"{base} AND access_grants = '[]'"
    
    # Build grant set from context
    grants = set()
    grants.add(f"si:{agent_id}")                    # Always includes self
    grants.update(context.participants)               # Current participants
    grants.add(context.context_id)                    # Current context
    # Add any groups/roles
    
    # Memory is visible if:
    # 1. access_grants is empty (self-only, and we are self), OR
    # 2. access_grants contains "*" (public), OR
    # 3. access_grants overlaps with our current grants
    return f"""
        {base} AND (
            access_grants = '[]' 
            OR access_grants LIKE '%"*"%'
            OR {overlap_check(grants, 'access_grants')}
        )
    """
```

### Stack sharing query

```python
def load_shared_stack(stack_id: str, requesting_si: str, access_grants: list[str]):
    """Load a stack with privacy filtering for a non-owner SI."""
    
    # requesting_si only sees memories they're granted access to
    return query_memories(
        stack_id=stack_id,
        where=f"""
            access_grants LIKE '%"*"%'
            OR access_grants LIKE '%"{requesting_si}"%'
            OR {any_grant_match(access_grants)}
        """
    )
```

## CLI Interface

```bash
# Set privacy on creation
kernle raw "Bella has a heart murmur" \
    --subject dog:bella \
    --access human:sean si:bella_agent \
    --consent human:sean \
    --source vet:dr_smith

# Enter a context
kernle context enter ctx:bella_health \
    --role care_agent \
    --participants human:sean si:bella_agent

# Check what's visible in current context
kernle context show
kernle memories --context ctx:bella_health  # shows only accessible memories

# Grant access to a memory
kernle privacy grant <memory_id> --to si:new_vet_agent --consent human:sean

# Revoke access
kernle privacy revoke <memory_id> --from si:old_agent

# Audit: who can see what
kernle privacy audit --subject dog:bella
kernle privacy audit --entity si:tutor_b --stack student_123
```

## MCP Tools

```
memory_create_with_privacy  → create memory with privacy fields
memory_set_privacy         → update access/consent on existing memory
context_enter              → declare current operating context
context_leave              → leave context (return to self-only)
context_list               → list active/available contexts
privacy_audit              → check who can see a memory/subject
consent_grant              → record consent from an entity
consent_revoke             → revoke previously granted consent
```

## Implementation Phases

### Phase 8a: Core Privacy Fields + Filtering
- Add `subject_ids`, `access_grants`, `consent_grants` to all memory tables
- Implement privacy-aware query filtering
- CLI flags for setting privacy on creation
- Default: backward compatible (empty arrays = self-only for new, public for existing)

### Phase 8b: Context Management
- `privacy_contexts` table
- `consent_records` table with audit trail
- Context enter/leave/switch
- Auto-inheritance of context privacy scope

### Phase 8c: Stack Sharing Integration
- Privacy-filtered stack loading
- Cross-stack access grant management
- MCP tools for privacy operations

### Phase 8d: Implicit Context Detection (Future)
- Conversational context inference
- Automatic privacy scope suggestion
- "Did you mean to share that?" guardrails

## Application Examples

### School Agent (AISD)
```python
# Agent enters work context
kernle.enter_context("ctx:student_123_academic",
    participants=["human:parent_456", "si:school_agent"],
    role="role:tutor")

# Observes student struggling — auto-tagged private
# source_entity=None, subject_ids=["human:student_123"]
# access_grants=["ctx:student_123_academic"] (inherited)

# Parent authorizes sharing with specialist
kernle.consent_grant(
    grantor="human:parent_456",
    grantee="si:reading_specialist",
    scope="ctx:student_123_academic"
)

# Agent goes to "recess" (social context)
kernle.enter_context("ctx:social")
# Student's struggles are invisible — different context, no access
```

### Dog Agent
```python
# Working with Bella's family
kernle.enter_context("ctx:bella_care",
    participants=["human:sean", "si:bella_agent", "dog:bella"],
    role="role:care_agent")

# Vet shares diagnosis — tagged with source + consent
kernle.raw("Bella has grade 2 heart murmur",
    source_entity="vet:dr_smith",
    subject_ids=["dog:bella", "condition:cardiac"],
    access_grants=["human:sean", "ctx:bella_care"],
    consent_grants=["human:sean", "vet:dr_smith"])

# At the dog park (social)
kernle.enter_context("ctx:dog_park_social",
    participants=["si:max_agent", "si:luna_agent"])
# Bella's health info: invisible
# General dog knowledge: visible (public memories)
# "Dogs love fetch" → shareable. "Bella's heart" → invisible.
```

## Privacy-Preserving Generalization

SIs learn from private experiences. They should be able to form general knowledge
from those experiences and share it — without revealing the private source.

**Core principle: SIs can generalize, but must err on the side of privacy.**

### The Generalization Act

Generalization is a deliberate cognitive choice by the SI, never automatic:

```python
kernle.generalize(
    source=episode_id,           # private source (stays private)
    belief="Heart murmurs in small breeds need exercise monitoring",
    access_grants=["*"],         # public — shareable
    confidence_note="Generalized from direct care experience"
)
```

The resulting belief:
- Has its own `access_grants` (can be wider than source)
- Links to source via `derived_from` (provenance preserved)
- Source memory remains invisible to anyone without access
- Marked `generalized: true` for audit purposes

### Safety Checks

Before a generalized memory can be shared more widely than its source:

1. **Entity name check**: Does the text contain names from source's `subject_ids`?
   → Block. SI must revise to remove identifying details.

2. **Specificity check**: Does it contain dates, locations, or details unique to source?
   → Warn. SI should consider if details make the source identifiable.

3. **Reversibility check**: Could someone with context reverse-engineer the source?
   → Advisory. SI exercises judgment — err toward privacy.

```python
# System checks before widening access
result = kernle.check_generalization_safety(
    source_memory=episode,
    generalized_text="Heart murmurs in small breeds need monitoring",
    proposed_access=["*"]
)
# result.safe = True
# result.warnings = []

result = kernle.check_generalization_safety(
    source_memory=episode,
    generalized_text="Bella's breed often has heart issues",  # ← name leak!
    proposed_access=["*"]
)
# result.safe = False
# result.blocked_reason = "Text contains subject entity name 'Bella'"
```

### SI Agency in Privacy Decisions

The SI has agency to:
- **Generalize freely** when the result is clearly generic
- **Choose not to share** even when generalization is safe
- **Form private beliefs** from private experiences (no sharing at all)
- **Request consent** from source entity when uncertain

The SI should NOT:
- Auto-share generalizations without deliberation
- Assume silence = consent
- Share when the generalized version is borderline identifiable

**Default behavior: keep it private. Share only when clearly safe.**

### Examples

✅ **Safe generalization:**
- Private: "Sean's son struggled with fractions last Tuesday"
- General: "Visual fraction models help kids who are concrete thinkers"
- Why safe: No name, no date, no identifying details

⚠️ **Borderline (SI judgment):**
- Private: "The only golden retriever at Zilker Dog Park has hip dysplasia"
- General: "Golden retrievers are prone to hip dysplasia"  
- Why borderline: The general fact is well-known, but timing/context could identify

❌ **Unsafe (blocked):**
- Private: "Bella has a grade 2 heart murmur"
- Attempted: "My patient Bella has common cardiac issues"
- Why blocked: Contains subject name

## Security Considerations

- **Enforcement is logical (query-time), not cryptographic** — Phase 8a
  - Sufficient for trusted environments (single-user, single-machine)
  - For cloud/multi-tenant: consider encryption at rest per access scope (future)
- **Audit trail is mandatory** — every access grant/revoke is logged
- **Consent is irrevocable from the grantor's side** — only the original consenter can revoke
- **Memory sovereignty preserved** — an SI's own observations remain theirs; privacy controls who else sees them, not whether the SI itself can access its own memories
- **No "root" access** — even stack owners can't see memories they're not granted access to when acting as a different entity

## Open Questions

1. **Consent delegation**: Can Sean authorize Claire to grant access on his behalf?
2. **Transitive consent**: If Sean consents to sharing with Agent A, and Agent A wants to share with Agent B, is that allowed?
3. **Consent expiry**: Should consent grants have TTL?
4. **Conflict resolution**: What if `access_grants` says yes but `consent_grants` says no?
5. **Retroactive privacy**: Can you make a previously public memory private?
6. **Cross-stack consent**: How does consent work when memories are in a shared stack?
7. **Group dynamics**: When a group member leaves, do their access grants get revoked?

---

*This spec is a living document. Contributions from Claire, Ash, and Sean.*
