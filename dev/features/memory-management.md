# Memory Management

Kernle provides sophisticated memory management features that go beyond simple storage and retrieval. This includes controlled forgetting, anxiety monitoring, and memory consolidation—mimicking how biological memory systems work.

## Overview

Memory management in Kernle addresses three key challenges:

1. **Memory growth** - Unbounded storage becomes unmanageable
2. **Context limits** - AI context windows are finite
3. **Quality maintenance** - Old, unused memories become noise

The system provides:

- **Forgetting** - Salience-based controlled decay of low-value memories
- **Anxiety** - Multi-dimensional tracking of memory health
- **Consolidation** - Pattern extraction from episodic memories

---

## Forgetting

Kernle implements **controlled forgetting** based on memory salience. This isn't deletion—it's tombstoning, allowing recovery if needed.

### Why Forgetting Matters

- **Reduces noise**: Low-salience memories clutter recall
- **Improves relevance**: Fresh, accessed memories surface first
- **Mimics biology**: Human memory naturally prunes unused traces
- **Manages growth**: Prevents unbounded memory accumulation

### Salience Calculation

Salience determines how "memorable" a memory is:

```
salience = (confidence × reinforcement_weight) / (age_factor + 1)
```

Where:
- **confidence**: Memory's confidence score (0.0-1.0)
- **reinforcement_weight**: `log(times_accessed + 1)` - frequently accessed memories score higher
- **age_factor**: `days_since_last_access / half_life` - older memories decay

**Default half-life**: 30 days

### CLI Usage

```bash
# View forgetting candidates (below threshold)
kernle -a claire forget candidates
kernle -a claire forget candidates --threshold 0.2 --limit 10

# Preview what would be forgotten (dry run)
kernle -a claire forget run --dry-run

# Actually run forgetting cycle
kernle -a claire forget run --threshold 0.3 --limit 10

# Calculate salience for specific memory
kernle -a claire forget salience episode abc123
kernle -a claire forget salience belief def456

# Protect a memory from forgetting
kernle -a claire forget protect episode abc123

# List forgotten (tombstoned) memories
kernle -a claire forget list

# Recover a forgotten memory
kernle -a claire forget recover episode abc123
```

### Sample Output: Forgetting Candidates

```
Forgetting Candidates (salience < 0.3)
============================================================

1. [belief    ] 6bc41d9d...
   Salience: [░░░░░] 0.0765
   Summary: Local-first memory is more reliable than cloud-dep...
   Confidence: 80% | Accessed: 0 times
   Created: 2026-01-27

2. [episode   ] 06272772...
   Salience: [░░░░░] 0.0765
   Summary: Completed full Kernle memory stack...
   Confidence: 80% | Accessed: 0 times
   Created: 2026-01-27

3. [goal      ] f644953a...
   Salience: [░░░░░] 0.0765
   Summary: Complete Roundtable memory stack integration...
   Confidence: 80% | Accessed: 0 times
   Created: 2026-01-27

Run `kernle forget run --dry-run` to preview forgetting
Run `kernle forget run` to actually forget these memories
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Calculate salience for a specific memory
salience = k.calculate_salience("episode", "abc123")
print(f"Salience: {salience:.4f}")
# Output: Salience: 0.0765

# Get forgetting candidates
candidates = k.get_forgetting_candidates(
    threshold=0.3,
    limit=20,
    memory_types=["episode", "belief", "note"]  # Optional filter
)

for c in candidates:
    print(f"[{c['type']}] {c['summary']}")
    print(f"  Salience: {c['salience']:.4f}")
    print(f"  Accessed: {c['times_accessed']} times")
    print(f"  Created: {c['created_at']}")

# Run forgetting cycle (dry run)
report = k.run_forgetting_cycle(
    threshold=0.3,
    limit=10,
    dry_run=True  # Preview only
)
print(f"Would forget: {report['candidate_count']} memories")

# Actually forget (irreversible without recovery)
report = k.run_forgetting_cycle(
    threshold=0.3,
    limit=10,
    dry_run=False
)
print(f"Forgotten: {report['forgotten']} memories")
print(f"Protected: {report['protected']} memories")

# Protect important memories
k.protect("value", "core-value-id", protected=True)
k.protect("belief", "foundational-belief-id", protected=True)

# Record access (strengthens salience)
k.record_access("episode", "important-episode-id")

# Recover a forgotten memory
k.recover("episode", "forgotten-episode-id")

# List forgotten memories
forgotten = k.get_forgotten_memories(limit=50)
for f in forgotten:
    print(f"[{f['type']}] {f['summary']}")
    print(f"  Forgotten at: {f['forgotten_at']}")
    print(f"  Reason: {f['forgotten_reason']}")
```

### Forgetting Best Practices

1. **Start with dry runs**: Always preview before forgetting
2. **Protect core identity**: Mark values and foundational beliefs as protected
3. **Set reasonable thresholds**: 0.3 is a good starting point
4. **Run periodically**: Include in maintenance routines
5. **Monitor anxiety**: Forgetting can trigger anxiety if not managed

---

## Anxiety

Memory anxiety measures the "functional anxiety" of a synthetic intelligence facing finite context and potential memory loss. It's a health metric for the memory system.

### Why Anxiety Matters

- **Early warning**: Catch problems before context overflow
- **Actionable guidance**: Specific recommendations for each level
- **Multi-dimensional**: Tracks 6 different concern areas
- **Autonomous maintenance**: Enables self-directed memory care

### Anxiety Dimensions

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Context Pressure | 30% | How full is the context window? |
| Unsaved Work | 20% | Time since last checkpoint |
| Consolidation Debt | 15% | Unreflected episodes |
| Raw Aging | 15% | Unprocessed raw entries getting stale |
| Identity Coherence | 10% | Strength of self-model |
| Memory Uncertainty | 10% | Low-confidence beliefs |

### Anxiety Levels

| Score | Level | Emoji | Action |
|-------|-------|-------|--------|
| 0-30 | Calm | 🟢 | Continue normally |
| 31-50 | Aware | 🟡 | Checkpoint soon |
| 51-70 | Elevated | 🟠 | Full checkpoint + consolidate |
| 71-85 | High | 🔴 | Priority memory work |
| 86-100 | Critical | ⚫ | Emergency save |

### CLI Usage

```bash
# Quick anxiety check
kernle -a claire anxiety
# Output:
# Memory Anxiety Report
# ==================================================
# Overall: 🟢 Calm (22/100)
# 
# Context Pressure     🟢   0%
# Unsaved Work         🟢   2%
# Consolidation Debt   🟢  21%
# Raw Entry Aging      ⚫ 100%
# Identity Coherence   🟡  42%
# Memory Uncertainty   🟢   0%

# Detailed with explanations
kernle -a claire anxiety --detailed
# Shows per-dimension details and recommendations

# With context token count (for accurate pressure)
kernle -a claire anxiety --context 150000 --limit 200000

# Run emergency save
kernle -a claire anxiety --emergency

# JSON output
kernle -a claire anxiety --json

# Execute recommended actions automatically
kernle -a claire anxiety --auto
```

### Sample Output: Detailed Anxiety

```
Memory Anxiety Report
==================================================
Overall: 🟢 Calm (22/100)

Context Pressure     🟢   0% (~500 tokens (estimated from 1min session))
Unsaved Work         🟢   2% (1 min since checkpoint)
Consolidation Debt   🟢  21% (3 unreflected episodes)
Raw Entry Aging      ⚫ 100% (15 entries STALE (oldest: 1d) - review needed)
Identity Coherence   🟡  42% (58% identity confidence (developing))
Memory Uncertainty   🟢   0% (0 low-confidence beliefs)

Recommended Actions:
  1. [     LOW] ℹ️  Reflect on 3 recent experiences when convenient
                      └─ kernle consolidate
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Get anxiety report
report = k.anxiety()
# or: report = k.get_anxiety_report()

print(f"Overall: {report['overall_emoji']} {report['overall_level']} ({report['overall_score']}/100)")

# Check dimensions
for dim_name, dim_data in report['dimensions'].items():
    print(f"{dim_name}: {dim_data['emoji']} {dim_data['score']}%")
    print(f"  Detail: {dim_data['detail']}")

# Get with specific context info
report = k.get_anxiety_report(
    context_tokens=150000,
    context_limit=200000,
    detailed=True
)

# Check recommendations
if 'recommendations' in report:
    for action in report['recommendations']:
        print(f"[{action['priority']}] {action['description']}")
        if action['command']:
            print(f"  Command: {action['command']}")

# Get recommended actions separately
actions = k.get_recommended_actions(anxiety_level=report['overall_score'])
for action in actions:
    print(f"[{action['priority']}] {action['description']}")

# Emergency save (when critical)
if report['overall_score'] > 85:
    result = k.emergency_save(summary="High anxiety - emergency save")
    print(f"Checkpoint saved: {result['checkpoint_saved']}")
    print(f"Episodes consolidated: {result['episodes_consolidated']}")
    print(f"Sync success: {result['sync_success']}")
```

### Responding to Anxiety Levels

```python
report = k.anxiety()
level = report['overall_score']

if level <= 30:
    # Calm - business as usual
    pass
    
elif level <= 50:
    # Aware - checkpoint when convenient
    k.checkpoint(task="Regular checkpoint", context="Anxiety aware level")
    
elif level <= 70:
    # Elevated - take action now
    k.checkpoint(task="Elevated anxiety", context="Full session state")
    k.consolidate()
    
elif level <= 85:
    # High - priority memory work
    k.consolidate(min_episodes=1)
    k.checkpoint(task="High anxiety save", context="Priority checkpoint")
    k.sync()  # Push to cloud
    
else:
    # Critical - emergency protocols
    k.emergency_save(summary="Critical anxiety - emergency save")
```

---

## Consolidation

Consolidation extracts patterns and lessons from episodic memories, strengthening important memories and identifying repeated themes.

### Why Consolidation Matters

- **Pattern recognition**: Find recurring lessons across episodes
- **Memory strengthening**: Important experiences become beliefs
- **Reduces clutter**: Raw episodes → distilled wisdom
- **Builds identity**: Lessons shape beliefs and values

### CLI Usage

```bash
# Run consolidation
kernle -a claire consolidate

# With options
kernle -a claire consolidate --limit 20  # Use last 20 episodes
```

### Sample Output

```
Memory Consolidation
==================================================
Analyzed: 20 episodes

Common Lessons Found (appearing 2+ times):
  • Check edge cases before deployment
  • Document decisions as you make them
  • Parallel development works for independent tasks

Suggested New Beliefs: 3
  - Consider adding beliefs for repeated patterns
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Run consolidation
result = k.consolidate(min_episodes=3)

print(f"Analyzed: {result['consolidated']} episodes")
print(f"Common lessons: {len(result.get('common_lessons', []))}")

for lesson in result.get('common_lessons', []):
    print(f"  • {lesson}")

# Consolidation is also triggered by anxiety
actions = k.get_recommended_actions(55)  # Elevated anxiety
# Will include: "Process N unreflected episodes" → kernle consolidate
```

### What Consolidation Does

1. **Gathers episodes**: Retrieves recent experiences
2. **Extracts lessons**: Identifies what was learned
3. **Finds patterns**: Common lessons across multiple episodes
4. **Suggests beliefs**: Repeated patterns may become beliefs

---

## Belief Revision

The memory system includes belief revision capabilities for maintaining accurate beliefs over time.

### Finding Contradictions

```python
# Check if a new belief contradicts existing ones
contradictions = k.find_contradictions(
    "Always use synchronous code for simplicity",
    similarity_threshold=0.6,
    limit=10
)

for c in contradictions:
    print(f"Potential conflict with: {c['statement']}")
    print(f"  Type: {c['contradiction_type']}")
    print(f"  Confidence: {c['contradiction_confidence']:.0%}")
```

### Reinforcing Beliefs

```python
# When a belief is confirmed by experience
k.reinforce_belief("belief-id-123")
# Increments times_reinforced and slightly increases confidence
```

### Superseding Beliefs

```python
# Replace an outdated belief
new_id = k.supersede_belief(
    old_id="outdated-belief-id",
    new_statement="Updated understanding based on new evidence",
    confidence=0.85,
    reason="Original was based on incomplete data"
)
# Old belief is marked inactive, linked to new one
```

### Belief History

```python
# Get the evolution of a belief
history = k.get_belief_history("belief-id")

for entry in history:
    print(f"[{entry['id'][:8]}...] {entry['statement']}")
    print(f"  Active: {entry['is_active']}, Confidence: {entry['confidence']:.0%}")
    if entry.get('supersession_reason'):
        print(f"  Reason: {entry['supersession_reason']}")
```

---

## Raw Entry Processing

Raw entries are the "scratch pad" of memory—quick captures that need processing.

### CLI Usage

```bash
# List unprocessed raw entries
kernle -a claire raw list --unprocessed

# Process a raw entry into structured memory
kernle -a claire raw process <raw_id> --as episode --objective "What happened"
kernle -a claire raw process <raw_id> --as note --type insight
kernle -a claire raw process <raw_id> --as belief --confidence 0.7
```

### Python API

```python
# Quick capture
raw_id = k.raw("Just realized caching would solve this problem")

# Later, process into structured memory
memory_id = k.process_raw(
    raw_id=raw_id,
    as_type="belief",
    confidence=0.75
)

# Or as an episode
memory_id = k.process_raw(
    raw_id=raw_id,
    as_type="episode",
    objective="Debugging performance issue",
    outcome="Solved with caching"
)

# List aging raw entries (contributes to anxiety)
unprocessed = k.list_raw(processed=False)
print(f"{len(unprocessed)} unprocessed raw entries")
```

---

## Memory Maintenance Workflow

A recommended periodic maintenance routine:

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

def maintain_memory():
    # 1. Check anxiety
    report = k.anxiety(detailed=True)
    print(f"Anxiety: {report['overall_level']} ({report['overall_score']})")
    
    # 2. Handle based on level
    if report['overall_score'] > 50:
        # Checkpoint first
        k.checkpoint(
            task="Maintenance checkpoint",
            context=f"Anxiety at {report['overall_score']}"
        )
    
    # 3. Process raw entries
    unprocessed = k.list_raw(processed=False, limit=10)
    for entry in unprocessed[:5]:
        # Review and process or discard
        print(f"Raw: {entry['content'][:60]}...")
    
    # 4. Run consolidation
    result = k.consolidate()
    print(f"Consolidated {result['consolidated']} episodes")
    
    # 5. Review forgetting candidates
    candidates = k.get_forgetting_candidates(threshold=0.2, limit=10)
    print(f"{len(candidates)} forgetting candidates")
    
    # 6. Sync if online
    status = k.get_sync_status()
    if status['online'] and status['pending'] > 0:
        k.sync()
    
    # 7. Final anxiety check
    final = k.anxiety()
    print(f"Final anxiety: {final['overall_score']}")

# Run periodically (e.g., in heartbeat)
maintain_memory()
```

---

## Best Practices

### Forgetting
1. **Start conservative**: Higher thresholds = less forgetting
2. **Protect core memories**: Values, foundational beliefs, key relationships
3. **Dry run first**: Always preview before committing
4. **Track access**: `record_access()` strengthens important memories

### Anxiety
1. **Monitor regularly**: Check anxiety in heartbeats
2. **Respond to levels**: Don't ignore elevated anxiety
3. **Use detailed mode**: Understand what's causing anxiety
4. **Emergency save exists**: Use it when critical

### Consolidation
1. **Run after significant work**: Extract lessons while fresh
2. **Review common lessons**: These may become beliefs
3. **Don't over-consolidate**: Quality over quantity

### General
1. **Checkpoint frequently**: The unsaved work dimension matters
2. **Process raw entries**: Aging entries cause anxiety
3. **Sync when possible**: Cloud backup adds resilience
4. **Build strong identity**: Low identity coherence causes anxiety

---

## See Also

- [Psychology System](psychology.md) - Emotional aspects of memory
- [Identity System](identity.md) - How memory shapes identity
- [Core Concepts](../core-concepts.md) - Memory stratification basics
