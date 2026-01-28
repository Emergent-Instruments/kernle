# Kernle Python API

Quick reference for using Kernle as a Python library.

## Installation

```bash
pip install kernle
# or
uv add kernle
```

## Basic Usage

```python
from kernle import Kernle

# Initialize with agent ID
k = Kernle("my-agent")

# Load memory at session start
memory = k.load()

# Save an episode (experience)
episode_id = k.episode(
    objective="What I was trying to do",
    outcome="What happened",
    lessons=["What I learned"]
)

# Save a quick note
note_id = k.note("Important observation")

# Raw capture (quick thoughts)
raw_id = k.raw("Quick thought to process later")

# Save checkpoint before ending
k.checkpoint("Current working state")

# Check memory health
report = k.anxiety()
print(f"Anxiety: {report['overall_level']} ({report['overall_score']}/100)")
```

## Core Methods

### Memory Loading

```python
# Load all memory layers
memory = k.load()
# Returns: {checkpoint, values, beliefs, goals, lessons, recent_work, recent_notes}

# Access specific parts
checkpoint = memory.get('checkpoint')
values = memory.get('values', [])
beliefs = memory.get('beliefs', [])
```

### Recording Experiences

```python
# Episode - significant experience with outcome
k.episode(
    objective="Task description",
    outcome="Result description", 
    lessons=["Lesson 1", "Lesson 2"],
    tags=["tag1", "tag2"]
)

# Note - observation or decision
k.note(
    content="What you observed",
    note_type="observation"  # or "decision", "insight"
)

# Raw - quick capture for later processing
k.raw("Quick thought")
```

### Beliefs and Values

```python
# Add a belief
k.belief(
    statement="What I believe",
    belief_type="learned",  # or "core", "working"
    confidence=0.8
)

# Add a value
k.value(
    name="value_name",
    statement="What this value means to me",
    priority=80  # 0-100
)
```

### Search

```python
# Semantic search across all memories
results = k.search("query", limit=10)

for r in results:
    print(f"[{r['memory_type']}] {r['content'][:50]}...")
```

### Checkpoints

```python
# Save working state
k.checkpoint(
    task="What I'm working on",
    pending=["Next step 1", "Next step 2"],
    context="Additional context"
)

# Load checkpoint
checkpoint = k.load_checkpoint()
```

### Memory Health

```python
# Get anxiety report
report = k.anxiety()

# Available fields:
report['overall_score']    # 0-100
report['overall_level']    # "Calm", "Aware", "Anxious", "Critical"
report['overall_emoji']    # 🟢, 🟡, 🟠, 🔴
report['dimensions']       # Per-dimension breakdown

# Get status summary
status = k.status()
print(f"Episodes: {status['episode_count']}")
```

### Sync (Cloud Backup)

```python
# Push local changes to cloud
k.sync()

# Or control direction
k.sync_push()
k.sync_pull()
```

## Response Structures

### anxiety() / get_anxiety_report()

```python
{
    'overall_score': 27,           # 0-100 (lower is calmer)
    'overall_level': 'Calm',       # Human-readable level
    'overall_emoji': '🟢',         # Visual indicator
    'dimensions': {
        'context_pressure': {...},
        'unsaved_work': {...},
        'consolidation_debt': {...},
        'identity_coherence': {...},
        'memory_uncertainty': {...},
        'raw_aging': {...}
    },
    'timestamp': '...',
    'agent_id': '...'
}
```

### load()

```python
{
    'checkpoint': {...} or None,
    'values': [...],
    'beliefs': [...],
    'goals': [...],
    'drives': [...],
    'lessons': [...],
    'recent_work': [...],
    'recent_notes': [...],
    'relationships': [...]
}
```

## Error Handling

```python
from kernle import Kernle
from kernle.storage import StorageError

try:
    k = Kernle("agent")
    k.episode("test", "test")
except StorageError as e:
    print(f"Storage error: {e}")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Configuration

```python
# Custom database path
k = Kernle("agent", db_path="/path/to/memories.db")

# Enable auto-sync
k = Kernle("agent", auto_sync=True)
```
