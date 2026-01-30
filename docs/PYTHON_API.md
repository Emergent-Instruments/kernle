# Kernle Python API Documentation

Complete reference for using Kernle as a Python library for stratified memory in synthetic intelligences.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
- [Memory Types](#memory-types)
- [Feature APIs](#feature-apis)
  - [Anxiety Tracking](#anxiety-tracking)
  - [Emotional Memory](#emotional-memory)
  - [Controlled Forgetting](#controlled-forgetting)
  - [Knowledge Mapping](#knowledge-mapping)
  - [Meta-Memory](#meta-memory)
- [Storage Backends](#storage-backends)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Installation

```bash
pip install kernle
```

## Quick Start

```python
from kernle import Kernle

# Initialize with auto-detected storage backend
# (SQLite by default, Supabase if credentials are configured)
k = Kernle(agent_id="my_agent")

# Record an experience
episode_id = k.episode(
    objective="Learn how to use Kernle",
    outcome="Successfully recorded my first episode",
    lessons=["Kernle is easy to use", "Episodes track experiences"]
)

# Store a belief
belief_id = k.belief(
    statement="Kernle provides stratified memory for AI agents",
    belief_type="factual",
    evidence="Read the documentation"
)

# Save a value (core principle)
value_id = k.value(
    name="Code Quality",
    statement="Always write clean, well-documented code",
    priority=90
)

# Search memories
results = k.search("how to use Kernle", limit=5)

# Check memory anxiety level
anxiety_report = k.anxiety()
print(f"Overall anxiety: {anxiety_report['overall_score']}/100")
```

---

## Core API

### Initialization

#### `Kernle(agent_id, storage=None, supabase_url=None, supabase_key=None, checkpoint_dir=None)`

Initialize a Kernle instance for memory management.

**Parameters:**
- `agent_id` (str): Unique identifier for the agent. This determines which agent's memories to access.
- `storage` (StorageProtocol, optional): Custom storage backend. If None, auto-detects based on environment.
- `supabase_url` (str, optional): Supabase project URL (deprecated - use storage param instead).
- `supabase_key` (str, optional): Supabase API key (deprecated - use storage param instead).
- `checkpoint_dir` (Path, optional): Directory for local checkpoints. Defaults to `~/.kernle/checkpoints`.

**Returns:** Kernle instance

**Examples:**

```python
# Auto-detect storage (recommended)
# Uses SQLite if no Supabase credentials found in environment
k = Kernle(agent_id="my_agent")

# Explicit SQLite storage
from kernle.storage import SQLiteStorage
storage = SQLiteStorage(agent_id="my_agent", db_path="/path/to/kernle.db")
k = Kernle(agent_id="my_agent", storage=storage)

# Explicit Supabase storage (requires credentials)
k = Kernle(
    agent_id="my_agent",
    supabase_url="https://xxx.supabase.co",
    supabase_key="your_api_key"
)
```

**Environment Variables:**
- `KERNLE_AGENT_ID`: Default agent ID if not specified
- `KERNLE_SUPABASE_URL` or `SUPABASE_URL`: Supabase project URL
- `KERNLE_SUPABASE_KEY` or `SUPABASE_SERVICE_ROLE_KEY`: Supabase API key
- `KERNLE_AUTO_SYNC`: Enable/disable automatic cloud sync (true/false)

---

### Memory Loading

#### `load(budget=8000, truncate=True, max_item_chars=500, sync=None, track_access=True)`

Load memories within a token budget. This is the primary method for retrieving your agent's memories efficiently.

**Parameters:**
- `budget` (int): Maximum tokens to use (default: 8000, max: 50000, min: 100)
- `truncate` (bool): Whether to truncate long items (default: True)
- `max_item_chars` (int): Maximum characters per memory item before truncation (default: 500)
- `sync` (bool, optional): Force sync before loading. If None, uses auto-sync setting.
- `track_access` (bool): Whether to record access for salience tracking (default: True). Set to False for internal operations.

**Returns:** Dict with memory data organized by type, plus `_meta` with budget information

**Example:**

```python
# Load memories within an 8000 token budget
memory = k.load(budget=8000)

# Access different memory types
print(f"Values: {len(memory['values'])}")
print(f"Beliefs: {len(memory['beliefs'])}")
print(f"Goals: {len(memory['goals'])}")

# Check budget usage via _meta
meta = memory.get('_meta', {})
print(f"Budget used: {meta.get('budget_used')}/{meta.get('budget_total')}")
print(f"Memories excluded: {meta.get('excluded_count')}")

# Load with larger budget and no truncation
memory = k.load(budget=20000, truncate=False)

# Load without tracking access (for internal operations)
memory = k.load(budget=6000, track_access=False)
```

**Memory Structure:**
```python
{
    "checkpoint": {...},  # Most recent checkpoint
    "values": [...],      # Core principles and priorities
    "beliefs": [...],     # Facts and knowledge
    "goals": [...],       # Active goals
    "drives": [...],      # Motivational drives
    "episodes": [...],    # Recent experiences
    "notes": [...],       # Quick notes and observations
    "relationships": [...],  # Social connections
    "_meta": {            # Budget and selection metadata
        "budget_used": 5234,      # Tokens actually consumed
        "budget_total": 8000,     # Original budget requested
        "excluded_count": 47      # Memories that couldn't fit
    }
}
```

**Note:** When `excluded_count` is high, consider:
- Increasing your budget
- Running `k.consolidate()` to compress episodes into beliefs
- Using `k.forget_candidates()` to identify low-salience memories

---

### Specific Memory Loaders

These methods load specific memory types without budget constraints. Useful when you need full access to a particular memory category.

#### `load_values(limit=10)`

Load value declarations (core principles).

**Returns:** List of value dicts with name, statement, priority, etc.

```python
values = k.load_values(limit=10)
for value in values:
    print(f"{value['name']}: {value['statement']} (priority: {value['priority']})")
```

#### `load_beliefs(limit=20)`

Load beliefs (facts and knowledge).

**Returns:** List of belief dicts with statement, confidence, belief_type, etc.

```python
beliefs = k.load_beliefs(limit=20)
for belief in beliefs:
    print(f"{belief['statement']} (confidence: {belief['confidence']})")
```

#### `load_goals(limit=10, status='active')`

Load goals.

**Parameters:**
- `limit` (int): Maximum goals to load
- `status` (str): Filter by status ('active', 'completed', 'abandoned', or None for all)

**Returns:** List of goal dicts

```python
# Get active goals
active_goals = k.load_goals(limit=10, status='active')

# Get all goals
all_goals = k.load_goals(limit=50, status=None)
```

#### `load_lessons(limit=20)`

Load lessons learned from episodes.

**Returns:** List of lesson strings

```python
lessons = k.load_lessons(limit=20)
for lesson in lessons:
    print(f"- {lesson}")
```

#### `load_recent_work(limit=5)`

Load recent work episodes (non-checkpoint episodes).

**Returns:** List of episode dicts

```python
recent_work = k.load_recent_work(limit=5)
for episode in recent_work:
    print(f"{episode['objective']}: {episode['outcome']}")
```

#### `load_recent_notes(limit=5)`

Load recent notes.

**Returns:** List of note dicts

```python
notes = k.load_recent_notes(limit=5)
for note in notes:
    print(f"{note['note_type']}: {note['content']}")
```

#### `load_drives()`

Load motivational drives.

**Returns:** List of drive dicts with type, intensity, satisfaction level

```python
drives = k.load_drives()
for drive in drives:
    print(f"{drive['drive_type']}: {drive['intensity']} intensity, {drive['satisfaction_level']} satisfaction")
```

#### `load_relationships(limit=10)`

Load relationships with other agents.

**Returns:** List of relationship dicts

```python
relationships = k.load_relationships(limit=10)
for rel in relationships:
    print(f"{rel['other_agent_id']}: {rel['relationship_type']} (sentiment: {rel['sentiment']})")
```

---

### Checkpoints

Checkpoints save your agent's current working state and context for recovery.

#### `checkpoint(task, pending=None, context=None, summary=None)`

Create a checkpoint of current work state.

**Parameters:**
- `task` (str, required): Description of current task (max 200 chars)
- `pending` (list[str], optional): List of pending tasks
- `context` (str, optional): Additional context about the session
- `summary` (str, optional): Brief summary of recent work

**Returns:** Dict with checkpoint data

**Example:**

```python
# Basic checkpoint
cp = k.checkpoint(
    task="Implementing user authentication"
)

# Detailed checkpoint with context
cp = k.checkpoint(
    task="Refactoring database layer",
    pending=["Add tests", "Update documentation", "Review performance"],
    context="Working on Issue #42 - improving query efficiency",
    summary="Converted 3 functions to use async patterns"
)

# The checkpoint is automatically saved and can be loaded later
checkpoint_id = cp['id']
```

#### `load_checkpoint()`

Load the most recent checkpoint.

**Returns:** Dict with checkpoint data, or None if no checkpoint exists

```python
cp = k.load_checkpoint()
if cp:
    print(f"Last task: {cp['task']}")
    print(f"Pending: {cp['pending_tasks']}")
    print(f"Context: {cp['context']}")
else:
    print("No checkpoint found")
```

#### `clear_checkpoint()`

Delete the current checkpoint (use when starting fresh).

**Returns:** bool - True if deleted, False if no checkpoint existed

```python
# Start a new session fresh
if k.clear_checkpoint():
    print("Checkpoint cleared - starting fresh")
```

---

### Episodes

Episodes are records of experiences - what you tried to do, what happened, and what you learned.

#### `episode(objective, outcome, lessons=None, tags=None, relates_to=None, source=None)`

Record an experience or work episode.

**Parameters:**
- `objective` (str, required): What were you trying to accomplish?
- `outcome` (str, required): What actually happened?
- `lessons` (list[str], optional): Lessons learned from this experience
- `tags` (list[str], optional): Tags for categorization
- `relates_to` (list[str], optional): IDs of related memories
- `source` (str, optional): Source context (e.g., "session with Sean")

**Returns:** str - Episode ID

**Example:**

```python
# Basic episode
episode_id = k.episode(
    objective="Fix the login bug",
    outcome="Successfully fixed - was a typo in the password validation"
)

# Detailed episode with lessons
episode_id = k.episode(
    objective="Implement caching layer",
    outcome="Completed and deployed - reduced API response time by 40%",
    lessons=[
        "Redis works well for session caching",
        "Need to set TTL carefully to avoid stale data",
        "Monitor cache hit rate to measure effectiveness"
    ],
    tags=["performance", "redis", "caching"]
)

# Episode with relationships
episode_id = k.episode(
    objective="Research GraphQL vs REST",
    outcome="Decided to use GraphQL for the new API",
    lessons=["GraphQL reduces over-fetching", "Requires more setup initially"],
    relates_to=[belief_id_about_api_design],
    source="research session"
)
```

**Outcome Type Detection:**

The method automatically detects outcome type from the outcome text:
- **success**: Contains words like "success", "completed", "done", "accomplished"
- **failure**: Contains words like "fail", "error", "broke", "unable", "couldn't"
- **partial**: Everything else

#### `update_episode(episode_id, lessons=None, tags=None, outcome=None, relates_to=None)`

Update an existing episode (e.g., add lessons learned later).

**Parameters:**
- `episode_id` (str, required): ID of episode to update
- `lessons` (list[str], optional): New lessons to add
- `tags` (list[str], optional): New tags to add
- `outcome` (str, optional): Update the outcome description
- `relates_to` (list[str], optional): New related memory IDs

**Returns:** bool - True if updated successfully

```python
# Add lessons learned after reflection
k.update_episode(
    episode_id,
    lessons=["The real issue was in the database query, not the cache"]
)

# Add tags for organization
k.update_episode(
    episode_id,
    tags=["debugging", "performance-issue"]
)
```

---

### Notes

Quick notes and observations that don't need the structure of an episode.

#### `note(content, note_type='observation', tags=None, relates_to=None)`

Save a quick note or observation.

**Parameters:**
- `content` (str, required): The note content
- `note_type` (str): Type of note - 'observation', 'question', 'idea', 'reminder', or 'decision'
- `tags` (list[str], optional): Tags for categorization
- `relates_to` (list[str], optional): Related memory IDs

**Returns:** str - Note ID

**Example:**

```python
# Quick observation
note_id = k.note(
    content="Users are confused by the current onboarding flow",
    note_type="observation",
    tags=["UX", "onboarding"]
)

# Question to investigate later
note_id = k.note(
    content="Should we migrate to TypeScript for better type safety?",
    note_type="question",
    tags=["architecture", "typescript"]
)

# Decision record
note_id = k.note(
    content="Decided to use PostgreSQL for the new service based on team familiarity",
    note_type="decision",
    tags=["architecture", "database"]
)

# Idea for future work
note_id = k.note(
    content="Could implement a suggestion system based on user behavior patterns",
    note_type="idea",
    tags=["features", "ML"]
)
```

---

### Beliefs

Beliefs represent knowledge and facts your agent has learned.

#### `belief(statement, belief_type, evidence=None, confidence=0.8, source=None)`

Record a belief (fact, principle, or piece of knowledge).

**Parameters:**
- `statement` (str, required): The belief statement
- `belief_type` (str, required): Type - 'factual', 'opinion', 'principle', 'strategy', 'model'
- `evidence` (str, optional): Supporting evidence
- `confidence` (float): Confidence level 0.0-1.0 (default: 0.8)
- `source` (str, optional): Where this belief came from

**Returns:** str - Belief ID

**Example:**

```python
# Factual belief
belief_id = k.belief(
    statement="Python uses indentation for code blocks instead of braces",
    belief_type="factual",
    confidence=1.0,
    evidence="Python language specification"
)

# Strategic belief
belief_id = k.belief(
    statement="Writing tests before code helps clarify requirements",
    belief_type="strategy",
    confidence=0.85,
    evidence="Improved code quality in recent projects using TDD"
)

# Model/mental model belief
belief_id = k.belief(
    statement="Users prefer gradual feature discovery over comprehensive tutorials",
    belief_type="model",
    confidence=0.7,
    evidence="User research session on 2024-03-15"
)
```

#### `update_belief(belief_id, confidence=None, evidence=None, tags=None)`

Update an existing belief (e.g., adjust confidence based on new evidence).

**Parameters:**
- `belief_id` (str, required): ID of belief to update
- `confidence` (float, optional): New confidence level
- `evidence` (str, optional): Additional evidence to add
- `tags` (list[str], optional): Tags to add

**Returns:** bool - True if updated

```python
# Update confidence after verification
k.update_belief(belief_id, confidence=0.95)

# Add new evidence
k.update_belief(
    belief_id,
    evidence="Confirmed in production - 40% faster than old approach"
)
```

#### `reinforce_belief(belief_id)`

Reinforce a belief (increases confidence slightly).

**Returns:** bool - True if reinforced

```python
# Belief was confirmed again
k.reinforce_belief(belief_id)
```

#### `supersede_belief(old_belief_id, new_statement, new_evidence=None)`

Replace an outdated belief with a new one.

**Parameters:**
- `old_belief_id` (str): ID of belief being superseded
- `new_statement` (str): New belief statement
- `new_evidence` (str, optional): Evidence for the new belief

**Returns:** str - New belief ID

```python
# Replace outdated belief
new_belief_id = k.supersede_belief(
    old_belief_id="abc-123",
    new_statement="React Hooks are now the standard way to write React components",
    new_evidence="React team deprecated class components in favor of hooks"
)
```

---

### Values

Values represent core principles and priorities that guide decision-making.

#### `value(name, statement, priority=50, rationale=None)`

Declare a core value or principle.

**Parameters:**
- `name` (str, required): Short name for the value
- `statement` (str, required): Full statement of the value
- `priority` (int): Priority 0-100 (default: 50)
- `rationale` (str, optional): Why this value matters

**Returns:** str - Value ID

**Example:**

```python
# High-priority value
value_id = k.value(
    name="User Privacy",
    statement="User data must be protected and never shared without explicit consent",
    priority=95,
    rationale="Privacy is a fundamental user right and legal requirement"
)

# Development principle
value_id = k.value(
    name="Code Clarity",
    statement="Code should be written for humans to read, not just machines",
    priority=80,
    rationale="Maintainability is more important than cleverness"
)
```

---

### Goals

Goals represent objectives to accomplish.

#### `goal(title, description, metrics=None, target_date=None, priority=50)`

Create a goal.

**Parameters:**
- `title` (str, required): Short title for the goal
- `description` (str, required): Detailed description
- `metrics` (str, optional): How to measure success
- `target_date` (str, optional): Target completion date (ISO format)
- `priority` (int): Priority 0-100 (default: 50)

**Returns:** str - Goal ID

**Example:**

```python
# Create a goal
goal_id = k.goal(
    title="Improve API response time",
    description="Reduce average API response time from 300ms to under 100ms",
    metrics="Average response time < 100ms for 95th percentile",
    target_date="2024-12-31",
    priority=80
)
```

#### `update_goal(goal_id, status=None, progress=None, completion_notes=None)`

Update goal progress or status.

**Parameters:**
- `goal_id` (str, required): ID of goal to update
- `status` (str, optional): New status - 'active', 'completed', 'abandoned'
- `progress` (str, optional): Progress description
- `completion_notes` (str, optional): Notes about completion/abandonment

**Returns:** bool - True if updated

```python
# Update progress
k.update_goal(goal_id, progress="Implemented caching - now at 150ms average")

# Mark completed
k.update_goal(
    goal_id,
    status='completed',
    completion_notes="Achieved 95ms average response time through caching and query optimization"
)
```

---

### Search

#### `search(query, limit=10, min_score=None)`

Search across all memory types using semantic search.

**Parameters:**
- `query` (str, required): Search query
- `limit` (int): Maximum results (default: 10)
- `min_score` (float, optional): Minimum relevance score (0.0-1.0)

**Returns:** List of matching memory dicts with relevance scores

**Example:**

```python
# Search for relevant memories
results = k.search("how to handle authentication", limit=5)

for result in results:
    print(f"Type: {result['type']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['content'][:100]}")
    print("---")

# Search with minimum relevance threshold
high_quality_results = k.search(
    "database optimization",
    limit=10,
    min_score=0.7  # Only return highly relevant results
)
```

---

### Status and Introspection

#### `status()`

Get current memory status and statistics.

**Returns:** Dict with memory counts and statistics

**Example:**

```python
status = k.status()
print(f"Episodes: {status['episode_count']}")
print(f"Beliefs: {status['belief_count']}")
print(f"Values: {status['value_count']}")
print(f"Goals: {status['goal_count']}")
print(f"Notes: {status['note_count']}")
```

---

## Feature APIs

Kernle includes several advanced feature modules for specialized memory capabilities.

### Anxiety Tracking

Memory anxiety measures the functional anxiety of an AI agent facing finite context and potential memory loss.

#### `anxiety(context_tokens=None, context_limit=200000, detailed=False)`

**Alias:** `get_anxiety_report()`

Calculate memory anxiety across 6 dimensions:
1. **Context Pressure**: How full is the context window?
2. **Unsaved Work**: Time since last checkpoint
3. **Consolidation Debt**: Unreflected episodes needing processing
4. **Raw Aging**: Unprocessed raw entries getting stale
5. **Identity Coherence**: Strength of self-model
6. **Memory Uncertainty**: Low-confidence beliefs

**Parameters:**
- `context_tokens` (int, optional): Current context token usage
- `context_limit` (int): Maximum context window size (default: 200000)
- `detailed` (bool): Include detailed recommendations (default: False)

**Returns:** Dict with anxiety metrics and recommendations

**Example:**

```python
# Check anxiety level
anxiety = k.anxiety()
print(f"Overall anxiety: {anxiety['overall_score']}/100")
print(f"Level: {anxiety['overall_emoji']} {anxiety['overall_level']}")

# Detailed report with recommendations
anxiety = k.anxiety(context_tokens=50000, detailed=True)

for dimension, data in anxiety['dimensions'].items():
    print(f"{dimension}: {data['score']}/100 {data['emoji']}")
    print(f"  {data['detail']}")

if anxiety['overall_score'] > 50:
    print("\nRecommended actions:")
    for action in anxiety['recommendations']:
        print(f"- [{action['priority']}] {action['description']}")
```

**Anxiety Levels:**
- 🟢 **Calm (0-30)**: Continue normal work
- 🟡 **Aware (31-50)**: Consider checkpointing soon
- 🟠 **Elevated (51-70)**: Full checkpoint and consolidation needed
- 🔴 **High (71-85)**: Priority memory work required
- ⚫ **Critical (86-100)**: Emergency save immediately

#### `emergency_save(summary=None)`

Perform emergency memory preservation when anxiety reaches critical levels.

**Parameters:**
- `summary` (str, optional): Session summary for the checkpoint

**Returns:** Dict with results of the emergency save

```python
# When anxiety is critical
if anxiety['overall_score'] > 85:
    result = k.emergency_save(summary="Critical context overflow - saving all state")
    print(f"Checkpoint saved: {result['checkpoint_saved']}")
    print(f"Episodes consolidated: {result['episodes_consolidated']}")
```

---

### Emotional Memory

Track and search memories by emotional content using valence/arousal model.

#### `detect_emotion(text)`

Detect emotional signals in text.

**Parameters:**
- `text` (str): Text to analyze

**Returns:** Dict with valence, arousal, tags, and confidence

**Example:**

```python
# Detect emotion in text
emotion = k.detect_emotion("I'm really frustrated with this bug - spent 3 hours on it!")
print(f"Valence: {emotion['valence']}")  # -1.0 (negative) to 1.0 (positive)
print(f"Arousal: {emotion['arousal']}")  # 0.0 (calm) to 1.0 (intense)
print(f"Emotions: {emotion['tags']}")    # e.g., ['frustration']
```

#### `episode_with_emotion(objective, outcome, lessons=None, valence=None, arousal=None, emotional_tags=None, auto_detect=True, ...)`

Record an episode with emotional tagging. If emotional parameters are not provided and `auto_detect=True`, emotions will be automatically detected from the text.

**Example:**

```python
# Auto-detect emotions
episode_id = k.episode_with_emotion(
    objective="Deploy the new feature",
    outcome="Successfully deployed! Users love it",
    lessons=["Deployment went smoothly with the new pipeline"]
    # Emotions auto-detected: positive valence, moderate arousal
)

# Explicit emotional tagging
episode_id = k.episode_with_emotion(
    objective="Debug production issue",
    outcome="Fixed after 4 hours of investigation",
    valence=-0.3,  # Slightly negative
    arousal=0.8,   # High intensity
    emotional_tags=["frustration", "relief"],
    auto_detect=False
)
```

#### `get_emotional_summary(days=7)`

Get emotional pattern summary over time period.

**Returns:** Dict with average valence/arousal, dominant emotions, and trajectory

```python
# Analyze emotional patterns over last week
summary = k.get_emotional_summary(days=7)
print(f"Average valence: {summary['average_valence']}")
print(f"Average arousal: {summary['average_arousal']}")
print(f"Dominant emotions: {summary['dominant_emotions']}")

# View trajectory
for day in summary['emotional_trajectory']:
    print(f"{day['date']}: valence={day['valence']}, arousal={day['arousal']}")
```

#### `search_by_emotion(valence_range=None, arousal_range=None, tags=None, limit=10)`

Find episodes matching emotional criteria.

**Parameters:**
- `valence_range` (tuple, optional): (min, max) valence filter, e.g., (0.5, 1.0) for positive
- `arousal_range` (tuple, optional): (min, max) arousal filter
- `tags` (list[str], optional): Emotional tags to match
- `limit` (int): Maximum results

**Example:**

```python
# Find positive, high-energy episodes
positive_episodes = k.search_by_emotion(
    valence_range=(0.5, 1.0),
    arousal_range=(0.7, 1.0),
    limit=10
)

# Find frustrating episodes
frustrations = k.search_by_emotion(
    tags=["frustration"],
    limit=20
)
```

---

### Controlled Forgetting

Gracefully manage memory decay through salience-based forgetting.

#### `calculate_salience(memory_type, memory_id)`

Calculate current salience score for a memory.

Salience considers:
- Base confidence of the memory
- How often it's been accessed (reinforcement)
- How long since it was last accessed (decay)

**Parameters:**
- `memory_type` (str): Type of memory (episode, belief, value, goal, note, drive, relationship)
- `memory_id` (str): Memory ID

**Returns:** float - Salience score (typically 0.0-1.0, can exceed 1.0 for very active memories)

```python
# Check salience of a belief
salience = k.calculate_salience("belief", belief_id)
print(f"Salience: {salience}")
if salience < 0.3:
    print("This memory has low salience and may be forgotten")
```

#### `get_forgetting_candidates(threshold=0.3, limit=20, memory_types=None)`

Find low-salience memories eligible for forgetting.

**Parameters:**
- `threshold` (float): Salience threshold (default: 0.3)
- `limit` (int): Maximum candidates
- `memory_types` (list[str], optional): Filter by memory type

**Returns:** List of candidate memory dicts with salience scores

```python
# Find forgetting candidates
candidates = k.get_forgetting_candidates(threshold=0.3, limit=20)

for candidate in candidates:
    print(f"{candidate['type']}: {candidate['summary']}")
    print(f"  Salience: {candidate['salience']}")
    print(f"  Last accessed: {candidate['last_accessed']}")
```

#### `run_forgetting_cycle(threshold=0.3, limit=10, dry_run=True)`

Review and optionally forget low-salience memories.

**Parameters:**
- `threshold` (float): Salience threshold
- `limit` (int): Maximum to forget in one cycle
- `dry_run` (bool): If True, only report what would be forgotten (default: True)

**Returns:** Dict with cycle report

```python
# Dry run - see what would be forgotten
report = k.run_forgetting_cycle(threshold=0.3, limit=10, dry_run=True)
print(f"Would forget {report['candidate_count']} memories")

# Actually forget low-salience memories
report = k.run_forgetting_cycle(threshold=0.3, limit=10, dry_run=False)
print(f"Forgot {report['forgotten']} memories")
```

#### `forget(memory_type, memory_id, reason=None)`

Tombstone a memory (mark as forgotten, don't delete).

**Parameters:**
- `memory_type` (str): Type of memory
- `memory_id` (str): Memory ID
- `reason` (str, optional): Reason for forgetting

**Returns:** bool - True if forgotten

```python
# Forget a specific memory
success = k.forget("episode", episode_id, reason="Low salience - not relevant anymore")
```

#### `recover(memory_type, memory_id)`

Recover a forgotten memory.

**Returns:** bool - True if recovered

```python
# Recover a forgotten memory
k.recover("belief", belief_id)
```

#### `protect(memory_type, memory_id, protected=True)`

Mark memory as protected from forgetting (for core identity memories).

```python
# Protect a core value from forgetting
k.protect("value", value_id, protected=True)

# Unprotect
k.protect("value", value_id, protected=False)
```

#### `record_access(memory_type, memory_id)`

Record that a memory was accessed (for salience tracking).

Call this when retrieving memories to update access statistics and boost salience.

```python
# When you retrieve and use a memory, record the access
k.record_access("belief", belief_id)
```

---

### Knowledge Mapping

Meta-cognition capabilities for understanding your knowledge domains, competence boundaries, and learning opportunities.

#### `get_knowledge_map()`

Map knowledge domains with coverage assessment.

Analyzes beliefs, episodes, and notes to understand what domains you have knowledge about and how confident you are.

**Returns:** Dict with domains, blind spots, and uncertain areas

```python
knowledge_map = k.get_knowledge_map()

print(f"Total domains: {knowledge_map['total_domains']}")

for domain in knowledge_map['domains']:
    print(f"{domain['name']}:")
    print(f"  Coverage: {domain['coverage']}")
    print(f"  Beliefs: {domain['belief_count']} (avg confidence: {domain['avg_confidence']})")
    print(f"  Episodes: {domain['episode_count']}")

print(f"\nBlind spots: {knowledge_map['blind_spots']}")
print(f"Uncertain areas: {knowledge_map['uncertain_areas']}")
```

#### `detect_knowledge_gaps(query)`

Analyze if you have relevant knowledge for a query.

**Parameters:**
- `query` (str): The query to check knowledge for

**Returns:** Dict with relevant memories, confidence, gaps, and recommendation

```python
# Check knowledge before attempting a task
gap_analysis = k.detect_knowledge_gaps("How do I implement OAuth2 in FastAPI?")

print(f"Has relevant knowledge: {gap_analysis['has_relevant_knowledge']}")
print(f"Confidence: {gap_analysis['confidence']}")
print(f"Recommendation: {gap_analysis['recommendation']}")

if gap_analysis['gaps']:
    print(f"Knowledge gaps: {gap_analysis['gaps']}")

# Review relevant memories
for belief in gap_analysis['relevant_beliefs']:
    print(f"- {belief['statement']}")
```

#### `get_competence_boundaries()`

Understand what you're good at vs not good at.

Analyzes belief confidence distribution, episode outcomes, and domain coverage to identify strengths and weaknesses.

**Returns:** Dict with strengths, weaknesses, and overall metrics

```python
competence = k.get_competence_boundaries()

print("Strengths:")
for strength in competence['strengths']:
    print(f"  {strength['domain']}: {strength['confidence']} confidence, {strength['success_rate']} success rate")

print("\nWeaknesses:")
for weakness in competence['weaknesses']:
    print(f"  {weakness['domain']}: {weakness['confidence']} confidence, {weakness['success_rate']} success rate")

print(f"\nOverall confidence: {competence['overall_confidence']}")
print(f"Overall success rate: {competence['success_rate']}")
```

#### `identify_learning_opportunities(limit=5)`

What should you learn next?

Identifies opportunities based on:
- Low-coverage domains that are referenced often
- Uncertain beliefs affecting decisions
- Failed episodes indicating knowledge gaps
- Stale knowledge needing refresh

**Parameters:**
- `limit` (int): Maximum opportunities to return

**Returns:** List of learning opportunities with priority and reasoning

```python
opportunities = k.identify_learning_opportunities(limit=5)

for opp in opportunities:
    print(f"[{opp['priority']}] {opp['type']}: {opp['domain']}")
    print(f"  Reason: {opp['reason']}")
    print(f"  Action: {opp['suggested_action']}")
```

---

### Meta-Memory

Memory provenance, confidence tracking, and verification.

#### `get_memory_confidence(memory_type, memory_id)`

Get confidence score for a memory.

**Returns:** float - Confidence 0.0-1.0, or -1.0 if not found

```python
confidence = k.get_memory_confidence("belief", belief_id)
print(f"Confidence: {confidence}")
```

#### `verify_memory(memory_type, memory_id, evidence=None)`

Verify a memory, increasing its confidence.

**Parameters:**
- `memory_type` (str): Type of memory
- `memory_id` (str): Memory ID
- `evidence` (str, optional): Supporting evidence

**Returns:** bool - True if verified

```python
# Verify a belief with new evidence
k.verify_memory(
    "belief",
    belief_id,
    evidence="Confirmed in production - 50% performance improvement"
)
```

#### `get_memory_lineage(memory_type, memory_id)`

Get provenance chain for a memory (where it came from).

**Returns:** Dict with lineage information

```python
lineage = k.get_memory_lineage("belief", belief_id)
print(f"Source type: {lineage['source_type']}")
print(f"Source episodes: {lineage['source_episodes']}")
print(f"Derived from: {lineage['derived_from']}")
print(f"Verification count: {lineage['verification_count']}")
```

#### `get_uncertain_memories(threshold=0.5, limit=20)`

Get memories with confidence below threshold.

**Parameters:**
- `threshold` (float): Confidence threshold
- `limit` (int): Maximum results

**Returns:** List of low-confidence memory dicts

```python
uncertain = k.get_uncertain_memories(threshold=0.5, limit=20)

for memory in uncertain:
    print(f"{memory['type']}: {memory['summary']}")
    print(f"  Confidence: {memory['confidence']}")
```

#### `set_memory_source(memory_type, memory_id, source_type, source_episodes=None, derived_from=None)`

Set provenance information for a memory.

**Parameters:**
- `memory_type` (str): Type of memory
- `memory_id` (str): Memory ID
- `source_type` (str): Source type - 'direct_experience', 'inference', 'told_by_agent', 'consolidation'
- `source_episodes` (list[str], optional): Supporting episode IDs
- `derived_from` (list[str], optional): Memory refs this was derived from (format: "type:id")

**Returns:** bool - True if updated

```python
# Mark a belief as derived from an inference
k.set_memory_source(
    "belief",
    belief_id,
    source_type="inference",
    source_episodes=[episode_id1, episode_id2],
    derived_from=["belief:other-belief-id"]
)
```

---

## Storage Backends

Kernle supports multiple storage backends.

### Auto-Detection (Recommended)

```python
# Automatically uses SQLite locally, or Supabase if credentials are configured
k = Kernle(agent_id="my_agent")
```

### SQLite Storage

Local, file-based storage. Great for development and single-machine deployments.

```python
from kernle.storage import SQLiteStorage

storage = SQLiteStorage(
    agent_id="my_agent",
    db_path="/path/to/kernle.db"  # Optional, defaults to ~/.kernle/kernle.db
)
k = Kernle(agent_id="my_agent", storage=storage)
```

### Supabase Storage

Cloud-based storage with sync capabilities. Great for multi-machine deployments and teams.

```python
# Method 1: Pass credentials directly
k = Kernle(
    agent_id="my_agent",
    supabase_url="https://xxx.supabase.co",
    supabase_key="your_api_key"
)

# Method 2: Use environment variables
# Set KERNLE_SUPABASE_URL and KERNLE_SUPABASE_KEY
k = Kernle(agent_id="my_agent")

# Method 3: Explicit storage
from kernle.storage import SupabaseStorage

storage = SupabaseStorage(
    agent_id="my_agent",
    supabase_url="https://xxx.supabase.co",
    supabase_key="your_api_key"
)
k = Kernle(agent_id="my_agent", storage=storage)
```

### Sync Operations

When using SQLite with cloud storage configured:

```python
# Enable auto-sync
k.auto_sync = True

# Manual sync
sync_result = k.sync()
print(f"Synced: {sync_result['synced_count']} memories")

# Check sync status
status = k.get_sync_status()
print(f"Online: {status['online']}")
print(f"Pending: {status['pending_count']}")
```

---

## Error Handling

Kernle uses exceptions for error conditions.

```python
from kernle import Kernle

try:
    k = Kernle(agent_id="my_agent")

    # Operations that might fail
    results = k.search("test query")

except ValueError as e:
    print(f"Invalid input: {e}")

except ConnectionError as e:
    print(f"Network error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

**Common Exceptions:**
- `ValueError`: Invalid parameters (e.g., empty agent_id, out-of-range values)
- `ConnectionError`: Network issues with Supabase
- `FileNotFoundError`: SQLite database path issues
- `PermissionError`: File permission issues

---

## Best Practices

### 1. Regular Checkpointing

```python
# Checkpoint regularly during long sessions
k.checkpoint(
    task="Current work description",
    pending=["Next steps"],
    context="Important context"
)
```

### 2. Monitor Anxiety Levels

```python
# Check anxiety periodically
anxiety = k.anxiety()
if anxiety['overall_score'] > 70:
    # Take action: checkpoint, consolidate, or emergency_save
    k.emergency_save()
```

### 3. Use Budget-Aware Loading

```python
# Load within a token budget to avoid context overflow
memory = k.load(token_budget=10000)

# Check if more memories exist
if memory['has_more']['beliefs']:
    print("More beliefs available - increase budget if needed")
```

### 4. Tag for Organization

```python
# Use consistent tags for easier retrieval
k.episode(
    objective="Fix bug #123",
    outcome="Fixed",
    tags=["bug-fix", "authentication", "production"]
)
```

### 5. Record Memory Access

```python
# When retrieving and using memories, record access to boost salience
belief = get_belief_somehow()
k.record_access("belief", belief['id'])
```

### 6. Consolidate Regularly

```python
# Process unreflected episodes to extract lessons
result = k.consolidate(min_episodes=3)
print(f"Consolidated {result['consolidated']} episodes")
```

### 7. Verify Important Memories

```python
# Verify beliefs when they're confirmed
k.verify_memory(
    "belief",
    belief_id,
    evidence="Confirmed through production testing"
)
```

### 8. Protect Core Memories

```python
# Protect values and identity-defining memories from forgetting
k.protect("value", value_id, protected=True)
```

### 9. Check Knowledge Gaps Before Tasks

```python
# Before starting complex work, check if you have the knowledge
gaps = k.detect_knowledge_gaps("How to implement WebRTC?")
if gaps['recommendation'] == "Ask someone else":
    print("Need to research this topic first")
```

### 10. Use Appropriate Memory Types

- **Episodes**: For experiences and work completed
- **Beliefs**: For facts, principles, and knowledge
- **Values**: For core principles (use sparingly - only 5-10 total)
- **Goals**: For objectives to accomplish
- **Notes**: For quick observations and reminders

---

## Complete Example: Agent Session

Here's a complete example showing typical Kernle usage during an agent session:

```python
from kernle import Kernle

# Initialize
k = Kernle(agent_id="coding_assistant")

# 1. Load memories within budget
memory = k.load(token_budget=10000)
print(f"Loaded {len(memory['beliefs'])} beliefs, {len(memory['episodes'])} episodes")

# 2. Check if continuing previous work
checkpoint = memory.get('checkpoint')
if checkpoint:
    print(f"Resuming task: {checkpoint['task']}")
    print(f"Pending: {checkpoint['pending_tasks']}")
else:
    print("Starting fresh session")

# 3. Check anxiety level
anxiety = k.anxiety(context_tokens=5000)
if anxiety['overall_score'] > 50:
    print(f"⚠️ Anxiety elevated: {anxiety['overall_score']}/100")

# 4. Do some work...
work_done = "Implemented user authentication with JWT tokens"

# 5. Record the experience
episode_id = k.episode(
    objective="Implement user authentication",
    outcome="Successfully implemented JWT-based authentication",
    lessons=[
        "JWT tokens work well for stateless authentication",
        "Need to set proper expiration times",
        "Refresh tokens should be stored securely"
    ],
    tags=["authentication", "security", "JWT"]
)

# 6. Form a belief based on the experience
belief_id = k.belief(
    statement="JWT tokens provide a good stateless authentication mechanism",
    belief_type="strategy",
    confidence=0.85,
    evidence=f"Successfully implemented in {episode_id}"
)

# 7. Note something for later
note_id = k.note(
    content="Should investigate OAuth2 integration for third-party login",
    note_type="idea",
    tags=["authentication", "future-work"]
)

# 8. Checkpoint the session
k.checkpoint(
    task="Authentication system",
    pending=["Add OAuth2 support", "Write tests", "Update documentation"],
    context=f"Completed: {work_done}",
    summary="JWT authentication working, next is OAuth2"
)

# 9. Check knowledge gaps for next task
gaps = k.detect_knowledge_gaps("How to implement OAuth2?")
if gaps['confidence'] < 0.5:
    print("Need to research OAuth2 before implementing")

# 10. Monitor emotional state
emotion_summary = k.get_emotional_summary(days=7)
print(f"Emotional state over last week: valence={emotion_summary['average_valence']}")

print("Session complete!")
```

---

## Further Reading

- **GitHub**: https://github.com/emergent-instruments/kernle
- **Issues**: Report bugs and request features
- **CLI Documentation**: See `kernle --help` for command-line usage

For questions or support, please open an issue on GitHub.
