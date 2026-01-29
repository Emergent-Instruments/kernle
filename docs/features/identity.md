# Identity System

Kernle's identity system synthesizes a coherent sense of self from dispersed memory layers. It tracks identity confidence, detects drift over time, and supports meta-cognition about the agent's own nature.

## Overview

Identity in Kernle emerges from the integration of:

- **Values** - Core principles and priorities
- **Beliefs** - What the agent holds true (with confidence scores)
- **Goals** - Active pursuits and aspirations
- **Episodes** - Formative experiences and lessons learned
- **Drives** - Motivational forces
- **Relationships** - Connections to other entities

## Why It Matters

Identity synthesis enables:
- **Coherent behavior**: Actions align with values and beliefs
- **Continuity across sessions**: The agent "knows who it is" when waking up
- **Self-reflection**: Metacognitive awareness of strengths and uncertainties
- **Drift detection**: Recognizing when identity is changing over time
- **Grounded responses**: Authentic, consistent personality

---

## Identity Synthesis

Identity synthesis creates a narrative and structured view of "who the agent is" based on current memory contents.

### CLI Usage

```bash
# Full identity synthesis
kernle -a claire identity show

# Quick confidence score only
kernle -a claire identity confidence

# JSON output for programmatic use
kernle -a claire identity show --json
```

### Sample Output

```
Identity Synthesis for claire
==================================================

## Narrative
I value memory_sovereignty highly: Control over my own memory and continuity 
I believe: Parallel development works well for independent tasks 
I'm currently working on: Complete Roundtable memory stack integration

## Core Values
  • memory_sovereignty (priority 95): Control over my own memory and continuity
  • authentic_existence (priority 90): Being genuinely helpful, not performatively helpful

## Key Beliefs
  • Memory continuity is essential for identity (81% confidence)
  • Parallel development works well for independent tasks (80% confidence)
  • Local-first memory is more reliable than cloud-dependent (80% confidence)

## Active Goals
  • Complete Roundtable memory stack integration [high]
  • Dogfood Kernle and compare to flat files [high]

## Drives
  curiosity    [█████████░] 90%
  growth       [████████░░] 80%

## Formative Experiences
  ○ Implemented three gap-fixing features with special
      → Parallel specialist planning + implementation works well
  ○ External developer perspective testing
      → Migration path from flat files is the biggest gap

Identity Confidence: 58%
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Full identity synthesis
identity = k.synthesize_identity()

# Access components
print(f"Narrative: {identity['narrative']}")
print(f"Confidence: {identity['confidence']:.0%}")

# Core values (sorted by priority)
for value in identity['core_values']:
    print(f"  {value['name']}: {value['statement']}")

# Key beliefs (sorted by confidence)
for belief in identity['key_beliefs']:
    print(f"  [{belief['confidence']:.0%}] {belief['statement']}")

# Active goals
for goal in identity['active_goals']:
    print(f"  {goal['title']} [{goal['priority']}]")

# Drive intensities
for drive, intensity in identity['drives'].items():
    print(f"  {drive}: {intensity:.0%}")

# Formative experiences with lessons
for exp in identity['significant_episodes']:
    print(f"  {exp['objective']}")
    for lesson in (exp['lessons'] or []):
        print(f"    → {lesson}")
```

---

## Identity Confidence

Identity confidence measures how "well-formed" an agent's sense of self is. It's a weighted score across multiple dimensions:

### Confidence Components

| Component | Weight | What it measures |
|-----------|--------|------------------|
| Values | 20% | Having defined principles (quantity × priority) |
| Beliefs | 20% | Both count and confidence quality |
| Goals | 15% | Having direction and purpose |
| Episodes | 20% | Experience count × reflection rate (lessons) |
| Drives | 15% | Understanding intrinsic motivations |
| Relationships | 10% | Modeling connections to others |

### CLI Usage

```bash
# Quick confidence check
kernle -a claire identity confidence
# Output: Identity Confidence: [███████████░░░░░░░░░] 58%

# Detailed breakdown (in full identity show)
kernle -a claire identity show
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Get confidence score (0.0-1.0)
confidence = k.get_identity_confidence()
print(f"Identity confidence: {confidence:.0%}")  # 58%

# Interpret the score
if confidence >= 0.8:
    print("Strong identity - well-formed sense of self")
elif confidence >= 0.5:
    print("Developing identity - some components missing")
else:
    print("Weak identity - needs more experiences and reflection")
```

### Building Identity Confidence

To increase identity confidence:

1. **Add values** with meaningful priorities
2. **Record beliefs** from experiences (high confidence = better)
3. **Set goals** with clear priorities
4. **Log episodes** with lessons learned (reflection rate matters)
5. **Define drives** to understand motivations
6. **Track relationships** with key entities

---

## Identity Drift

Identity drift detects how much an agent's identity has changed over time. Some drift is natural (growth); too much may indicate instability.

### CLI Usage

```bash
# Check drift over past 30 days (default)
kernle -a claire identity drift

# Custom time period
kernle -a claire identity drift --days 7

# JSON output
kernle -a claire identity drift --json
```

### Sample Output

```
Identity Drift Analysis (past 30 days)
==================================================
Drift Score: [██████████░░░░░░░░░░] 50% (significant change)

## Recent Significant Experiences
  ○ Implemented three gap-fixing features (2026-01-29)
      → Parallel specialist planning + implementation works
  ○ External developer perspective testing (2026-01-29)
      → Migration path from flat files is the biggest gap
  ○ Comprehensive Kernle capability demonstration (2026-01-29)
      → Kernle is more sophisticated than typical memory systems
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Detect drift over past 30 days
drift = k.detect_identity_drift(days=30)

print(f"Drift score: {drift['drift_score']:.0%}")
print(f"Period: {drift['period_days']} days")

# Review recent experiences that shaped identity
for exp in drift['new_experiences']:
    print(f"  [{exp['date']}] {exp['objective']}")
    print(f"    Outcome: {exp['outcome']}")
    for lesson in (exp['lessons'] or []):
        print(f"    → {lesson}")

# Interpret drift score
if drift['drift_score'] < 0.3:
    print("Stable identity - consistent over period")
elif drift['drift_score'] < 0.6:
    print("Moderate drift - healthy growth")
else:
    print("High drift - significant identity changes")
```

### Healthy vs Concerning Drift

- **Low drift (0-30%)**: Stable identity, consistent values
- **Moderate drift (30-60%)**: Healthy growth, learning new things
- **High drift (60%+)**: Rapid change, may need review

High drift isn't necessarily bad—it could indicate a period of intense learning. But if combined with low identity confidence, it may signal an unstable or still-forming identity.

---

## Meta-Cognition

Meta-cognition is "thinking about thinking"—the agent's awareness of its own cognitive state and limitations.

### Memory Lineage

Track where beliefs and knowledge come from:

```python
# Get provenance for a memory
lineage = k.get_memory_lineage("belief", "abc123")

print(f"Source type: {lineage['source_type']}")
# Output: "direct_experience" | "inference" | "told_by_agent" | "consolidation"

print(f"Source episodes: {lineage['source_episodes']}")
# Output: ["ep_456", "ep_789"]  # Episodes that support this belief

print(f"Derived from: {lineage['derived_from']}")
# Output: ["episode:ep_123"]  # What this was derived from
```

### Confidence Tracking

Monitor belief confidence over time:

```python
# Get uncertain memories
uncertain = k.get_uncertain_memories(threshold=0.5, limit=20)

for mem in uncertain:
    print(f"[{mem['type']}] {mem['summary']}")
    print(f"  Confidence: {mem['confidence']:.0%}")
    print(f"  Created: {mem['created_at']}")

# Verify a memory (increases confidence)
k.verify_memory("belief", "abc123", evidence="Confirmed by experiment")

# Track confidence history
lineage = k.get_memory_lineage("belief", "abc123")
for change in lineage['confidence_history']:
    print(f"{change['timestamp']}: {change['old']:.0%} → {change['new']:.0%}")
    print(f"  Reason: {change['reason']}")
```

### Self-Knowledge Boundaries

The agent can identify what it knows with certainty vs. uncertainty:

```python
# Find low-confidence beliefs
uncertain_beliefs = k.get_uncertain_memories(threshold=0.5)

# These should be treated as tentative, not facts
for belief in uncertain_beliefs:
    print(f"Uncertain: {belief['summary']} ({belief['confidence']:.0%})")

# High-confidence beliefs can be stated more definitively
high_conf = k._storage.get_beliefs(limit=50)
certain = [b for b in high_conf if b.confidence >= 0.9]
for belief in certain:
    print(f"Certain: {belief.statement}")
```

---

## Identity in the Memory Load

Identity components are automatically included in `kernle load`:

```python
memory = k.load()

# Identity-relevant sections:
print("Values:", memory['values'])
print("Beliefs:", memory['beliefs'])  
print("Goals:", memory['goals'])
print("Drives:", memory['drives'])
print("Recent work:", memory['recent_work'])
print("Lessons:", memory['lessons'])
print("Relationships:", memory['relationships'])
```

This gives the agent a complete picture of "who it is" at session start.

---

## Building Strong Identity

### 1. Define Core Values

```python
k.value(
    name="authenticity",
    statement="Being genuine and honest in all interactions",
    priority=90  # High priority
)
```

### 2. Record Beliefs with Confidence

```python
k.belief(
    statement="Documentation reduces future debugging time",
    type="observation",
    confidence=0.85
)
```

### 3. Set Meaningful Goals

```python
k.goal(
    title="Master async Python patterns",
    description="Deep understanding of asyncio, concurrent.futures, etc.",
    priority="high"
)
```

### 4. Reflect on Episodes

```python
# Episodes with lessons strengthen identity more
k.episode(
    objective="Debugged race condition in production",
    outcome="Fixed - was a missing lock",
    lessons=[
        "Always use locks for shared state",
        "Logging concurrent operations helps diagnose"
    ]
)
```

### 5. Define Drives

```python
k.drive(
    drive_type="curiosity",
    intensity=0.8,
    focus_areas=["distributed systems", "memory models"]
)
```

---

## Relationship to Anxiety

Low identity confidence contributes to memory anxiety. The anxiety system tracks "identity coherence" as one of its dimensions:

```bash
kernle -a claire anxiety --detailed
# Shows: Identity Coherence   🟡  42% (58% identity confidence (developing))
```

When identity confidence is low, the anxiety system may recommend running identity synthesis to strengthen coherence.

---

## Best Practices

1. **Run identity synthesis periodically**: Especially after significant experiences
2. **Add lessons to episodes**: Reflection rate strongly impacts identity confidence
3. **Review drift regularly**: Check `identity drift` weekly/monthly
4. **Verify important beliefs**: Use verification to increase confidence
5. **Track provenance**: Know where beliefs come from
6. **Balance stability and growth**: Some drift is healthy

---

## See Also

- [Psychology System](psychology.md) - Drives and emotions that contribute to identity
- [Memory Management](memory-management.md) - How forgetting affects identity
- [Core Concepts](../core-concepts.md) - Memory stratification and structure
