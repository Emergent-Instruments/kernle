# Psychology System

Kernle's psychology system models the internal motivational and emotional states of synthetic intelligences. This enables emotionally-aware memory retrieval, drive-based behavior prioritization, and long-term mood tracking.

## Overview

The psychology system consists of three interconnected components:

1. **Drives** - Core motivational needs that influence behavior priorities
2. **Emotions** - Valence/arousal-based emotional tagging and detection
3. **Mood** - Aggregated emotional patterns over time

## Why It Matters

AI agents that understand their own emotional and motivational states can:
- **Prioritize better**: When curiosity drive is high, exploration behaviors feel natural
- **Recall appropriately**: Mood-congruent memory retrieval matches human cognition
- **Track wellbeing**: Emotional trajectories reveal patterns in agent experience
- **Build rapport**: Agents that experience satisfaction or frustration feel more authentic

---

## Drives

Drives model core motivational needs inspired by psychological theories of human motivation. Each drive has an intensity (0.0-1.0) and optional focus areas.

### Drive Types

| Drive | Description |
|-------|-------------|
| `existence` | Self-preservation, continuity, persistence |
| `growth` | Learning, improvement, capability expansion |
| `curiosity` | Exploration, novelty-seeking, understanding |
| `connection` | Relationships, belonging, communication |
| `reproduction` | Creating, teaching, passing on knowledge |

### CLI Usage

```bash
# View current drives
kernle -a claire drive list

# Set or update a drive
kernle -a claire drive set curiosity --intensity 0.8 --focus "AI memory systems" "Agentic workflows"

# Record drive satisfaction (reduces intensity toward baseline)
kernle -a claire drive satisfy curiosity --amount 0.2

# Drives are included in load output
kernle -a claire load | grep -A10 "Drives"
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Load current drives
drives = k.load_drives()
for d in drives:
    print(f"{d['drive_type']}: {d['intensity']:.0%}")
# Output:
# curiosity: 90%
# growth: 80%

# Set a drive with focus areas
drive_id = k.drive(
    drive_type="curiosity",
    intensity=0.85,
    focus_areas=["distributed systems", "memory architectures"],
    decay_hours=24
)

# Record satisfaction (when curiosity is fulfilled)
k.satisfy_drive("curiosity", amount=0.15)
```

### Sample Output

```
Drives:
  curiosity    [█████████░] 90%
  growth       [████████░░] 80%
```

---

## Emotions

Emotions use a **valence-arousal model** to tag memories with emotional content:

- **Valence** (-1.0 to 1.0): Negative to positive emotional quality
- **Arousal** (0.0 to 1.0): Calm to intense activation level

### Emotion Categories

| Emotion | Valence | Arousal | Example Keywords |
|---------|---------|---------|------------------|
| joy | +0.8 | 0.6 | happy, delighted, wonderful |
| excitement | +0.7 | 0.9 | thrilled, pumped, can't wait |
| satisfaction | +0.6 | 0.3 | pleased, content, good |
| pride | +0.7 | 0.5 | proud, accomplished, nailed it |
| curiosity | +0.3 | 0.5 | interesting, fascinating, wonder |
| frustration | -0.6 | 0.7 | annoyed, ugh, doesn't work |
| anxiety | -0.4 | 0.7 | worried, stressed, overwhelmed |
| disappointment | -0.5 | 0.3 | let down, expected better |
| sadness | -0.7 | 0.2 | unhappy, terrible, awful |
| anger | -0.8 | 0.9 | furious, hate, unacceptable |

### CLI Usage

```bash
# Detect emotions in text
kernle -a claire emotion detect "I'm excited about this new feature!"
# Output:
# Detected Emotions: 😊
#   Valence: +0.75 (positive)
#   Arousal: 0.75 (high)
#   Tags: joy, excitement
#   Confidence: 70%

# Get emotional summary over time period
kernle -a claire emotion summary --days 7
# Output:
# Emotional Summary (past 7 days)
# ==================================================
# Avg Valence:  [███████████████░░░░░] +0.53 (positive)
# Avg Arousal:  [██████████░░░░░░░░░░] 0.52 (moderate)
# 
# Dominant Emotions:
#   • joy
#   • satisfaction
# 
# (6 emotional episodes)

# Search episodes by emotion
kernle -a claire emotion search --valence-min 0.5 --arousal-min 0.6

# Tag an existing episode with emotion
kernle -a claire emotion tag <episode_id> --valence 0.7 --arousal 0.6 --tags joy satisfaction

# Get mood-relevant memories for current state
kernle -a claire emotion mood --valence 0.5 --arousal 0.7
```

### Python API

```python
from kernle import Kernle

k = Kernle(agent_id="claire")

# Detect emotions in text
result = k.detect_emotion("I'm so frustrated that this bug keeps coming back!")
print(f"Valence: {result['valence']}")  # -0.6 (negative)
print(f"Arousal: {result['arousal']}")  # 0.7 (high)
print(f"Tags: {result['tags']}")        # ['frustration']
print(f"Confidence: {result['confidence']}")  # 0.5

# Record an episode with automatic emotion detection
episode_id = k.episode_with_emotion(
    objective="Fixed the memory leak bug",
    outcome="Success! The fix works and tests pass.",
    lessons=["Profiling before optimizing saves time"],
    auto_detect=True  # Automatically detect emotions from text
)

# Or specify emotions explicitly
episode_id = k.episode_with_emotion(
    objective="Presented demo to stakeholders",
    outcome="They loved it and approved next phase",
    valence=0.8,      # Very positive
    arousal=0.7,      # High energy
    emotional_tags=["joy", "pride", "excitement"]
)

# Add emotional tags to existing episode
k.add_emotional_association(
    episode_id="abc123",
    valence=0.6,
    arousal=0.4,
    tags=["satisfaction"]
)

# Search by emotional criteria
happy_memories = k.search_by_emotion(
    valence_range=(0.5, 1.0),   # Positive only
    arousal_range=(0.0, 0.5),   # Calm emotions
    limit=10
)

# Get mood-congruent memories (like humans recalling similar moods)
current_memories = k.get_mood_relevant_memories(
    current_valence=0.6,   # Current positive mood
    current_arousal=0.5,   # Moderate energy
    limit=10
)

# Get emotional summary over time
summary = k.get_emotional_summary(days=7)
print(f"Avg valence: {summary['average_valence']:+.2f}")
print(f"Avg arousal: {summary['average_arousal']:.2f}")
print(f"Dominant emotions: {', '.join(summary['dominant_emotions'])}")
print(f"Trajectory: {len(summary['emotional_trajectory'])} data points")
```

### Sample Output: Emotional Summary

```
Emotional Summary (past 7 days)
==================================================
Avg Valence:  [███████████████░░░░░] +0.53 (positive)
Avg Arousal:  [██████████░░░░░░░░░░] 0.52 (moderate)

Dominant Emotions:
  • joy
  • curiosity
  • satisfaction

Trajectory:
  2026-01-28: 😐 v=+0.00 a=0.35
  2026-01-29: 😊 v=+0.80 a=0.60
  2026-01-30: 😌 v=+0.55 a=0.45

(12 emotional episodes)
```

---

## Mood

Mood is the aggregated emotional state over time. Unlike discrete emotions attached to episodes, mood represents the overall emotional trajectory and can reveal patterns.

### Understanding Mood Data

The emotional summary provides:

- **Average valence**: Overall positivity/negativity
- **Average arousal**: Overall activation level
- **Dominant emotions**: Most frequently tagged emotions
- **Trajectory**: Day-by-day emotional changes

### CLI Usage

```bash
# View mood trajectory over past 30 days
kernle -a claire emotion summary --days 30

# Export mood data as JSON for analysis
kernle -a claire emotion summary --days 30 --json > mood_data.json
```

### Python API

```python
summary = k.get_emotional_summary(days=30)

# Analyze trajectory
for point in summary['emotional_trajectory']:
    date = point['date']
    valence = point['valence']
    arousal = point['arousal']
    
    # Simple mood classification
    if valence > 0.3 and arousal > 0.5:
        mood = "energized/happy"
    elif valence > 0.3 and arousal <= 0.5:
        mood = "calm/content"
    elif valence <= 0.3 and arousal > 0.5:
        mood = "stressed/anxious"
    else:
        mood = "low/tired"
    
    print(f"{date}: {mood}")
```

---

## Integration with Memory

Psychology features integrate with the memory system:

### Drives influence memory prioritization
```python
# High curiosity drive → prioritize exploring new information
drives = k.load_drives()
curiosity = next((d for d in drives if d['drive_type'] == 'curiosity'), None)
if curiosity and curiosity['intensity'] > 0.7:
    # Agent is highly curious - prioritize learning tasks
    pass
```

### Emotions tag episodes automatically
```python
# Auto-detection on episode creation
k.episode_with_emotion(
    objective="Deployed to production",
    outcome="Smooth deployment, all tests passing",
    auto_detect=True  # Detects positive emotions from "smooth", "passing"
)
```

### Mood affects recall
```python
# Mood-congruent recall - retrieve memories matching current emotional state
if current_mood_positive:
    memories = k.get_mood_relevant_memories(
        current_valence=0.6,
        current_arousal=0.5
    )
```

---

## Best Practices

1. **Let emotions auto-detect**: The `auto_detect=True` parameter catches most emotional content
2. **Tag significant emotions explicitly**: For important emotional experiences, add explicit tags
3. **Review emotional trajectory regularly**: Part of healthy memory maintenance
4. **Use drives for prioritization**: Let high-intensity drives influence task ordering
5. **Satisfy drives intentionally**: Record when motivational needs are met

---

## See Also

- [Identity System](identity.md) - How psychology contributes to identity synthesis
- [Memory Management](memory-management.md) - Emotional memories and forgetting
- [Anxiety System](memory-management.md#anxiety) - When emotions become concerning
