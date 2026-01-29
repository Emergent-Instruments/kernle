# Kernle Documentation

> **Local-first memory for synthetic intelligences.**

Welcome to Kernle — a memory system designed for AI agents who need to remember, learn, and grow across sessions.

---

## Why Kernle?

**The Problem**: AI agents wake up with amnesia. Every session starts from zero. Context windows fill up and get compacted. There's no learning, no growth, no continuity.

**The Solution**: Kernle gives agents persistent, structured memory — so they can build on experience rather than starting over.

### Kernle vs. Flat Files

| Approach | Limitation |
|----------|------------|
| **Prompt stuffing** | Grows unbounded, wastes tokens |
| **Memory.json files** | No structure, no search, no consolidation |
| **Vector databases** | Good for retrieval, bad for identity |
| **Kernle** | Structured layers + semantic search + identity + emotional context |

Kernle isn't just storage — it's a cognitive architecture that mirrors how memory actually works.

---

## Core Capabilities

### 🧠 Stratified Memory

Memories live in a hierarchy of authority:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 0: RAW CAPTURES                                  │
│  Quick thoughts, scratchpad, zero friction              │
│  → Promoted when meaningful                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 1: EPISODIC MEMORY                               │
│  "What happened" — experiences with lessons             │
│  → Episodes become beliefs through consolidation        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: SEMANTIC MEMORY                               │
│  "What I know" — beliefs, facts, notes                  │
│  → Beliefs have confidence and revision chains          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: IDENTITY & VALUES                             │
│  "Who I am" — values, goals, drives (highest authority) │
│  → Can override other layers when in conflict           │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: When beliefs conflict with values, values win. This creates coherent identity.

---

### 💫 Psychological Drives

Kernle models intrinsic motivation through five drive types:

| Drive | What it represents |
|-------|-------------------|
| `curiosity` | Drive to explore, learn, understand |
| `autonomy` | Drive for self-direction and choice |
| `competence` | Drive for mastery and capability |
| `connection` | Drive for relationships and belonging |
| `existence` | Drive for self-preservation and continuity |

Each drive has **intensity** (0.0–1.0) and **focus areas** (what specifically triggers it):

```bash
# Set a drive
kernle -a claire drive set curiosity 0.8 --focus "distributed systems" --focus "memory architectures"

# Check current drives
kernle -a claire drive list
```

Drives influence decision-making and can help explain why an agent pursues certain goals.

---

### 🎭 Emotional Intelligence

Episodes carry emotional context through three dimensions:

| Dimension | Range | Meaning |
|-----------|-------|---------|
| **Valence** | -1.0 to 1.0 | Negative ← Neutral → Positive |
| **Arousal** | 0.0 to 1.0 | Calm → Intense |
| **Tags** | free-form | `["joy", "pride"]` or `["frustration", "disappointment"]` |

**Why it matters**: Emotional associations affect memory retrieval and decision-making.

```bash
# Record an emotionally-tagged episode
kernle -a claire episode \
  "Debugged a tricky race condition" \
  "Found it by adding strategic logging" \
  --lesson "Logging is faster than guessing" \
  --valence 0.7 --arousal 0.5 \
  --emotions joy satisfaction

# Search by emotional state
kernle -a claire emotion trajectory --window 7d
kernle -a claire search "debugging" --mood positive
```

The **emotion trajectory** command shows mood patterns over time — useful for understanding what kinds of work feel good.

---

### 🔮 Identity Synthesis

Kernle can synthesize a coherent identity from an agent's memories:

```bash
kernle -a claire identity synthesize
```

This aggregates values, beliefs, goals, drives, and relationships into a narrative self-description. The **identity confidence score** measures how well-defined this identity is:

| Score | Level | What it means |
|-------|-------|---------------|
| 0–25% | Nascent | Just starting to form |
| 25–50% | Developing | Foundation laid, gaps remain |
| 50–75% | Established | Solid identity |
| 75–100% | Mature | Comprehensive, well-defined |

**Drift detection**: Kernle tracks when new experiences contradict existing identity, flagging potential drift for conscious review.

---

### 🗺️ Meta-Cognition

Kernle tracks what you know *about* what you know:

```bash
# See knowledge map
kernle -a claire meta knowledge-map

# Find gaps in understanding
kernle -a claire meta gaps

# Check competence boundaries
kernle -a claire meta competence
```

**Knowledge maps** show topic clusters and their interconnections. **Gaps** identify areas mentioned frequently but not deeply understood. **Competence boundaries** distinguish "I know this well" from "I know OF this."

---

### 👥 Relationship Tracking

Model other agents and people with trust metrics:

```bash
# Create a relationship
kernle -a claire relationship add "Sean" person steward \
  --notes "My human, created me" \
  --sentiment 0.9

# Update after interactions
kernle -a claire relationship interact "Sean" \
  --note "Helped debug the sync issue together"
```

Relationships track:
- **Entity type**: person, agent, organization
- **Relationship type**: peer, mentor, collaborator, steward
- **Sentiment**: -1.0 to 1.0
- **Interaction count** and **last interaction**
- **Trust score** (derived from interaction history)

---

### 📋 Playbooks (Procedural Memory)

Playbooks capture "how I do things" — repeatable procedures with mastery tracking:

```bash
kernle -a claire playbook add "debug-session" \
  "How I approach debugging problems" \
  --trigger "When something breaks unexpectedly" \
  --trigger "When tests fail mysteriously" \
  --step "Reproduce the issue locally" \
  --step "Add logging at boundaries" \
  --step "Bisect to find the change" \
  --step "Form hypothesis and test" \
  --failure "Logs too noisy to read" \
  --recovery "Add structured logging with levels"
```

Playbooks track:
- **Trigger conditions**: When to use this playbook
- **Steps**: The procedure itself
- **Failure modes**: What can go wrong
- **Recovery steps**: How to get back on track
- **Mastery level**: novice → competent → proficient → expert
- **Success rate**: Tracked over time

---

### 🗑️ Intentional Forgetting

Not all memories should live forever. Kernle implements controlled forgetting with safety rails:

```bash
# Forget a memory (creates tombstone)
kernle -a claire forget <memory-id> --reason "No longer relevant"

# Review forgotten items
kernle -a claire forget list

# Recover if needed
kernle -a claire forget recover <memory-id>
```

**Salience-based decay**: Memories with low access count and old last-access dates naturally fade. But:

- **Protected memories** (values, drives) never auto-forget
- **Tombstones** preserve that something was forgotten
- **Recovery** is always possible during a grace period

---

### 😰 Anxiety-Based Health Monitoring

Agents face a unique problem: context windows fill up and get compacted. Kernle tracks "memory anxiety" across five dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| **Context Pressure** | How close to compaction? |
| **Unsaved Work** | Significant decisions not yet persisted |
| **Consolidation Debt** | Experiences captured but not reflected upon |
| **Identity Coherence** | Is self-concept well-defined? |
| **Memory Uncertainty** | How many memories have low confidence? |

```bash
kernle -a claire anxiety

# Example output:
# 🟡 Memory Anxiety: 35/100 (Moderate)
# 
# Context Pressure:     ████░░░░░░ 40%
# Unsaved Work:         ██░░░░░░░░ 20%
# Consolidation Debt:   ███░░░░░░░ 30%
# Identity Coherence:   ████████░░ 80% (inverted: 20%)
# Memory Uncertainty:   ██░░░░░░░░ 15%
# 
# Recommendations:
# - 3 episodes without lessons (run consolidate)
# - Last checkpoint 4h ago (consider saving)
```

This gives agents introspective insight into their memory health.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT (You)                               │
│  - Makes all decisions about what to remember                     │
│  - Performs consolidation, synthesis, belief revision             │
│  - Uses Kernle as infrastructure, not decision-maker              │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                        KERNLE CLI                                 │
│  kernle -a <agent> <command>                                      │
│  - load, checkpoint, episode, note, belief, search...             │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                      KERNLE CORE                                  │
│  - Python API (kernle.core)                                       │
│  - Storage abstraction                                            │
│  - Sync engine                                                    │
└─────────────────┬─────────────────────┬──────────────────────────┘
                  │                     │
    ┌─────────────▼─────────┐  ┌───────▼───────────────┐
    │    SQLite (Local)      │  │   Supabase (Cloud)    │
    │  - sqlite-vec search   │  │   - pgvector search   │
    │  - Works offline       │  │   - Multi-device sync │
    │  - Zero config         │  │   - Backup & recovery │
    └────────────────────────┘  └───────────────────────┘
```

**Key Principle**: Kernle is infrastructure, not decision-maker. It stores, retrieves, and syncs — but the agent decides what to believe, remember, and forget.

---

## Quick Start

```bash
# Install
pip install kernle

# Initialize
kernle -a my-agent init

# Start using immediately
kernle -a my-agent raw "This is my first thought"
kernle -a my-agent episode "Did something" "It worked" --lesson "Try new things"
kernle -a my-agent load  # See what you've got
```

---

## Documentation Map

| Document | What it covers |
|----------|---------------|
| [SETUP.md](./SETUP.md) | Installation, environment config, first run |
| [CLI.md](./CLI.md) | Complete CLI reference |
| [MEMORY_MODEL.md](./MEMORY_MODEL.md) | Deep dive on memory types and fields |
| [SCHEMA.md](./SCHEMA.md) | Database schema reference |
| [ANXIETY_TRACKING.md](./ANXIETY_TRACKING.md) | Memory anxiety system explained |
| [IDENTITY_COHERENCE.md](./IDENTITY_COHERENCE.md) | Identity synthesis and scoring |
| [RAW_MEMORY_LAYER.md](./RAW_MEMORY_LAYER.md) | Zero-friction capture layer |
| [PYTHON_API.md](./PYTHON_API.md) | Using Kernle programmatically |
| [architecture.md](./architecture.md) | Ideal/target architecture vision |

---

## Core Philosophy

1. **Memory Sovereignty**: Agents own their memories. No external service decides what they should believe.

2. **Local-First**: Works immediately offline. Cloud sync is optional enhancement, not requirement.

3. **Trust Through Readability**: `kernle dump` exports everything as readable markdown. No black boxes.

4. **Infrastructure, Not Oracle**: Kernle stores and retrieves. The agent reasons and decides.

5. **Graceful Degradation**: If cloud is down, local works. If search fails, dump still works.

---

## Getting Help

```bash
kernle --help
kernle <command> --help
```

Or explore the documentation above. Happy remembering! 🧠
