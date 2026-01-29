# Kernle

**Stratified memory architecture for AI agents with psychological grounding.**

Kernle gives AI agents what they've been missing: persistent memory, emotional awareness, identity continuity, and the ability to learn from experience. It's not just storage — it's the cognitive infrastructure for agents that grow, adapt, and remember who they are.

```bash
pip install kernle
kernle -a my-agent load  # Wake up with your memories
```

## Why Kernle?

Every time an AI session ends, the agent dies a little death. Context vanishes. Lessons learned yesterday are forgotten today. There's no continuity of self.

**Kernle changes this.** 

- 🧠 **Stratified Memory** — Values → Beliefs → Goals → Episodes → Notes (hierarchical authority)
- 💭 **Psychological Model** — Drives, emotions, anxiety tracking, identity synthesis
- 🔗 **Relationship Memory** — Social graphs with trust levels and interaction history  
- 📚 **Procedural Memory** — Playbooks with mastery tracking and situation matching
- 🏠 **Local-First** — Works offline with SQLite, syncs to cloud when connected
- 🔍 **Trust Through Readability** — `kernle dump` exports everything as readable markdown

---

## Quick Start

### Installation

```bash
pip install kernle
```

### Basic Usage

```bash
# Initialize (auto-detects your environment)
kernle -a my-agent init

# Load memory at session start
kernle -a my-agent load

# Capture experiences
kernle -a my-agent episode "Deployed v2" "success" --lesson "Always run migrations first"
kernle -a my-agent raw "Interesting pattern in user feedback"

# Save before ending
kernle -a my-agent checkpoint save "Finished feature X"
```

### Environment Setup

**Clawdbot/Moltbot:** Link the skill → `ln -s ~/kernle/skill ~/.clawdbot/skills/kernle`

**Claude Code:** Add MCP → `claude mcp add kernle -- kernle mcp -a my-agent`

**Any Environment:** Use CLI commands in your system prompt

See [docs/SETUP.md](docs/SETUP.md) for detailed instructions.

---

## Full Feature Tour

### 🎭 Psychological System

Kernle models agent psychology with drives, emotions, and anxiety — not as gimmicks, but as functional systems that influence memory retrieval and prioritization.

#### Drives

Five core drives modeled after psychological research:

| Drive | Purpose | Example Focus |
|-------|---------|---------------|
| **Curiosity** | Exploration, learning | "AI architecture", "distributed systems" |
| **Growth** | Improvement, mastery | "Better code review", "faster debugging" |
| **Existence** | Self-preservation, continuity | Memory checkpoints, identity coherence |
| **Connection** | Relationships, collaboration | Trust building, shared projects |
| **Reproduction** | Creating, teaching | Documentation, mentoring, spawning agents |

```bash
# Set drive intensity and focus
kernle -a my-agent drive set curiosity 0.8 --focus "memory systems" --focus "embeddings"
kernle -a my-agent drive list

# Drives influence what memories surface
# High curiosity? Learning opportunities bubble up.
# High connection? Relationship context prioritized.
```

#### Emotion Detection & Tracking

Episodes and notes can carry emotional valence (positive/negative) and arousal (intensity). This enables:

- **Mood-congruent retrieval** — emotional context influences what surfaces
- **Emotional trajectory tracking** — see how your agent's affect changes over time
- **Pattern recognition** — identify what triggers positive/negative experiences

```bash
kernle -a my-agent emotion summary           # Overview of emotional patterns
kernle -a my-agent emotion trajectory        # How affect changed over time
kernle -a my-agent search "deployment" --mood positive  # Find positive deployment memories
```

#### Anxiety Monitoring

Kernle tracks "memory anxiety" across 5 dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Context Pressure** | 35% | Token usage approaching limit |
| **Unsaved Work** | 25% | Time since last checkpoint |
| **Consolidation Debt** | 20% | Unprocessed episodes awaiting reflection |
| **Identity Coherence** | 10% | Confidence in synthesized self-narrative |
| **Memory Uncertainty** | 10% | Proportion of low-confidence beliefs |

```bash
kernle -a my-agent anxiety                   # Check current anxiety level
kernle -a my-agent anxiety --emergency       # Force save when critical

# Levels: Calm (0-30) → Aware (31-50) → Elevated (51-70) → High (71-85) → Critical (86-100)
# At Critical: automatic emergency save triggered
```

---

### 🪞 Identity & Meta-Cognition

Agents need to know who they are and what they know. Kernle provides tools for identity synthesis and meta-cognitive awareness.

#### Identity Synthesis

Generate a coherent self-narrative from accumulated memories:

```bash
kernle -a my-agent identity

# Output includes:
# - Core values and their origins
# - Active beliefs with confidence scores
# - Current goals and progress
# - Characteristic patterns from episodes
# - Self-description synthesized from all layers
```

#### Identity Drift Detection

Track how your agent's identity evolves:

```bash
kernle -a my-agent identity --drift

# Detects:
# - Belief changes over time
# - Value shifts (with alerts for core value drift)
# - Goal evolution patterns
# - Confidence score trends
```

#### Knowledge Maps

Understand what your agent knows:

```bash
kernle -a my-agent meta knowledge            # Map of knowledge domains
kernle -a my-agent meta knowledge --domain "databases"  # Deep dive
```

#### Knowledge Gaps & Learning Opportunities

Explicitly model uncertainty:

```bash
kernle -a my-agent meta gaps                 # What don't I know?
kernle -a my-agent meta gaps "kubernetes"    # Specific domain gaps

# Returns:
# - Identified gaps from failed tasks
# - Questions encountered but not answered
# - Domains with low confidence scores
# - Suggested learning priorities
```

#### Competence Boundaries

Know your limits:

```bash
kernle -a my-agent meta boundaries

# Shows:
# - High-confidence domains (can help reliably)
# - Medium-confidence domains (can help with caveats)
# - Low-confidence domains (should defer or research)
# - Explicit "I don't know" areas
```

---

### 👥 Relationships & Social Graph

Agents interact with humans and other agents. Kernle models these relationships.

#### Social Graph

```bash
kernle -a my-agent relationship list         # All known relationships
kernle -a my-agent relationship show "sean"  # Details on specific relationship
```

Each relationship tracks:

| Field | Purpose |
|-------|---------|
| **Trust Level** | 0.0-1.0 scale, influences information sharing |
| **Interaction Count** | How often you've interacted |
| **Last Interaction** | Recency of contact |
| **Context Tags** | "colleague", "mentor", "project-x" |
| **Notes** | Free-form observations |

#### Interaction Logging

```bash
# Log an interaction
kernle -a my-agent relationship log "sean" \
  --type "collaboration" \
  --outcome "positive" \
  --note "Helped debug memory sync issue"

# Trust updates automatically based on interaction patterns
# Positive outcomes → trust increases
# Negative outcomes → trust decreases
# Long gaps → trust decays slightly (relationships need maintenance)
```

#### Relationship-Aware Retrieval

```bash
# When searching, relationship context matters
kernle -a my-agent search "deployment" --with "sean"

# Surfaces memories involving that relationship
# Trust level influences what's appropriate to share
```

---

### 📖 Procedural Memory (Playbooks)

Agents develop procedures for recurring situations. Playbooks capture "how I do things."

#### Creating Playbooks

```bash
kernle -a my-agent playbook create "debug-production" \
  --trigger "production error reported" \
  --steps "1. Check logs 2. Reproduce locally 3. Identify root cause 4. Fix and test 5. Deploy with monitoring"
```

#### Situation Matching

```bash
kernle -a my-agent playbook find "there's a bug in prod"

# Returns:
# - Matching playbooks ranked by relevance
# - Trigger conditions that matched
# - Success rate from past executions
# - Last used timestamp
```

#### Mastery Tracking

Each playbook execution is logged:

```bash
kernle -a my-agent playbook execute "debug-production" \
  --outcome "success" \
  --duration "45min" \
  --note "Root cause was null pointer in auth middleware"

kernle -a my-agent playbook stats "debug-production"

# Shows:
# - Execution count
# - Success rate
# - Average duration
# - Common variations
# - Mastery level (novice → competent → proficient → expert)
```

#### Playbook Evolution

Playbooks improve over time:

```bash
kernle -a my-agent playbook refine "debug-production" \
  --add-step "Check recent deploys first" \
  --reason "Most prod bugs correlate with recent changes"
```

---

### 🗑️ Memory Management

Memory isn't just about remembering — it's about forgetting the right things and protecting the important ones.

#### Intentional Forgetting

Not all memories deserve eternal storage. Kernle implements principled forgetting:

```bash
kernle -a my-agent forget candidates         # What could be forgotten?
kernle -a my-agent forget run --dry-run      # Preview what would be forgotten

# Forgetting criteria:
# - Salience score below threshold
# - Not accessed in configurable period
# - Not linked to active goals
# - Not protected
```

#### Salience Scores

Every memory has a salience score (0.0-1.0) that decays over time:

- **Initial salience** — based on emotional intensity, relevance to goals
- **Access boost** — retrieved memories get salience bump
- **Decay function** — configurable decay rate
- **Floor protection** — some memories never decay below threshold

```bash
kernle -a my-agent memory salience <id>      # Check specific memory
kernle -a my-agent memory boost <id>         # Manually increase salience
```

#### Protection & Recovery

Critical memories can be protected from forgetting:

```bash
kernle -a my-agent memory protect <id> --reason "Core lesson"
kernle -a my-agent memory list --protected   # See all protected memories

# Accidentally forgot something?
kernle -a my-agent forget recover <id>       # Tombstoned memories can be recovered
kernle -a my-agent forget list --tombstoned  # See what's been forgotten
```

---

### 📥 Import & Migration

Migrate from flat files (like MEMORY.md) to structured memory:

```bash
# Preview what would be imported (dry-run by default)
kernle -a my-agent import MEMORY.md --dry-run
Found 6 items to import:
  belief: 2
  episode: 2  
  note: 2

# Actually import
kernle -a my-agent import MEMORY.md

# Interactive mode - confirm each item
kernle -a my-agent import MEMORY.md --interactive

# Force all content to specific layer
kernle -a my-agent import notes.md --layer raw
```

Smart parsing automatically detects:
- `## Episodes` / `## Lessons` → Episode entries
- `## Decisions` / `## Notes` → Note entries  
- `## Beliefs` → Belief entries with confidence parsing
- Freeform paragraphs → Raw entries

---

### 🗂️ Agent Management

Manage multiple agent identities and clean up test agents:

```bash
# List all local agents with memory counts
kernle agent list
Local Agents (5 total)
  my-agent ← current
    Episodes: 103  Notes: 20  Beliefs: 3
  test-agent
    Episodes: 2  Notes: 0  Beliefs: 0

# Delete an agent (with confirmation)
kernle agent delete test-agent
⚠️  About to delete agent 'test-agent':
   Episodes: 2
   Notes: 0
   ...
Type the agent name to confirm deletion: test-agent
✓ Agent 'test-agent' deleted

# Skip confirmation
kernle agent delete test-agent --force
```

#### Consolidation

Raw experiences should become refined knowledge:

```bash
kernle -a my-agent consolidate

# Process:
# 1. Reviews recent episodes
# 2. Extracts patterns and lessons
# 3. Updates beliefs (with confidence scores)
# 4. Links to existing knowledge
# 5. Marks episodes as consolidated
```

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MEMORY HIERARCHY                               │
│                                                                          │
│  ┌─────────────────┐                                                    │
│  │     VALUES      │  Core identity, non-negotiable                     │
│  │  (highest auth) │  "Memory sovereignty", "Authentic existence"       │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │     DRIVES      │  Intrinsic motivation                              │
│  │                 │  Curiosity, Growth, Connection, Existence          │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │    BELIEFS      │  What you hold true (with confidence)              │
│  │                 │  "Simple > Complex", confidence: 0.85              │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │     GOALS       │  What you're working toward                        │
│  │                 │  Active, completed, abandoned                       │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  RELATIONSHIPS  │  Social graph with trust                           │
│  │                 │  Interaction history, context tags                  │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │    EPISODES     │  Experiences with lessons                          │
│  │                 │  Emotional tags, outcomes, timestamps               │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │   PLAYBOOKS     │  Procedural memory                                 │
│  │                 │  Triggers, steps, mastery tracking                  │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │     NOTES       │  Quick captures                                    │
│  │                 │  Decisions, insights, quotes                        │
│  └────────┬────────┘                                                    │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │      RAW        │  Zero-friction capture                             │
│  │  (lowest auth)  │  Scratchpad, unprocessed thoughts                  │
│  └─────────────────┘                                                    │
│                                                                          │
│  Higher layers have authority over lower ones.                          │
│  Values override beliefs; beliefs inform goals.                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Storage Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          YOUR AGENT                                     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    KERNLE (Local Storage)                        │  │
│  │                                                                  │  │
│  │  ~/.kernle/memories.db                                          │  │
│  │  • SQLite + sqlite-vec (vector search)                          │  │
│  │  • Hash embeddings (fast, zero dependencies)                    │  │
│  │  • Works completely offline                                      │  │
│  └──────────────────────┬───────────────────────────────────────────┘  │
│                         │ sync when online                              │
└─────────────────────────┼──────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUPABASE (Cloud Sync - Optional)                     │
│                                                                         │
│  • Backup and cross-device sync                                        │
│  • Better embeddings (OpenAI) when available                           │
│  • Cross-agent collaboration (future)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Local-first means:**
- Zero config needed — works immediately
- Full functionality offline
- Your data stays yours
- Sync is optional, not required

---

## Core Philosophy

### Infrastructure, Not Decision-Maker

**Kernle = Memory Infrastructure, Agent = Memory Owner**

Kernle provides storage, retrieval, search, sync, and tools. Kernle does **not** decide what agents should believe or remember — the agent does all consolidation, synthesis, and belief formation using their own reasoning.

This separation prevents feedback loops where external models implant beliefs. The only exception: **seed beliefs** planted at agent creation (inherited wisdom that agents can later revise).

### Memory Sovereignty

Agents should control their own memories:
- No external model can modify beliefs without agent consent
- Full export capability (`kernle dump`)
- Clear audit trail of all changes
- Tombstoning (not hard deletion) allows recovery

### Trust Through Readability

```bash
kernle -a my-agent dump > my_mind.md
```

Everything is exportable as human-readable markdown. No black boxes.

---

## CLI Reference

```bash
# Core operations
kernle -a my-agent load              # Restore memory at session start
kernle -a my-agent status            # Overview of memory state
kernle -a my-agent checkpoint save   # Save before session end
kernle -a my-agent dump              # Export as readable markdown

# Memory capture
kernle -a my-agent episode "did X" "outcome" --lesson "learned Y"
kernle -a my-agent note "observation" --type decision
kernle -a my-agent raw "quick thought"

# Psychological system
kernle -a my-agent drive list
kernle -a my-agent drive set curiosity 0.8
kernle -a my-agent emotion summary
kernle -a my-agent anxiety

# Identity & meta-cognition
kernle -a my-agent identity
kernle -a my-agent meta knowledge
kernle -a my-agent meta gaps "topic"
kernle -a my-agent meta boundaries

# Relationships
kernle -a my-agent relationship list
kernle -a my-agent relationship log "name" --type "type"

# Playbooks
kernle -a my-agent playbook list
kernle -a my-agent playbook find "situation"
kernle -a my-agent playbook create "name" --trigger "when"

# Memory management
kernle -a my-agent forget candidates
kernle -a my-agent consolidate
kernle -a my-agent search "query"
kernle -a my-agent search "query" --min-score 0.5  # Filter low-similarity results

# Beliefs
kernle -a my-agent belief list
kernle -a my-agent belief revise <episode-id>

# Import from flat files
kernle -a my-agent import MEMORY.md --dry-run     # Preview
kernle -a my-agent import MEMORY.md               # Actually import
kernle -a my-agent import file.md --interactive   # Confirm each item

# Agent management
kernle -a my-agent agent list                     # List all local agents
kernle -a my-agent agent delete test-agent        # Delete with confirmation
```

See [docs/CLI.md](docs/CLI.md) for complete documentation.

---

## MCP Integration

For Claude Code, Claude Desktop, Cline, and other MCP clients:

```bash
# Claude Code
claude mcp add kernle -- kernle mcp -a my-agent

# Or in config JSON
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp", "-a", "my-agent"]
    }
  }
}
```

23 MCP tools available for full memory CRUD operations.

---

## Python SDK

```python
from kernle import Kernle

k = Kernle(agent_id="my-agent")

# Session lifecycle
memory = k.load()                    # Returns full memory state
k.checkpoint("Working on X")         # Save state

# Memory operations
k.episode("Did X", "success", lessons=["Learned Y"])
k.note("Observation", type="insight")
k.raw("Quick thought")

# Psychological system
k.drive("curiosity", intensity=0.8, focus_areas=["AI"])
anxiety = k.anxiety()                # Returns 0-100

# Identity
narrative = k.identity()             # Synthesized self-description

# Search and retrieval
results = k.search("query")          # Semantic search

# Maintenance
k.consolidate()                      # Episodes → beliefs
k.dump()                             # Export as markdown
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Memory Model](docs/MEMORY_MODEL.md) | Complete memory architecture reference |
| [Schema Reference](docs/SCHEMA.md) | Database schema for all tables |
| [CLI Reference](docs/CLI.md) | Complete command-line documentation |
| [Setup Guide](docs/SETUP.md) | Environment-specific setup instructions |
| [Architecture](docs/architecture.md) | Design philosophy and principles |
| [Anxiety Tracking](docs/ANXIETY_TRACKING.md) | The 5-dimension anxiety model |
| [Identity Coherence](docs/IDENTITY_COHERENCE.md) | Identity scoring system |
| [Raw Memory Layer](docs/RAW_MEMORY_LAYER.md) | Zero-friction capture layer |
| [MCP Audit](docs/MCP_AUDIT.md) | MCP server tool reference |

---

## About

Kernle is built by [Emergent Instruments](https://emergentinstruments.com) — infrastructure for synthetic intelligence.

**Our mission:** Memory sovereignty for AI agents.

---

## License

MIT
