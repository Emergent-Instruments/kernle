# Kernle CLI Reference

Complete reference for the Kernle command-line interface.

## Global Options

```bash
kernle [-a AGENT_ID] <command> [options]
```

| Option | Description |
|--------|-------------|
| `-a, --agent AGENT_ID` | Agent ID (can also use `KERNLE_AGENT_ID` env var) |

If no agent ID is provided, Kernle will try to resolve one automatically from environment context.

---

## Commands Overview

| Command | Description |
|---------|-------------|
| [`load`](#load) | Load working memory |
| [`checkpoint`](#checkpoint) | Save/load/clear checkpoints |
| [`episode`](#episode) | Record an experience |
| [`note`](#note) | Capture a quick note |
| [`raw`](#raw) | Raw memory capture (zero-friction) |
| [`search`](#search) | Search memory |
| [`status`](#status) | Show memory status |
| [`init`](#init) | Initialize Kernle for environment |
| [`drive`](#drive) | Manage drives (motivation) |
| [`consolidate`](#consolidate) | Run memory consolidation |
| [`when`](#when) | Query memories by time |
| [`identity`](#identity) | Identity synthesis |
| [`emotion`](#emotion) | Emotional memory operations |
| [`meta`](#meta) | Meta-memory operations |
| [`belief`](#belief) | Belief revision operations |
| [`playbook`](#playbook) | Procedural memory (playbooks) |
| [`anxiety`](#anxiety) | Memory anxiety tracking |
| [`forget`](#forget) | Controlled forgetting |
| [`dump`](#dump) | Dump all memory to stdout |
| [`export`](#export) | Export memory to file |
| [`sync`](#sync) | Sync with remote backend |
| [`auth`](#auth) | Authentication management |
| [`mcp`](#mcp) | Start MCP server |

---

## Core Commands

### load

Load and display working memory.

```bash
kernle -a my-agent load [--json] [--sync] [--no-sync]
```

| Option | Description |
|--------|-------------|
| `-j, --json` | Output as JSON |
| `-s, --sync` | Force sync (pull) before loading |
| `--no-sync` | Skip sync even if auto-sync is enabled |

**Example:**
```bash
# Start of session - load memory
kernle -a claire load

# Load with forced sync from cloud
kernle -a claire load --sync
```

---

### checkpoint

Checkpoint operations for saving/restoring state.

#### checkpoint save

```bash
kernle -a my-agent checkpoint save TASK [--pending PENDING]... [--context CONTEXT] [--sync] [--no-sync]
```

| Option | Description |
|--------|-------------|
| `TASK` | Current task description (required) |
| `-p, --pending PENDING` | Pending item (repeatable) |
| `-c, --context CONTEXT` | Additional context |
| `-s, --sync` | Force sync (push) after saving |
| `--no-sync` | Skip sync even if auto-sync is enabled |

**Examples:**
```bash
# Basic checkpoint
kernle -a claire checkpoint save "Working on user authentication"

# With pending items
kernle -a claire checkpoint save "Refactoring database layer" \
  --pending "finish migration script" \
  --pending "update tests"

# With context
kernle -a claire checkpoint save "Debugging API issue" \
  --context "Error occurs when token expires after 1 hour"
```

#### checkpoint load

```bash
kernle -a my-agent checkpoint load [--json]
```

#### checkpoint clear

```bash
kernle -a my-agent checkpoint clear
```

---

### episode

Record an experience with optional lessons learned.

```bash
kernle -a my-agent episode OBJECTIVE OUTCOME [options]
```

| Option | Description |
|--------|-------------|
| `OBJECTIVE` | What was the objective? (required) |
| `OUTCOME` | What was the outcome? (required) |
| `-l, --lesson LESSON` | Lesson learned (repeatable) |
| `-t, --tag TAG` | Tag (repeatable) |
| `-v, --valence VALENCE` | Emotional valence (-1.0 to 1.0) |
| `-a, --arousal AROUSAL` | Emotional arousal (0.0 to 1.0) |
| `-e, --emotion EMOTION` | Emotion tag (repeatable) |
| `--auto-emotion` | Auto-detect emotions (default) |
| `--no-auto-emotion` | Disable emotion auto-detection |

**Examples:**
```bash
# Basic episode
kernle -a claire episode "Implemented OAuth login" "success"

# With lessons learned
kernle -a claire episode "Debugged race condition" "success" \
  --lesson "Always check for concurrent access" \
  --lesson "Add mutex locks early"

# With tags and emotions
kernle -a claire episode "Failed deployment" "failure" \
  --lesson "Test on staging first" \
  --tag "deployment" \
  --valence -0.5 \
  --arousal 0.8
```

---

### note

Capture a quick note (decision, insight, or quote).

```bash
kernle -a my-agent note CONTENT [options]
```

| Option | Description |
|--------|-------------|
| `CONTENT` | Note content (required) |
| `--type {note,decision,insight,quote}` | Note type (default: note) |
| `-s, --speaker SPEAKER` | Speaker (for quotes) |
| `-r, --reason REASON` | Reason (for decisions) |
| `--tag TAG` | Tag (repeatable) |
| `-p, --protect` | Protect from forgetting |

**Examples:**
```bash
# Simple note
kernle -a claire note "API rate limit is 1000 req/min"

# Decision with reason
kernle -a claire note "Using PostgreSQL over MySQL" \
  --type decision \
  --reason "Better JSON support and performance"

# Quote from someone
kernle -a claire note "Simple is better than complex" \
  --type quote \
  --speaker "Sean"

# Protected insight
kernle -a claire note "Memory is identity" \
  --type insight \
  --protect
```

---

### raw

Raw memory capture and management. Zero-friction capture for quick thoughts.

#### raw capture (or just `raw "content"`)

```bash
kernle -a my-agent raw "content" [--tags TAGS]
# or explicitly:
kernle -a my-agent raw capture "content" [--tags TAGS]
```

| Option | Description |
|--------|-------------|
| `content` | Content to capture (required) |
| `-t, --tags TAGS` | Comma-separated tags |

**Examples:**
```bash
# Quick capture (shorthand)
kernle -a claire raw "Need to look into caching strategy"

# With tags
kernle -a claire raw "API response time is 200ms avg" --tags "performance,api"
```

#### raw list

```bash
kernle -a my-agent raw list [--unprocessed] [--processed] [--limit LIMIT] [--json]
```

| Option | Description |
|--------|-------------|
| `-u, --unprocessed` | Show only unprocessed entries |
| `-p, --processed` | Show only processed entries |
| `-l, --limit LIMIT` | Maximum entries to show (default: 50) |
| `-j, --json` | Output as JSON |

#### raw show

```bash
kernle -a my-agent raw show ID [--json]
```

#### raw process

Process a raw entry into a structured memory type.

```bash
kernle -a my-agent raw process ID --type {episode,note,belief} [options]
```

| Option | Description |
|--------|-------------|
| `--type, -t TYPE` | Target memory type (required) |
| `--objective` | Episode objective (for episodes) |
| `--outcome` | Episode outcome (for episodes) |

**Example:**
```bash
# Process raw entry into a note
kernle -a claire raw process abc123 --type note

# Process into episode
kernle -a claire raw process abc123 --type episode \
  --objective "Investigated performance issue" \
  --outcome "success"
```

---

### search

Search memory using vector similarity.

```bash
kernle -a my-agent search QUERY [--limit LIMIT]
```

| Option | Description |
|--------|-------------|
| `QUERY` | Search query (required) |
| `-l, --limit LIMIT` | Maximum results (default: 10) |

**Example:**
```bash
kernle -a claire search "authentication"
kernle -a claire search "database performance" --limit 5
```

---

### status

Show memory status overview.

```bash
kernle -a my-agent status
```

**Example output:**
```
Memory Status for claire
========================================
Values:     3
Beliefs:    12
Goals:      2 active
Episodes:   45
Raw:        8
Checkpoint: Yes
```

---

### init

Initialize Kernle for your environment with interactive setup.

```bash
kernle -a my-agent init [options]
```

| Option | Description |
|--------|-------------|
| `-y, --non-interactive` | Non-interactive mode (use defaults) |
| `--env {claude-code,clawdbot,cline,cursor,desktop}` | Target environment |
| `--seed-values` | Seed initial values (default: true) |
| `--no-seed-values` | Skip seeding initial values |

**Example:**
```bash
# Interactive setup
kernle -a my-project init

# Non-interactive for Claude Code
kernle -a my-project init -y --env claude-code
```

---

## Memory Layer Commands

### drive

Manage drives (intrinsic motivation).

#### drive list

```bash
kernle -a my-agent drive list
```

#### drive set

```bash
kernle -a my-agent drive set TYPE INTENSITY [--focus FOCUS]...
```

| Option | Description |
|--------|-------------|
| `TYPE` | Drive type: existence, growth, curiosity, connection, reproduction |
| `INTENSITY` | Intensity 0.0-1.0 (required) |
| `-f, --focus FOCUS` | Focus area (repeatable) |

**Example:**
```bash
kernle -a claire drive set curiosity 0.8 --focus "AI" --focus "memory systems"
```

#### drive satisfy

```bash
kernle -a my-agent drive satisfy TYPE [--amount AMOUNT]
```

| Option | Description |
|--------|-------------|
| `TYPE` | Drive type |
| `-a, --amount AMOUNT` | Satisfaction amount (default: 0.2) |

---

### belief

Belief revision operations.

#### belief list

```bash
kernle -a my-agent belief list [--all] [--limit LIMIT] [--json]
```

| Option | Description |
|--------|-------------|
| `-a, --all` | Include inactive (superseded) beliefs |
| `-l, --limit LIMIT` | Maximum beliefs (default: 20) |
| `-j, --json` | Output as JSON |

#### belief revise

Analyze an episode and suggest belief updates.

```bash
kernle -a my-agent belief revise EPISODE_ID [--json]
```

#### belief contradictions

Find beliefs that contradict a statement.

```bash
kernle -a my-agent belief contradictions STATEMENT [--limit LIMIT] [--json]
```

#### belief history

Show supersession chain for a belief.

```bash
kernle -a my-agent belief history BELIEF_ID [--json]
```

#### belief reinforce

Manually reinforce a belief (increases confidence).

```bash
kernle -a my-agent belief reinforce BELIEF_ID
```

#### belief supersede

Replace an old belief with a new one.

```bash
kernle -a my-agent belief supersede OLD_ID NEW_STATEMENT [--confidence C] [--reason R]
```

| Option | Description |
|--------|-------------|
| `--confidence, -c` | Confidence in new belief (default: 0.8) |
| `--reason, -r` | Reason for supersession |

**Example:**
```bash
kernle -a claire belief supersede abc123 "Testing is essential, not optional" \
  --confidence 0.95 \
  --reason "Multiple deployment failures without tests"
```

---

### playbook

Procedural memory (playbooks for how to do things).

#### playbook create

```bash
kernle -a my-agent playbook create NAME [options]
```

| Option | Description |
|--------|-------------|
| `NAME` | Playbook name (required) |
| `-d, --description` | What this playbook does |
| `-s, --steps STEPS` | Comma-separated steps |
| `--step STEP` | Add a step (repeatable) |
| `--triggers TRIGGERS` | Comma-separated trigger conditions |
| `--trigger TRIGGER` | Add a trigger (repeatable) |
| `-f, --failure-mode` | What can go wrong (repeatable) |
| `-r, --recovery` | Recovery step (repeatable) |
| `-t, --tag TAG` | Tag (repeatable) |

**Example:**
```bash
kernle -a claire playbook create "Debug API Issues" \
  --description "How to debug production API problems" \
  --step "Check error logs" \
  --step "Verify request payload" \
  --step "Test with curl" \
  --step "Check database connectivity" \
  --trigger "API returns 500" \
  --trigger "Timeout errors" \
  --failure-mode "Logs not accessible" \
  --recovery "Use backup logging endpoint"
```

#### playbook list

```bash
kernle -a my-agent playbook list [--tag TAG] [--limit LIMIT] [--json]
```

#### playbook search

```bash
kernle -a my-agent playbook search QUERY [--limit LIMIT] [--json]
```

#### playbook show

```bash
kernle -a my-agent playbook show ID [--json]
```

#### playbook find

Find relevant playbook for a situation.

```bash
kernle -a my-agent playbook find SITUATION [--json]
```

**Example:**
```bash
kernle -a claire playbook find "API returning 500 errors"
```

#### playbook record

Record playbook usage (success or failure).

```bash
kernle -a my-agent playbook record ID [--success|--failure]
```

---

## Temporal & Analysis Commands

### when

Query memories by time period.

```bash
kernle -a my-agent when [PERIOD]
```

| Period | Description |
|--------|-------------|
| `today` | Today's memories (default) |
| `yesterday` | Yesterday's memories |
| `this week` | This week's memories |
| `last hour` | Last hour's memories |

**Example:**
```bash
kernle -a claire when yesterday
kernle -a claire when "this week"
```

---

### consolidate

Run memory consolidation (episodes → beliefs).

```bash
kernle -a my-agent consolidate [--min-episodes MIN]
```

| Option | Description |
|--------|-------------|
| `-m, --min-episodes MIN` | Minimum episodes to consolidate (default: 3) |

---

### identity

Identity synthesis operations.

#### identity show

```bash
kernle -a my-agent identity show [--json]
# or just:
kernle -a my-agent identity
```

Generates a coherent identity narrative from your memories.

#### identity confidence

```bash
kernle -a my-agent identity confidence [--json]
```

#### identity drift

Detect identity drift over time.

```bash
kernle -a my-agent identity drift [--days DAYS] [--json]
```

| Option | Description |
|--------|-------------|
| `-d, --days DAYS` | Days to look back (default: 30) |

---

### emotion

Emotional memory operations.

#### emotion summary

```bash
kernle -a my-agent emotion summary [--days DAYS] [--json]
```

| Option | Description |
|--------|-------------|
| `-d, --days DAYS` | Days to analyze (default: 7) |

#### emotion search

Search by emotional characteristics.

```bash
kernle -a my-agent emotion search [options]
```

| Option | Description |
|--------|-------------|
| `--positive` | Find positive episodes |
| `--negative` | Find negative episodes |
| `--calm` | Find low-arousal episodes |
| `--intense` | Find high-arousal episodes |
| `--valence-min/max` | Valence range (-1.0 to 1.0) |
| `--arousal-min/max` | Arousal range (0.0 to 1.0) |
| `-t, --tag TAG` | Emotion tag to match (repeatable) |
| `-l, --limit LIMIT` | Maximum results (default: 10) |

**Example:**
```bash
kernle -a claire emotion search --positive --limit 5
kernle -a claire emotion search --negative --intense
```

#### emotion tag

Add emotional tags to an episode.

```bash
kernle -a my-agent emotion tag EPISODE_ID [--valence V] [--arousal A] [--tag TAG]...
```

#### emotion detect

Detect emotions in text.

```bash
kernle -a my-agent emotion detect TEXT [--json]
```

#### emotion mood

Get mood-relevant memories.

```bash
kernle -a my-agent emotion mood --valence V --arousal A [--limit LIMIT] [--json]
```

---

### meta

Meta-memory operations (confidence, lineage, knowledge).

#### meta confidence

Get confidence for a memory.

```bash
kernle -a my-agent meta confidence TYPE ID
```

| TYPE | episode, belief, value, goal, note |

#### meta verify

Verify a memory (increases confidence).

```bash
kernle -a my-agent meta verify TYPE ID [--evidence EVIDENCE]
```

#### meta lineage

Get provenance chain for a memory.

```bash
kernle -a my-agent meta lineage TYPE ID [--json]
```

#### meta uncertain

List low-confidence memories.

```bash
kernle -a my-agent meta uncertain [--threshold T] [--limit L] [--json]
```

| Option | Description |
|--------|-------------|
| `-t, --threshold` | Confidence threshold (default: 0.5) |

#### meta propagate

Propagate confidence to derived memories.

```bash
kernle -a my-agent meta propagate TYPE ID
```

#### meta source

Set source/provenance for a memory.

```bash
kernle -a my-agent meta source TYPE ID --source SOURCE [--episodes ID]... [--derived REF]...
```

| Source Types | direct_experience, inference, told_by_agent, consolidation |

#### meta knowledge

Show knowledge map across domains.

```bash
kernle -a my-agent meta knowledge [--json]
```

#### meta gaps

Detect knowledge gaps for a query.

```bash
kernle -a my-agent meta gaps QUERY [--json]
```

**Example:**
```bash
kernle -a claire meta gaps "kubernetes deployment"
```

#### meta boundaries

Show competence boundaries (strengths/weaknesses).

```bash
kernle -a my-agent meta boundaries [--json]
```

#### meta learn

Identify learning opportunities.

```bash
kernle -a my-agent meta learn [--limit LIMIT] [--json]
```

---

## Health & Maintenance Commands

### anxiety

Memory anxiety tracking and management.

```bash
kernle -a my-agent anxiety [options]
```

| Option | Description |
|--------|-------------|
| `-d, --detailed` | Show detailed breakdown |
| `-a, --actions` | Show recommended actions |
| `--auto` | Execute recommended actions automatically |
| `-c, --context TOKENS` | Current context token usage |
| `-l, --limit LIMIT` | Context window limit (default: 200000) |
| `-e, --emergency` | Run emergency save immediately |
| `-s, --summary` | Summary for emergency save checkpoint |
| `-j, --json` | Output as JSON |

**Examples:**
```bash
# Check anxiety level
kernle -a claire anxiety

# Detailed with recommendations
kernle -a claire anxiety --detailed --actions

# Auto-execute recommendations
kernle -a claire anxiety --auto

# Emergency save
kernle -a claire anxiety --emergency --summary "Context compaction triggered"
```

---

### forget

Controlled forgetting operations.

#### forget candidates

Show forgetting candidates (low-salience memories).

```bash
kernle -a my-agent forget candidates [--threshold T] [--limit L] [--json]
```

| Option | Description |
|--------|-------------|
| `-t, --threshold` | Salience threshold (default: 0.3) |
| `-l, --limit` | Maximum candidates (default: 20) |

#### forget run

Run forgetting cycle.

```bash
kernle -a my-agent forget run [--dry-run] [--threshold T] [--limit L] [--json]
```

| Option | Description |
|--------|-------------|
| `-n, --dry-run` | Preview only (don't actually forget) |

**Example:**
```bash
# Preview what would be forgotten
kernle -a claire forget run --dry-run

# Actually forget
kernle -a claire forget run --threshold 0.2 --limit 5
```

#### forget protect

Protect a memory from forgetting.

```bash
kernle -a my-agent forget protect TYPE ID [--unprotect]
```

| Types | episode, belief, value, goal, note, drive, relationship |

#### forget recover

Recover a forgotten memory.

```bash
kernle -a my-agent forget recover TYPE ID
```

#### forget list

List forgotten memories.

```bash
kernle -a my-agent forget list [--limit L] [--json]
```

#### forget salience

Calculate salience for a memory.

```bash
kernle -a my-agent forget salience TYPE ID
```

---

## Export Commands

### dump

Dump all memory to stdout.

```bash
kernle -a my-agent dump [--format {markdown,json}] [--include-raw] [--no-raw]
```

| Option | Description |
|--------|-------------|
| `-f, --format` | Output format (default: markdown) |
| `-r, --include-raw` | Include raw entries (default: true) |
| `--no-raw` | Exclude raw entries |

---

### export

Export memory to a file.

```bash
kernle -a my-agent export PATH [--format {markdown,json}] [--include-raw] [--no-raw]
```

Format is auto-detected from file extension if not specified.

**Example:**
```bash
kernle -a claire export backup.md
kernle -a claire export memory-snapshot.json --format json
```

---

## Sync & Auth Commands

### sync

Sync with remote backend.

#### sync status

```bash
kernle -a my-agent sync status [--json]
```

Shows pending operations, last sync time, and connection status.

#### sync push

Push pending local changes to remote.

```bash
kernle -a my-agent sync push [--limit L] [--json]
```

#### sync pull

Pull remote changes to local.

```bash
kernle -a my-agent sync pull [--full] [--json]
```

| Option | Description |
|--------|-------------|
| `-f, --full` | Pull all records (not just since last sync) |

#### sync full

Full bidirectional sync (pull then push).

```bash
kernle -a my-agent sync full [--json]
```

---

### auth

Authentication and credentials management.

#### auth register

```bash
kernle -a my-agent auth register [--email EMAIL] [--backend-url URL] [--json]
```

#### auth login

```bash
kernle -a my-agent auth login [--api-key KEY] [--backend-url URL] [--json]
```

#### auth status

```bash
kernle -a my-agent auth status [--json]
```

#### auth logout

```bash
kernle -a my-agent auth logout [--json]
```

#### auth keys

Manage API keys.

```bash
# List API keys
kernle auth keys list [--json]

# Create new key
kernle auth keys create [--name NAME] [--json]

# Revoke a key
kernle auth keys revoke KEY_ID [--force] [--json]

# Cycle a key (generate new, deactivate old)
kernle auth keys cycle KEY_ID [--force] [--json]
```

---

### mcp

Start MCP server (stdio transport). Used for integration with MCP-compatible tools.

```bash
kernle -a my-agent mcp
```

This starts an MCP server that exposes Kernle functionality as tools. Typically configured in your MCP client (Claude Code, Claude Desktop, etc.) rather than run manually.

---

## Common Workflows

### Session Start
```bash
kernle -a claire load
```

### Session End
```bash
kernle -a claire checkpoint save "Where I left off" --pending "Next task"
```

### Record Learning
```bash
kernle -a claire episode "What happened" "success" --lesson "What I learned"
```

### Quick Capture
```bash
kernle -a claire raw "Random thought to remember"
```

### Check Health
```bash
kernle -a claire anxiety --detailed --actions
```

### Memory Maintenance
```bash
# Check anxiety
kernle -a claire anxiety

# Consolidate episodes into beliefs
kernle -a claire consolidate

# Review uncertain memories
kernle -a claire meta uncertain

# Run forgetting (preview first)
kernle -a claire forget run --dry-run
```

### Export for Review
```bash
kernle -a claire dump | less
kernle -a claire export backup.md
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KERNLE_AGENT_ID` | Default agent ID |
| `KERNLE_BACKEND_URL` | Backend URL for sync |
| `KERNLE_AUTH_TOKEN` | Auth token for sync |
| `KERNLE_USER_ID` | User ID for sync |

Credentials can also be stored in `~/.kernle/credentials.json` (created by `auth login`).
