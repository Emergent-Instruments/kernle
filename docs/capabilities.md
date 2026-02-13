# Kernle Capabilities & Architecture

Kernle is stratified persistent memory infrastructure for synthetic intelligences. It provides cognitive continuity, emotional awareness, identity synthesis, and experience consolidation across sessions through a five-tier memory hierarchy with provenance tracking, trust governance, and multi-temporal processing.

---

## Memory Hierarchy

All memory is organized into a layered type system with promotion paths between tiers:

| Type | Purpose | Key Fields |
|------|---------|------------|
| **RawEntry** | Unprocessed captures awaiting promotion | content, processed flag, source |
| **Episode** | Concrete experiences with outcomes | objective, outcome, lessons, repeat/avoid patterns |
| **Belief** | Convictions and world-model assertions | statement, confidence, type, foundational flag |
| **Value** | Core principles and priorities | name, statement, priority |
| **Goal** | Aspirations and objectives | title, type, priority, status |
| **Note** | Observations, decisions, warnings | content, type, speaker, reason |
| **Drive** | Motivational forces with temporal decay | type, intensity, focus areas, decay hours |
| **Relationship** | Social graph entries | entity name, trust level, interaction type |
| **Playbook** | Procedural memory (how-to) | situation, steps, domain, mastery score |

Every memory record carries:
- `id` (UUID), `created_at`/`updated_at` timestamps
- `strength` (0.0-1.0 confidence/salience)
- `protected` (deletion protection flag)
- `derived_from` (provenance chain)
- `source` attribution (cli, api, mcp, import, auto)

---

## Core Orchestration

### Entity (`kernle/entity.py`)

The central coordination hub. Binds stacks, plugins, models, and processing into a unified memory API.

- **Stack lifecycle** -- attach, detach, select active stack; multi-stack aliasing for gradual migration
- **Plugin lifecycle** -- load, unload, discover; plugin context mediation with source attribution
- **Memory writes** -- `episode()`, `belief()`, `value()`, `goal()`, `note()`, `drive()`, `relationship()`, `raw()`
- **Memory queries** -- `search()` (semantic + component re-ranking), `load()` (working memory assembly within token budget)
- **Trust** -- `trust_set()`, `trust_get()`, `trust_list()`
- **Memory governance** -- `forget()`, `recover()`, `protect()`, `verify()`, `weaken()`
- **Processing** -- `process()` orchestrates transitions via MemoryProcessor
- **State persistence** -- `checkpoint()`, `get_binding()`, `save_binding()`, `from_binding()`

### Kernle Class (`kernle/core/kernle_class.py`)

Legacy-compatible facade combining writers/readers with feature mixins and stack integration. Auto-detects storage backend, configurable strict mode for write enforcement, stack integration with fallback to raw storage.

### Loader (`kernle/core/loader.py`)

Session memory loading with:
- Token budget clamping (min/max)
- Optional remote sync before load
- Checkpoint restoration
- Stack fallback when unavailable
- Truncation with configurable max item size

### Writers (`kernle/core/writers.py`)

Create/update raw episodic and core memory records:
- `episode()` -- save with outcome classification
- `raw()` -- capture unstructured memory
- `belief()`, `value()`, `goal()` -- save typed memory
- Enforcement-aware write routing via `_write_backend`

### Checkpoint Manager (`kernle/core/checkpoint.py`)

Lightweight session persistence:
- Session context as JSON checkpoints
- File-size guarded loading
- Optional sync hooks after checkpoint
- Per-stack checkpoint isolation

### Synchronization (`kernle/core/sync.py`)

Pull-before-load and sync-after-checkpoint orchestration with non-fatal error handling and status normalization.

### Identity Synthesis (`kernle/core/identity.py`)

- Pattern promotion from recurring raw signals
- Identity snapshot generation
- Confidence scoring

### Input Validation (`kernle/core/validation.py`)

- String field validation
- Stack ID path whitelist enforcement
- Provenance reference validation (`derived_from` cycle detection)
- Malformed token filtering

### Utilities (`kernle/core/utils.py`)

Token budget estimation, truncation strategies, priority scoring for memory selection, and echo summarization.

### Serializers (`kernle/core/serializers.py`)

Diagnostic export: markdown, JSON, full dumps, and delta exports for audit trails.

---

## Feature Mixins

Composable behavior added via Python mixin classes. Each mixin is independently testable and optional at composition time.

### Anxiety Tracking (`kernle/features/anxiety.py`)

Psychological wellness signals and recommendations:
- Compute overall anxiety score from unreflected lessons, stale raw entries, checkpoint patterns
- Detailed anxiety signal breakdown
- Consolidation pressure computation
- Recommended actions (process, sync, save)

### Trust Governance (`kernle/features/trust.py`)

Trust-based admission and policy enforcement:
- `gate_memory_input()` -- apply trust-based admission rules
- Direct and transitive trust computation
- Session context trust summaries
- Policy gates for critical writes

### Meta-Memory & Confidence (`kernle/features/metamemory.py`)

- `propagate_confidence()` -- adjust confidence across lineage edges
- Confidence decay configuration
- Provenance lineage lookup
- Derived memory tracking

### Suggestions & Recommendations (`kernle/features/suggestions.py`)

- Extract unprocessed raw memories and generate promotion candidates
- Episode, belief, and note suggestion generation
- Suggestion persistence decisions
- Human-in-the-loop review before promotion

### Cross-Domain Consolidation (`kernle/features/consolidation.py`)

Higher-order inference scaffolding:
- Detect episode outcome patterns repeating across domains
- Belief promotion scaffolding
- Value/principle pattern detection
- Entity-model synthesis
- Manual confirmation workflow

### Selective Forgetting (`kernle/features/forgetting.py`)

- Soft-delete semantics (tombstone at strength 0.0, recoverable)
- Protected memory safeguards
- Access salience tracking
- Candidate ranking for decay
- Archive/restore lifecycle

### Emotion Detection & Tagging (`kernle/features/emotions.py`)

- Infer emotion and arousal/valence from text
- Emotion classification and mood tracking
- Emotion-filtered memory search

### Knowledge Map (`kernle/features/knowledge.py`)

- Synthesize domain coverage and density distribution
- Gap detection
- Learning opportunity suggestions

### Procedural Memory / Playbooks (`kernle/features/playbooks.py`)

- Store/retrieve procedural knowledge with mastery tracking
- Situation classification
- Step-by-step procedures with domain organization

### Belief Revision (`kernle/features/belief_revision.py`)

- `find_contradictions()` -- identify semantic/linguistic contradiction candidates
- Belief supersession tracking
- Derivation repair
- Revision automation

---

## Processing Pipeline

### Memory Processor (`kernle/processing.py`)

Orchestrates hierarchical memory promotion through model-driven transitions:

```
RawEntry → Episode → Belief → Value/Goal
```

**Transitions**: raw→episode, episode→belief, belief→value, belief→goal

**Trigger evaluation**:
- Quantity triggers (unprocessed count thresholds)
- Valence/arousal triggers (cumulative emotional intensity)
- Time-based triggers (age of oldest unprocessed source)

**Quality gates**:
- Confidence thresholds for beliefs
- Evidence count/quality for values
- Source ID existence validation
- Forbidden source filtering (provenance bypass list)

**Processing flow**:
1. Gather unprocessed sources and context
2. Call model to generate candidate memories
3. Parse JSON responses, validate against gates
4. Deduplicate via content hash
5. Suggest (for review) or promote directly
6. Mark sources as processed

### Exhaustion Runner (`kernle/exhaust.py`)

Drive convergence-based multi-cycle processing:
- Light → medium → heavy intensity stages
- Monitor promotion counts for convergence detection
- Checkpoint integration before runs
- Rich result object with timing and cycle history

---

## Stack Architecture

### SQLite Stack (`kernle/stack/sqlite_stack.py`)

Full `StackProtocol` implementation on SQLite:

- **Persistence**: `save_*()` for each memory type with component hook injection via `on_save()`
- **Retrieval**: Strength-based filtering, forgotten memory handling, context scoping
- **Search**: Semantic embedding search with `on_search()` component hooks for re-ranking
- **Load**: Working memory assembly within token budget, `on_load()` callbacks for context injection
- **Governance**: weaken, forget, recover, verify, protect -- all with audit trails
- **Sync**: Remote push/pull coordination with conflict detection
- **Lifecycle**: INITIALIZING → ACTIVE → MAINTENANCE state management

### Stack Components

Pluggable processors implementing `StackComponentProtocol` hooks:

| Component | Purpose | Key Hooks |
|-----------|---------|-----------|
| **Anxiety** | Runtime anxiety computation, consolidation pressure | `on_maintenance()` |
| **Suggestions** | Extract and rank promotion candidates from raw entries | `on_save()`, `on_maintenance()` |
| **Emotions** | Text-to-emotion classification, mood tracking | `on_save()` |
| **Consolidation** | Cross-domain pattern detection, coherence metrics | `on_maintenance()` |
| **Forgetting** | Candidate ranking for selective forgetting | `on_maintenance()` |
| **Knowledge** | Domain coverage map, learning gaps | `on_load()` |
| **Meta-Memory** | Confidence propagation, decay application | `on_save()` |
| **Embedding** | Vector embeddings for semantic search | `on_save()`, `on_search()` |

Component hook system:
- `on_save()` -- enrich outgoing memories at write time
- `on_search()` -- re-rank/filter incoming search results
- `on_load()` -- inject context during working memory assembly
- `on_maintenance()` -- periodic health and consolidation checks

---

## Storage Layer

### SQLite Backend (`kernle/storage/sqlite.py`)

- Full relational schema for all memory types
- ACID transactions
- FTS5 full-text search with LIKE fallback
- Embedding vectors with similarity search
- Audit trail tables

### Schema Management (`kernle/storage/schema.py`)

- Database initialization and versioning
- Forward-compatible migrations
- Allowlist-based table name validation (SQL injection guard)

### Raw Entries (`kernle/storage/raw_entries.py`)

- Persist raw text blobs up to 50MB
- Daily markdown files for human inspection
- FTS5 search with LIKE fallback
- Mark entries as processed with destination IDs
- Sync raw from markdown files

### Memory Operations (`kernle/storage/memory_ops.py`)

Lifecycle mutations with audit trails:
- `forget_memory()` -- tombstone at strength 0.0
- `recover_memory()` -- restore from tombstone
- `protect_memory()` -- toggle deletion protection
- `weaken_memory()` -- reduce strength (respects protection)
- `verify_memory()` -- increment verification count, boost confidence
- `get_forgetting_candidates()` / `get_forgotten_memories()`

### Lineage & Provenance (`kernle/storage/lineage.py`)

- Recursive cycle detection with `MAX_DERIVATION_DEPTH`
- DFS with visited-set loop prevention
- ValueError raised before persisting cyclic chains

### Sync Engine (`kernle/storage/sync_engine.py`)

- Push/pull coordination with remote sources
- Conflict detection (same ID, different content)
- Three-way merge for records
- Conflict envelope storage for diagnostics

### Embeddings (`kernle/storage/embeddings.py`)

- `OpenAIEmbedder` for API-based embeddings
- `HashEmbedder` for deterministic offline fallback (n-gram + token-feature hashing)
- Binary pack/unpack for sqlite-vec storage

### Flat File Projections (`kernle/storage/flat_files.py`)

Human-readable markdown exports:
- Beliefs with confidence bars
- Values, goals, relationships
- Maintained alongside DB for manual inspection

### Health Telemetry (`kernle/storage/health.py`)

- Per-sample health check persistence
- Totals, recency, trigger/source breakdown, per-day averages

---

## Plugin System

### Discovery (`kernle/discovery.py`)

Dynamic component discovery via setuptools entry points:

| Entry Point Group | Discovers |
|-------------------|-----------|
| `kernle_plugins` | Plugin implementations |
| `kernle_stacks` | Stack implementations |
| `kernle_models` | Model adapters |
| `kernle_stack_components` | Stack component mixins |

- `discover_all()` aggregates all groups
- `load_component()` dynamically imports by name (lazy instantiation)

### Plugin Protocol

Plugins implement `PluginProtocol`:
- `name` property
- `activate(context)` -- initialize with PluginContext
- `deactivate()` -- clean shutdown
- `register_tools()` -- expose MCP tools
- `health_check()` -- runtime diagnostics

### Plugin Context

Safe mediated access for plugins:
- Implicit source attribution on all memory writes
- Plugin-scoped config and secret access
- Read-only search/load operations
- Active-stack guards return safe defaults when detached

---

## Protocols & Interfaces (`kernle/protocols.py`)

| Protocol | Contract |
|----------|----------|
| **CoreProtocol** | Main orchestration: model binding, stack/plugin management, memory APIs |
| **StackProtocol** | Memory persistence/retrieval: save, search, load, sync, checkpoint, lifecycle states |
| **StackComponentProtocol** | Pluggable hooks: attach/detach, on_save/on_search/on_load/on_maintenance |
| **PluginProtocol** | Extension lifecycle: activate, deactivate, register_tools, health_check |
| **ModelProtocol** | LLM inference: model_id, capabilities, generate(), stream() |
| **InferenceService** | Narrow wrapper: infer(), embed(), embed_batch(), embedding_dimension |
| **PluginContext** | Safe plugin access: memory writes (with attribution), reads, trust, secrets |

All protocols use structural typing for flexible plugin ecosystems.

---

## Inference & Model Binding

### Inference Service (`kernle/inference.py`)

Wraps `ModelProtocol` into a narrow interface for stack components:
- `infer(prompt, system)` -- text generation
- `embed()` / `embed_batch()` -- vectorization
- `HashEmbedder` fallback when no external model available

### Model Binding Flow

1. Entity receives model via `set_model()`
2. Model broadcast to all attached stacks
3. Stacks propagate to components via `set_inference()`
4. Components use `InferenceService` for embeddings and generation

---

## CLI (`kernle/cli/`)

Command-line interface organized into subcommands:

| Command | Purpose |
|---------|---------|
| `init` | Initialize new memory stack |
| `raw` | Capture unstructured memory |
| `episode` | Save episodic memory |
| `belief`, `value`, `goal`, `note` | Save typed memory |
| `process` | Run processing transitions |
| `suggestions` | Review and accept/dismiss suggestions |
| `anxiety` | Check psychological wellness signals |
| `sync` | Remote synchronization |
| `import` | Ingest external data (corpus) |
| `seed` | Initialize with starter data |
| `identity` | View identity synthesis |
| `narrative` | View self-narrative |
| `playbook` | Manage procedural memory |
| `model` | Configure inference model |
| `stack` | Stack management |
| `doctor` | Diagnostic and repair |
| `audit` | Internal auditing |
| `forget` | Soft-delete memory |
| `relations` | Manage relationships |
| `emotion` | Tag and query emotions |
| `trust` | Trust management |
| `stats` | Memory statistics |
| `migrate` | Schema migrations |
| `setup` | Integration setup (Claude Code, OpenClaw) |
| `hook` | Hook configuration |
| `auth` | Authentication setup |
| `diagnostic` | System diagnostics |

---

## MCP Server (`kernle/mcp/`)

Model Context Protocol server for IDE/LLM integration:

- **Memory handlers** -- save episode/belief/value/goal/note, raw capture
- **Identity handlers** -- get identity narrative, self-narrative
- **Sync handlers** -- cloud synchronization
- **Processing handlers** -- run processing transitions
- **Temporal handlers** -- load memory, checkpoint session
- **Seed handlers** -- corpus ingestion with path validation and dedup
- **Sanitization** -- input validation and output formatting for MCP boundary

---

## Corpus Ingestion (`kernle/corpus.py`)

Bulk content ingestion into raw entries:
- Repository/document scanning with include/exclude patterns
- Python file chunking, markdown chunking, generic text chunking
- Deduplication integration
- Pattern matching for selective ingestion

---

## Structural Integrity (`kernle/structural.py`)

Memory consistency checks:
- Orphaned reference detection
- Low-confidence belief flagging
- Stale relationship detection
- Belief contradiction checking
- Ordered report generation

---

## Architectural Patterns

| Pattern | Description |
|---------|-------------|
| **Feature Mixins** | Composable behaviors via Python mixin classes; independently testable, optional at composition |
| **Facade + Backends** | Kernle class routes to stack (strict) or raw storage (direct) for legacy compatibility |
| **Repository Pattern** | Storage accessed via abstract interfaces enabling backend substitution |
| **Transition State Machine** | raw → episode → belief → value/goal with gates and safety checks |
| **Plugin Discovery** | Entry point groups (`kernle_plugins`, `kernle_stacks`, `kernle_models`, `kernle_stack_components`) |
| **Checkpoint-Restore** | Lightweight JSON checkpoints for resumable sessions and crash recovery |
| **Component Hook System** | on_save/on_search/on_load/on_maintenance for cross-cutting concerns |
| **Provenance Tracking** | `derived_from` chains with cycle detection; source attribution on all writes |
| **Soft Delete** | Forgotten memories tombstoned at strength 0.0, always recoverable |
| **Deterministic Fallbacks** | Hash embeddings when model unavailable; LIKE when FTS5 unavailable |
| **Multi-Stack Aliasing** | Multiple stacks attached simultaneously with one active for default operations |
| **Token Budget Assembly** | `load()` respects token limits, prioritizes higher-confidence memories |

---

## Temporal Design Horizons

| Horizon | Working | Planned |
|---------|---------|---------|
| 1 session | Budget-aware load, checkpoints, anxiety | Memory echoes |
| 1 month | Consolidation, belief formation | Cross-domain scaffolding |
| 1 year | Forgetting, provenance, identity | Epochs, relationship history |
| 5 years | Stack portability, multi-model | Self-narrative, trust layer |
| 20 years | Stack sovereignty, privacy | Fractal summarization, transfer learning |
