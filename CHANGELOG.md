# Changelog

All notable changes to Kernle will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-02-07

### Added

#### Provenance Wiring
- **CLI provenance**: `derived_from` parameter on relationship add/update, all import commands, value/goal/relationship core methods
- **MCP provenance**: `derived_from`, `source`, `source_type` on all 6 memory creation tools with validation
- **`export-full` command**: Complete agent context export as markdown or JSON with configurable limits (up to 10,000 entries)

#### Strength Cascade
- **Reverse-lineage lookup**: Find all memories derived from a given source
- **Forget/weaken cascade**: Cascading strength changes through `derived_from` lineage with configurable flags
- **Verify boost to sources**: Verifying a memory boosts the strength of its source memories
- **`get_ungrounded_memories()`**: Find beliefs/values whose source memories have all been forgotten or weakened

#### Component Inference
- **EmotionalTagging inference**: Calls bound model for emotion detection when available, falls back to keyword matching
- **Consolidation inference**: Calls bound model for pattern extraction when available, falls back to keyword matching
- **Belief.processed field**: Tracks whether a belief has been considered for promotion

#### Memory Processing
- **`memory_process` MCP tool**: Trigger memory processing from MCP clients
- **`memory_process_status` MCP tool**: Check processing queue status
- **`kernle process run` CLI command**: Run memory processing from the command line
- **`kernle process status` CLI command**: View processing queue status

### Changed

- **`enforce_provenance` defaults to `True`**: All memory creation requires valid `derived_from` references (except raw entries and seed writes)
- **`strict` mode defaults to `True`**: Source type and provenance requirements enforced by default
- **Strength-tier gating**: `load()` excludes weak (< 0.5) and dormant (< 0.2) memories by default; `include_weak=True` to include them
- **`on_save` mutation persistence**: Component mutations during `on_save` hooks are now persisted to storage

### Fixed

- **`get_ungrounded_memories`**: Fixed handling of raw `type:id` reference format
- **Export limits**: Raised from 1000 to 10,000 entries for `export-full`
- **RawEntry fields**: Use canonical field names in export
- **Format detection**: Fixed JSON/markdown format detection in export
- **`belief_to_value` idempotency**: Duplicate promotion attempts now handled gracefully

## [0.9.0] - 2026-02-05

### Added

#### Provenance Enforcement
- **Hierarchy rules**: `derived_from` required on all memory types except raw entries; hierarchy enforced (Raw → Episode/Note → Belief → Value, etc.)
- **`ProvenanceError` exception**: Raised when hierarchy rules are violated
- **Reference validation**: Each `derived_from` reference checked for `type:id` format, allowed source types, and existence in stack

#### Stack Lifecycle
- **Stack states**: `INITIALIZING` → `ACTIVE` → `MAINTENANCE` state machine
- **Seed writes**: Only permitted during `INITIALIZING` state with `source_type="seed"`; no provenance required
- **State transitions**: Stack moves to `ACTIVE` on first `on_attach()` call

#### Continuous Memory Strength
- **`strength` field** (0.0–1.0) on all memory types, replacing binary `is_forgotten`
- **5 strength tiers**: Strong (0.8–1.0), Fading (0.5–0.8), Weak (0.2–0.5), Dormant (0.0–0.2), Forgotten (0.0)
- **Strength-based queries**: `load()` and `search()` filter by strength tier
- **Removed**: `is_forgotten`, `forgotten_at`, `forgotten_reason` fields (replaced by strength + audit trail)

#### Controlled Access & Audit Trail
- **Named operations**: `weaken(memory_id, amount, reason)`, `forget(memory_id, reason)`, `recover(memory_id)`, `verify(memory_id)`, `protect(memory_id)` on Entity
- **`memory_audit` table**: All mutation operations logged with actor, operation type, and details
- **Audit queries**: Retrieve audit history for any memory

#### Settings
- **`stack_settings` table**: Persistent per-stack configuration
- **Live settings sync**: Settings changes propagate to stack components in real-time

### Changed

- **Entity method signatures**: `derived_from` parameter now required (not Optional) on `belief()`, `goal()`, `value()`, `drive()`, `relationship()`, `note()`, `episode()`
- **`raw()` is the entry point**: Only raw entries can be created without citing sources
- **`source_entity` populated**: Now set correctly for Goal, Drive, Relationship saves
- **Plugin registration**: Plugins receive live settings updates via stack component sync

### Fixed

- **MCP `validate_tool_input`**: No longer rejects plugin tool names with dots
- **`source_entity` missing**: Goal, Drive, Relationship now properly set `source_entity`

### Breaking Changes

- `is_forgotten` field removed from all memory types (use `strength == 0.0`)
- `forgotten_at` and `forgotten_reason` removed (moved to audit trail)
- `derived_from` required on Entity methods (was optional)
- Direct `save_*()` calls bypass provenance only if stack is in `INITIALIZING` state

## [0.7.0] - 2026-02-02

### Added

#### Plugin Extraction
- **chainbased package**: Extracted as independent `PluginProtocol` implementation (wallet, jobs, skills, escrow); 280 lines, 45 tests
- **fatline package**: New communications plugin with AgentRegistry + Ed25519 crypto identity; 1664 lines, 70 tests
- **Entry point registration**: Both plugins register as `kernle.plugins` entry points

### Changed

- **Comms module removed** from kernle core (moved to fatline)
- **fatline uses own SQLite DB** in plugin data dir — no private kernle API usage

## [0.6.0] - 2026-02-01

### Added

#### Model Providers
- **AnthropicModel**: `ModelProtocol` implementation for Claude (238 lines)
- **OllamaModel**: `ModelProtocol` implementation for local models (211 lines)
- **Entry point registration**: Both registered as `kernle.models` entry points

#### Plugin Wiring
- **Plugin CLI registration**: `Entity.load_plugin()` registers tools + optional CLI commands
- **MCP tool dispatch**: Plugin tools namespaced as `{plugin_name}.{tool_name}` in MCP server
- **`Entity.unload_plugin()`**: Clean plugin removal

### Fixed

- **MCP `validate_tool_input`**: Fixed rejection of plugin tool names containing dots

## [0.5.0] - 2026-01-31

### Added

#### Stack Components
- **`StackComponentProtocol`**: New protocol for composable stack extensions with `on_save`, `on_search`, `on_load` hooks
- **InferenceService**: Wraps `ModelProtocol` with `HashEmbedder` fallback for environments without a model
- **EmbeddingComponent**: Vector embedding via `StackComponentProtocol` (155 lines, 65 tests)
- **7 feature mixin components**: Forgetting, Consolidation, Emotions, Anxiety, Suggestions, MetaMemory, Knowledge (1253 lines, 146 tests)
- **Component discovery**: Auto-loading of 8 default components via `kernle.stack_components` entry points
- **Configurable composition**: `components=[]` for bare stack, explicit list for custom, default for all 8

## [0.4.0] - 2026-01-31

### Added

#### Core/Stack Split
- **Entity** (`CoreProtocol` implementation): Coordinator/bus, provenance routing, InferenceService creation (833 lines, 76 tests)
- **SQLiteStack** (`StackProtocol` implementation): Wraps SQLiteStorage + feature mixins + component registry (1022 lines, 61 tests)
- **Kernle compat layer**: Lazy `.entity` and `.stack` properties for backward compatibility (17 tests)
- **Contract tests**: 163 tests for `StackProtocol` + `CoreProtocol` conformance
- **CLI migration**: Composition info in `kernle status`, plugin discovery

### Changed

- **Architecture split**: Single monolithic class split into Entity (coordinator) + Stack (memory container)
- **Composability**: Entity and Stack can evolve independently; multiple stack implementations possible

## [0.3.0] - 2026-01-30

### Added

#### Protocol System
- **5 protocols**: `CoreProtocol`, `StackProtocol`, `PluginProtocol`, `ModelProtocol`, `StackComponentProtocol`
- **Shared types** (`types.py`): All memory dataclasses centralized
- **Discovery system** (`discovery.py`): Entry point discovery via `importlib.metadata`
- **`InferenceService` protocol**: Abstraction for model inference with embedding support

### Changed

- **Architecture**: Moved from monolithic design to protocol-based composition
- **Entry points**: `kernle.plugins`, `kernle.stacks`, `kernle.models`, `kernle.stack_components` groups defined

## [0.2.1] - 2026-01-30

### Fixed

- **Migration ordering**: Fixed raw blob migration where data migration check ran before `ALTER TABLE` statements, causing "no such column: blob" warning on upgrade from 0.1.x
- **pyenv compatibility**: Documented `pyenv rehash` requirement for console script updates

### Added

- **Upgrade guide**: New troubleshooting documentation for upgrading Kernle and handling migrations

## [0.2.0] - 2026-01-30

### Added

#### Context Management
- **Budget-aware loading**: `kernle load --budget 6000` limits memory loading to prevent context overflow
- **Budget metadata**: `_meta` field in load() returns `budget_used`, `budget_total`, `excluded_count`
- **Priority-based selection**: Memories loaded by priority (values > beliefs > goals > episodes > notes)
- **Truncation control**: `--no-truncate` flag to disable content truncation

#### Memory Stack
- **Automatic access tracking**: `load()` and `search()` now track access for salience-based forgetting
- **Batch access recording**: `record_access_batch()` for efficient bulk updates
- **Array field merging**: Sync now merges arrays (tags, lessons, etc.) instead of last-write-wins
- **Array size limits**: MAX_SYNC_ARRAY_SIZE (500) prevents resource exhaustion

#### Raw Layer
- **Blob storage**: Raw entries refactored to use blob storage
- **FTS5 search**: Full-text search on raw entries via `kernle raw search`

#### Playbooks
- **Postgres support**: Full CRUD operations in Postgres backend
- **Sync support**: Playbooks now sync between local and cloud
- **Mastery tracking**: Track usage and success rate

#### Forgetting
- **Postgres implementation**: `forget_memory()`, `recover_memory()`, `protect_memory()`
- **Salience calculation**: `get_forgetting_candidates()` in Postgres
- **Access tracking**: `record_access()` in Postgres

#### CLI Commands
- `kernle load --budget N` - Budget-aware memory loading
- `kernle raw search QUERY` - FTS5 search on raw entries
- `kernle forget salience TYPE ID` - Check memory salience score
- `kernle suggestions list/approve/reject` - Manage memory suggestions
- `kernle agent list/delete` - Manage local agents
- `kernle doctor` - Validate boot sequence compliance
- `kernle stats health-checks` - View health check compliance

#### MCP Tools
- 33 MCP tools (up from ~12 in 0.1.0)
- Added: `memory_sync`, `memory_raw_search`, `memory_note_search`, `memory_when`
- Added: `memory_belief_list`, `memory_value_list`, `memory_goal_list`, `memory_drive_list`
- Added: `memory_episode_update`, `memory_goal_update`, `memory_belief_update`
- Added: `memory_suggestions_extract`, `memory_suggestions_promote`

#### Authentication
- **User-centric auth**: Separated users from agents (migration 017)
- **JWKS OAuth**: Public key verification for Supabase tokens
- **API key format**: `knl_sk_` prefix for API keys

### Changed

- `load()` parameter renamed: `token_budget` → `budget`
- Sync conflict resolution now includes array merging
- Default budget is 8000 tokens (was unlimited)

### Fixed

- TOCTOU race condition in `save_drive()` - now uses atomic upsert
- OAuth issuer validation - strict equality check
- Sync queue atomicity - uses INSERT OR REPLACE
- Memory pressure visibility - agents now know when memories are excluded

### Security

- Array merge size limits prevent DoS via unbounded growth
- Debug logging for invalid memory types
- Mass assignment protection in sync push
- CSRF protection with SameSite=Strict cookies

## [0.1.0] - 2026-01-15

### Added

- Initial release
- Core memory types: Episodes, Beliefs, Values, Goals, Notes, Drives, Relationships
- SQLite local storage with sqlite-vec
- Supabase cloud storage with pgvector
- Basic sync engine (local-first, last-write-wins)
- CLI with basic commands
- MCP server with ~12 tools
- Anxiety tracking
- Emotional memory tagging
- Meta-memory (confidence, provenance)
- Controlled forgetting (salience-based)
- Identity synthesis
