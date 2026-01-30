# Changelog

All notable changes to Kernle will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
