"""SQLite storage backend for Kernle.

Local-first storage with:
- SQLite for structured data
- sqlite-vec for vector search (semantic search)
- Sync metadata for cloud synchronization
"""

import contextlib
import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import (
    Belief,
    Drive,
    Episode,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    QueuedChange,
    RawEntry,
    Relationship,
    SearchResult,
    SyncConflict,
    SyncResult,
    Value,
    VersionConflictError,
    parse_datetime,
    utc_now,
)
from .embeddings import (
    EmbeddingProvider,
    HashEmbedder,
    pack_embedding,
)

if TYPE_CHECKING:
    from .base import Storage as StorageProtocol

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 12  # Bumped for memory suggestions table

# Maximum size for merged arrays during sync to prevent resource exhaustion
MAX_SYNC_ARRAY_SIZE = 500

# Array fields that should be merged (set union) during sync rather than overwritten
# These are fields where both local and cloud additions should be preserved
SYNC_ARRAY_FIELDS: Dict[str, List[str]] = {
    "episodes": [
        "lessons",
        "tags",
        "emotional_tags",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "beliefs": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "notes": [
        "tags",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "drives": [
        "focus_areas",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "agent_values": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "relationships": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "goals": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "playbooks": [
        "trigger_conditions",
        "failure_modes",
        "recovery_steps",
        "source_episodes",
        "tags",
    ],
}

# Allowed table names for SQL queries (security: prevents SQL injection via table names)
ALLOWED_TABLES = frozenset(
    {
        "episodes",
        "beliefs",
        "agent_values",
        "goals",
        "notes",
        "drives",
        "relationships",
        "playbooks",
        "raw_entries",
        "checkpoints",
        "memory_suggestions",
        "schema_version",
        "sync_queue",
        "sync_conflicts",
        "embeddings",
        "health_check_events",
    }
)


def validate_table_name(table: str) -> str:
    """Validate table name against allowlist to prevent SQL injection.

    Args:
        table: Table name to validate

    Returns:
        The validated table name

    Raises:
        ValueError: If table name is not in allowlist
    """
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    return table


SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Episodes (experiences/work logs)
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    outcome TEXT NOT NULL,
    outcome_type TEXT,
    lessons TEXT,  -- JSON array
    tags TEXT,     -- JSON array
    created_at TEXT NOT NULL,
    -- Emotional memory fields
    emotional_valence REAL DEFAULT 0.0,  -- -1.0 (negative) to 1.0 (positive)
    emotional_arousal REAL DEFAULT 0.0,  -- 0.0 (calm) to 1.0 (intense)
    emotional_tags TEXT,  -- JSON array ["joy", "frustration", "curiosity"]
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,  -- JSON array of episode IDs
    derived_from TEXT,     -- JSON array of memory IDs (format: type:id)
    last_verified TEXT,    -- ISO timestamp
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,  -- JSON array of confidence changes
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_sync ON episodes(cloud_synced_at);
CREATE INDEX IF NOT EXISTS idx_episodes_valence ON episodes(emotional_valence);
CREATE INDEX IF NOT EXISTS idx_episodes_arousal ON episodes(emotional_arousal);
CREATE INDEX IF NOT EXISTS idx_episodes_confidence ON episodes(confidence);
CREATE INDEX IF NOT EXISTS idx_episodes_source_type ON episodes(source_type);
CREATE INDEX IF NOT EXISTS idx_episodes_is_forgotten ON episodes(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_episodes_is_protected ON episodes(is_protected);

-- Beliefs
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    belief_type TEXT DEFAULT 'fact',
    confidence REAL DEFAULT 0.8,
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,  -- JSON array of episode IDs
    derived_from TEXT,     -- JSON array of memory IDs
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Belief revision fields
    supersedes TEXT,           -- ID of belief this replaced
    superseded_by TEXT,        -- ID of belief that replaced this
    times_reinforced INTEGER DEFAULT 0,  -- How many times confirmed
    is_active INTEGER DEFAULT 1,  -- 0 if superseded/archived
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_beliefs_agent ON beliefs(agent_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);
CREATE INDEX IF NOT EXISTS idx_beliefs_source_type ON beliefs(source_type);
CREATE INDEX IF NOT EXISTS idx_beliefs_is_active ON beliefs(is_active);
CREATE INDEX IF NOT EXISTS idx_beliefs_supersedes ON beliefs(supersedes);
CREATE INDEX IF NOT EXISTS idx_beliefs_superseded_by ON beliefs(superseded_by);
CREATE INDEX IF NOT EXISTS idx_beliefs_is_forgotten ON beliefs(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_beliefs_is_protected ON beliefs(is_protected);

-- Values
CREATE TABLE IF NOT EXISTS agent_values (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    statement TEXT NOT NULL,
    priority INTEGER DEFAULT 50,
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.9,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 1,  -- Values protected by default
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_values_agent ON agent_values(agent_id);
CREATE INDEX IF NOT EXISTS idx_values_confidence ON agent_values(confidence);
CREATE INDEX IF NOT EXISTS idx_values_is_forgotten ON agent_values(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_values_is_protected ON agent_values(is_protected);

-- Goals
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_goals_agent ON goals(agent_id);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_confidence ON goals(confidence);
CREATE INDEX IF NOT EXISTS idx_goals_is_forgotten ON goals(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_goals_is_protected ON goals(is_protected);

-- Notes (memories)
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    note_type TEXT DEFAULT 'note',
    speaker TEXT,
    reason TEXT,
    tags TEXT,  -- JSON array
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notes_agent ON notes(agent_id);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at);
CREATE INDEX IF NOT EXISTS idx_notes_confidence ON notes(confidence);
CREATE INDEX IF NOT EXISTS idx_notes_is_forgotten ON notes(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_notes_is_protected ON notes(is_protected);

-- Drives
CREATE TABLE IF NOT EXISTS drives (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    drive_type TEXT NOT NULL,
    intensity REAL DEFAULT 0.5,
    focus_areas TEXT,  -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 1,  -- Drives protected by default
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(agent_id, drive_type)
);
CREATE INDEX IF NOT EXISTS idx_drives_agent ON drives(agent_id);
CREATE INDEX IF NOT EXISTS idx_drives_confidence ON drives(confidence);
CREATE INDEX IF NOT EXISTS idx_drives_is_forgotten ON drives(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_drives_is_protected ON drives(is_protected);

-- Relationships
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    notes TEXT,
    sentiment REAL DEFAULT 0.0,
    interaction_count INTEGER DEFAULT 0,
    last_interaction TEXT,
    created_at TEXT NOT NULL,
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,
    -- Forgetting fields
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,
    -- Context/scope fields
    context TEXT,
    context_tags TEXT,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(agent_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_relationships_agent ON relationships(agent_id);
CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON relationships(confidence);
CREATE INDEX IF NOT EXISTS idx_relationships_is_forgotten ON relationships(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_relationships_is_protected ON relationships(is_protected);

-- Playbooks (procedural memory - "how I do things")
CREATE TABLE IF NOT EXISTS playbooks (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger_conditions TEXT NOT NULL,  -- JSON array
    steps TEXT NOT NULL,               -- JSON array of {action, details, adaptations}
    failure_modes TEXT NOT NULL,       -- JSON array
    recovery_steps TEXT,               -- JSON array (optional)
    mastery_level TEXT DEFAULT 'novice',  -- novice/competent/proficient/expert
    times_used INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    source_episodes TEXT,              -- JSON array of episode IDs
    tags TEXT,                         -- JSON array
    -- Meta-memory fields
    confidence REAL DEFAULT 0.8,
    last_used TEXT,
    created_at TEXT NOT NULL,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_playbooks_agent ON playbooks(agent_id);
CREATE INDEX IF NOT EXISTS idx_playbooks_mastery ON playbooks(mastery_level);
CREATE INDEX IF NOT EXISTS idx_playbooks_times_used ON playbooks(times_used);
CREATE INDEX IF NOT EXISTS idx_playbooks_confidence ON playbooks(confidence);

-- Raw entries (unstructured blob capture for later processing)
-- The blob field is the primary storage - unvalidated, high limit brain dumps
CREATE TABLE IF NOT EXISTS raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    blob TEXT NOT NULL,  -- Unstructured brain dump (primary field)
    captured_at TEXT NOT NULL,  -- When captured (was: timestamp)
    source TEXT DEFAULT 'unknown',  -- Auto-populated: cli|mcp|sdk|import|unknown
    processed INTEGER DEFAULT 0,
    processed_into TEXT,  -- JSON array of memory refs (type:id)
    -- DEPRECATED columns (kept for migration, will be removed)
    content TEXT,  -- Use blob instead
    timestamp TEXT,  -- Use captured_at instead
    tags TEXT,  -- Include in blob text instead
    confidence REAL DEFAULT 1.0,  -- Not meaningful for raw
    source_type TEXT DEFAULT 'direct_experience',  -- Meta-memory, not for raw
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_raw_agent ON raw_entries(agent_id);
CREATE INDEX IF NOT EXISTS idx_raw_processed ON raw_entries(agent_id, processed);
CREATE INDEX IF NOT EXISTS idx_raw_captured_at ON raw_entries(agent_id, captured_at);
-- Partial index for anxiety system queries (unprocessed entries)
CREATE INDEX IF NOT EXISTS idx_raw_unprocessed ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;

-- FTS5 virtual table for raw blob keyword search (safety net for backlogs)
-- Note: FTS5 table is created separately in _ensure_fts5_tables() due to SQLite version differences

-- Memory suggestions (auto-extracted suggestions awaiting review)
CREATE TABLE IF NOT EXISTS memory_suggestions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- episode, belief, note
    content TEXT NOT NULL,  -- JSON object with structured data
    confidence REAL DEFAULT 0.5,
    source_raw_ids TEXT NOT NULL,  -- JSON array of raw entry IDs
    status TEXT DEFAULT 'pending',  -- pending, promoted, modified, rejected
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution_reason TEXT,
    promoted_to TEXT,  -- Format: type:id (e.g., episode:abc123)
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_suggestions_agent ON memory_suggestions(agent_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_status ON memory_suggestions(agent_id, status);
CREATE INDEX IF NOT EXISTS idx_suggestions_type ON memory_suggestions(agent_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_suggestions_created ON memory_suggestions(created_at);

-- Health check events (compliance tracking)
CREATE TABLE IF NOT EXISTS health_check_events (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    anxiety_score INTEGER,
    source TEXT DEFAULT 'cli',  -- cli, mcp
    triggered_by TEXT DEFAULT 'manual'  -- boot, heartbeat, manual
);
CREATE INDEX IF NOT EXISTS idx_health_check_agent ON health_check_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_health_check_checked_at ON health_check_events(checked_at);

-- Sync queue for offline changes (enhanced v2)
CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    operation TEXT NOT NULL,  -- insert, update, delete
    data TEXT,  -- JSON payload of the record data
    local_updated_at TEXT NOT NULL,
    synced INTEGER DEFAULT 0,  -- 0 = pending, 1 = synced
    payload TEXT,  -- Legacy: JSON payload (kept for backward compatibility)
    queued_at TEXT  -- Legacy: kept for backward compatibility
);
CREATE INDEX IF NOT EXISTS idx_sync_queue_table ON sync_queue(table_name);
CREATE INDEX IF NOT EXISTS idx_sync_queue_record ON sync_queue(record_id);
CREATE INDEX IF NOT EXISTS idx_sync_queue_synced ON sync_queue(synced);
-- Unique partial index for atomic UPSERT on unsynced entries
CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_queue_unsynced_unique
    ON sync_queue(table_name, record_id) WHERE synced = 0;

-- Sync metadata (tracks last sync time and state)
CREATE TABLE IF NOT EXISTS sync_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Sync conflict history (tracks resolved conflicts for user visibility)
CREATE TABLE IF NOT EXISTS sync_conflicts (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    local_version TEXT NOT NULL,   -- JSON snapshot of local version
    cloud_version TEXT NOT NULL,   -- JSON snapshot of cloud version
    resolution TEXT NOT NULL,      -- "local_wins" or "cloud_wins"
    resolved_at TEXT NOT NULL,
    local_summary TEXT,            -- Human-readable summary
    cloud_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_resolved ON sync_conflicts(resolved_at);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_record ON sync_conflicts(table_name, record_id);

-- Embedding metadata (tracks what's been embedded)
CREATE TABLE IF NOT EXISTS embedding_meta (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- To detect when re-embedding needed
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embedding_meta_record ON embedding_meta(table_name, record_id);
"""

# Virtual table for vector search (created when sqlite-vec is available)
VECTOR_SCHEMA = """
-- Vector table using sqlite-vec
-- dimension should match embedding provider
CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""


class SQLiteStorage:
    """SQLite-based local storage for Kernle.

    Features:
    - Zero-config local storage
    - Semantic search with sqlite-vec (when available)
    - Sync metadata for cloud synchronization
    - Offline-first with automatic queue when disconnected
    """

    # Connectivity check timeout (seconds)
    CONNECTIVITY_TIMEOUT = 5.0
    # Cloud search timeout (seconds)
    CLOUD_SEARCH_TIMEOUT = 3.0

    def __init__(
        self,
        agent_id: str,
        db_path: Optional[Path] = None,
        cloud_storage: Optional["StorageProtocol"] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        self.agent_id = agent_id
        self.db_path = self._validate_db_path(db_path or Path.home() / ".kernle" / "memories.db")
        self.cloud_storage = cloud_storage  # For sync

        # Connectivity cache
        self._last_connectivity_check: Optional[datetime] = None
        self._is_online_cached: bool = False
        self._connectivity_cache_ttl = 30  # seconds

        # Cloud search credentials cache
        self._cloud_credentials: Optional[Dict[str, str]] = None
        self._cloud_credentials_loaded: bool = False

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for sqlite-vec first
        self._has_vec = self._check_sqlite_vec()

        # Initialize embedder
        self._embedder = embedder or (HashEmbedder() if not embedder else embedder)

        # Initialize flat file directories for all memory layers
        self._agent_dir = self.db_path.parent / agent_id
        self._agent_dir.mkdir(parents=True, exist_ok=True)

        self._raw_dir = self._agent_dir / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        # Identity files (single files, always complete)
        self._beliefs_file = self._agent_dir / "beliefs.md"
        self._values_file = self._agent_dir / "values.md"
        self._relationships_file = self._agent_dir / "relationships.md"
        self._goals_file = self._agent_dir / "goals.md"

        # Initialize database
        self._init_db()

        # Initialize flat files from existing data
        self._init_flat_files()

        if not self._has_vec:
            logger.info("sqlite-vec not available, semantic search will use text matching")

    def _init_flat_files(self) -> None:
        """Initialize flat files from existing database data.

        Called on startup to ensure flat files exist and are in sync.
        """
        try:
            # Only sync if files don't exist or are empty
            if not self._beliefs_file.exists() or self._beliefs_file.stat().st_size == 0:
                self._sync_beliefs_to_file()
            if not self._values_file.exists() or self._values_file.stat().st_size == 0:
                self._sync_values_to_file()
            if (
                not self._relationships_file.exists()
                or self._relationships_file.stat().st_size == 0
            ):
                self._sync_relationships_to_file()
            if not self._goals_file.exists() or self._goals_file.stat().st_size == 0:
                self._sync_goals_to_file()
        except Exception as e:
            logger.warning(f"Failed to initialize flat files: {e}")

    def _validate_db_path(self, db_path: Path) -> Path:
        """Validate database path to prevent path traversal attacks."""
        import tempfile

        try:
            # Resolve to absolute path to prevent directory traversal
            resolved_path = db_path.resolve()

            # Ensure it's within a safe directory (user's home, system temp, or /tmp)
            home_path = Path.home().resolve()
            tmp_path = Path("/tmp").resolve()
            system_temp = Path(tempfile.gettempdir()).resolve()

            # Use is_relative_to() for secure path validation (Python 3.9+)
            is_safe = (
                resolved_path.is_relative_to(home_path)
                or resolved_path.is_relative_to(tmp_path)
                or resolved_path.is_relative_to(system_temp)
            )

            # Also allow /var/folders on macOS (where tempfile creates dirs)
            if not is_safe:
                try:
                    var_folders = Path("/var/folders").resolve()
                    private_var_folders = Path("/private/var/folders").resolve()
                    is_safe = resolved_path.is_relative_to(
                        var_folders
                    ) or resolved_path.is_relative_to(private_var_folders)
                except (OSError, ValueError):
                    pass

            if not is_safe:
                raise ValueError("Database path must be within user home or temp directory")

            return resolved_path

        except (OSError, ValueError) as e:
            logger.error(f"Invalid database path: {e}")
            raise ValueError(f"Invalid database path: {e}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with sqlite-vec loaded if available.

        IMPORTANT: Callers should use this with contextlib.closing() or
        call conn.close() explicitly to avoid resource warnings:

            from contextlib import closing
            with closing(self._get_conn()) as conn:
                ...

        Or use the _connect() context manager which handles both
        commit/rollback and close.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if self._has_vec:
            self._load_vec(conn)
        return conn

    @contextlib.contextmanager
    def _connect(self):
        """Context manager that handles transactions AND closes connection.

        Use this instead of `with self._connect() as conn:` to avoid
        unclosed connection warnings. This handles:
        - Transaction commit on success
        - Transaction rollback on exception
        - Connection close in all cases
        """
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self):
        """Close any resources.

        Since we create connections per-operation with context managers,
        this primarily exists for API compatibility and explicit cleanup.
        """
        pass  # No persistent connections to close

    # === Cloud Search Methods ===

    def _load_cloud_credentials(self) -> Optional[Dict[str, str]]:
        """Load cloud credentials from config files or environment variables.

        Priority:
        1. ~/.kernle/credentials.json
        2. Environment variables (KERNLE_BACKEND_URL, KERNLE_AUTH_TOKEN)
        3. ~/.kernle/config.json (legacy)

        Returns:
            Dict with 'backend_url' and 'auth_token', or None if not configured.
        """
        import os

        if self._cloud_credentials_loaded:
            return self._cloud_credentials

        self._cloud_credentials_loaded = True
        backend_url = None
        auth_token = None

        # Try credentials.json first
        credentials_path = Path.home() / ".kernle" / "credentials.json"
        if credentials_path.exists():
            try:
                with open(credentials_path) as f:
                    creds = json.load(f)
                    backend_url = creds.get("backend_url")
                    # Accept both "auth_token" (preferred) and "token" (legacy) for compatibility
                    auth_token = creds.get("auth_token") or creds.get("token")
            except (json.JSONDecodeError, OSError):
                pass

        # Override with environment variables
        backend_url = os.environ.get("KERNLE_BACKEND_URL") or backend_url
        auth_token = os.environ.get("KERNLE_AUTH_TOKEN") or auth_token

        # Try config.json as fallback
        if not backend_url or not auth_token:
            config_path = Path.home() / ".kernle" / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        backend_url = backend_url or config.get("backend_url")
                        auth_token = auth_token or config.get("auth_token")
                except (json.JSONDecodeError, OSError):
                    pass

        if backend_url and auth_token:
            self._cloud_credentials = {
                "backend_url": backend_url.rstrip("/"),
                "auth_token": auth_token,
            }
        else:
            self._cloud_credentials = None

        return self._cloud_credentials

    def has_cloud_credentials(self) -> bool:
        """Check if cloud credentials are available.

        Returns:
            True if backend_url and auth_token are configured.
        """
        creds = self._load_cloud_credentials()
        return creds is not None

    def cloud_health_check(self, timeout: float = 3.0) -> Dict[str, Any]:
        """Test cloud backend connectivity.

        Args:
            timeout: Request timeout in seconds (default 3s)

        Returns:
            Dict with keys:
            - 'healthy': bool indicating if cloud is reachable
            - 'latency_ms': response time in milliseconds (if healthy)
            - 'error': error message (if not healthy)
        """
        import time

        creds = self._load_cloud_credentials()
        if not creds:
            return {
                "healthy": False,
                "error": "No cloud credentials configured",
            }

        try:
            import urllib.error
            import urllib.request

            url = f"{creds['backend_url']}/health"
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {creds['auth_token']}",
                    "Content-Type": "application/json",
                },
                method="GET",
            )

            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                latency_ms = (time.time() - start) * 1000
                if resp.status == 200:
                    return {
                        "healthy": True,
                        "latency_ms": round(latency_ms, 2),
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {resp.status}",
                    }
        except urllib.error.URLError as e:
            return {
                "healthy": False,
                "error": f"Connection failed: {e.reason}",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }

    def _cloud_search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[List[SearchResult]]:
        """Search memories via cloud backend.

        Args:
            query: Search query
            limit: Maximum results
            record_types: Filter by type (episode, note, belief, etc.)
            timeout: Request timeout in seconds (default: CLOUD_SEARCH_TIMEOUT)

        Returns:
            List of SearchResult, or None if cloud search failed/unavailable.
        """
        creds = self._load_cloud_credentials()
        if not creds:
            return None

        timeout = timeout or self.CLOUD_SEARCH_TIMEOUT

        try:
            import urllib.error
            import urllib.request

            url = f"{creds['backend_url']}/memories/search"
            payload = {
                "query": query,
                "limit": limit,
            }
            if record_types:
                # Map internal types to backend table names
                type_map = {
                    "episode": "episodes",
                    "note": "notes",
                    "belief": "beliefs",
                    "value": "values",
                    "goal": "goals",
                }
                payload["memory_types"] = [type_map.get(t, t) for t in record_types]

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {creds['auth_token']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    response_data = json.loads(resp.read().decode("utf-8"))
                    return self._parse_cloud_search_results(response_data)
                else:
                    logger.debug(f"Cloud search returned HTTP {resp.status}")
                    return None

        except urllib.error.URLError as e:
            logger.debug(f"Cloud search failed: {e.reason}")
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"Cloud search response parse error: {e}")
            return None
        except Exception as e:
            logger.debug(f"Cloud search error: {e}")
            return None

    def _parse_cloud_search_results(self, response_data: Dict[str, Any]) -> List[SearchResult]:
        """Parse cloud search response into SearchResult objects.

        Args:
            response_data: Response from /memories/search endpoint

        Returns:
            List of SearchResult objects
        """
        results = []
        cloud_results = response_data.get("results", [])

        # Map backend types to internal types
        type_map = {
            "episodes": "episode",
            "notes": "note",
            "beliefs": "belief",
            "values": "value",
            "goals": "goal",
        }

        for item in cloud_results:
            memory_type = type_map.get(item.get("memory_type"), item.get("memory_type"))
            metadata = item.get("metadata", {})

            # Create a minimal record object based on type
            record = self._create_record_from_cloud(memory_type, item, metadata)

            if record:
                results.append(
                    SearchResult(
                        record=record,
                        record_type=memory_type,
                        score=item.get("score", 1.0),
                    )
                )

        return results

    def _create_record_from_cloud(
        self, memory_type: str, item: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Optional[Any]:
        """Create a record object from cloud search result.

        Args:
            memory_type: Type of memory (episode, note, etc.)
            item: Search result item
            metadata: Additional metadata from result

        Returns:
            Record object or None
        """
        record_id = item.get("id", "")
        created_at = parse_datetime(item.get("created_at"))

        if memory_type == "episode":
            return Episode(
                id=record_id,
                agent_id=self.agent_id,
                objective=metadata.get("objective", item.get("content", "")),
                outcome=metadata.get("outcome", ""),
                outcome_type=metadata.get("outcome_type"),
                lessons=metadata.get("lessons_learned"),
                tags=metadata.get("tags"),
                created_at=created_at,
            )
        elif memory_type == "note":
            return Note(
                id=record_id,
                agent_id=self.agent_id,
                content=item.get("content", ""),
                note_type=metadata.get("note_type", "note"),
                tags=metadata.get("tags"),
                created_at=created_at,
            )
        elif memory_type == "belief":
            return Belief(
                id=record_id,
                agent_id=self.agent_id,
                statement=item.get("content", ""),
                belief_type=metadata.get("belief_type", "fact"),
                confidence=metadata.get("confidence", 0.8),
                created_at=created_at,
            )
        elif memory_type == "value":
            return Value(
                id=record_id,
                agent_id=self.agent_id,
                name=metadata.get("name", ""),
                statement=item.get("content", ""),
                priority=metadata.get("priority", 50),
                created_at=created_at,
            )
        elif memory_type == "goal":
            return Goal(
                id=record_id,
                agent_id=self.agent_id,
                title=metadata.get("title", item.get("content", "")),
                description=metadata.get("description"),
                priority=metadata.get("priority", "medium"),
                status=metadata.get("status", "active"),
                created_at=created_at,
            )

        return None

    def _init_db(self):
        """Initialize the database schema."""
        with self._connect() as conn:
            # First, run migrations if needed (before executing full schema)
            self._migrate_schema(conn)

            # Now execute full schema (CREATE TABLE IF NOT EXISTS is safe)
            conn.executescript(SCHEMA)

            # Check/set schema version
            cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cur.fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            else:
                # Update schema version
                conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

            # Create vector table if sqlite-vec is available
            if self._has_vec:
                self._load_vec(conn)
                vec_schema = VECTOR_SCHEMA.format(dim=self._embedder.dimension)
                try:
                    conn.executescript(vec_schema)
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Could not create vector table: {e}")

            # Create FTS5 table for raw blob keyword search
            self._ensure_raw_fts5(conn)

            conn.commit()

        # Set secure file permissions (owner read/write only)
        import os

        try:
            os.chmod(self.db_path, 0o600)
            os.chmod(self.db_path.parent, 0o700)
            if self._agent_dir.exists():
                os.chmod(self._agent_dir, 0o700)
        except OSError as e:
            logger.warning(f"Could not set secure permissions: {e}")

    def _ensure_raw_fts5(self, conn: sqlite3.Connection):
        """Create FTS5 virtual table for raw blob keyword search.

        FTS5 provides fast keyword search on raw blobs as a safety net
        when backlogs accumulate. This is separate from semantic search.
        """
        try:
            # Check if FTS5 table already exists
            result = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='raw_fts'
            """).fetchone()

            if result is None:
                # Create FTS5 content table (external content mode)
                # This links to raw_entries.blob without duplicating data
                conn.execute("""
                    CREATE VIRTUAL TABLE raw_fts USING fts5(
                        blob,
                        content=raw_entries,
                        content_rowid=rowid
                    )
                """)
                logger.info("Created raw_fts FTS5 table for keyword search")

                # Populate FTS index from existing data
                conn.execute("INSERT INTO raw_fts(raw_fts) VALUES('rebuild')")
                logger.info("Rebuilt raw_fts index")

        except sqlite3.OperationalError as e:
            # FTS5 might not be available in all SQLite builds
            if "no such module: fts5" in str(e).lower():
                logger.warning("FTS5 not available in this SQLite build - keyword search disabled")
            elif "already exists" in str(e).lower():
                pass  # Table already exists, that's fine
            else:
                logger.warning(f"Could not create FTS5 table: {e}")
        except Exception as e:
            logger.warning(f"FTS5 setup failed: {e}")

    def _migrate_schema(self, conn: sqlite3.Connection):
        """Run schema migrations for existing databases.

        Handles adding new columns to existing tables.
        """
        # Check if tables exist first
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}

        if "episodes" not in table_names:
            # Fresh database, no migration needed
            return

        # Get current columns for each table
        def get_columns(table: str) -> set:
            try:
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                return {c[1] for c in cols}
            except (TypeError, ValueError):
                return set()

        # Migrations for episodes table
        episode_cols = get_columns("episodes")
        migrations = []

        if "emotional_valence" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN emotional_valence REAL DEFAULT 0.0")
        if "emotional_arousal" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN emotional_arousal REAL DEFAULT 0.0")
        if "emotional_tags" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN emotional_tags TEXT")
        if "confidence" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN confidence REAL DEFAULT 0.8")
        if "source_type" not in episode_cols:
            migrations.append(
                "ALTER TABLE episodes ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
            )
        if "source_episodes" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN source_episodes TEXT")
        if "derived_from" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN derived_from TEXT")
        if "last_verified" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN last_verified TEXT")
        if "verification_count" not in episode_cols:
            migrations.append(
                "ALTER TABLE episodes ADD COLUMN verification_count INTEGER DEFAULT 0"
            )
        if "confidence_history" not in episode_cols:
            migrations.append("ALTER TABLE episodes ADD COLUMN confidence_history TEXT")

        # Migrations for beliefs table
        belief_cols = get_columns("beliefs")
        if "beliefs" in table_names:
            if "source_type" not in belief_cols:
                migrations.append(
                    "ALTER TABLE beliefs ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
                )
            if "source_episodes" not in belief_cols:
                migrations.append("ALTER TABLE beliefs ADD COLUMN source_episodes TEXT")
            if "derived_from" not in belief_cols:
                migrations.append("ALTER TABLE beliefs ADD COLUMN derived_from TEXT")
            if "last_verified" not in belief_cols:
                migrations.append("ALTER TABLE beliefs ADD COLUMN last_verified TEXT")
            if "verification_count" not in belief_cols:
                migrations.append(
                    "ALTER TABLE beliefs ADD COLUMN verification_count INTEGER DEFAULT 0"
                )
            if "confidence_history" not in belief_cols:
                migrations.append("ALTER TABLE beliefs ADD COLUMN confidence_history TEXT")
            # Belief revision fields
            if "supersedes" not in belief_cols:
                migrations.append("ALTER TABLE beliefs ADD COLUMN supersedes TEXT")
            if "superseded_by" not in belief_cols:
                migrations.append("ALTER TABLE beliefs ADD COLUMN superseded_by TEXT")
            if "times_reinforced" not in belief_cols:
                migrations.append(
                    "ALTER TABLE beliefs ADD COLUMN times_reinforced INTEGER DEFAULT 0"
                )
            if "is_active" not in belief_cols:
                migrations.append("ALTER TABLE beliefs ADD COLUMN is_active INTEGER DEFAULT 1")

        # Migrations for values table
        value_cols = get_columns("agent_values")
        if "agent_values" in table_names:
            if "confidence" not in value_cols:
                migrations.append("ALTER TABLE agent_values ADD COLUMN confidence REAL DEFAULT 0.9")
            if "source_type" not in value_cols:
                migrations.append(
                    "ALTER TABLE agent_values ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
                )
            if "source_episodes" not in value_cols:
                migrations.append("ALTER TABLE agent_values ADD COLUMN source_episodes TEXT")
            if "derived_from" not in value_cols:
                migrations.append("ALTER TABLE agent_values ADD COLUMN derived_from TEXT")
            if "last_verified" not in value_cols:
                migrations.append("ALTER TABLE agent_values ADD COLUMN last_verified TEXT")
            if "verification_count" not in value_cols:
                migrations.append(
                    "ALTER TABLE agent_values ADD COLUMN verification_count INTEGER DEFAULT 0"
                )
            if "confidence_history" not in value_cols:
                migrations.append("ALTER TABLE agent_values ADD COLUMN confidence_history TEXT")

        # Migrations for goals table
        goal_cols = get_columns("goals")
        if "goals" in table_names:
            if "confidence" not in goal_cols:
                migrations.append("ALTER TABLE goals ADD COLUMN confidence REAL DEFAULT 0.8")
            if "source_type" not in goal_cols:
                migrations.append(
                    "ALTER TABLE goals ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
                )
            if "source_episodes" not in goal_cols:
                migrations.append("ALTER TABLE goals ADD COLUMN source_episodes TEXT")
            if "derived_from" not in goal_cols:
                migrations.append("ALTER TABLE goals ADD COLUMN derived_from TEXT")
            if "last_verified" not in goal_cols:
                migrations.append("ALTER TABLE goals ADD COLUMN last_verified TEXT")
            if "verification_count" not in goal_cols:
                migrations.append(
                    "ALTER TABLE goals ADD COLUMN verification_count INTEGER DEFAULT 0"
                )
            if "confidence_history" not in goal_cols:
                migrations.append("ALTER TABLE goals ADD COLUMN confidence_history TEXT")

        # Migrations for notes table
        note_cols = get_columns("notes")
        if "notes" in table_names:
            if "confidence" not in note_cols:
                migrations.append("ALTER TABLE notes ADD COLUMN confidence REAL DEFAULT 0.8")
            if "source_type" not in note_cols:
                migrations.append(
                    "ALTER TABLE notes ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
                )
            if "source_episodes" not in note_cols:
                migrations.append("ALTER TABLE notes ADD COLUMN source_episodes TEXT")
            if "derived_from" not in note_cols:
                migrations.append("ALTER TABLE notes ADD COLUMN derived_from TEXT")
            if "last_verified" not in note_cols:
                migrations.append("ALTER TABLE notes ADD COLUMN last_verified TEXT")
            if "verification_count" not in note_cols:
                migrations.append(
                    "ALTER TABLE notes ADD COLUMN verification_count INTEGER DEFAULT 0"
                )
            if "confidence_history" not in note_cols:
                migrations.append("ALTER TABLE notes ADD COLUMN confidence_history TEXT")

        # Migrations for drives table
        drive_cols = get_columns("drives")
        if "drives" in table_names:
            if "confidence" not in drive_cols:
                migrations.append("ALTER TABLE drives ADD COLUMN confidence REAL DEFAULT 0.8")
            if "source_type" not in drive_cols:
                migrations.append(
                    "ALTER TABLE drives ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
                )
            if "source_episodes" not in drive_cols:
                migrations.append("ALTER TABLE drives ADD COLUMN source_episodes TEXT")
            if "derived_from" not in drive_cols:
                migrations.append("ALTER TABLE drives ADD COLUMN derived_from TEXT")
            if "last_verified" not in drive_cols:
                migrations.append("ALTER TABLE drives ADD COLUMN last_verified TEXT")
            if "verification_count" not in drive_cols:
                migrations.append(
                    "ALTER TABLE drives ADD COLUMN verification_count INTEGER DEFAULT 0"
                )
            if "confidence_history" not in drive_cols:
                migrations.append("ALTER TABLE drives ADD COLUMN confidence_history TEXT")

        # Migrations for relationships table
        rel_cols = get_columns("relationships")
        if "relationships" in table_names:
            if "confidence" not in rel_cols:
                migrations.append(
                    "ALTER TABLE relationships ADD COLUMN confidence REAL DEFAULT 0.8"
                )
            if "source_type" not in rel_cols:
                migrations.append(
                    "ALTER TABLE relationships ADD COLUMN source_type TEXT DEFAULT 'direct_experience'"
                )
            if "source_episodes" not in rel_cols:
                migrations.append("ALTER TABLE relationships ADD COLUMN source_episodes TEXT")
            if "derived_from" not in rel_cols:
                migrations.append("ALTER TABLE relationships ADD COLUMN derived_from TEXT")
            if "last_verified" not in rel_cols:
                migrations.append("ALTER TABLE relationships ADD COLUMN last_verified TEXT")
            if "verification_count" not in rel_cols:
                migrations.append(
                    "ALTER TABLE relationships ADD COLUMN verification_count INTEGER DEFAULT 0"
                )
            if "confidence_history" not in rel_cols:
                migrations.append("ALTER TABLE relationships ADD COLUMN confidence_history TEXT")

        # Migrations for sync_queue table (enhanced v2)
        sync_cols = get_columns("sync_queue")
        if "sync_queue" in table_names:
            if "payload" not in sync_cols:
                migrations.append("ALTER TABLE sync_queue ADD COLUMN payload TEXT")
            if "data" not in sync_cols:
                migrations.append("ALTER TABLE sync_queue ADD COLUMN data TEXT")
            if "local_updated_at" not in sync_cols:
                migrations.append("ALTER TABLE sync_queue ADD COLUMN local_updated_at TEXT")
            if "synced" not in sync_cols:
                migrations.append("ALTER TABLE sync_queue ADD COLUMN synced INTEGER DEFAULT 0")

        # === Forgetting field migrations ===
        # Add forgetting fields to all memory tables
        forgetting_tables = [
            ("episodes", False),  # (table_name, protected_by_default)
            ("beliefs", False),
            ("agent_values", True),  # Values protected by default
            ("goals", False),
            ("notes", False),
            ("drives", True),  # Drives protected by default
            ("relationships", False),
        ]

        for table, protected_default in forgetting_tables:
            if table not in table_names:
                continue
            cols = get_columns(table)
            protected_val = 1 if protected_default else 0

            if "times_accessed" not in cols:
                migrations.append(
                    f"ALTER TABLE {table} ADD COLUMN times_accessed INTEGER DEFAULT 0"
                )
            if "last_accessed" not in cols:
                migrations.append(f"ALTER TABLE {table} ADD COLUMN last_accessed TEXT")
            if "is_protected" not in cols:
                migrations.append(
                    f"ALTER TABLE {table} ADD COLUMN is_protected INTEGER DEFAULT {protected_val}"
                )
            if "is_forgotten" not in cols:
                migrations.append(f"ALTER TABLE {table} ADD COLUMN is_forgotten INTEGER DEFAULT 0")
            if "forgotten_at" not in cols:
                migrations.append(f"ALTER TABLE {table} ADD COLUMN forgotten_at TEXT")
            if "forgotten_reason" not in cols:
                migrations.append(f"ALTER TABLE {table} ADD COLUMN forgotten_reason TEXT")

        # === Context/scope field migrations ===
        # Add context fields to all memory tables
        context_tables = [
            "episodes",
            "beliefs",
            "agent_values",
            "goals",
            "notes",
            "drives",
            "relationships",
        ]

        for table in context_tables:
            if table not in table_names:
                continue
            cols = get_columns(table)

            if "context" not in cols:
                migrations.append(f"ALTER TABLE {table} ADD COLUMN context TEXT")
            if "context_tags" not in cols:
                migrations.append(f"ALTER TABLE {table} ADD COLUMN context_tags TEXT")

        # === Raw entries blob migration ===
        # Migrate from structured content/tags to blob-based storage
        raw_cols = get_columns("raw_entries")
        if "raw_entries" in table_names:
            # Add blob column if it doesn't exist
            if "blob" not in raw_cols:
                migrations.append("ALTER TABLE raw_entries ADD COLUMN blob TEXT")
            # Add captured_at column if it doesn't exist
            if "captured_at" not in raw_cols:
                migrations.append("ALTER TABLE raw_entries ADD COLUMN captured_at TEXT")

            # Migrate data from content/tags to blob with natural language format
            # Only run if blob column exists but has NULL values and content has data
            if "blob" in raw_cols or "blob" not in raw_cols:
                # This will run after the ALTER TABLE above
                try:
                    # Check if migration is needed (blob is NULL but content exists)
                    needs_migration = conn.execute("""
                        SELECT COUNT(*) FROM raw_entries
                        WHERE (blob IS NULL OR blob = '') AND content IS NOT NULL AND content != ''
                    """).fetchone()[0]

                    if needs_migration > 0:
                        # Migrate content + tags to blob in natural language format
                        conn.execute("""
                            UPDATE raw_entries SET blob =
                                content ||
                                CASE WHEN source IS NOT NULL AND source != 'manual' AND source != '' AND source != 'unknown'
                                     THEN ' (from ' || source || ')' ELSE '' END ||
                                CASE WHEN tags IS NOT NULL AND tags != '[]' AND tags != 'null' AND tags != ''
                                     THEN ' [tags: ' ||
                                          REPLACE(REPLACE(REPLACE(tags, '["', ''), '"]', ''), '","', ', ') ||
                                          ']'
                                     ELSE '' END
                            WHERE (blob IS NULL OR blob = '') AND content IS NOT NULL
                        """)
                        # Copy timestamp to captured_at
                        conn.execute("""
                            UPDATE raw_entries SET captured_at = timestamp
                            WHERE captured_at IS NULL AND timestamp IS NOT NULL
                        """)
                        # Normalize source to enum values
                        conn.execute("""
                            UPDATE raw_entries SET source =
                                CASE
                                    WHEN source IN ('cli', 'mcp', 'sdk', 'import') THEN source
                                    WHEN source = 'manual' THEN 'cli'
                                    WHEN source LIKE '%auto%' THEN 'sdk'
                                    ELSE 'unknown'
                                END
                        """)
                        logger.info(f"Migrated {needs_migration} raw entries to blob format")
                except Exception as e:
                    logger.warning(f"Raw blob migration check failed: {e}")

        # Create health_check_events table if it doesn't exist
        if "health_check_events" not in table_names:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_check_events (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    anxiety_score INTEGER,
                    source TEXT DEFAULT 'cli',
                    triggered_by TEXT DEFAULT 'manual'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_health_check_agent ON health_check_events(agent_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_health_check_checked_at ON health_check_events(checked_at)"
            )
            logger.info("Created health_check_events table")

        # Execute migrations
        for migration in migrations:
            try:
                conn.execute(migration)
                logger.debug(f"Migration applied: {migration}")
            except Exception as e:
                logger.warning(f"Migration failed (may already exist): {migration} - {e}")

        if migrations:
            conn.commit()
            logger.info(f"Applied {len(migrations)} schema migrations")

    def _check_sqlite_vec(self) -> bool:
        """Check if sqlite-vec extension is available."""
        try:
            import sqlite_vec

            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.close()
            return True
        except ImportError:
            logger.debug("sqlite-vec package not installed")
            return False
        except Exception as e:
            logger.debug(f"sqlite-vec not available: {e}")
            return False

    def _load_vec(self, conn: sqlite3.Connection):
        """Load sqlite-vec extension into connection."""
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception as e:
            logger.warning(f"Could not load sqlite-vec: {e}")

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return utc_now()

    def _parse_datetime(self, s: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        return parse_datetime(s)

    def _to_json(self, data: Any) -> Optional[str]:
        """Convert to JSON string."""
        if data is None:
            return None
        return json.dumps(data)

    def _from_json(self, s: Optional[str]) -> Any:
        """Parse JSON string."""
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    def _safe_get(self, row: sqlite3.Row, key: str, default: Any = None) -> Any:
        """Safely get a value from a row, returning default if column missing.

        Useful for backwards compatibility when schema is migrated.
        """
        try:
            value = row[key]
            return value if value is not None else default
        except (IndexError, KeyError):
            return default

    def _record_to_dict(self, record: Any) -> Dict[str, Any]:
        """Serialize a record to a dictionary for sync queue data.

        Handles datetime conversion and nested objects.
        """
        import dataclasses

        if not dataclasses.is_dataclass(record):
            return {}

        result = {}
        for field in dataclasses.fields(record):
            value = getattr(record, field.name)
            if value is None:
                result[field.name] = None
            elif isinstance(value, datetime):
                result[field.name] = value.isoformat()
            elif isinstance(value, (list, dict)):
                result[field.name] = value
            else:
                result[field.name] = value
        return result

    def _queue_sync(
        self,
        conn: sqlite3.Connection,
        table: str,
        record_id: str,
        operation: str,
        payload: Optional[str] = None,
        data: Optional[str] = None,
    ):
        """Queue a change for sync.

        Deduplicates by (table, record_id) - only keeps latest operation.
        Uses UPSERT (INSERT ... ON CONFLICT) for atomic operation to prevent
        race conditions between concurrent writes.
        """
        now = self._now()

        # Use atomic UPSERT to prevent race condition between SELECT and UPDATE/INSERT
        # This requires a unique index on (table_name, record_id) where synced = 0
        # We use INSERT ... ON CONFLICT DO UPDATE for atomicity
        conn.execute(
            """INSERT INTO sync_queue
               (table_name, record_id, operation, data, local_updated_at, synced, payload, queued_at)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT(table_name, record_id) WHERE synced = 0
               DO UPDATE SET
                   operation = excluded.operation,
                   data = excluded.data,
                   local_updated_at = excluded.local_updated_at,
                   payload = excluded.payload,
                   queued_at = excluded.queued_at""",
            (table, record_id, operation, data, now, payload, now),
        )

    def _content_hash(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _save_embedding(self, conn: sqlite3.Connection, table: str, record_id: str, content: str):
        """Save embedding for a record."""
        if not self._has_vec:
            return

        content_hash = self._content_hash(content)
        vec_id = f"{table}:{record_id}"

        # Check if embedding exists and is current
        existing = conn.execute(
            "SELECT content_hash FROM embedding_meta WHERE id = ?", (vec_id,)
        ).fetchone()

        if existing and existing["content_hash"] == content_hash:
            return  # Already up to date

        # Generate embedding
        try:
            embedding = self._embedder.embed(content)
            packed = pack_embedding(embedding)

            # Upsert into vector table
            conn.execute(
                "INSERT OR REPLACE INTO vec_embeddings (id, embedding) VALUES (?, ?)",
                (vec_id, packed),
            )

            # Update metadata
            conn.execute(
                """INSERT OR REPLACE INTO embedding_meta
                   (id, table_name, record_id, content_hash, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (vec_id, table, record_id, content_hash, self._now()),
            )
        except Exception as e:
            logger.warning(f"Failed to save embedding for {vec_id}: {e}")

    def _get_searchable_content(self, record_type: str, record: Any) -> str:
        """Get searchable text content from a record."""
        if record_type == "episode":
            parts = [record.objective, record.outcome]
            if record.lessons:
                parts.extend(record.lessons)
            return " ".join(filter(None, parts))
        elif record_type == "note":
            return record.content
        elif record_type == "belief":
            return record.statement
        elif record_type == "value":
            return f"{record.name}: {record.statement}"
        elif record_type == "goal":
            return f"{record.title} {record.description or ''}"
        return ""

    # === Episodes ===

    def save_episode(self, episode: Episode) -> str:
        """Save an episode."""
        if not episode.id:
            episode.id = str(uuid.uuid4())

        now = self._now()
        episode.local_updated_at = self._parse_datetime(now)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                (id, agent_id, objective, outcome, outcome_type, lessons, tags,
                 emotional_valence, emotional_arousal, emotional_tags,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 times_accessed, last_accessed, is_protected, is_forgotten,
                 forgotten_at, forgotten_reason, context, context_tags,
                 created_at, local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    episode.id,
                    self.agent_id,
                    episode.objective,
                    episode.outcome,
                    episode.outcome_type,
                    self._to_json(episode.lessons),
                    self._to_json(episode.tags),
                    episode.emotional_valence,
                    episode.emotional_arousal,
                    self._to_json(episode.emotional_tags),
                    episode.confidence,
                    episode.source_type,
                    self._to_json(episode.source_episodes),
                    self._to_json(episode.derived_from),
                    episode.last_verified.isoformat() if episode.last_verified else None,
                    episode.verification_count,
                    self._to_json(episode.confidence_history),
                    episode.times_accessed,
                    episode.last_accessed.isoformat() if episode.last_accessed else None,
                    1 if episode.is_protected else 0,
                    1 if episode.is_forgotten else 0,
                    episode.forgotten_at.isoformat() if episode.forgotten_at else None,
                    episode.forgotten_reason,
                    episode.context,
                    self._to_json(episode.context_tags),
                    episode.created_at.isoformat() if episode.created_at else now,
                    now,
                    episode.cloud_synced_at.isoformat() if episode.cloud_synced_at else None,
                    episode.version,
                    1 if episode.deleted else 0,
                ),
            )
            # Queue for sync with record data
            episode_data = self._to_json(self._record_to_dict(episode))
            self._queue_sync(conn, "episodes", episode.id, "upsert", data=episode_data)

            # Save embedding for search
            content = self._get_searchable_content("episode", episode)
            self._save_embedding(conn, "episodes", episode.id, content)

            conn.commit()

        return episode.id

    def update_episode_atomic(
        self, episode: Episode, expected_version: Optional[int] = None
    ) -> bool:
        """Update an episode with optimistic concurrency control.

        This method performs an atomic update that:
        1. Checks if the current version matches expected_version
        2. Increments the version atomically
        3. Updates all other fields

        Args:
            episode: The episode with updated fields
            expected_version: The version we expect the record to have.
                             If None, uses episode.version.

        Returns:
            True if update succeeded

        Raises:
            VersionConflictError: If the record's version doesn't match expected
        """
        if expected_version is None:
            expected_version = episode.version

        now = self._now()

        with self._connect() as conn:
            # First check current version to provide better error message
            current = conn.execute(
                "SELECT version FROM episodes WHERE id = ? AND agent_id = ?",
                (episode.id, self.agent_id),
            ).fetchone()

            if not current:
                return False  # Record doesn't exist

            current_version = current["version"]
            if current_version != expected_version:
                raise VersionConflictError(
                    "episodes", episode.id, expected_version, current_version
                )

            # Atomic update with version increment
            cursor = conn.execute(
                """
                UPDATE episodes SET
                    objective = ?,
                    outcome = ?,
                    outcome_type = ?,
                    lessons = ?,
                    tags = ?,
                    emotional_valence = ?,
                    emotional_arousal = ?,
                    emotional_tags = ?,
                    confidence = ?,
                    source_type = ?,
                    source_episodes = ?,
                    derived_from = ?,
                    last_verified = ?,
                    verification_count = ?,
                    confidence_history = ?,
                    times_accessed = ?,
                    last_accessed = ?,
                    is_protected = ?,
                    is_forgotten = ?,
                    forgotten_at = ?,
                    forgotten_reason = ?,
                    context = ?,
                    context_tags = ?,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND agent_id = ? AND version = ?
                """,
                (
                    episode.objective,
                    episode.outcome,
                    episode.outcome_type,
                    self._to_json(episode.lessons),
                    self._to_json(episode.tags),
                    episode.emotional_valence,
                    episode.emotional_arousal,
                    self._to_json(episode.emotional_tags),
                    episode.confidence,
                    episode.source_type,
                    self._to_json(episode.source_episodes),
                    self._to_json(episode.derived_from),
                    episode.last_verified.isoformat() if episode.last_verified else None,
                    episode.verification_count,
                    self._to_json(episode.confidence_history),
                    episode.times_accessed,
                    episode.last_accessed.isoformat() if episode.last_accessed else None,
                    1 if episode.is_protected else 0,
                    1 if episode.is_forgotten else 0,
                    episode.forgotten_at.isoformat() if episode.forgotten_at else None,
                    episode.forgotten_reason,
                    episode.context,
                    self._to_json(episode.context_tags),
                    now,
                    episode.id,
                    self.agent_id,
                    expected_version,
                ),
            )

            if cursor.rowcount == 0:
                # Version changed between check and update (rare but possible)
                conn.rollback()
                new_current = conn.execute(
                    "SELECT version FROM episodes WHERE id = ? AND agent_id = ?",
                    (episode.id, self.agent_id),
                ).fetchone()
                actual = new_current["version"] if new_current else -1
                raise VersionConflictError("episodes", episode.id, expected_version, actual)

            # Queue for sync
            episode.version = expected_version + 1
            episode_data = self._to_json(self._record_to_dict(episode))
            self._queue_sync(conn, "episodes", episode.id, "upsert", data=episode_data)

            # Update embedding
            content = self._get_searchable_content("episode", episode)
            self._save_embedding(conn, "episodes", episode.id, content)

            conn.commit()

        return True

    def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        """Save multiple episodes in a single transaction."""
        if not episodes:
            return []
        now = self._now()
        ids = []
        with self._connect() as conn:
            for episode in episodes:
                if not episode.id:
                    episode.id = str(uuid.uuid4())
                ids.append(episode.id)
                episode.local_updated_at = self._parse_datetime(now)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO episodes
                    (id, agent_id, objective, outcome, outcome_type, lessons, tags,
                     emotional_valence, emotional_arousal, emotional_tags,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     times_accessed, last_accessed, is_protected, is_forgotten,
                     forgotten_at, forgotten_reason, context, context_tags,
                     created_at, local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        episode.id,
                        self.agent_id,
                        episode.objective,
                        episode.outcome,
                        episode.outcome_type,
                        self._to_json(episode.lessons),
                        self._to_json(episode.tags),
                        episode.emotional_valence,
                        episode.emotional_arousal,
                        self._to_json(episode.emotional_tags),
                        episode.confidence,
                        episode.source_type,
                        self._to_json(episode.source_episodes),
                        self._to_json(episode.derived_from),
                        episode.last_verified.isoformat() if episode.last_verified else None,
                        episode.verification_count,
                        self._to_json(episode.confidence_history),
                        episode.times_accessed,
                        episode.last_accessed.isoformat() if episode.last_accessed else None,
                        1 if episode.is_protected else 0,
                        1 if episode.is_forgotten else 0,
                        episode.forgotten_at.isoformat() if episode.forgotten_at else None,
                        episode.forgotten_reason,
                        episode.context,
                        self._to_json(episode.context_tags),
                        episode.created_at.isoformat() if episode.created_at else now,
                        now,
                        episode.cloud_synced_at.isoformat() if episode.cloud_synced_at else None,
                        episode.version,
                        1 if episode.deleted else 0,
                    ),
                )
                episode_data = self._to_json(self._record_to_dict(episode))
                self._queue_sync(conn, "episodes", episode.id, "upsert", data=episode_data)
                content = self._get_searchable_content("episode", episode)
                self._save_embedding(conn, "episodes", episode.id, content)
            conn.commit()
        return ids

    def get_episodes(
        self, limit: int = 100, since: Optional[datetime] = None, tags: Optional[List[str]] = None
    ) -> List[Episode]:
        """Get episodes."""
        query = "SELECT * FROM episodes WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]

        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        episodes = [self._row_to_episode(row) for row in rows]

        # Filter by tags in Python (SQLite JSON support is limited)
        if tags:
            episodes = [e for e in episodes if e.tags and any(t in e.tags for t in tags)]

        return episodes

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ? AND agent_id = ?", (episode_id, self.agent_id)
            ).fetchone()

        return self._row_to_episode(row) if row else None

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert a row to an Episode."""
        return Episode(
            id=row["id"],
            agent_id=row["agent_id"],
            objective=row["objective"],
            outcome=row["outcome"],
            outcome_type=row["outcome_type"],
            lessons=self._from_json(row["lessons"]),
            tags=self._from_json(row["tags"]),
            created_at=self._parse_datetime(row["created_at"]),
            emotional_valence=(
                row["emotional_valence"] if row["emotional_valence"] is not None else 0.0
            ),
            emotional_arousal=(
                row["emotional_arousal"] if row["emotional_arousal"] is not None else 0.0
            ),
            emotional_tags=self._from_json(row["emotional_tags"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Meta-memory fields
            confidence=self._safe_get(row, "confidence", 0.8),
            source_type=self._safe_get(row, "source_type", "direct_experience"),
            source_episodes=self._from_json(self._safe_get(row, "source_episodes", None)),
            derived_from=self._from_json(self._safe_get(row, "derived_from", None)),
            last_verified=self._parse_datetime(self._safe_get(row, "last_verified", None)),
            verification_count=self._safe_get(row, "verification_count", 0),
            confidence_history=self._from_json(self._safe_get(row, "confidence_history", None)),
            # Forgetting fields
            times_accessed=self._safe_get(row, "times_accessed", 0),
            last_accessed=self._parse_datetime(self._safe_get(row, "last_accessed", None)),
            is_protected=bool(self._safe_get(row, "is_protected", 0)),
            is_forgotten=bool(self._safe_get(row, "is_forgotten", 0)),
            forgotten_at=self._parse_datetime(self._safe_get(row, "forgotten_at", None)),
            forgotten_reason=self._safe_get(row, "forgotten_reason", None),
            # Context/scope fields
            context=self._safe_get(row, "context", None),
            context_tags=self._from_json(self._safe_get(row, "context_tags", None)),
        )

    def update_episode_emotion(
        self, episode_id: str, valence: float, arousal: float, tags: Optional[List[str]] = None
    ) -> bool:
        """Update emotional associations for an episode.

        Args:
            episode_id: The episode to update
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal (0.0 to 1.0)
            tags: Emotional tags (e.g., ["joy", "excitement"])

        Returns:
            True if updated, False if episode not found
        """
        # Clamp values to valid ranges
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))

        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE episodes SET
                   emotional_valence = ?,
                   emotional_arousal = ?,
                   emotional_tags = ?,
                   local_updated_at = ?,
                   version = version + 1
                   WHERE id = ? AND agent_id = ? AND deleted = 0""",
                (valence, arousal, self._to_json(tags), now, episode_id, self.agent_id),
            )
            if cursor.rowcount > 0:
                self._queue_sync(conn, "episodes", episode_id, "upsert")
                conn.commit()
                return True
        return False

    def search_by_emotion(
        self,
        valence_range: Optional[tuple] = None,
        arousal_range: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Episode]:
        """Find episodes matching emotional criteria.

        Args:
            valence_range: (min, max) valence filter, e.g. (0.5, 1.0) for positive
            arousal_range: (min, max) arousal filter, e.g. (0.7, 1.0) for high arousal
            tags: Emotional tags to match (any match)
            limit: Maximum results

        Returns:
            List of matching episodes
        """
        query = "SELECT * FROM episodes WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]

        if valence_range:
            query += " AND emotional_valence >= ? AND emotional_valence <= ?"
            params.extend([valence_range[0], valence_range[1]])

        if arousal_range:
            query += " AND emotional_arousal >= ? AND emotional_arousal <= ?"
            params.extend([arousal_range[0], arousal_range[1]])

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit * 2 if tags else limit)  # Get more if we need to filter by tags

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        episodes = [self._row_to_episode(row) for row in rows]

        # Filter by emotional tags in Python
        if tags:
            episodes = [
                e for e in episodes if e.emotional_tags and any(t in e.emotional_tags for t in tags)
            ][:limit]

        return episodes

    def get_emotional_episodes(self, days: int = 7, limit: int = 100) -> List[Episode]:
        """Get episodes with emotional data for summary calculations.

        Args:
            days: Number of days to look back
            limit: Maximum episodes to retrieve

        Returns:
            Episodes with non-zero emotional data
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        query = """SELECT * FROM episodes
                   WHERE agent_id = ? AND deleted = 0
                   AND created_at >= ?
                   AND (emotional_valence != 0.0 OR emotional_arousal != 0.0 OR emotional_tags IS NOT NULL)
                   ORDER BY created_at DESC
                   LIMIT ?"""

        with self._connect() as conn:
            rows = conn.execute(query, (self.agent_id, cutoff, limit)).fetchall()

        return [self._row_to_episode(row) for row in rows]

    # === Beliefs ===

    def save_belief(self, belief: Belief) -> str:
        """Save a belief."""
        if not belief.id:
            belief.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO beliefs
                (id, agent_id, statement, belief_type, confidence, created_at,
                 source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 supersedes, superseded_by, times_reinforced, is_active,
                 context, context_tags,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    belief.id,
                    self.agent_id,
                    belief.statement,
                    belief.belief_type,
                    belief.confidence,
                    belief.created_at.isoformat() if belief.created_at else now,
                    belief.source_type,
                    self._to_json(belief.source_episodes),
                    self._to_json(belief.derived_from),
                    belief.last_verified.isoformat() if belief.last_verified else None,
                    belief.verification_count,
                    self._to_json(belief.confidence_history),
                    belief.supersedes,
                    belief.superseded_by,
                    belief.times_reinforced,
                    1 if belief.is_active else 0,
                    belief.context,
                    self._to_json(belief.context_tags),
                    now,
                    belief.cloud_synced_at.isoformat() if belief.cloud_synced_at else None,
                    belief.version,
                    1 if belief.deleted else 0,
                ),
            )
            # Queue for sync with record data
            belief_data = self._to_json(self._record_to_dict(belief))
            self._queue_sync(conn, "beliefs", belief.id, "upsert", data=belief_data)

            # Save embedding for search
            self._save_embedding(conn, "beliefs", belief.id, belief.statement)

            conn.commit()

        # Sync to flat file
        self._sync_beliefs_to_file()

        return belief.id

    def update_belief_atomic(self, belief: Belief, expected_version: Optional[int] = None) -> bool:
        """Update a belief with optimistic concurrency control.

        Args:
            belief: The belief with updated fields
            expected_version: The version we expect the record to have.
                             If None, uses belief.version.

        Returns:
            True if update succeeded

        Raises:
            VersionConflictError: If the record's version doesn't match expected
        """
        if expected_version is None:
            expected_version = belief.version

        now = self._now()

        with self._connect() as conn:
            # Check current version
            current = conn.execute(
                "SELECT version FROM beliefs WHERE id = ? AND agent_id = ?",
                (belief.id, self.agent_id),
            ).fetchone()

            if not current:
                return False

            current_version = current["version"]
            if current_version != expected_version:
                raise VersionConflictError("beliefs", belief.id, expected_version, current_version)

            # Atomic update with version increment
            cursor = conn.execute(
                """
                UPDATE beliefs SET
                    statement = ?,
                    belief_type = ?,
                    confidence = ?,
                    source_type = ?,
                    source_episodes = ?,
                    derived_from = ?,
                    last_verified = ?,
                    verification_count = ?,
                    confidence_history = ?,
                    supersedes = ?,
                    superseded_by = ?,
                    times_reinforced = ?,
                    is_active = ?,
                    context = ?,
                    context_tags = ?,
                    local_updated_at = ?,
                    deleted = ?,
                    version = version + 1
                WHERE id = ? AND agent_id = ? AND version = ?
                """,
                (
                    belief.statement,
                    belief.belief_type,
                    belief.confidence,
                    belief.source_type,
                    self._to_json(belief.source_episodes),
                    self._to_json(belief.derived_from),
                    belief.last_verified.isoformat() if belief.last_verified else None,
                    belief.verification_count,
                    self._to_json(belief.confidence_history),
                    belief.supersedes,
                    belief.superseded_by,
                    belief.times_reinforced,
                    1 if belief.is_active else 0,
                    belief.context,
                    self._to_json(belief.context_tags),
                    now,
                    1 if belief.deleted else 0,
                    belief.id,
                    self.agent_id,
                    expected_version,
                ),
            )

            if cursor.rowcount == 0:
                conn.rollback()
                new_current = conn.execute(
                    "SELECT version FROM beliefs WHERE id = ? AND agent_id = ?",
                    (belief.id, self.agent_id),
                ).fetchone()
                actual = new_current["version"] if new_current else -1
                raise VersionConflictError("beliefs", belief.id, expected_version, actual)

            # Queue for sync
            belief.version = expected_version + 1
            belief_data = self._to_json(self._record_to_dict(belief))
            self._queue_sync(conn, "beliefs", belief.id, "upsert", data=belief_data)

            # Update embedding
            self._save_embedding(conn, "beliefs", belief.id, belief.statement)

            conn.commit()

        # Sync to flat file
        self._sync_beliefs_to_file()

        return True

    def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]:
        """Save multiple beliefs in a single transaction."""
        if not beliefs:
            return []
        now = self._now()
        ids = []
        with self._connect() as conn:
            for belief in beliefs:
                if not belief.id:
                    belief.id = str(uuid.uuid4())
                ids.append(belief.id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO beliefs
                    (id, agent_id, statement, belief_type, confidence, created_at,
                     source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     supersedes, superseded_by, times_reinforced, is_active,
                     times_accessed, last_accessed, is_protected, is_forgotten,
                     forgotten_at, forgotten_reason, context, context_tags,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        belief.id,
                        self.agent_id,
                        belief.statement,
                        belief.belief_type,
                        belief.confidence,
                        belief.created_at.isoformat() if belief.created_at else now,
                        belief.source_type,
                        self._to_json(belief.source_episodes),
                        self._to_json(belief.derived_from),
                        belief.last_verified.isoformat() if belief.last_verified else None,
                        belief.verification_count,
                        self._to_json(belief.confidence_history),
                        belief.supersedes,
                        belief.superseded_by,
                        belief.times_reinforced,
                        1 if belief.is_active else 0,
                        belief.times_accessed,
                        belief.last_accessed.isoformat() if belief.last_accessed else None,
                        1 if belief.is_protected else 0,
                        1 if belief.is_forgotten else 0,
                        belief.forgotten_at.isoformat() if belief.forgotten_at else None,
                        belief.forgotten_reason,
                        belief.context,
                        self._to_json(belief.context_tags),
                        now,
                        belief.cloud_synced_at.isoformat() if belief.cloud_synced_at else None,
                        belief.version,
                        1 if belief.deleted else 0,
                    ),
                )
                belief_data = self._to_json(self._record_to_dict(belief))
                self._queue_sync(conn, "beliefs", belief.id, "upsert", data=belief_data)
                self._save_embedding(conn, "beliefs", belief.id, belief.statement)
            conn.commit()
        # Sync to flat file (once, after all saves)
        self._sync_beliefs_to_file()
        return ids

    def _sync_beliefs_to_file(self) -> None:
        """Write all active beliefs to flat file."""
        try:
            beliefs = self.get_beliefs(limit=500, include_inactive=False)
            lines = ["# Beliefs", f"_Last updated: {self._now()}_", ""]

            for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True):
                conf_bar = "█" * int(b.confidence * 5) + "░" * (5 - int(b.confidence * 5))
                lines.append(f"## [{conf_bar}] {int(b.confidence * 100)}% - {b.id[:8]}")
                lines.append(b.statement)
                if b.source_episodes:
                    lines.append(f"Sources: {', '.join(b.source_episodes[:3])}")
                lines.append("")

            with open(self._beliefs_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            # Secure permissions
            import os

            os.chmod(self._beliefs_file, 0o600)
        except Exception as e:
            logger.warning(f"Failed to sync beliefs to file: {e}")

    def get_beliefs(self, limit: int = 100, include_inactive: bool = False) -> List[Belief]:
        """Get beliefs.

        Args:
            limit: Maximum number of beliefs to return
            include_inactive: If True, include superseded/archived beliefs
        """
        with self._connect() as conn:
            if include_inactive:
                rows = conn.execute(
                    "SELECT * FROM beliefs WHERE agent_id = ? AND deleted = 0 ORDER BY created_at DESC LIMIT ?",
                    (self.agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM beliefs WHERE agent_id = ? AND deleted = 0 AND (is_active = 1 OR is_active IS NULL) ORDER BY created_at DESC LIMIT ?",
                    (self.agent_id, limit),
                ).fetchall()

        return [self._row_to_belief(row) for row in rows]

    def find_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE agent_id = ? AND statement = ? AND deleted = 0",
                (self.agent_id, statement),
            ).fetchone()

        return self._row_to_belief(row) if row else None

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        """Convert a row to a Belief."""
        is_active_val = self._safe_get(row, "is_active", 1)
        return Belief(
            id=row["id"],
            agent_id=row["agent_id"],
            statement=row["statement"],
            belief_type=row["belief_type"],
            confidence=row["confidence"],
            created_at=self._parse_datetime(row["created_at"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Meta-memory fields
            source_type=self._safe_get(row, "source_type", "direct_experience"),
            source_episodes=self._from_json(self._safe_get(row, "source_episodes", None)),
            derived_from=self._from_json(self._safe_get(row, "derived_from", None)),
            last_verified=self._parse_datetime(self._safe_get(row, "last_verified", None)),
            verification_count=self._safe_get(row, "verification_count", 0),
            confidence_history=self._from_json(self._safe_get(row, "confidence_history", None)),
            # Belief revision fields
            supersedes=self._safe_get(row, "supersedes", None),
            superseded_by=self._safe_get(row, "superseded_by", None),
            times_reinforced=self._safe_get(row, "times_reinforced", 0),
            is_active=bool(is_active_val) if is_active_val is not None else True,
            # Forgetting fields
            times_accessed=self._safe_get(row, "times_accessed", 0),
            last_accessed=self._parse_datetime(self._safe_get(row, "last_accessed", None)),
            is_protected=bool(self._safe_get(row, "is_protected", 0)),
            is_forgotten=bool(self._safe_get(row, "is_forgotten", 0)),
            forgotten_at=self._parse_datetime(self._safe_get(row, "forgotten_at", None)),
            forgotten_reason=self._safe_get(row, "forgotten_reason", None),
            # Context/scope fields
            context=self._safe_get(row, "context", None),
            context_tags=self._from_json(self._safe_get(row, "context_tags", None)),
        )

    # === Values ===

    def save_value(self, value: Value) -> str:
        """Save a value."""
        if not value.id:
            value.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_values
                (id, agent_id, name, statement, priority, created_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 context, context_tags,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    value.id,
                    self.agent_id,
                    value.name,
                    value.statement,
                    value.priority,
                    value.created_at.isoformat() if value.created_at else now,
                    value.confidence,
                    value.source_type,
                    self._to_json(value.source_episodes),
                    self._to_json(value.derived_from),
                    value.last_verified.isoformat() if value.last_verified else None,
                    value.verification_count,
                    self._to_json(value.confidence_history),
                    value.context,
                    self._to_json(value.context_tags),
                    now,
                    value.cloud_synced_at.isoformat() if value.cloud_synced_at else None,
                    value.version,
                    1 if value.deleted else 0,
                ),
            )
            # Queue for sync with record data
            value_data = self._to_json(self._record_to_dict(value))
            self._queue_sync(conn, "agent_values", value.id, "upsert", data=value_data)

            # Save embedding for search
            content = f"{value.name}: {value.statement}"
            self._save_embedding(conn, "agent_values", value.id, content)

            conn.commit()

        # Sync to flat file
        self._sync_values_to_file()

        return value.id

    def _sync_values_to_file(self) -> None:
        """Write all values to flat file."""
        try:
            values = self.get_values(limit=100)
            lines = ["# Values", f"_Last updated: {self._now()}_", ""]

            for v in sorted(values, key=lambda x: x.priority, reverse=True):
                lines.append(f"## {v.name} (priority: {v.priority}) - {v.id[:8]}")
                lines.append(v.statement)
                lines.append("")

            with open(self._values_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to sync values to file: {e}")

    def get_values(self, limit: int = 100) -> List[Value]:
        """Get values ordered by priority."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_values WHERE agent_id = ? AND deleted = 0 ORDER BY priority DESC LIMIT ?",
                (self.agent_id, limit),
            ).fetchall()

        return [self._row_to_value(row) for row in rows]

    def _row_to_value(self, row: sqlite3.Row) -> Value:
        """Convert a row to a Value."""
        return Value(
            id=row["id"],
            agent_id=row["agent_id"],
            name=row["name"],
            statement=row["statement"],
            priority=row["priority"],
            created_at=self._parse_datetime(row["created_at"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Meta-memory fields
            confidence=self._safe_get(row, "confidence", 0.9),
            source_type=self._safe_get(row, "source_type", "direct_experience"),
            source_episodes=self._from_json(self._safe_get(row, "source_episodes", None)),
            derived_from=self._from_json(self._safe_get(row, "derived_from", None)),
            last_verified=self._parse_datetime(self._safe_get(row, "last_verified", None)),
            verification_count=self._safe_get(row, "verification_count", 0),
            confidence_history=self._from_json(self._safe_get(row, "confidence_history", None)),
            # Forgetting fields (values protected by default)
            times_accessed=self._safe_get(row, "times_accessed", 0),
            last_accessed=self._parse_datetime(self._safe_get(row, "last_accessed", None)),
            is_protected=bool(self._safe_get(row, "is_protected", 1)),  # Default protected
            is_forgotten=bool(self._safe_get(row, "is_forgotten", 0)),
            forgotten_at=self._parse_datetime(self._safe_get(row, "forgotten_at", None)),
            forgotten_reason=self._safe_get(row, "forgotten_reason", None),
            # Context/scope fields
            context=self._safe_get(row, "context", None),
            context_tags=self._from_json(self._safe_get(row, "context_tags", None)),
        )

    # === Goals ===

    def save_goal(self, goal: Goal) -> str:
        """Save a goal."""
        if not goal.id:
            goal.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO goals
                (id, agent_id, title, description, priority, status, created_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 context, context_tags,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    goal.id,
                    self.agent_id,
                    goal.title,
                    goal.description,
                    goal.priority,
                    goal.status,
                    goal.created_at.isoformat() if goal.created_at else now,
                    goal.confidence,
                    goal.source_type,
                    self._to_json(goal.source_episodes),
                    self._to_json(goal.derived_from),
                    goal.last_verified.isoformat() if goal.last_verified else None,
                    goal.verification_count,
                    self._to_json(goal.confidence_history),
                    goal.context,
                    self._to_json(goal.context_tags),
                    now,
                    goal.cloud_synced_at.isoformat() if goal.cloud_synced_at else None,
                    goal.version,
                    1 if goal.deleted else 0,
                ),
            )
            # Queue for sync with record data
            goal_data = self._to_json(self._record_to_dict(goal))
            self._queue_sync(conn, "goals", goal.id, "upsert", data=goal_data)

            # Save embedding for search
            content = f"{goal.title} {goal.description or ''}"
            self._save_embedding(conn, "goals", goal.id, content)

            conn.commit()

        # Sync to flat file
        self._sync_goals_to_file()

        return goal.id

    def _sync_goals_to_file(self) -> None:
        """Write all active goals to flat file."""
        try:
            goals = self.get_goals(status=None, limit=100)  # All goals
            lines = ["# Goals", f"_Last updated: {self._now()}_", ""]

            # Group by status
            for status in ["active", "completed", "paused"]:
                status_goals = [g for g in goals if g.status == status]
                if status_goals:
                    lines.append(f"## {status.title()}")
                    for g in sorted(status_goals, key=lambda x: x.priority or "", reverse=True):
                        priority = f" [{g.priority}]" if g.priority else ""
                        status_icon = (
                            "○" if status == "active" else "✓" if status == "completed" else "⏸"
                        )
                        lines.append(f"- {status_icon} {g.title}{priority} ({g.id[:8]})")
                        if g.description:
                            lines.append(f"  {g.description[:100]}")
                    lines.append("")

            with open(self._goals_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to sync goals to file: {e}")

    def get_goals(self, status: Optional[str] = "active", limit: int = 100) -> List[Goal]:
        """Get goals."""
        query = "SELECT * FROM goals WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_goal(row) for row in rows]

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        """Convert a row to a Goal."""
        return Goal(
            id=row["id"],
            agent_id=row["agent_id"],
            title=row["title"],
            description=row["description"],
            priority=row["priority"],
            status=row["status"],
            created_at=self._parse_datetime(row["created_at"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Meta-memory fields
            confidence=self._safe_get(row, "confidence", 0.8),
            source_type=self._safe_get(row, "source_type", "direct_experience"),
            source_episodes=self._from_json(self._safe_get(row, "source_episodes", None)),
            derived_from=self._from_json(self._safe_get(row, "derived_from", None)),
            last_verified=self._parse_datetime(self._safe_get(row, "last_verified", None)),
            verification_count=self._safe_get(row, "verification_count", 0),
            confidence_history=self._from_json(self._safe_get(row, "confidence_history", None)),
            # Forgetting fields
            times_accessed=self._safe_get(row, "times_accessed", 0),
            last_accessed=self._parse_datetime(self._safe_get(row, "last_accessed", None)),
            is_protected=bool(self._safe_get(row, "is_protected", 0)),
            is_forgotten=bool(self._safe_get(row, "is_forgotten", 0)),
            forgotten_at=self._parse_datetime(self._safe_get(row, "forgotten_at", None)),
            forgotten_reason=self._safe_get(row, "forgotten_reason", None),
            # Context/scope fields
            context=self._safe_get(row, "context", None),
            context_tags=self._from_json(self._safe_get(row, "context_tags", None)),
        )

    # === Notes ===

    def save_note(self, note: Note) -> str:
        """Save a note."""
        if not note.id:
            note.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notes
                (id, agent_id, content, note_type, speaker, reason, tags, created_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 context, context_tags,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    note.id,
                    self.agent_id,
                    note.content,
                    note.note_type,
                    note.speaker,
                    note.reason,
                    self._to_json(note.tags),
                    note.created_at.isoformat() if note.created_at else now,
                    note.confidence,
                    note.source_type,
                    self._to_json(note.source_episodes),
                    self._to_json(note.derived_from),
                    note.last_verified.isoformat() if note.last_verified else None,
                    note.verification_count,
                    self._to_json(note.confidence_history),
                    note.context,
                    self._to_json(note.context_tags),
                    now,
                    note.cloud_synced_at.isoformat() if note.cloud_synced_at else None,
                    note.version,
                    1 if note.deleted else 0,
                ),
            )
            # Queue for sync with record data
            note_data = self._to_json(self._record_to_dict(note))
            self._queue_sync(conn, "notes", note.id, "upsert", data=note_data)

            # Save embedding for search
            self._save_embedding(conn, "notes", note.id, note.content)

            conn.commit()

        return note.id

    def save_notes_batch(self, notes: List[Note]) -> List[str]:
        """Save multiple notes in a single transaction."""
        if not notes:
            return []
        now = self._now()
        ids = []
        with self._connect() as conn:
            for note in notes:
                if not note.id:
                    note.id = str(uuid.uuid4())
                ids.append(note.id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO notes
                    (id, agent_id, content, note_type, speaker, reason, tags, created_at,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     times_accessed, last_accessed, is_protected, is_forgotten,
                     forgotten_at, forgotten_reason, context, context_tags,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        note.id,
                        self.agent_id,
                        note.content,
                        note.note_type,
                        note.speaker,
                        note.reason,
                        self._to_json(note.tags),
                        note.created_at.isoformat() if note.created_at else now,
                        note.confidence,
                        note.source_type,
                        self._to_json(note.source_episodes),
                        self._to_json(note.derived_from),
                        note.last_verified.isoformat() if note.last_verified else None,
                        note.verification_count,
                        self._to_json(note.confidence_history),
                        note.times_accessed,
                        note.last_accessed.isoformat() if note.last_accessed else None,
                        1 if note.is_protected else 0,
                        1 if note.is_forgotten else 0,
                        note.forgotten_at.isoformat() if note.forgotten_at else None,
                        note.forgotten_reason,
                        note.context,
                        self._to_json(note.context_tags),
                        now,
                        note.cloud_synced_at.isoformat() if note.cloud_synced_at else None,
                        note.version,
                        1 if note.deleted else 0,
                    ),
                )
                note_data = self._to_json(self._record_to_dict(note))
                self._queue_sync(conn, "notes", note.id, "upsert", data=note_data)
                self._save_embedding(conn, "notes", note.id, note.content)
            conn.commit()
        return ids

    def get_notes(
        self, limit: int = 100, since: Optional[datetime] = None, note_type: Optional[str] = None
    ) -> List[Note]:
        """Get notes."""
        query = "SELECT * FROM notes WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]

        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        if note_type:
            query += " AND note_type = ?"
            params.append(note_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_note(row) for row in rows]

    def _row_to_note(self, row: sqlite3.Row) -> Note:
        """Convert a row to a Note."""
        return Note(
            id=row["id"],
            agent_id=row["agent_id"],
            content=row["content"],
            note_type=row["note_type"],
            speaker=row["speaker"],
            reason=row["reason"],
            tags=self._from_json(row["tags"]),
            created_at=self._parse_datetime(row["created_at"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Meta-memory fields
            confidence=self._safe_get(row, "confidence", 0.8),
            source_type=self._safe_get(row, "source_type", "direct_experience"),
            source_episodes=self._from_json(self._safe_get(row, "source_episodes", None)),
            derived_from=self._from_json(self._safe_get(row, "derived_from", None)),
            last_verified=self._parse_datetime(self._safe_get(row, "last_verified", None)),
            verification_count=self._safe_get(row, "verification_count", 0),
            confidence_history=self._from_json(self._safe_get(row, "confidence_history", None)),
            # Forgetting fields
            times_accessed=self._safe_get(row, "times_accessed", 0),
            last_accessed=self._parse_datetime(self._safe_get(row, "last_accessed", None)),
            is_protected=bool(self._safe_get(row, "is_protected", 0)),
            is_forgotten=bool(self._safe_get(row, "is_forgotten", 0)),
            forgotten_at=self._parse_datetime(self._safe_get(row, "forgotten_at", None)),
            forgotten_reason=self._safe_get(row, "forgotten_reason", None),
            # Context/scope fields
            context=self._safe_get(row, "context", None),
            context_tags=self._from_json(self._safe_get(row, "context_tags", None)),
        )

    # === Drives ===

    def save_drive(self, drive: Drive) -> str:
        """Save or update a drive."""
        if not drive.id:
            drive.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM drives WHERE agent_id = ? AND drive_type = ?",
                (self.agent_id, drive.drive_type),
            ).fetchone()

            if existing:
                drive.id = existing["id"]
                conn.execute(
                    """
                    UPDATE drives SET
                        intensity = ?, focus_areas = ?, updated_at = ?,
                        confidence = ?, source_type = ?, source_episodes = ?,
                        derived_from = ?, last_verified = ?, verification_count = ?,
                        confidence_history = ?, context = ?, context_tags = ?,
                        local_updated_at = ?, version = version + 1
                    WHERE id = ?
                """,
                    (
                        drive.intensity,
                        self._to_json(drive.focus_areas),
                        now,
                        drive.confidence,
                        drive.source_type,
                        self._to_json(drive.source_episodes),
                        self._to_json(drive.derived_from),
                        drive.last_verified.isoformat() if drive.last_verified else None,
                        drive.verification_count,
                        self._to_json(drive.confidence_history),
                        drive.context,
                        self._to_json(drive.context_tags),
                        now,
                        drive.id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO drives
                    (id, agent_id, drive_type, intensity, focus_areas, created_at, updated_at,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     context, context_tags,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        drive.id,
                        self.agent_id,
                        drive.drive_type,
                        drive.intensity,
                        self._to_json(drive.focus_areas),
                        now,
                        now,
                        drive.confidence,
                        drive.source_type,
                        self._to_json(drive.source_episodes),
                        self._to_json(drive.derived_from),
                        drive.last_verified.isoformat() if drive.last_verified else None,
                        drive.verification_count,
                        self._to_json(drive.confidence_history),
                        drive.context,
                        self._to_json(drive.context_tags),
                        now,
                        None,
                        1,
                        0,
                    ),
                )

            # Queue for sync with record data
            drive_data = self._to_json(self._record_to_dict(drive))
            self._queue_sync(conn, "drives", drive.id, "upsert", data=drive_data)
            conn.commit()

        return drive.id

    def get_drives(self) -> List[Drive]:
        """Get all drives."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM drives WHERE agent_id = ? AND deleted = 0", (self.agent_id,)
            ).fetchall()

        return [self._row_to_drive(row) for row in rows]

    def get_drive(self, drive_type: str) -> Optional[Drive]:
        """Get a specific drive."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM drives WHERE agent_id = ? AND drive_type = ? AND deleted = 0",
                (self.agent_id, drive_type),
            ).fetchone()

        return self._row_to_drive(row) if row else None

    def _row_to_drive(self, row: sqlite3.Row) -> Drive:
        """Convert a row to a Drive."""
        return Drive(
            id=row["id"],
            agent_id=row["agent_id"],
            drive_type=row["drive_type"],
            intensity=row["intensity"],
            focus_areas=self._from_json(row["focus_areas"]),
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Meta-memory fields
            confidence=self._safe_get(row, "confidence", 0.8),
            source_type=self._safe_get(row, "source_type", "direct_experience"),
            source_episodes=self._from_json(self._safe_get(row, "source_episodes", None)),
            derived_from=self._from_json(self._safe_get(row, "derived_from", None)),
            last_verified=self._parse_datetime(self._safe_get(row, "last_verified", None)),
            verification_count=self._safe_get(row, "verification_count", 0),
            confidence_history=self._from_json(self._safe_get(row, "confidence_history", None)),
            # Forgetting fields (drives protected by default)
            times_accessed=self._safe_get(row, "times_accessed", 0),
            last_accessed=self._parse_datetime(self._safe_get(row, "last_accessed", None)),
            is_protected=bool(self._safe_get(row, "is_protected", 1)),  # Default protected
            is_forgotten=bool(self._safe_get(row, "is_forgotten", 0)),
            forgotten_at=self._parse_datetime(self._safe_get(row, "forgotten_at", None)),
            forgotten_reason=self._safe_get(row, "forgotten_reason", None),
            # Context/scope fields
            context=self._safe_get(row, "context", None),
            context_tags=self._from_json(self._safe_get(row, "context_tags", None)),
        )

    # === Relationships ===

    def save_relationship(self, relationship: Relationship) -> str:
        """Save or update a relationship."""
        if not relationship.id:
            relationship.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM relationships WHERE agent_id = ? AND entity_name = ?",
                (self.agent_id, relationship.entity_name),
            ).fetchone()

            if existing:
                relationship.id = existing["id"]
                conn.execute(
                    """
                    UPDATE relationships SET
                        entity_type = ?, relationship_type = ?, notes = ?,
                        sentiment = ?, interaction_count = ?, last_interaction = ?,
                        confidence = ?, source_type = ?, source_episodes = ?,
                        derived_from = ?, last_verified = ?, verification_count = ?,
                        confidence_history = ?, context = ?, context_tags = ?,
                        local_updated_at = ?, version = version + 1
                    WHERE id = ?
                """,
                    (
                        relationship.entity_type,
                        relationship.relationship_type,
                        relationship.notes,
                        relationship.sentiment,
                        relationship.interaction_count,
                        (
                            relationship.last_interaction.isoformat()
                            if relationship.last_interaction
                            else None
                        ),
                        relationship.confidence,
                        relationship.source_type,
                        self._to_json(relationship.source_episodes),
                        self._to_json(relationship.derived_from),
                        (
                            relationship.last_verified.isoformat()
                            if relationship.last_verified
                            else None
                        ),
                        relationship.verification_count,
                        self._to_json(relationship.confidence_history),
                        relationship.context,
                        self._to_json(relationship.context_tags),
                        now,
                        relationship.id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO relationships
                    (id, agent_id, entity_name, entity_type, relationship_type, notes,
                     sentiment, interaction_count, last_interaction, created_at,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     context, context_tags,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        relationship.id,
                        self.agent_id,
                        relationship.entity_name,
                        relationship.entity_type,
                        relationship.relationship_type,
                        relationship.notes,
                        relationship.sentiment,
                        relationship.interaction_count,
                        (
                            relationship.last_interaction.isoformat()
                            if relationship.last_interaction
                            else None
                        ),
                        now,
                        relationship.confidence,
                        relationship.source_type,
                        self._to_json(relationship.source_episodes),
                        self._to_json(relationship.derived_from),
                        (
                            relationship.last_verified.isoformat()
                            if relationship.last_verified
                            else None
                        ),
                        relationship.verification_count,
                        self._to_json(relationship.confidence_history),
                        relationship.context,
                        self._to_json(relationship.context_tags),
                        now,
                        None,
                        1,
                        0,
                    ),
                )

            # Queue for sync with record data
            relationship_data = self._to_json(self._record_to_dict(relationship))
            self._queue_sync(
                conn, "relationships", relationship.id, "upsert", data=relationship_data
            )
            conn.commit()

        # Sync to flat file
        self._sync_relationships_to_file()

        return relationship.id

    def _sync_relationships_to_file(self) -> None:
        """Write all relationships to flat file."""
        try:
            relationships = self.get_relationships()
            lines = ["# Relationships", f"_Last updated: {self._now()}_", ""]

            for r in sorted(relationships, key=lambda x: x.sentiment, reverse=True):
                trust_pct = int(((r.sentiment + 1) / 2) * 100)
                trust_bar = "█" * (trust_pct // 10) + "░" * (10 - trust_pct // 10)
                lines.append(f"## {r.entity_name} ({r.entity_type}) - {r.id[:8]}")
                lines.append(f"Trust: [{trust_bar}] {trust_pct}%")
                lines.append(f"Interactions: {r.interaction_count}")
                if r.last_interaction:
                    lines.append(f"Last: {r.last_interaction.strftime('%Y-%m-%d')}")
                if r.notes:
                    lines.append(f"Notes: {r.notes}")
                lines.append("")

            with open(self._relationships_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to sync relationships to file: {e}")

    def get_relationships(self, entity_type: Optional[str] = None) -> List[Relationship]:
        """Get relationships."""
        query = "SELECT * FROM relationships WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_relationship(row) for row in rows]

    def get_relationship(self, entity_name: str) -> Optional[Relationship]:
        """Get a specific relationship."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationships WHERE agent_id = ? AND entity_name = ? AND deleted = 0",
                (self.agent_id, entity_name),
            ).fetchone()

        return self._row_to_relationship(row) if row else None

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        """Convert a row to a Relationship."""
        return Relationship(
            id=row["id"],
            agent_id=row["agent_id"],
            entity_name=row["entity_name"],
            entity_type=row["entity_type"],
            relationship_type=row["relationship_type"],
            notes=row["notes"],
            sentiment=row["sentiment"],
            interaction_count=row["interaction_count"],
            last_interaction=self._parse_datetime(row["last_interaction"]),
            created_at=self._parse_datetime(row["created_at"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Meta-memory fields
            confidence=self._safe_get(row, "confidence", 0.8),
            source_type=self._safe_get(row, "source_type", "direct_experience"),
            source_episodes=self._from_json(self._safe_get(row, "source_episodes", None)),
            derived_from=self._from_json(self._safe_get(row, "derived_from", None)),
            last_verified=self._parse_datetime(self._safe_get(row, "last_verified", None)),
            verification_count=self._safe_get(row, "verification_count", 0),
            confidence_history=self._from_json(self._safe_get(row, "confidence_history", None)),
            # Forgetting fields
            times_accessed=self._safe_get(row, "times_accessed", 0),
            last_accessed=self._parse_datetime(self._safe_get(row, "last_accessed", None)),
            is_protected=bool(self._safe_get(row, "is_protected", 0)),
            is_forgotten=bool(self._safe_get(row, "is_forgotten", 0)),
            forgotten_at=self._parse_datetime(self._safe_get(row, "forgotten_at", None)),
            forgotten_reason=self._safe_get(row, "forgotten_reason", None),
            # Context/scope fields
            context=self._safe_get(row, "context", None),
            context_tags=self._from_json(self._safe_get(row, "context_tags", None)),
        )

    # === Playbooks (Procedural Memory) ===

    def save_playbook(self, playbook: Playbook) -> str:
        """Save a playbook. Returns the playbook ID."""
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO playbooks
                (id, agent_id, name, description, trigger_conditions, steps, failure_modes,
                 recovery_steps, mastery_level, times_used, success_rate, source_episodes, tags,
                 confidence, last_used, created_at, local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    playbook.id,
                    self.agent_id,
                    playbook.name,
                    playbook.description,
                    self._to_json(playbook.trigger_conditions),
                    self._to_json(playbook.steps),
                    self._to_json(playbook.failure_modes),
                    self._to_json(playbook.recovery_steps),
                    playbook.mastery_level,
                    playbook.times_used,
                    playbook.success_rate,
                    self._to_json(playbook.source_episodes),
                    self._to_json(playbook.tags),
                    playbook.confidence,
                    playbook.last_used.isoformat() if playbook.last_used else None,
                    playbook.created_at.isoformat() if playbook.created_at else now,
                    now,
                    None,  # cloud_synced_at
                    playbook.version,
                    0,  # deleted
                ),
            )

            # Queue for sync with record data
            playbook_data = self._to_json(self._record_to_dict(playbook))
            self._queue_sync(conn, "playbooks", playbook.id, "upsert", data=playbook_data)

            # Add embedding for search
            content = (
                f"{playbook.name} {playbook.description} {' '.join(playbook.trigger_conditions)}"
            )
            self._save_embedding(conn, "playbooks", playbook.id, content)

            conn.commit()

        return playbook.id

    def get_playbook(self, playbook_id: str) -> Optional[Playbook]:
        """Get a specific playbook by ID."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM playbooks WHERE id = ? AND agent_id = ? AND deleted = 0",
                (playbook_id, self.agent_id),
            )
            row = cur.fetchone()

        return self._row_to_playbook(row) if row else None

    def list_playbooks(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Playbook]:
        """Get playbooks, optionally filtered by tags."""
        with self._connect() as conn:
            query = """
                SELECT * FROM playbooks
                WHERE agent_id = ? AND deleted = 0
                ORDER BY times_used DESC, created_at DESC
                LIMIT ?
            """
            cur = conn.execute(query, (self.agent_id, limit))
            rows = cur.fetchall()

        playbooks = [self._row_to_playbook(row) for row in rows]

        # Filter by tags if provided
        if tags:
            tags_set = set(tags)
            playbooks = [p for p in playbooks if p.tags and tags_set.intersection(p.tags)]

        return playbooks

    def search_playbooks(self, query: str, limit: int = 10) -> List[Playbook]:
        """Search playbooks by name, description, or triggers using semantic search."""
        if self._has_vec:
            # Use vector search
            embedding = self._embedder.embed(query)
            packed = pack_embedding(embedding)

            with self._connect() as conn:
                cur = conn.execute(
                    """
                    SELECT e.id, e.embedding, distance
                    FROM vec_embeddings e
                    WHERE e.id LIKE 'playbooks:%'
                    ORDER BY distance
                    LIMIT ?
                """.replace("distance", f"vec_distance_L2(e.embedding, X'{packed.hex()}')"),
                    (limit * 2,),
                )

                vec_results = cur.fetchall()

            playbook_ids = [r[0].replace("playbooks:", "") for r in vec_results]

            playbooks = []
            for pid in playbook_ids:
                playbook = self.get_playbook(pid)
                if playbook:
                    playbooks.append(playbook)
                if len(playbooks) >= limit:
                    break

            return playbooks
        else:
            # Fall back to text search
            with self._connect() as conn:
                search_pattern = f"%{query}%"
                cur = conn.execute(
                    """
                    SELECT * FROM playbooks
                    WHERE agent_id = ? AND deleted = 0
                    AND (name LIKE ? OR description LIKE ? OR trigger_conditions LIKE ?)
                    ORDER BY times_used DESC
                    LIMIT ?
                """,
                    (self.agent_id, search_pattern, search_pattern, search_pattern, limit),
                )
                rows = cur.fetchall()

            return [self._row_to_playbook(row) for row in rows]

    def update_playbook_usage(self, playbook_id: str, success: bool) -> bool:
        """Update playbook usage statistics."""
        playbook = self.get_playbook(playbook_id)
        if not playbook:
            return False

        now = self._now()

        # Calculate new success rate
        new_times_used = playbook.times_used + 1
        if playbook.times_used == 0:
            new_success_rate = 1.0 if success else 0.0
        else:
            # Running average
            total_successes = playbook.success_rate * playbook.times_used
            total_successes += 1.0 if success else 0.0
            new_success_rate = total_successes / new_times_used

        # Update mastery level based on usage and success rate
        new_mastery = playbook.mastery_level
        if new_times_used >= 20 and new_success_rate >= 0.9:
            new_mastery = "expert"
        elif new_times_used >= 10 and new_success_rate >= 0.8:
            new_mastery = "proficient"
        elif new_times_used >= 5 and new_success_rate >= 0.7:
            new_mastery = "competent"

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE playbooks SET
                    times_used = ?,
                    success_rate = ?,
                    mastery_level = ?,
                    last_used = ?,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND agent_id = ?
            """,
                (
                    new_times_used,
                    new_success_rate,
                    new_mastery,
                    now,
                    now,
                    playbook_id,
                    self.agent_id,
                ),
            )

            self._queue_sync(conn, "playbooks", playbook_id, "upsert")
            conn.commit()

        return True

    def _row_to_playbook(self, row: sqlite3.Row) -> Playbook:
        """Convert a row to a Playbook."""
        return Playbook(
            id=row["id"],
            agent_id=row["agent_id"],
            name=row["name"],
            description=row["description"],
            trigger_conditions=self._from_json(row["trigger_conditions"]) or [],
            steps=self._from_json(row["steps"]) or [],
            failure_modes=self._from_json(row["failure_modes"]) or [],
            recovery_steps=self._from_json(row["recovery_steps"]),
            mastery_level=row["mastery_level"],
            times_used=row["times_used"],
            success_rate=row["success_rate"],
            source_episodes=self._from_json(row["source_episodes"]),
            tags=self._from_json(row["tags"]),
            confidence=self._safe_get(row, "confidence", 0.8),
            last_used=self._parse_datetime(self._safe_get(row, "last_used", None)),
            created_at=self._parse_datetime(row["created_at"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
        )

    # === Raw Entries ===

    def save_raw(
        self,
        blob: Optional[str] = None,
        source: str = "unknown",
        # DEPRECATED parameters - kept for backward compatibility
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Save a raw entry for later processing.

        The raw layer is designed for zero-friction brain dumps. The agent dumps
        whatever they want into the blob field; the system only tracks housekeeping
        metadata.

        Args:
            blob: The raw brain dump content (primary parameter).
            source: Auto-populated source identifier (cli|mcp|sdk|import|unknown).

        Deprecated Args:
            content: Use blob instead. Will be removed in future version.
            tags: Include tags in blob text instead. Will be removed in future version.

        Returns:
            The raw entry ID.
        """
        import warnings

        # Handle deprecated parameters
        if content is not None:
            warnings.warn(
                "The 'content' parameter is deprecated. Use 'blob' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if blob is None:
                blob = content

        if tags is not None:
            warnings.warn(
                "The 'tags' parameter is deprecated. Include tags in blob text instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if blob is None:
            raise ValueError("blob parameter is required")

        # Normalize source to valid enum values
        valid_sources = {"cli", "mcp", "sdk", "import", "unknown"}
        if source == "manual":
            source = "cli"
        elif source not in valid_sources:
            if "auto" in source.lower():
                source = "sdk"
            else:
                source = "unknown"

        # Size warnings (don't reject, let anxiety system handle)
        blob_size = len(blob.encode("utf-8"))
        if blob_size > 50 * 1024 * 1024:  # 50MB - reject
            raise ValueError(
                f"Raw entry too large ({blob_size / 1024 / 1024:.1f}MB). "
                "Consider breaking into smaller chunks or processing immediately."
            )
        elif blob_size > 10 * 1024 * 1024:  # 10MB
            logger.warning(f"Extremely large raw entry ({blob_size / 1024 / 1024:.1f}MB)")
        elif blob_size > 1 * 1024 * 1024:  # 1MB
            logger.warning(f"Very large raw entry ({blob_size / 1024:.0f}KB) - consider processing")
        elif blob_size > 100 * 1024:  # 100KB
            logger.info(f"Large raw entry ({blob_size / 1024:.0f}KB)")

        raw_id = str(uuid.uuid4())
        now = self._now()

        # 1. Write to flat file (blob acts as flat file content)
        self._append_raw_to_file(raw_id, blob, now, source, tags)

        # 2. Index in SQLite with both blob and legacy columns for compatibility
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO raw_entries
                (id, agent_id, blob, captured_at, source, processed, processed_into,
                 content, timestamp, tags, confidence, source_type,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    raw_id,
                    self.agent_id,
                    blob,  # Primary blob field
                    now,  # captured_at
                    source,
                    0,  # processed = False
                    None,  # processed_into
                    blob,  # Legacy content field (same as blob for new entries)
                    now,  # Legacy timestamp field (same as captured_at)
                    self._to_json(tags),  # Legacy tags field
                    1.0,  # confidence (deprecated)
                    "direct_experience",  # source_type (deprecated)
                    now,
                    None,
                    1,
                    0,
                ),
            )

            # Update FTS index for keyword search
            self._update_raw_fts(conn, raw_id, blob)

            # Queue for sync (if raw sync is enabled - off by default)
            if self._should_sync_raw():
                raw_data = self._to_json(
                    {
                        "id": raw_id,
                        "agent_id": self.agent_id,
                        "blob": blob,
                        "captured_at": now,
                        "source": source,
                        "processed": False,
                    }
                )
                self._queue_sync(conn, "raw_entries", raw_id, "upsert", data=raw_data)

            # Save embedding for search (on blob content)
            self._save_embedding(conn, "raw_entries", raw_id, blob)

            conn.commit()

        return raw_id

    def _should_sync_raw(self) -> bool:
        """Check if raw entries should be synced to cloud.

        Raw sync is OFF by default for security (raw blobs often contain
        accidental secrets). Users must explicitly enable it.
        """
        import os

        # Check environment variable
        raw_sync_env = os.environ.get("KERNLE_RAW_SYNC", "").lower()
        if raw_sync_env in ("true", "1", "yes", "on"):
            return True
        if raw_sync_env in ("false", "0", "no", "off"):
            return False

        # Check config file
        config_path = Path.home() / ".kernle" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    return config.get("sync", {}).get("raw", False)
            except (json.JSONDecodeError, OSError):
                pass

        # Default: OFF for security
        return False

    def _update_raw_fts(self, conn: sqlite3.Connection, raw_id: str, blob: str):
        """Update FTS5 index for a raw entry."""
        try:
            # Get rowid for the entry
            result = conn.execute(
                "SELECT rowid FROM raw_entries WHERE id = ?", (raw_id,)
            ).fetchone()
            if result:
                rowid = result[0]
                # Insert into FTS index
                conn.execute("INSERT INTO raw_fts(rowid, blob) VALUES (?, ?)", (rowid, blob))
        except sqlite3.OperationalError as e:
            # FTS5 might not be available
            if "no such table" not in str(e).lower():
                logger.debug(f"FTS update failed: {e}")

    def _append_raw_to_file(
        self,
        raw_id: str,
        content: str,
        timestamp: str,
        source: str,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Append a raw entry to the daily flat file.

        File format (greppable, human-readable):
        ```
        ## HH:MM:SS [id_prefix] source
        Content goes here
        Tags: tag1, tag2

        ```
        """
        try:
            # Parse date from timestamp for filename
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")

            daily_file = self._raw_dir / f"{date_str}.md"

            # Build entry
            lines = []
            lines.append(f"## {time_str} [{raw_id[:8]}] {source}")
            lines.append(content)
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")
            lines.append("")  # Blank line separator

            # Append to file
            with open(daily_file, "a", encoding="utf-8") as f:
                # Add header if new file
                if daily_file.stat().st_size == 0 if daily_file.exists() else True:
                    f.write(f"# Raw Captures - {date_str}\n\n")
                f.write("\n".join(lines) + "\n")

        except Exception as e:
            logger.warning(f"Failed to write raw entry to flat file: {e}")
            # Don't fail - SQLite is the backup

    def get_raw_dir(self) -> Path:
        """Get the path to the raw flat files directory."""
        return self._raw_dir

    def get_raw_files(self) -> List[Path]:
        """Get list of raw flat files, sorted by date descending."""
        if not self._raw_dir.exists():
            return []
        files = sorted(self._raw_dir.glob("*.md"), reverse=True)
        return files

    def sync_raw_from_files(self) -> Dict[str, Any]:
        """Sync raw entries from flat files into SQLite.

        Parses flat files and imports any entries not already in SQLite.
        This enables bidirectional editing - add entries via vim, then sync.

        Returns:
            Dict with imported_count, skipped_count, errors
        """
        import re

        result = {
            "imported": 0,
            "skipped": 0,
            "errors": [],
            "files_processed": 0,
        }

        files = self.get_raw_files()

        # Get existing IDs for quick lookup
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM raw_entries WHERE agent_id = ?", (self.agent_id,)
            ).fetchall()
        existing_ids = {row["id"] for row in rows}

        # Pattern to match entry headers: ## HH:MM:SS [id_prefix] source
        # ID can be alphanumeric (for manually created entries)
        header_pattern = re.compile(r"^## (\d{2}:\d{2}:\d{2}) \[([a-zA-Z0-9]+)\] ([\w-]+)$")

        for file_path in files:
            result["files_processed"] += 1
            try:
                # Extract date from filename (2026-01-28.md)
                date_str = file_path.stem  # "2026-01-28"

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Split into entries
                lines = content.split("\n")
                current_entry = None

                for line in lines:
                    header_match = header_pattern.match(line)

                    if header_match:
                        # Save previous entry if exists
                        if current_entry and current_entry.get("content_lines"):
                            current_entry["content"] = "\n".join(current_entry["content_lines"])
                            self._import_raw_entry(current_entry, existing_ids, result)

                        # Start new entry
                        time_str, id_prefix, source = header_match.groups()
                        current_entry = {
                            "id_prefix": id_prefix,
                            "timestamp": f"{date_str}T{time_str}",
                            "source": source,
                            "content_lines": [],
                            "tags": None,
                        }
                    elif current_entry is not None:
                        # Check for tags line
                        if line.startswith("Tags: "):
                            current_entry["tags"] = [t.strip() for t in line[6:].split(",")]
                        elif line.startswith("# Raw Captures"):
                            pass  # Skip header
                        elif line.strip():  # Non-empty content
                            current_entry["content_lines"].append(line)

                # Don't forget last entry
                if current_entry and current_entry.get("content_lines"):
                    current_entry["content"] = "\n".join(current_entry["content_lines"])
                    self._import_raw_entry(current_entry, existing_ids, result)

            except Exception as e:
                result["errors"].append(f"{file_path.name}: {str(e)}")

        return result

    def _import_raw_entry(
        self,
        entry: Dict[str, Any],
        existing_ids: set,
        result: Dict[str, Any],
    ) -> None:
        """Import a single raw entry if not already in SQLite."""
        # Check if any existing ID starts with this prefix
        id_prefix = entry.get("id_prefix", "")
        matching_ids = [eid for eid in existing_ids if eid.startswith(id_prefix)]

        if matching_ids:
            result["skipped"] += 1
            return

        # Generate new ID (use prefix + random suffix to maintain traceability)
        new_id = id_prefix + str(uuid.uuid4())[8:]  # Keep prefix, new suffix
        content = "\n".join(entry.get("content_lines", []))

        if not content.strip():
            result["skipped"] += 1
            return

        try:
            now = self._now()
            timestamp = entry.get("timestamp", now)

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO raw_entries
                    (id, agent_id, content, timestamp, source, processed, processed_into, tags,
                     confidence, source_type, local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        new_id,
                        self.agent_id,
                        content,
                        timestamp,
                        entry.get("source", "file-sync"),
                        0,
                        None,
                        self._to_json(entry.get("tags")),
                        1.0,
                        "file_import",
                        now,
                        None,
                        1,
                        0,
                    ),
                )
                self._save_embedding(conn, "raw_entries", new_id, content)
                conn.commit()

            existing_ids.add(new_id)
            result["imported"] += 1

        except Exception as e:
            result["errors"].append(f"Entry {id_prefix}: {str(e)}")

    def get_raw(self, raw_id: str) -> Optional[RawEntry]:
        """Get a specific raw entry by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM raw_entries WHERE id = ? AND agent_id = ? AND deleted = 0",
                (raw_id, self.agent_id),
            ).fetchone()

        return self._row_to_raw_entry(row) if row else None

    def list_raw(self, processed: Optional[bool] = None, limit: int = 100) -> List[RawEntry]:
        """Get raw entries, optionally filtered by processed state."""
        query = "SELECT * FROM raw_entries WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]

        if processed is not None:
            query += " AND processed = ?"
            params.append(1 if processed else 0)

        # Use COALESCE to handle both new (captured_at) and legacy (timestamp) schemas
        query += " ORDER BY COALESCE(captured_at, timestamp) DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_raw_entry(row) for row in rows]

    def search_raw_fts(self, query: str, limit: int = 50) -> List[RawEntry]:
        """Search raw entries using FTS5 keyword search.

        This is a safety net for when backlogs accumulate. For semantic search,
        use the regular search() method instead.

        Args:
            query: FTS5 search query (supports AND, OR, NOT, phrases in quotes)
            limit: Maximum number of results

        Returns:
            List of matching RawEntry objects, ordered by relevance.
        """
        with self._connect() as conn:
            try:
                # FTS5 MATCH query with relevance ranking
                rows = conn.execute(
                    """
                    SELECT r.* FROM raw_entries r
                    JOIN raw_fts f ON r.rowid = f.rowid
                    WHERE r.agent_id = ? AND r.deleted = 0
                    AND raw_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (self.agent_id, query, limit),
                ).fetchall()
                return [self._row_to_raw_entry(row) for row in rows]
            except sqlite3.OperationalError as e:
                # FTS5 not available, fall back to LIKE search
                if "no such table" in str(e).lower() or "fts5" in str(e).lower():
                    logger.debug("FTS5 not available, using LIKE fallback")
                    # Escape LIKE pattern special characters to prevent pattern injection
                    escaped_query = self._escape_like_pattern(query)
                    # Use COALESCE to search both blob and content fields
                    rows = conn.execute(
                        """
                        SELECT * FROM raw_entries
                        WHERE agent_id = ? AND deleted = 0
                        AND (COALESCE(blob, content, '') LIKE ? ESCAPE '\\')
                        ORDER BY COALESCE(captured_at, timestamp) DESC
                        LIMIT ?
                        """,
                        (self.agent_id, f"%{escaped_query}%", limit),
                    ).fetchall()
                    return [self._row_to_raw_entry(row) for row in rows]
                raise

    def _escape_like_pattern(self, pattern: str) -> str:
        """Escape LIKE pattern special characters to prevent pattern injection.

        LIKE pattern metacharacters (%, _, [) can be exploited to alter search
        behavior unexpectedly or cause performance issues.
        """
        return pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def mark_raw_processed(self, raw_id: str, processed_into: List[str]) -> bool:
        """Mark a raw entry as processed into other memories."""
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE raw_entries SET
                    processed = 1,
                    processed_into = ?,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND agent_id = ? AND deleted = 0
            """,
                (self._to_json(processed_into), now, raw_id, self.agent_id),
            )
            if cursor.rowcount > 0:
                self._queue_sync(conn, "raw_entries", raw_id, "upsert")
                conn.commit()
                return True
        return False

    def delete_raw(self, raw_id: str) -> bool:
        """Delete a raw entry (soft delete by marking deleted=1)."""
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE raw_entries SET
                    deleted = 1,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND agent_id = ? AND deleted = 0
            """,
                (now, raw_id, self.agent_id),
            )
            if cursor.rowcount > 0:
                self._queue_sync(conn, "raw_entries", raw_id, "delete")
                conn.commit()
                return True
        return False

    def _row_to_raw_entry(self, row: sqlite3.Row) -> RawEntry:
        """Convert a row to a RawEntry.

        Handles both new (blob/captured_at) and legacy (content/timestamp) schemas.
        """
        # Get blob - prefer blob field, fall back to content for legacy data
        blob = self._safe_get(row, "blob", None) or self._safe_get(row, "content", "")

        # Get captured_at - prefer captured_at, fall back to timestamp for legacy data
        captured_at_str = self._safe_get(row, "captured_at", None) or self._safe_get(
            row, "timestamp", None
        )
        captured_at = self._parse_datetime(captured_at_str)

        return RawEntry(
            id=row["id"],
            agent_id=row["agent_id"],
            blob=blob,
            captured_at=captured_at,
            source=row["source"],
            processed=bool(row["processed"]),
            processed_into=self._from_json(row["processed_into"]),
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
            # Legacy fields (deprecated)
            content=self._safe_get(row, "content", None),
            timestamp=self._parse_datetime(self._safe_get(row, "timestamp", None)),
            tags=self._from_json(self._safe_get(row, "tags", None)),
            confidence=self._safe_get(row, "confidence", 1.0),
            source_type=self._safe_get(row, "source_type", "direct_experience"),
        )

    # === Memory Suggestions ===

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        """Save a memory suggestion. Returns the suggestion ID."""
        suggestion_id = suggestion.id or str(uuid.uuid4())
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_suggestions
                (id, agent_id, memory_type, content, confidence, source_raw_ids,
                 status, created_at, resolved_at, resolution_reason, promoted_to,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    suggestion_id,
                    self.agent_id,
                    suggestion.memory_type,
                    self._to_json(suggestion.content),
                    suggestion.confidence,
                    self._to_json(suggestion.source_raw_ids),
                    suggestion.status,
                    suggestion.created_at.isoformat() if suggestion.created_at else now,
                    suggestion.resolved_at.isoformat() if suggestion.resolved_at else None,
                    suggestion.resolution_reason,
                    suggestion.promoted_to,
                    now,
                    None,
                    suggestion.version,
                    0,
                ),
            )
            self._queue_sync(conn, "memory_suggestions", suggestion_id, "upsert")
            conn.commit()

        return suggestion_id

    def get_suggestion(self, suggestion_id: str) -> Optional[MemorySuggestion]:
        """Get a specific suggestion by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_suggestions WHERE id = ? AND agent_id = ? AND deleted = 0",
                (suggestion_id, self.agent_id),
            ).fetchone()

        return self._row_to_suggestion(row) if row else None

    def get_suggestions(
        self,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemorySuggestion]:
        """Get suggestions, optionally filtered."""
        query = "SELECT * FROM memory_suggestions WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]

        if status is not None:
            query += " AND status = ?"
            params.append(status)

        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_suggestion(row) for row in rows]

    def update_suggestion_status(
        self,
        suggestion_id: str,
        status: str,
        resolution_reason: Optional[str] = None,
        promoted_to: Optional[str] = None,
    ) -> bool:
        """Update the status of a suggestion."""
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_suggestions SET
                    status = ?,
                    resolved_at = ?,
                    resolution_reason = ?,
                    promoted_to = ?,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND agent_id = ? AND deleted = 0
            """,
                (status, now, resolution_reason, promoted_to, now, suggestion_id, self.agent_id),
            )
            if cursor.rowcount > 0:
                self._queue_sync(conn, "memory_suggestions", suggestion_id, "upsert")
                conn.commit()
                return True
        return False

    def delete_suggestion(self, suggestion_id: str) -> bool:
        """Delete a suggestion (soft delete)."""
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_suggestions SET
                    deleted = 1,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND agent_id = ? AND deleted = 0
            """,
                (now, suggestion_id, self.agent_id),
            )
            if cursor.rowcount > 0:
                self._queue_sync(conn, "memory_suggestions", suggestion_id, "delete")
                conn.commit()
                return True
        return False

    def _row_to_suggestion(self, row: sqlite3.Row) -> MemorySuggestion:
        """Convert a row to a MemorySuggestion."""
        return MemorySuggestion(
            id=row["id"],
            agent_id=row["agent_id"],
            memory_type=row["memory_type"],
            content=self._from_json(row["content"]) or {},
            confidence=row["confidence"],
            source_raw_ids=self._from_json(row["source_raw_ids"]) or [],
            status=row["status"],
            created_at=self._parse_datetime(row["created_at"]),
            resolved_at=self._parse_datetime(row["resolved_at"]),
            resolution_reason=row["resolution_reason"],
            promoted_to=row["promoted_to"],
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"]),
        )

    # === Search ===

    def search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        prefer_cloud: bool = True,
    ) -> List[SearchResult]:
        """Search across memories using hybrid cloud/local strategy.

        Strategy:
        1. If cloud credentials are configured and prefer_cloud=True,
           try cloud search first with 3s timeout
        2. On cloud failure or no credentials, fall back to local search
        3. Local search uses sqlite-vec (if available) or text matching

        Args:
            query: Search query string
            limit: Maximum results to return
            record_types: Filter by memory type (episode, note, belief, value, goal)
            prefer_cloud: If True, try cloud search first (default True)

        Returns:
            List of SearchResult objects
        """
        types = record_types or ["episode", "note", "belief", "value", "goal"]

        # Try cloud search first if configured and preferred
        if prefer_cloud and self.has_cloud_credentials():
            cloud_results = self._cloud_search(query, limit, types)
            if cloud_results is not None:
                logger.debug(f"Cloud search returned {len(cloud_results)} results")
                return cloud_results
            logger.debug("Cloud search failed, falling back to local search")

        # Fall back to local search
        return self._local_search(query, limit, types)

    def _local_search(self, query: str, limit: int, types: List[str]) -> List[SearchResult]:
        """Local search using sqlite-vec or text matching.

        Args:
            query: Search query
            limit: Maximum results
            types: Memory types to search

        Returns:
            List of SearchResult
        """
        if self._has_vec:
            return self._vector_search(query, limit, types)
        else:
            return self._text_search(query, limit, types)

    def _vector_search(self, query: str, limit: int, types: List[str]) -> List[SearchResult]:
        """Semantic search using sqlite-vec."""
        results = []

        # Embed query
        query_embedding = self._embedder.embed(query)
        query_packed = pack_embedding(query_embedding)

        # Map types to table names
        table_map = {
            "episode": "episodes",
            "note": "notes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
        }

        with self._connect() as conn:
            # Build table prefix filter
            table_prefixes = [table_map[t] for t in types if t in table_map]

            # Query vector table for nearest neighbors
            # Use KNN search with sqlite-vec
            try:
                rows = conn.execute(
                    """SELECT id, distance
                       FROM vec_embeddings
                       WHERE embedding MATCH ?
                       ORDER BY distance
                       LIMIT ?""",
                    (query_packed, limit * 2),  # Get more to filter by type
                ).fetchall()
            except Exception as e:
                logger.warning(f"Vector search failed: {e}, falling back to text search")
                return self._text_search(query, limit, types)

            # Fetch actual records
            for row in rows:
                vec_id = row["id"]
                distance = row["distance"]

                # Parse table:record_id format
                if ":" not in vec_id:
                    continue
                table_name, record_id = vec_id.split(":", 1)

                # Filter by requested types
                if table_name not in table_prefixes:
                    continue

                # Convert distance to similarity score (lower distance = higher score)
                # For cosine distance, range is [0, 2], so we normalize
                score = max(0.0, 1.0 - distance / 2.0)

                # Fetch the actual record
                record, record_type = self._fetch_record(conn, table_name, record_id)
                if record:
                    results.append(
                        SearchResult(record=record, record_type=record_type, score=score)
                    )

                if len(results) >= limit:
                    break

        return results

    def _fetch_record(self, conn: sqlite3.Connection, table: str, record_id: str) -> tuple:
        """Fetch a record by table and ID."""
        type_map = {
            "episodes": ("episode", self._row_to_episode),
            "notes": ("note", self._row_to_note),
            "beliefs": ("belief", self._row_to_belief),
            "agent_values": ("value", self._row_to_value),
            "goals": ("goal", self._row_to_goal),
        }

        if table not in type_map:
            return None, None

        record_type, converter = type_map[table]
        validate_table_name(table)  # Security: validate before SQL use

        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND agent_id = ? AND deleted = 0",
            (record_id, self.agent_id),
        ).fetchone()

        if row:
            return converter(row), record_type
        return None, None

    def _text_search(self, query: str, limit: int, types: List[str]) -> List[SearchResult]:
        """Fallback text-based search using LIKE."""
        results = []
        search_term = f"%{query}%"

        with self._connect() as conn:
            if "episode" in types:
                rows = conn.execute(
                    """SELECT * FROM episodes
                       WHERE agent_id = ? AND deleted = 0
                       AND (objective LIKE ? OR outcome LIKE ? OR lessons LIKE ?)
                       LIMIT ?""",
                    (self.agent_id, search_term, search_term, search_term, limit),
                ).fetchall()
                for row in rows:
                    results.append(
                        SearchResult(
                            record=self._row_to_episode(row), record_type="episode", score=1.0
                        )
                    )

            if "note" in types:
                rows = conn.execute(
                    """SELECT * FROM notes
                       WHERE agent_id = ? AND deleted = 0
                       AND content LIKE ?
                       LIMIT ?""",
                    (self.agent_id, search_term, limit),
                ).fetchall()
                for row in rows:
                    results.append(
                        SearchResult(record=self._row_to_note(row), record_type="note", score=1.0)
                    )

            if "belief" in types:
                rows = conn.execute(
                    """SELECT * FROM beliefs
                       WHERE agent_id = ? AND deleted = 0
                       AND statement LIKE ?
                       LIMIT ?""",
                    (self.agent_id, search_term, limit),
                ).fetchall()
                for row in rows:
                    results.append(
                        SearchResult(
                            record=self._row_to_belief(row), record_type="belief", score=1.0
                        )
                    )

            if "value" in types:
                rows = conn.execute(
                    """SELECT * FROM agent_values
                       WHERE agent_id = ? AND deleted = 0
                       AND (name LIKE ? OR statement LIKE ?)
                       LIMIT ?""",
                    (self.agent_id, search_term, search_term, limit),
                ).fetchall()
                for row in rows:
                    results.append(
                        SearchResult(record=self._row_to_value(row), record_type="value", score=1.0)
                    )

            if "goal" in types:
                rows = conn.execute(
                    """SELECT * FROM goals
                       WHERE agent_id = ? AND deleted = 0
                       AND (title LIKE ? OR description LIKE ?)
                       LIMIT ?""",
                    (self.agent_id, search_term, search_term, limit),
                ).fetchall()
                for row in rows:
                    results.append(
                        SearchResult(record=self._row_to_goal(row), record_type="goal", score=1.0)
                    )

        return results[:limit]

    # === Stats ===

    def get_stats(self) -> Dict[str, int]:
        """Get counts of each record type."""
        with self._connect() as conn:
            stats = {}
            for table, key in [
                ("episodes", "episodes"),
                ("beliefs", "beliefs"),
                ("agent_values", "values"),
                ("goals", "goals"),
                ("notes", "notes"),
                ("drives", "drives"),
                ("relationships", "relationships"),
                ("raw_entries", "raw"),
                ("memory_suggestions", "suggestions"),
            ]:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE agent_id = ? AND deleted = 0",
                    (self.agent_id,),
                ).fetchone()[0]
                stats[key] = count

            # Add count of pending suggestions specifically
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM memory_suggestions WHERE agent_id = ? AND status = 'pending' AND deleted = 0",
                (self.agent_id,),
            ).fetchone()[0]
            stats["pending_suggestions"] = pending_count

        return stats

    # === Batch Loading ===

    def load_all(
        self,
        values_limit: Optional[int] = 10,
        beliefs_limit: Optional[int] = 20,
        goals_limit: Optional[int] = 10,
        goals_status: str = "active",
        episodes_limit: Optional[int] = 20,
        notes_limit: Optional[int] = 5,
        drives_limit: Optional[int] = None,
        relationships_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load all memory types in a single database connection.

        This optimizes the common pattern of loading working memory context
        by batching all queries into a single connection, avoiding N+1 query
        patterns where each memory type requires a separate connection.

        Args:
            values_limit: Max values to load (None = 1000 for budget loading)
            beliefs_limit: Max beliefs to load (None = 1000 for budget loading)
            goals_limit: Max goals to load (None = 1000 for budget loading)
            goals_status: Goal status filter ("active", "all", etc.)
            episodes_limit: Max episodes to load (None = 1000 for budget loading)
            notes_limit: Max notes to load (None = 1000 for budget loading)
            drives_limit: Max drives to load (None = all drives)
            relationships_limit: Max relationships to load (None = all relationships)

        Returns:
            Dict with keys: values, beliefs, goals, drives, episodes, notes, relationships
        """
        # Use high limit (1000) when None is passed - for budget-based loading
        high_limit = 1000
        _values_limit = values_limit if values_limit is not None else high_limit
        _beliefs_limit = beliefs_limit if beliefs_limit is not None else high_limit
        _goals_limit = goals_limit if goals_limit is not None else high_limit
        _episodes_limit = episodes_limit if episodes_limit is not None else high_limit
        _notes_limit = notes_limit if notes_limit is not None else high_limit

        result = {
            "values": [],
            "beliefs": [],
            "goals": [],
            "drives": [],
            "episodes": [],
            "notes": [],
            "relationships": [],
        }

        with self._connect() as conn:
            # Values - ordered by priority, exclude forgotten
            rows = conn.execute(
                "SELECT * FROM agent_values WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 ORDER BY priority DESC LIMIT ?",
                (self.agent_id, _values_limit),
            ).fetchall()
            result["values"] = [self._row_to_value(row) for row in rows]

            # Beliefs - ordered by confidence, exclude forgotten
            rows = conn.execute(
                "SELECT * FROM beliefs WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 AND (is_active = 1 OR is_active IS NULL) ORDER BY confidence DESC LIMIT ?",
                (self.agent_id, _beliefs_limit),
            ).fetchall()
            result["beliefs"] = [self._row_to_belief(row) for row in rows]

            # Goals - filtered by status, exclude forgotten
            if goals_status and goals_status != "all":
                rows = conn.execute(
                    "SELECT * FROM goals WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 AND status = ? ORDER BY created_at DESC LIMIT ?",
                    (self.agent_id, goals_status, _goals_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM goals WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 ORDER BY created_at DESC LIMIT ?",
                    (self.agent_id, _goals_limit),
                ).fetchall()
            result["goals"] = [self._row_to_goal(row) for row in rows]

            # Drives - all for agent (or limited), exclude forgotten
            if drives_limit is not None:
                rows = conn.execute(
                    "SELECT * FROM drives WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 LIMIT ?",
                    (self.agent_id, drives_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM drives WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0",
                    (self.agent_id,),
                ).fetchall()
            result["drives"] = [self._row_to_drive(row) for row in rows]

            # Episodes - most recent, exclude forgotten
            rows = conn.execute(
                "SELECT * FROM episodes WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 ORDER BY created_at DESC LIMIT ?",
                (self.agent_id, _episodes_limit),
            ).fetchall()
            result["episodes"] = [self._row_to_episode(row) for row in rows]

            # Notes - most recent, exclude forgotten
            rows = conn.execute(
                "SELECT * FROM notes WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 ORDER BY created_at DESC LIMIT ?",
                (self.agent_id, _notes_limit),
            ).fetchall()
            result["notes"] = [self._row_to_note(row) for row in rows]

            # Relationships - all for agent (or limited), exclude forgotten
            if relationships_limit is not None:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0 LIMIT ?",
                    (self.agent_id, relationships_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE agent_id = ? AND deleted = 0 AND is_forgotten = 0",
                    (self.agent_id,),
                ).fetchall()
            result["relationships"] = [self._row_to_relationship(row) for row in rows]

        return result

    # === Meta-Memory ===

    def get_memory(self, memory_type: str, memory_id: str) -> Optional[Any]:
        """Get a memory by type and ID.

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship)
            memory_id: ID of the memory

        Returns:
            The memory record or None if not found
        """
        getters = {
            "episode": lambda: self.get_episode(memory_id),
            "belief": lambda: self._get_belief_by_id(memory_id),
            "value": lambda: self._get_value_by_id(memory_id),
            "goal": lambda: self._get_goal_by_id(memory_id),
            "note": lambda: self._get_note_by_id(memory_id),
            "drive": lambda: self._get_drive_by_id(memory_id),
            "relationship": lambda: self._get_relationship_by_id(memory_id),
        }

        getter = getters.get(memory_type)
        return getter() if getter else None

    def _get_belief_by_id(self, belief_id: str) -> Optional[Belief]:
        """Get a belief by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE id = ? AND agent_id = ? AND deleted = 0",
                (belief_id, self.agent_id),
            ).fetchone()
        return self._row_to_belief(row) if row else None

    def _get_value_by_id(self, value_id: str) -> Optional[Value]:
        """Get a value by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ? AND agent_id = ? AND deleted = 0",
                (value_id, self.agent_id),
            ).fetchone()
        return self._row_to_value(row) if row else None

    def _get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ? AND agent_id = ? AND deleted = 0",
                (goal_id, self.agent_id),
            ).fetchone()
        return self._row_to_goal(row) if row else None

    def _get_note_by_id(self, note_id: str) -> Optional[Note]:
        """Get a note by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ? AND agent_id = ? AND deleted = 0",
                (note_id, self.agent_id),
            ).fetchone()
        return self._row_to_note(row) if row else None

    def _get_drive_by_id(self, drive_id: str) -> Optional[Drive]:
        """Get a drive by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM drives WHERE id = ? AND agent_id = ? AND deleted = 0",
                (drive_id, self.agent_id),
            ).fetchone()
        return self._row_to_drive(row) if row else None

    def _get_relationship_by_id(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationships WHERE id = ? AND agent_id = ? AND deleted = 0",
                (relationship_id, self.agent_id),
            ).fetchone()
        return self._row_to_relationship(row) if row else None

    def update_memory_meta(
        self,
        memory_type: str,
        memory_id: str,
        confidence: Optional[float] = None,
        source_type: Optional[str] = None,
        source_episodes: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
        last_verified: Optional[datetime] = None,
        verification_count: Optional[int] = None,
        confidence_history: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Update meta-memory fields for a memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            confidence: New confidence value
            source_type: New source type
            source_episodes: New source episodes list
            derived_from: New derived_from list
            last_verified: New verification timestamp
            verification_count: New verification count
            confidence_history: New confidence history

        Returns:
            True if updated, False if memory not found
        """
        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        # Build update query dynamically
        updates = []
        params = []

        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if source_type is not None:
            updates.append("source_type = ?")
            params.append(source_type)
        if source_episodes is not None:
            updates.append("source_episodes = ?")
            params.append(self._to_json(source_episodes))
        if derived_from is not None:
            updates.append("derived_from = ?")
            params.append(self._to_json(derived_from))
        if last_verified is not None:
            updates.append("last_verified = ?")
            params.append(last_verified.isoformat())
        if verification_count is not None:
            updates.append("verification_count = ?")
            params.append(verification_count)
        if confidence_history is not None:
            updates.append("confidence_history = ?")
            params.append(self._to_json(confidence_history))

        if not updates:
            return False

        # Also update local_updated_at
        updates.append("local_updated_at = ?")
        params.append(self._now())

        # Add version increment
        updates.append("version = version + 1")

        # Add WHERE clause params
        params.extend([memory_id, self.agent_id])

        query = (
            f"UPDATE {table} SET {', '.join(updates)} WHERE id = ? AND agent_id = ? AND deleted = 0"
        )

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            if cursor.rowcount > 0:
                self._queue_sync(conn, table, memory_id, "upsert")
                conn.commit()
                return True
        return False

    def get_memories_by_confidence(
        self,
        threshold: float,
        below: bool = True,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by confidence threshold.

        Args:
            threshold: Confidence threshold
            below: If True, get memories below threshold; if False, above
            memory_types: Filter by type (episode, belief, etc.)
            limit: Maximum results

        Returns:
            List of matching memories with their types
        """
        results = []
        op = "<" if below else ">="
        types = memory_types or [
            "episode",
            "belief",
            "value",
            "goal",
            "note",
            "drive",
            "relationship",
        ]

        table_map = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "drive": ("drives", self._row_to_drive),
            "relationship": ("relationships", self._row_to_relationship),
        }

        with self._connect() as conn:
            for memory_type in types:
                if memory_type not in table_map:
                    continue

                table, converter = table_map[memory_type]
                validate_table_name(table)  # Security: validate before SQL use
                query = f"""
                    SELECT * FROM {table}
                    WHERE agent_id = ? AND deleted = 0
                    AND confidence {op} ?
                    ORDER BY confidence {"ASC" if below else "DESC"}
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.agent_id, threshold, limit)).fetchall()
                    for row in rows:
                        results.append(
                            SearchResult(
                                record=converter(row),
                                record_type=memory_type,
                                score=self._safe_get(row, "confidence", 0.8),
                            )
                        )
                except Exception as e:
                    # Column might not exist in old schema
                    logger.debug(f"Could not query {table} by confidence: {e}")

        # Sort by confidence
        results.sort(key=lambda x: x.score, reverse=not below)
        return results[:limit]

    def get_memories_by_source(
        self,
        source_type: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by source type.

        Args:
            source_type: Source type to filter by
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of matching memories
        """
        results = []
        types = memory_types or [
            "episode",
            "belief",
            "value",
            "goal",
            "note",
            "drive",
            "relationship",
        ]

        table_map = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "drive": ("drives", self._row_to_drive),
            "relationship": ("relationships", self._row_to_relationship),
        }

        with self._connect() as conn:
            for memory_type in types:
                if memory_type not in table_map:
                    continue

                table, converter = table_map[memory_type]
                validate_table_name(table)  # Security: validate before SQL use
                query = f"""
                    SELECT * FROM {table}
                    WHERE agent_id = ? AND deleted = 0
                    AND source_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.agent_id, source_type, limit)).fetchall()
                    for row in rows:
                        results.append(
                            SearchResult(
                                record=converter(row),
                                record_type=memory_type,
                                score=self._safe_get(row, "confidence", 0.8),
                            )
                        )
                except Exception as e:
                    # Column might not exist in old schema
                    logger.debug(f"Could not query {table} by source_type: {e}")

        return results[:limit]

    # === Forgetting ===

    def record_access(self, memory_type: str, memory_id: str) -> bool:
        """Record that a memory was accessed (for salience tracking).

        Increments times_accessed and updates last_accessed timestamp.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if updated, False if memory not found
        """
        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET times_accessed = COALESCE(times_accessed, 0) + 1,
                       last_accessed = ?,
                       local_updated_at = ?
                   WHERE id = ? AND agent_id = ?""",
                (now, now, memory_id, self.agent_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def record_access_batch(self, accesses: List[tuple[str, str]]) -> int:
        """Record multiple memory accesses in a single transaction.

        This is more efficient than calling record_access() for each item
        because it uses a single database connection and transaction.

        Args:
            accesses: List of (memory_type, memory_id) tuples

        Returns:
            Number of memories successfully updated
        """
        if not accesses:
            return 0

        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        # Group by table for efficient batch updates
        by_table: Dict[str, List[str]] = {}
        for memory_type, memory_id in accesses:
            table = table_map.get(memory_type)
            if table:
                if table not in by_table:
                    by_table[table] = []
                by_table[table].append(memory_id)

        if not by_table:
            return 0

        now = self._now()
        total_updated = 0

        with self._connect() as conn:
            for table, ids in by_table.items():
                # Validate table name for safety
                validate_table_name(table)

                # Update all IDs in this table at once
                placeholders = ",".join("?" * len(ids))
                cursor = conn.execute(
                    f"""UPDATE {table}
                       SET times_accessed = COALESCE(times_accessed, 0) + 1,
                           last_accessed = ?,
                           local_updated_at = ?
                       WHERE id IN ({placeholders}) AND agent_id = ?""",
                    (now, now, *ids, self.agent_id),
                )
                total_updated += cursor.rowcount

            conn.commit()

        return total_updated

    def forget_memory(
        self,
        memory_type: str,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Tombstone a memory (mark as forgotten, don't delete).

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            reason: Optional reason for forgetting

        Returns:
            True if forgotten, False if not found, already forgotten, or protected
        """
        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False
        validate_table_name(table)  # Security: validate before SQL use

        now = self._now()

        with self._connect() as conn:
            # Check if memory exists and is not protected
            row = conn.execute(
                f"SELECT is_protected, is_forgotten FROM {table} WHERE id = ? AND agent_id = ?",
                (memory_id, self.agent_id),
            ).fetchone()

            if not row:
                return False

            if self._safe_get(row, "is_protected", 0):
                logger.debug(f"Cannot forget protected memory {memory_type}:{memory_id}")
                return False

            if self._safe_get(row, "is_forgotten", 0):
                return False  # Already forgotten

            cursor = conn.execute(
                f"""UPDATE {table}
                   SET is_forgotten = 1,
                       forgotten_at = ?,
                       forgotten_reason = ?,
                       local_updated_at = ?
                   WHERE id = ? AND agent_id = ?""",
                (now, reason, now, memory_id, self.agent_id),
            )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if recovered, False if not found or not forgotten
        """
        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        now = self._now()

        with self._connect() as conn:
            # Check if memory is forgotten
            row = conn.execute(
                f"SELECT is_forgotten FROM {table} WHERE id = ? AND agent_id = ?",
                (memory_id, self.agent_id),
            ).fetchone()

            if not row or not self._safe_get(row, "is_forgotten", 0):
                return False

            cursor = conn.execute(
                f"""UPDATE {table}
                   SET is_forgotten = 0,
                       forgotten_at = NULL,
                       forgotten_reason = NULL,
                       local_updated_at = ?
                   WHERE id = ? AND agent_id = ?""",
                (now, memory_id, self.agent_id),
            )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def protect_memory(self, memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark a memory as protected from forgetting.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            protected: True to protect, False to unprotect

        Returns:
            True if updated, False if memory not found
        """
        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET is_protected = ?,
                       local_updated_at = ?
                   WHERE id = ? AND agent_id = ?""",
                (1 if protected else 0, now, memory_id, self.agent_id),
            )
            self._queue_sync(conn, table, memory_id, "update")
            conn.commit()
            return cursor.rowcount > 0

    def get_forgetting_candidates(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories that are candidates for forgetting.

        Returns memories that are:
        - Not protected
        - Not already forgotten
        - Sorted by computed salience (lowest first)

        Salience formula:
        salience = (confidence × reinforcement_weight) / (age_factor + 1)
        where:
            reinforcement_weight = log(times_accessed + 1)
            age_factor = days_since_last_access / half_life (30 days)

        Args:
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of candidate memories with computed salience scores
        """
        import math

        results = []
        types = memory_types or ["episode", "belief", "goal", "note", "relationship"]
        # Exclude values and drives by default since they're protected by default

        table_map = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "drive": ("drives", self._row_to_drive),
            "relationship": ("relationships", self._row_to_relationship),
        }

        now = datetime.now(timezone.utc)
        half_life = 30.0  # days

        with self._connect() as conn:
            for memory_type in types:
                if memory_type not in table_map:
                    continue

                table, converter = table_map[memory_type]
                query = f"""
                    SELECT * FROM {table}
                    WHERE agent_id = ?
                    AND deleted = 0
                    AND COALESCE(is_protected, 0) = 0
                    AND COALESCE(is_forgotten, 0) = 0
                    ORDER BY created_at ASC
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.agent_id, limit * 2)).fetchall()
                    for row in rows:
                        record = converter(row)

                        # Calculate salience
                        confidence = self._safe_get(row, "confidence", 0.8)
                        times_accessed = self._safe_get(row, "times_accessed", 0) or 0

                        # Get last access time
                        last_accessed_str = self._safe_get(row, "last_accessed", None)
                        if last_accessed_str:
                            last_accessed = self._parse_datetime(last_accessed_str)
                        else:
                            # Use created_at if never accessed
                            created_at_str = row["created_at"]
                            last_accessed = self._parse_datetime(created_at_str)

                        # Calculate age factor
                        if last_accessed:
                            days_since = (now - last_accessed).total_seconds() / 86400
                        else:
                            days_since = 365  # Very old if unknown

                        age_factor = days_since / half_life
                        reinforcement_weight = math.log(times_accessed + 1)

                        # Salience calculation
                        salience = (confidence * (reinforcement_weight + 0.1)) / (age_factor + 1)

                        results.append(
                            SearchResult(record=record, record_type=memory_type, score=salience)
                        )
                except Exception as e:
                    logger.debug(f"Could not get forgetting candidates from {table}: {e}")

        # Sort by salience (lowest first = best candidates for forgetting)
        results.sort(key=lambda x: x.score)
        return results[:limit]

    def get_forgotten_memories(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get all forgotten (tombstoned) memories.

        Args:
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of forgotten memories
        """
        results = []
        types = memory_types or [
            "episode",
            "belief",
            "value",
            "goal",
            "note",
            "drive",
            "relationship",
        ]

        table_map = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "drive": ("drives", self._row_to_drive),
            "relationship": ("relationships", self._row_to_relationship),
        }

        with self._connect() as conn:
            for memory_type in types:
                if memory_type not in table_map:
                    continue

                table, converter = table_map[memory_type]
                query = f"""
                    SELECT * FROM {table}
                    WHERE agent_id = ?
                    AND deleted = 0
                    AND COALESCE(is_forgotten, 0) = 1
                    ORDER BY forgotten_at DESC
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.agent_id, limit)).fetchall()
                    for row in rows:
                        results.append(
                            SearchResult(
                                record=converter(row),
                                record_type=memory_type,
                                score=0.0,  # Forgotten memories have 0 active salience
                            )
                        )
                except Exception as e:
                    logger.debug(f"Could not get forgotten memories from {table}: {e}")

        return results[:limit]

    # === Sync Queue ===

    def queue_sync_operation(
        self, operation: str, table: str, record_id: str, data: Optional[Dict[str, Any]] = None
    ) -> int:
        """Queue a sync operation for later synchronization.

        Uses atomic UPSERT (INSERT ... ON CONFLICT DO UPDATE) to prevent
        race conditions between concurrent writes to the same record.

        Args:
            operation: Operation type ('insert', 'update', 'delete')
            table: Table name (e.g., 'episodes', 'notes')
            record_id: ID of the record
            data: Optional JSON-serializable data payload

        Returns:
            The queue entry ID (or 0 if upsert updated existing row)
        """
        now = self._now()
        data_json = self._to_json(data) if data else None

        with self._connect() as conn:
            # Use atomic UPSERT to prevent race condition between SELECT and UPDATE/INSERT
            # This requires a unique partial index on (table_name, record_id) where synced = 0
            cursor = conn.execute(
                """INSERT INTO sync_queue
                   (table_name, record_id, operation, data, local_updated_at, synced, queued_at)
                   VALUES (?, ?, ?, ?, ?, 0, ?)
                   ON CONFLICT(table_name, record_id) WHERE synced = 0
                   DO UPDATE SET
                       operation = excluded.operation,
                       data = excluded.data,
                       local_updated_at = excluded.local_updated_at,
                       queued_at = excluded.queued_at""",
                (table, record_id, operation, data_json, now, now),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_pending_sync_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all unsynced operations from the queue.

        Returns:
            List of dicts with: id, operation, table_name, record_id, data, local_updated_at
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, operation, table_name, record_id, data, local_updated_at
                   FROM sync_queue
                   WHERE synced = 0
                   ORDER BY id
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "operation": row["operation"],
                "table_name": row["table_name"],
                "record_id": row["record_id"],
                "data": self._from_json(row["data"]) if row["data"] else None,
                "local_updated_at": self._parse_datetime(row["local_updated_at"]),
            }
            for row in rows
        ]

    def mark_synced(self, ids: List[int]) -> int:
        """Mark sync queue entries as synced.

        Args:
            ids: List of sync queue entry IDs to mark as synced

        Returns:
            Number of entries marked as synced
        """
        if not ids:
            return 0

        with self._connect() as conn:
            placeholders = ",".join("?" * len(ids))
            cursor = conn.execute(
                f"UPDATE sync_queue SET synced = 1 WHERE id IN ({placeholders})", ids
            )
            conn.commit()
            return cursor.rowcount

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync queue status with counts.

        Returns:
            Dict with:
                - pending: count of unsynced operations
                - synced: count of synced operations
                - total: total queue entries
                - by_table: breakdown by table name
                - by_operation: breakdown by operation type
        """
        with self._connect() as conn:
            # Overall counts
            pending = conn.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 0").fetchone()[0]
            synced = conn.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 1").fetchone()[0]

            # Breakdown by table (pending only)
            table_rows = conn.execute("""SELECT table_name, COUNT(*) as count
                   FROM sync_queue WHERE synced = 0
                   GROUP BY table_name""").fetchall()
            by_table = {row["table_name"]: row["count"] for row in table_rows}

            # Breakdown by operation (pending only)
            op_rows = conn.execute("""SELECT operation, COUNT(*) as count
                   FROM sync_queue WHERE synced = 0
                   GROUP BY operation""").fetchall()
            by_operation = {row["operation"]: row["count"] for row in op_rows}

        return {
            "pending": pending,
            "synced": synced,
            "total": pending + synced,
            "by_table": by_table,
            "by_operation": by_operation,
        }

    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 0").fetchone()[0]
        return count

    def get_queued_changes(self, limit: int = 100) -> List[QueuedChange]:
        """Get queued changes for sync (legacy method, returns unsynced items)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, operation, payload, queued_at
                   FROM sync_queue
                   WHERE synced = 0
                   ORDER BY id
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            QueuedChange(
                id=row["id"],
                table_name=row["table_name"],
                record_id=row["record_id"],
                operation=row["operation"],
                payload=row["payload"],
                queued_at=self._parse_datetime(row["queued_at"]),
            )
            for row in rows
        ]

    def _clear_queued_change(self, conn: sqlite3.Connection, queue_id: int):
        """Mark a change as synced (legacy behavior kept for compatibility)."""
        conn.execute("UPDATE sync_queue SET synced = 1 WHERE id = ?", (queue_id,))

    def _get_sync_meta(self, key: str) -> Optional[str]:
        """Get a sync metadata value."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM sync_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_sync_meta(self, key: str, value: str):
        """Set a sync metadata value."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sync_meta (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, self._now()),
            )
            conn.commit()

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful sync."""
        value = self._get_sync_meta("last_sync_time")
        return self._parse_datetime(value) if value else None

    def get_sync_conflicts(self, limit: int = 100) -> List[SyncConflict]:
        """Get recent sync conflict history.

        Args:
            limit: Maximum number of conflicts to return

        Returns:
            List of SyncConflict records, most recent first
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, local_version, cloud_version,
                          resolution, resolved_at, local_summary, cloud_summary
                   FROM sync_conflicts
                   ORDER BY resolved_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            SyncConflict(
                id=row["id"],
                table=row["table_name"],
                record_id=row["record_id"],
                local_version=json.loads(row["local_version"]),
                cloud_version=json.loads(row["cloud_version"]),
                resolution=row["resolution"],
                resolved_at=self._parse_datetime(row["resolved_at"]) or datetime.now(timezone.utc),
                local_summary=row["local_summary"],
                cloud_summary=row["cloud_summary"],
            )
            for row in rows
        ]

    def save_sync_conflict(self, conflict: SyncConflict) -> str:
        """Save a sync conflict record.

        Args:
            conflict: The conflict to save

        Returns:
            The conflict ID
        """
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sync_conflicts
                   (id, table_name, record_id, local_version, cloud_version,
                    resolution, resolved_at, local_summary, cloud_summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conflict.id,
                    conflict.table,
                    conflict.record_id,
                    json.dumps(conflict.local_version),
                    json.dumps(conflict.cloud_version),
                    conflict.resolution,
                    (
                        conflict.resolved_at.isoformat()
                        if isinstance(conflict.resolved_at, datetime)
                        else conflict.resolved_at
                    ),
                    conflict.local_summary,
                    conflict.cloud_summary,
                ),
            )
            conn.commit()
        return conflict.id

    def clear_sync_conflicts(self, before: Optional[datetime] = None) -> int:
        """Clear sync conflict history.

        Args:
            before: If provided, only clear conflicts before this timestamp.
                    If None, clear all conflicts.

        Returns:
            Number of conflicts cleared
        """
        with self._connect() as conn:
            if before:
                cursor = conn.execute(
                    "DELETE FROM sync_conflicts WHERE resolved_at < ?", (before.isoformat(),)
                )
            else:
                cursor = conn.execute("DELETE FROM sync_conflicts")
            conn.commit()
            return cursor.rowcount

    def is_online(self) -> bool:
        """Check if cloud storage is reachable.

        Uses cached result if checked recently.
        Returns True if connected, False if offline or no cloud configured.
        """
        if not self.cloud_storage:
            return False

        # Check cache
        now = datetime.now(timezone.utc)
        if self._last_connectivity_check:
            elapsed = (now - self._last_connectivity_check).total_seconds()
            if elapsed < self._connectivity_cache_ttl:
                return self._is_online_cached

        # Perform connectivity check
        try:
            # Try a lightweight operation - get stats with timeout
            import socket

            old_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(self.CONNECTIVITY_TIMEOUT)
                # A simple operation to verify connectivity
                self.cloud_storage.get_stats()
                self._is_online_cached = True
            except Exception as e:
                logger.debug(f"Connectivity check failed: {e}")
                self._is_online_cached = False
            finally:
                socket.setdefaulttimeout(old_timeout)
        except Exception as e:
            logger.debug(f"Connectivity check error: {e}")
            self._is_online_cached = False

        self._last_connectivity_check = now
        return self._is_online_cached

    def _mark_synced(self, conn: sqlite3.Connection, table: str, record_id: str):
        """Mark a record as synced with the cloud."""
        validate_table_name(table)  # Security: validate before SQL use
        now = self._now()
        conn.execute(f"UPDATE {table} SET cloud_synced_at = ? WHERE id = ?", (now, record_id))

    def _get_record_for_push(self, table: str, record_id: str) -> Optional[Any]:
        """Get a record by table and ID for pushing to cloud."""
        validate_table_name(table)  # Security: validate before SQL use
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND agent_id = ?", (record_id, self.agent_id)
            ).fetchone()

        if not row:
            return None

        converters = {
            "episodes": self._row_to_episode,
            "notes": self._row_to_note,
            "beliefs": self._row_to_belief,
            "agent_values": self._row_to_value,
            "goals": self._row_to_goal,
            "drives": self._row_to_drive,
            "relationships": self._row_to_relationship,
            "playbooks": self._row_to_playbook,
            "raw_entries": self._row_to_raw_entry,
        }

        converter = converters.get(table)
        return converter(row) if converter else None

    def _push_record(self, table: str, record: Any) -> bool:
        """Push a single record to cloud storage.

        Returns True if successful, False otherwise.
        """
        if not self.cloud_storage:
            return False

        try:
            if table == "episodes":
                self.cloud_storage.save_episode(record)
            elif table == "notes":
                self.cloud_storage.save_note(record)
            elif table == "beliefs":
                self.cloud_storage.save_belief(record)
            elif table == "agent_values":
                self.cloud_storage.save_value(record)
            elif table == "goals":
                self.cloud_storage.save_goal(record)
            elif table == "drives":
                self.cloud_storage.save_drive(record)
            elif table == "relationships":
                self.cloud_storage.save_relationship(record)
            elif table == "playbooks":
                self.cloud_storage.save_playbook(record)
            else:
                logger.warning(f"Unknown table for push: {table}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to push record {table}:{record.id}: {e}")
            return False

    def sync(self) -> SyncResult:
        """Sync with cloud storage.

        Pushes queued local changes to cloud, then pulls remote changes.
        Uses last-write-wins for conflict resolution.

        Returns:
            SyncResult with counts of pushed/pulled records and any errors.
        """
        result = SyncResult()

        if not self.cloud_storage:
            logger.debug("No cloud storage configured, skipping sync")
            return result

        # Check connectivity first
        if not self.is_online():
            logger.info("Offline - sync skipped, changes queued")
            result.errors.append("Offline - cannot reach cloud storage")
            return result

        # Phase 1: Push queued changes
        queued = self.get_queued_changes()
        logger.debug(f"Pushing {len(queued)} queued changes")

        with self._connect() as conn:
            for change in queued:
                # Get the current record
                record = self._get_record_for_push(change.table_name, change.record_id)

                if record is None:
                    # Record was deleted or doesn't exist
                    if change.operation == "delete":
                        # TODO: Handle soft delete in cloud
                        self._clear_queued_change(conn, change.id)
                        result.pushed += 1
                    else:
                        # Record no longer exists locally, clear from queue
                        self._clear_queued_change(conn, change.id)
                    continue

                # Push to cloud
                if self._push_record(change.table_name, record):
                    self._mark_synced(conn, change.table_name, change.record_id)
                    self._clear_queued_change(conn, change.id)
                    result.pushed += 1
                else:
                    result.errors.append(f"Failed to push {change.table_name}:{change.record_id}")

            conn.commit()

        # Phase 2: Pull remote changes
        pull_result = self.pull_changes()
        result.pulled = pull_result.pulled
        result.conflicts = pull_result.conflicts
        result.errors.extend(pull_result.errors)

        # Update last sync time
        if result.success or (result.pushed > 0 or result.pulled > 0):
            self._set_sync_meta("last_sync_time", self._now())

        logger.info(
            f"Sync complete: pushed={result.pushed}, pulled={result.pulled}, conflicts={result.conflict_count}"
        )
        return result

    def pull_changes(self, since: Optional[datetime] = None) -> SyncResult:
        """Pull changes from cloud since the given timestamp.

        Uses last-write-wins for conflict resolution:
        - If cloud record is newer (cloud_synced_at > local_updated_at), use cloud version
        - If local record is newer, keep local version (it will be pushed on next sync)

        Args:
            since: Pull changes since this time. If None, uses last sync time.

        Returns:
            SyncResult with pulled count and any conflicts.
        """
        result = SyncResult()

        if not self.cloud_storage:
            return result

        # Determine the since timestamp
        if since is None:
            since = self.get_last_sync_time()

        # Pull from each table
        tables_and_getters = [
            ("episodes", self.cloud_storage.get_episodes, self._merge_episode),
            ("notes", self.cloud_storage.get_notes, self._merge_note),
            ("beliefs", self.cloud_storage.get_beliefs, self._merge_belief),
            ("agent_values", self.cloud_storage.get_values, self._merge_value),
            ("goals", self.cloud_storage.get_goals, self._merge_goal),
            ("drives", self.cloud_storage.get_drives, self._merge_drive),
            ("relationships", self.cloud_storage.get_relationships, self._merge_relationship),
            ("playbooks", self.cloud_storage.list_playbooks, self._merge_playbook),
        ]

        for table, getter, merger in tables_and_getters:
            try:
                # Get records from cloud (filtered by since if supported)
                if table == "episodes":
                    cloud_records = getter(limit=1000, since=since)
                elif table == "notes":
                    cloud_records = getter(limit=1000, since=since)
                elif table == "goals":
                    cloud_records = getter(status=None, limit=1000)  # Get all statuses
                else:
                    cloud_records = getter(limit=1000) if callable(getter) else getter()

                for cloud_record in cloud_records:
                    pull_count, conflict = merger(cloud_record)
                    result.pulled += pull_count
                    if conflict:
                        result.conflicts.append(conflict)

            except Exception as e:
                logger.error(f"Failed to pull from {table}: {e}")
                result.errors.append(f"Failed to pull {table}: {str(e)}")

        return result

    def _merge_episode(self, cloud_record: Episode) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud episode with local. Returns (pulled, conflict_or_none)."""
        return self._merge_record("episodes", cloud_record, self.get_episode)

    def _merge_note(self, cloud_record: Note) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud note with local. Returns (pulled, conflict_or_none)."""
        local = None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ? AND agent_id = ?",
                (cloud_record.id, self.agent_id),
            ).fetchone()
            if row:
                local = self._row_to_note(row)
        return self._merge_generic(
            "notes", cloud_record, local, lambda: self.save_note(cloud_record)
        )

    def _merge_belief(self, cloud_record: Belief) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud belief with local. Returns (pulled, conflict_or_none)."""
        local = None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE id = ? AND agent_id = ?",
                (cloud_record.id, self.agent_id),
            ).fetchone()
            if row:
                local = self._row_to_belief(row)
        return self._merge_generic(
            "beliefs", cloud_record, local, lambda: self.save_belief(cloud_record)
        )

    def _merge_value(self, cloud_record: Value) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud value with local. Returns (pulled, conflict_or_none)."""
        local = None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ? AND agent_id = ?",
                (cloud_record.id, self.agent_id),
            ).fetchone()
            if row:
                local = self._row_to_value(row)
        return self._merge_generic(
            "agent_values", cloud_record, local, lambda: self.save_value(cloud_record)
        )

    def _merge_goal(self, cloud_record: Goal) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud goal with local. Returns (pulled, conflict_or_none)."""
        local = None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ? AND agent_id = ?",
                (cloud_record.id, self.agent_id),
            ).fetchone()
            if row:
                local = self._row_to_goal(row)
        return self._merge_generic(
            "goals", cloud_record, local, lambda: self.save_goal(cloud_record)
        )

    def _merge_drive(self, cloud_record: Drive) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud drive with local. Returns (pulled, conflict_or_none)."""
        local = self.get_drive(cloud_record.drive_type)
        return self._merge_generic(
            "drives", cloud_record, local, lambda: self.save_drive(cloud_record)
        )

    def _merge_relationship(self, cloud_record: Relationship) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud relationship with local. Returns (pulled, conflict_or_none)."""
        local = self.get_relationship(cloud_record.entity_name)
        return self._merge_generic(
            "relationships", cloud_record, local, lambda: self.save_relationship(cloud_record)
        )

    def _merge_playbook(self, cloud_record: Playbook) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud playbook with local. Returns (pulled, conflict_or_none)."""
        local = self.get_playbook(cloud_record.id)
        return self._merge_generic(
            "playbooks", cloud_record, local, lambda: self.save_playbook(cloud_record)
        )

    def _merge_record(
        self, table: str, cloud_record: Any, get_local
    ) -> tuple[int, Optional[SyncConflict]]:
        """Generic merge for records with an ID-based getter."""
        local = get_local(cloud_record.id)
        return self._merge_generic(
            table, cloud_record, local, lambda: self._save_from_cloud(table, cloud_record)
        )

    def _merge_array_fields(self, table: str, winner: Any, loser: Any) -> Any:
        """Merge array fields from loser into winner using set union.

        This preserves array items from both records rather than losing data
        when one record wins via last-write-wins. For example, if local has
        tags=["A", "B"] and cloud has tags=["C"], the merged result will be
        tags=["A", "B", "C"].

        Args:
            table: The table name (used to look up which fields to merge)
            winner: The record that won timestamp comparison (will be modified)
            loser: The record that lost (arrays will be merged from here)

        Returns:
            The winner record with merged array fields
        """
        from dataclasses import replace

        array_fields = SYNC_ARRAY_FIELDS.get(table, [])
        if not array_fields:
            return winner

        updates = {}
        for field_name in array_fields:
            if not hasattr(winner, field_name) or not hasattr(loser, field_name):
                continue

            winner_val = getattr(winner, field_name)
            loser_val = getattr(loser, field_name)

            # Skip if both are None/empty or loser has nothing to add
            if not loser_val:
                continue
            if not winner_val:
                # Winner has nothing, use loser's array
                updates[field_name] = list(loser_val)
                continue

            # Both have values - merge using set union
            # Handle both string arrays and dict arrays (like steps in playbooks)
            try:
                if winner_val and isinstance(winner_val[0], dict):
                    # For dict arrays (like playbook steps), use JSON for dedup
                    seen = set()
                    merged = []
                    for item in winner_val:
                        key = json.dumps(item, sort_keys=True)
                        if key not in seen:
                            seen.add(key)
                            merged.append(item)
                    for item in loser_val:
                        key = json.dumps(item, sort_keys=True)
                        if key not in seen:
                            seen.add(key)
                            merged.append(item)
                    # Enforce size limit to prevent resource exhaustion
                    if len(merged) > MAX_SYNC_ARRAY_SIZE:
                        logger.warning(
                            f"Array field {field_name} exceeded max size ({len(merged)} > {MAX_SYNC_ARRAY_SIZE}), truncating"
                        )
                        merged = merged[:MAX_SYNC_ARRAY_SIZE]
                    updates[field_name] = merged
                else:
                    # For string arrays, use simple set union
                    merged = list(set(winner_val) | set(loser_val))
                    # Enforce size limit to prevent resource exhaustion
                    if len(merged) > MAX_SYNC_ARRAY_SIZE:
                        logger.warning(
                            f"Array field {field_name} exceeded max size ({len(merged)} > {MAX_SYNC_ARRAY_SIZE}), truncating"
                        )
                        merged = merged[:MAX_SYNC_ARRAY_SIZE]
                    updates[field_name] = merged
            except (TypeError, KeyError):
                # If anything goes wrong, just keep winner's value
                logger.warning(f"Failed to merge array field {field_name}, keeping winner's value")
                continue

        if updates:
            # Create a new record with merged fields
            return replace(winner, **updates)
        return winner

    def _merge_generic(
        self, table: str, cloud_record: Any, local_record: Optional[Any], save_fn
    ) -> tuple[int, Optional[SyncConflict]]:
        """Generic merge logic with last-write-wins for scalar fields, set union for arrays.

        Returns (pulled_count, conflict_or_none).
        If a conflict occurred, returns a SyncConflict with details.

        Array fields (like tags, lessons, focus_areas) are merged using set union
        rather than last-write-wins to preserve data from both sides.
        """
        if local_record is None:
            # No local record - just save the cloud version
            save_fn()
            # Mark as synced (don't queue for push)
            with self._connect() as conn:
                self._mark_synced(conn, table, cloud_record.id)
                # Remove from queue if present
                conn.execute(
                    "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                    (table, cloud_record.id),
                )
                conn.commit()
            return (1, None)

        # Both exist - use last-write-wins for scalar fields, merge arrays
        cloud_time = cloud_record.cloud_synced_at or cloud_record.local_updated_at
        local_time = local_record.local_updated_at

        if cloud_time and local_time:
            if cloud_time > local_time:
                # Cloud is newer - cloud wins for scalar fields
                # But merge array fields from local into cloud
                merged_record = self._merge_array_fields(table, cloud_record, local_record)

                # Create conflict record before overwriting (uses original records for history)
                conflict = self._create_conflict(
                    table, cloud_record.id, local_record, cloud_record, "cloud_wins_arrays_merged"
                )

                # Save the merged record
                self._save_from_cloud(table, merged_record)

                with self._connect() as conn:
                    self._mark_synced(conn, table, cloud_record.id)
                    conn.execute(
                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                        (table, cloud_record.id),
                    )
                    conn.commit()
                # Save conflict to history
                self.save_sync_conflict(conflict)
                return (1, conflict)  # Pulled with conflict
            else:
                # Local is newer - local wins for scalar fields
                # But merge array fields from cloud into local
                merged_record = self._merge_array_fields(table, local_record, cloud_record)

                # If arrays were merged, save the updated local record
                if merged_record is not local_record:
                    self._save_from_cloud(table, merged_record)

                conflict = self._create_conflict(
                    table, cloud_record.id, local_record, cloud_record, "local_wins_arrays_merged"
                )
                # Save conflict to history
                self.save_sync_conflict(conflict)
                return (0, conflict)  # Conflict, but local wins (with merged arrays)
        elif cloud_time:
            # Only cloud has timestamp - use cloud
            save_fn()
            with self._connect() as conn:
                self._mark_synced(conn, table, cloud_record.id)
                conn.commit()
            return (1, None)
        else:
            # Only local has timestamp or neither - keep local
            return (0, None)

    def _create_conflict(
        self, table: str, record_id: str, local_record: Any, cloud_record: Any, resolution: str
    ) -> SyncConflict:
        """Create a SyncConflict record with human-readable summaries."""
        import uuid

        local_summary = self._get_record_summary(table, local_record)
        cloud_summary = self._get_record_summary(table, cloud_record)
        local_dict = self._record_to_dict(local_record)
        cloud_dict = self._record_to_dict(cloud_record)
        return SyncConflict(
            id=str(uuid.uuid4()),
            table=table,
            record_id=record_id,
            local_version=local_dict,
            cloud_version=cloud_dict,
            resolution=resolution,
            resolved_at=datetime.now(timezone.utc),
            local_summary=local_summary,
            cloud_summary=cloud_summary,
        )

    def _get_record_summary(self, table: str, record: Any) -> str:
        """Get a human-readable summary of a record for conflict display."""
        if table == "episodes":
            return record.objective[:50] + "..." if len(record.objective) > 50 else record.objective
        elif table == "notes":
            return record.content[:50] + "..." if len(record.content) > 50 else record.content
        elif table == "beliefs":
            return record.statement[:50] + "..." if len(record.statement) > 50 else record.statement
        elif table == "agent_values":
            stmt = record.statement[:40] + "..." if len(record.statement) > 40 else record.statement
            return f"{record.name}: {stmt}"
        elif table == "goals":
            return record.title[:50] + "..." if len(record.title) > 50 else record.title
        elif table == "drives":
            return f"{record.drive_type} (intensity: {record.intensity})"
        elif table == "relationships":
            return f"{record.entity_name} ({record.relationship_type})"
        return f"{table}:{record.id}"

    def _record_to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert a dataclass record to a dict for JSON storage."""
        from dataclasses import asdict

        try:
            d = asdict(record)
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            return d
        except Exception:
            return {"id": getattr(record, "id", "unknown")}

    def _save_from_cloud(self, table: str, record: Any):
        """Save a record that came from cloud (used in _merge_record)."""
        if table == "episodes":
            self.save_episode(record)
        elif table == "notes":
            self.save_note(record)
        elif table == "beliefs":
            self.save_belief(record)
        elif table == "agent_values":
            self.save_value(record)
        elif table == "goals":
            self.save_goal(record)
        elif table == "drives":
            self.save_drive(record)
        elif table == "relationships":
            self.save_relationship(record)

    # === Health Check Compliance Tracking ===

    def log_health_check(
        self, anxiety_score: Optional[int] = None, source: str = "cli", triggered_by: str = "manual"
    ) -> str:
        """Log a health check event for compliance tracking.

        Args:
            anxiety_score: The anxiety score at time of check (0-100)
            source: Where the check originated from (cli, mcp)
            triggered_by: What triggered the check (boot, heartbeat, manual)

        Returns:
            The ID of the logged event
        """
        event_id = str(uuid.uuid4())
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO health_check_events
                   (id, agent_id, checked_at, anxiety_score, source, triggered_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event_id, self.agent_id, now, anxiety_score, source, triggered_by),
            )
            conn.commit()

        logger.debug(
            f"Logged health check: score={anxiety_score}, source={source}, triggered_by={triggered_by}"
        )
        return event_id

    def get_health_check_stats(self) -> Dict[str, Any]:
        """Get health check compliance statistics.

        Returns:
            Dict with:
            - total_checks: Total number of health checks
            - avg_per_day: Average checks per day
            - last_check_at: Timestamp of last check
            - last_anxiety_score: Anxiety score at last check
            - checks_by_source: Breakdown by source (cli/mcp)
            - checks_by_trigger: Breakdown by trigger (boot/heartbeat/manual)
        """
        with self._connect() as conn:
            # Total checks
            cur = conn.execute(
                "SELECT COUNT(*) FROM health_check_events WHERE agent_id = ?", (self.agent_id,)
            )
            total_checks = cur.fetchone()[0]

            if total_checks == 0:
                return {
                    "total_checks": 0,
                    "avg_per_day": 0.0,
                    "last_check_at": None,
                    "last_anxiety_score": None,
                    "checks_by_source": {},
                    "checks_by_trigger": {},
                }

            # Get first and last check for calculating avg per day
            cur = conn.execute(
                """SELECT MIN(checked_at), MAX(checked_at)
                   FROM health_check_events WHERE agent_id = ?""",
                (self.agent_id,),
            )
            first_check, last_check = cur.fetchone()

            # Calculate days spanned
            if first_check and last_check:
                first_dt = parse_datetime(first_check)
                last_dt = parse_datetime(last_check)
                if first_dt and last_dt:
                    days_spanned = max(1, (last_dt - first_dt).days + 1)
                    avg_per_day = total_checks / days_spanned
                else:
                    avg_per_day = float(total_checks)
            else:
                avg_per_day = float(total_checks)

            # Last check details
            cur = conn.execute(
                """SELECT checked_at, anxiety_score FROM health_check_events
                   WHERE agent_id = ? ORDER BY checked_at DESC LIMIT 1""",
                (self.agent_id,),
            )
            row = cur.fetchone()
            last_check_at = row[0] if row else None
            last_anxiety_score = row[1] if row else None

            # Checks by source
            cur = conn.execute(
                """SELECT source, COUNT(*) FROM health_check_events
                   WHERE agent_id = ? GROUP BY source""",
                (self.agent_id,),
            )
            checks_by_source = {row[0]: row[1] for row in cur.fetchall()}

            # Checks by trigger
            cur = conn.execute(
                """SELECT triggered_by, COUNT(*) FROM health_check_events
                   WHERE agent_id = ? GROUP BY triggered_by""",
                (self.agent_id,),
            )
            checks_by_trigger = {row[0]: row[1] for row in cur.fetchall()}

            return {
                "total_checks": total_checks,
                "avg_per_day": round(avg_per_day, 2),
                "last_check_at": last_check_at,
                "last_anxiety_score": last_anxiety_score,
                "checks_by_source": checks_by_source,
                "checks_by_trigger": checks_by_trigger,
            }
