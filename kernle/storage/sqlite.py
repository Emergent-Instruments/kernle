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

from kernle.utils import get_kernle_home

from .base import (
    Belief,
    DiagnosticReport,
    DiagnosticSession,
    Drive,
    EntityModel,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    Relationship,
    RelationshipHistoryEntry,
    SearchResult,
    SelfNarrative,
    Summary,
    TrustAssessment,
    Value,
    VersionConflictError,
    parse_datetime,
    utc_now,
)
from .cloud import CloudClient
from .embeddings import (
    EmbeddingProvider,
    HashEmbedder,
    pack_embedding,
)
from .flat_files import (
    init_flat_files,
    sync_beliefs_to_file,
    sync_goals_to_file,
    sync_relationships_to_file,
    sync_values_to_file,
)
from .health import get_health_check_stats as _get_health_check_stats
from .health import log_health_check as _log_health_check
from .lineage import check_derived_from_cycle
from .memory_crud import (
    _row_to_belief as _mc_row_to_belief,
)
from .memory_crud import (
    _row_to_diagnostic_report as _mc_row_to_diagnostic_report,
)
from .memory_crud import (
    _row_to_diagnostic_session as _mc_row_to_diagnostic_session,
)
from .memory_crud import (
    _row_to_drive as _mc_row_to_drive,
)
from .memory_crud import (
    _row_to_entity_model as _mc_row_to_entity_model,
)
from .memory_crud import (
    _row_to_episode as _mc_row_to_episode,
)
from .memory_crud import (
    _row_to_epoch as _mc_row_to_epoch,
)
from .memory_crud import (
    _row_to_goal as _mc_row_to_goal,
)
from .memory_crud import (
    _row_to_note as _mc_row_to_note,
)
from .memory_crud import (
    _row_to_playbook as _mc_row_to_playbook,
)
from .memory_crud import (
    _row_to_relationship as _mc_row_to_relationship,
)
from .memory_crud import (
    _row_to_relationship_history as _mc_row_to_relationship_history,
)
from .memory_crud import (
    _row_to_self_narrative as _mc_row_to_self_narrative,
)
from .memory_crud import (
    _row_to_suggestion as _mc_row_to_suggestion,
)
from .memory_crud import (
    _row_to_summary as _mc_row_to_summary,
)
from .memory_crud import (
    _row_to_value as _mc_row_to_value,
)
from .memory_ops import MemoryOps
from .raw_entries import (
    append_raw_to_file,
    escape_like_pattern,
    row_to_raw_entry,
    should_sync_raw,
    update_raw_fts,
)
from .raw_entries import (
    delete_raw as _delete_raw,
)
from .raw_entries import (
    get_all_stack_settings as _get_all_stack_settings,
)
from .raw_entries import (
    get_processing_config as _get_processing_config,
)
from .raw_entries import (
    get_raw as _get_raw,
)
from .raw_entries import (
    get_raw_files as _get_raw_files,
)
from .raw_entries import (
    get_stack_setting as _get_stack_setting,
)
from .raw_entries import (
    list_raw as _list_raw,
)
from .raw_entries import (
    mark_processed as _mark_processed,
)
from .raw_entries import (
    mark_raw_processed as _mark_raw_processed,
)
from .raw_entries import (
    save_raw as _save_raw,
)
from .raw_entries import (
    search_raw_fts as _search_raw_fts,
)
from .raw_entries import (
    set_processing_config as _set_processing_config,
)
from .raw_entries import (
    set_stack_setting as _set_stack_setting,
)
from .raw_entries import (
    sync_raw_from_files as _sync_raw_from_files,
)
from .schema import (
    ensure_raw_fts5,
    init_db,
    migrate_schema,
    validate_table_name,
)
from .sync_engine import SyncEngine

if TYPE_CHECKING:
    from .base import Storage as StorageProtocol

logger = logging.getLogger(__name__)


# NOTE: SCHEMA_VERSION, ALLOWED_TABLES, validate_table_name, SCHEMA, and VECTOR_SCHEMA
# have been moved to storage/schema.py and are imported at the top of this file.


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
        stack_id: str,
        db_path: Optional[Path] = None,
        cloud_storage: Optional["StorageProtocol"] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        # Defense-in-depth: reject path traversal in stack_id before using in paths
        if not stack_id or not stack_id.strip():
            raise ValueError("Stack ID cannot be empty")
        if "/" in stack_id or "\\" in stack_id:
            raise ValueError("Stack ID must not contain path separators")
        if stack_id.strip() in (".", ".."):
            raise ValueError("Stack ID must not be a relative path component")
        if ".." in stack_id.split("."):
            raise ValueError("Stack ID must not contain path traversal sequences")

        self.stack_id = stack_id
        self.db_path = self._resolve_db_path(db_path)
        self.cloud_storage = cloud_storage  # For sync

        # Connectivity cache
        self._last_connectivity_check: Optional[datetime] = None
        self._is_online_cached: bool = False
        self._connectivity_cache_ttl = 30  # seconds

        # Cloud search client
        self._cloud = CloudClient(stack_id, self.CLOUD_SEARCH_TIMEOUT)

        # Memory lifecycle operations
        self._memory_ops = MemoryOps(
            connect_fn=self._connect,
            stack_id=stack_id,
            now_fn=self._now,
            safe_get_fn=self._safe_get,
            queue_sync_fn=self._queue_sync,
            validate_table_name_fn=validate_table_name,
        )

        # Sync engine
        self._sync_engine = SyncEngine(self, validate_table_name)

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for sqlite-vec first
        self._has_vec = self._check_sqlite_vec()

        # Initialize embedder
        self._embedder = embedder or (HashEmbedder() if not embedder else embedder)

        # Initialize flat file directories for all memory layers
        self._agent_dir = self.db_path.parent / stack_id
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
        """Initialize flat files from existing database data."""
        init_flat_files(
            self._beliefs_file,
            self._values_file,
            self._relationships_file,
            self._goals_file,
            self._sync_beliefs_to_file,
            self._sync_values_to_file,
            self._sync_goals_to_file,
            self._sync_relationships_to_file,
        )

    def _resolve_db_path(self, db_path: Optional[Path]) -> Path:
        """Resolve the database path, falling back to temp dir if home is not writable."""
        import tempfile

        if db_path is not None:
            return self._validate_db_path(db_path)

        default_path = get_kernle_home() / "memories.db"
        try:
            default_path.parent.mkdir(parents=True, exist_ok=True)
            return self._validate_db_path(default_path)
        except (OSError, PermissionError) as e:
            # Home dir not writable (sandboxed/container/CI environment)
            fallback_dir = Path(tempfile.gettempdir()) / ".kernle"
            fallback_path = fallback_dir / "memories.db"
            logger.warning(
                f"Cannot write to {default_path.parent} ({e}), " f"falling back to {fallback_dir}"
            )
            return self._validate_db_path(fallback_path)

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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
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
        except Exception as e:
            logger.debug(f"Transaction failed, rolling back: {e}")
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

    # === Cloud Search Methods (delegated to CloudClient) ===

    def has_cloud_credentials(self) -> bool:
        """Check if cloud credentials are available."""
        return self._cloud.has_cloud_credentials()

    def cloud_health_check(self, timeout: float = 3.0) -> Dict[str, Any]:
        """Test cloud backend connectivity."""
        return self._cloud.cloud_health_check(timeout)

    def _cloud_search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[List[SearchResult]]:
        """Search memories via cloud backend."""
        return self._cloud._cloud_search(query, limit, record_types, timeout)

    def _init_db(self):
        """Initialize the database schema. Delegates to schema.init_db()."""
        with self._connect() as conn:
            init_db(
                conn=conn,
                stack_id=self.stack_id,
                has_vec=self._has_vec,
                embedder_dimension=self._embedder.dimension,
                load_vec_fn=self._load_vec,
                db_path=self.db_path,
                agent_dir=self._agent_dir,
            )

    def _ensure_raw_fts5(self, conn: sqlite3.Connection):
        """Create FTS5 virtual table. Delegates to schema.ensure_raw_fts5()."""
        ensure_raw_fts5(conn)

    def _migrate_schema(self, conn: sqlite3.Connection):
        """Run schema migrations. Delegates to schema.migrate_schema()."""
        migrate_schema(conn, self.stack_id)

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
        from dataclasses import asdict

        try:
            d = asdict(record)
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            return d
        except Exception as e:
            logger.debug(f"Failed to serialize record, using fallback: {e}")
            return {"id": getattr(record, "id", "unknown")}

    def _build_access_filter(
        self, requesting_entity: Optional[str] = None
    ) -> tuple[str, List[Any]]:
        """Build SQL filter for privacy access control.

        Args:
            requesting_entity: Entity requesting access. None means self-access (see everything).

        Returns:
            Tuple of (where_clause, params) for SQL query.

        Logic:
            - If requesting_entity is None → no filter (self-access, see everything)
            - If requesting_entity is set → filter records where:
              - access_grants IS NULL (private to self only), OR
              - access_grants = '[]' (private to self only), OR
              - access_grants contains requesting_entity
        """
        if requesting_entity is None:
            # Self-access: see everything
            return ("", [])

        # External access: only show records where requesting_entity is in access_grants
        # NULL or empty access_grants = private to self only
        where_clause = """
            AND (access_grants IS NOT NULL
                 AND access_grants != '[]'
                 AND access_grants LIKE ?)
        """
        params = [f'%"{requesting_entity}"%']

        return (where_clause, params)

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

        Stores data in both `data` and `payload` columns for consistency.
        The `payload` column is the canonical source; `data` is kept for
        backward compatibility.
        """
        now = self._now()

        # Normalize: use whichever is provided, store in both columns
        effective_payload = payload or data
        effective_data = data or payload

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
            (table, record_id, operation, effective_data, now, effective_payload, now),
        )

    def _content_hash(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _save_embedding(self, conn: sqlite3.Connection, table: str, record_id: str, content: str):
        """Save embedding for a record."""
        if not self._has_vec:
            return

        content_hash = self._content_hash(content)
        # Include stack_id in vec_id for isolation (security: prevents cross-agent timing leaks)
        vec_id = f"{self.stack_id}:{table}:{record_id}"

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

        if episode.derived_from:
            check_derived_from_cycle(self, "episode", episode.id, episode.derived_from)

        now = self._now()
        episode.local_updated_at = self._parse_datetime(now)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                (id, stack_id, objective, outcome, outcome_type, lessons, tags,
                 emotional_valence, emotional_arousal, emotional_tags,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 times_accessed, last_accessed, is_protected, strength,
                 processed, context, context_tags,
                 source_entity, subject_ids, access_grants, consent_grants,
                 epoch_id, repeat, avoid,
                 created_at, local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    episode.id,
                    self.stack_id,
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
                    episode.strength,
                    1 if episode.processed else 0,
                    episode.context,
                    self._to_json(episode.context_tags),
                    getattr(episode, "source_entity", None),
                    self._to_json(getattr(episode, "subject_ids", None)),
                    self._to_json(getattr(episode, "access_grants", None)),
                    self._to_json(getattr(episode, "consent_grants", None)),
                    episode.epoch_id,
                    self._to_json(episode.repeat),
                    self._to_json(episode.avoid),
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

        Security: Provenance fields have special handling:
        - source_type: Write-once (preserved from original)
        - derived_from: Append-only (merged with original)
        - confidence_history: Append-only (merged with original)

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
            # First check current version and get original provenance fields
            current = conn.execute(
                """SELECT version, source_type, derived_from, confidence_history
                   FROM episodes WHERE id = ? AND stack_id = ?""",
                (episode.id, self.stack_id),
            ).fetchone()

            if not current:
                return False  # Record doesn't exist

            current_version = current["version"]
            if current_version != expected_version:
                raise VersionConflictError(
                    "episodes", episode.id, expected_version, current_version
                )

            # Security: Preserve provenance fields (write-once / append-only)
            # source_type is write-once - always use original
            original_source_type = current["source_type"] or episode.source_type

            # derived_from is append-only - merge lists
            original_derived = self._from_json(current["derived_from"]) or []
            new_derived = episode.derived_from or []
            merged_derived = list(set(original_derived) | set(new_derived))

            # confidence_history is append-only - merge lists
            original_history = self._from_json(current["confidence_history"]) or []
            new_history = episode.confidence_history or []
            # For history, append new entries that aren't already present
            merged_history = original_history + [
                h for h in new_history if h not in original_history
            ]

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
                    strength = ?,
                    context = ?,
                    context_tags = ?,
                    local_updated_at = ?,
                    version = version + 1
                WHERE id = ? AND stack_id = ? AND version = ?
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
                    original_source_type,  # Write-once: preserve original
                    self._to_json(episode.source_episodes),
                    self._to_json(merged_derived),  # Append-only: merged
                    episode.last_verified.isoformat() if episode.last_verified else None,
                    episode.verification_count,
                    self._to_json(merged_history),  # Append-only: merged
                    episode.times_accessed,
                    episode.last_accessed.isoformat() if episode.last_accessed else None,
                    1 if episode.is_protected else 0,
                    episode.strength,
                    episode.context,
                    self._to_json(episode.context_tags),
                    now,
                    episode.id,
                    self.stack_id,
                    expected_version,
                ),
            )

            if cursor.rowcount == 0:
                # Version changed between check and update (rare but possible)
                conn.rollback()
                new_current = conn.execute(
                    "SELECT version FROM episodes WHERE id = ? AND stack_id = ?",
                    (episode.id, self.stack_id),
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
                    (id, stack_id, objective, outcome, outcome_type, lessons, tags,
                     emotional_valence, emotional_arousal, emotional_tags,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     times_accessed, last_accessed, is_protected, strength,
                     processed, context, context_tags,
                     epoch_id, repeat, avoid,
                     created_at, local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        episode.id,
                        self.stack_id,
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
                        episode.strength,
                        1 if episode.processed else 0,
                        episode.context,
                        self._to_json(episode.context_tags),
                        episode.epoch_id,
                        self._to_json(episode.repeat),
                        self._to_json(episode.avoid),
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
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        requesting_entity: Optional[str] = None,
    ) -> List[Episode]:
        """Get episodes."""
        query = "SELECT * FROM episodes WHERE stack_id = ? AND deleted = 0"
        params: List[Any] = [self.stack_id]

        # Apply privacy filter
        access_filter, access_params = self._build_access_filter(requesting_entity)
        query += access_filter
        params.extend(access_params)

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

    def memory_exists(self, memory_type: str, memory_id: str) -> bool:
        """Check if a memory record exists in the stack.

        Args:
            memory_type: Type of memory (episode, belief, note, raw, etc.)
            memory_id: ID of the memory record

        Returns:
            True if the record exists and is not deleted
        """
        table_map = {
            "episode": "episodes",
            "belief": "beliefs",
            "value": "agent_values",
            "goal": "goals",
            "note": "notes",
            "drive": "drives",
            "relationship": "relationships",
            "raw": "raw_entries",
            "playbook": "playbooks",
        }
        table = table_map.get(memory_type)
        if not table:
            return False
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT 1 FROM {table} WHERE id = ? AND stack_id = ? AND deleted = 0",
                (memory_id, self.stack_id),
            ).fetchone()
        return row is not None

    def get_episode(
        self, episode_id: str, requesting_entity: Optional[str] = None
    ) -> Optional[Episode]:
        """Get a specific episode."""
        query = "SELECT * FROM episodes WHERE id = ? AND stack_id = ?"
        params: List[Any] = [episode_id, self.stack_id]

        # Apply privacy filter
        access_filter, access_params = self._build_access_filter(requesting_entity)
        query += access_filter
        params.extend(access_params)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        return self._row_to_episode(row) if row else None

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert row to Episode. Delegates to memory_crud._row_to_episode()."""
        return _mc_row_to_episode(row)

    def get_episodes_by_source_entity(self, source_entity: str, limit: int = 500) -> List[Episode]:
        """Get episodes associated with a source entity for trust computation."""
        query = """
            SELECT * FROM episodes
            WHERE stack_id = ? AND source_entity = ? AND deleted = 0 AND strength > 0.0
            ORDER BY created_at DESC LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, (self.stack_id, source_entity, limit)).fetchall()
        return [self._row_to_episode(row) for row in rows]

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
                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                (valence, arousal, self._to_json(tags), now, episode_id, self.stack_id),
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
        query = "SELECT * FROM episodes WHERE stack_id = ? AND deleted = 0"
        params: List[Any] = [self.stack_id]

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
                   WHERE stack_id = ? AND deleted = 0
                   AND created_at >= ?
                   AND (emotional_valence != 0.0 OR emotional_arousal != 0.0 OR emotional_tags IS NOT NULL)
                   ORDER BY created_at DESC
                   LIMIT ?"""

        with self._connect() as conn:
            rows = conn.execute(query, (self.stack_id, cutoff, limit)).fetchall()

        return [self._row_to_episode(row) for row in rows]

    # === Beliefs ===

    def save_belief(self, belief: Belief) -> str:
        """Save a belief."""
        if not belief.id:
            belief.id = str(uuid.uuid4())

        if belief.derived_from:
            check_derived_from_cycle(self, "belief", belief.id, belief.derived_from)

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO beliefs
                (id, stack_id, statement, belief_type, confidence, created_at,
                 source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 supersedes, superseded_by, times_reinforced, is_active,
                 strength,
                 context, context_tags, source_entity, subject_ids, access_grants, consent_grants,
                 processed,
                 belief_scope, source_domain, cross_domain_applications, abstraction_level,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    belief.id,
                    self.stack_id,
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
                    belief.strength,
                    belief.context,
                    self._to_json(belief.context_tags),
                    getattr(belief, "source_entity", None),
                    self._to_json(getattr(belief, "subject_ids", None)),
                    self._to_json(getattr(belief, "access_grants", None)),
                    self._to_json(getattr(belief, "consent_grants", None)),
                    1 if belief.processed else 0,
                    getattr(belief, "belief_scope", "world"),
                    getattr(belief, "source_domain", None),
                    self._to_json(getattr(belief, "cross_domain_applications", None)),
                    getattr(belief, "abstraction_level", "specific"),
                    belief.epoch_id,
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
                "SELECT version FROM beliefs WHERE id = ? AND stack_id = ?",
                (belief.id, self.stack_id),
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
                    belief_scope = ?,
                    source_domain = ?,
                    cross_domain_applications = ?,
                    abstraction_level = ?,
                    local_updated_at = ?,
                    deleted = ?,
                    version = version + 1
                WHERE id = ? AND stack_id = ? AND version = ?
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
                    getattr(belief, "belief_scope", "world"),
                    getattr(belief, "source_domain", None),
                    self._to_json(getattr(belief, "cross_domain_applications", None)),
                    getattr(belief, "abstraction_level", "specific"),
                    now,
                    1 if belief.deleted else 0,
                    belief.id,
                    self.stack_id,
                    expected_version,
                ),
            )

            if cursor.rowcount == 0:
                conn.rollback()
                new_current = conn.execute(
                    "SELECT version FROM beliefs WHERE id = ? AND stack_id = ?",
                    (belief.id, self.stack_id),
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
                    (id, stack_id, statement, belief_type, confidence, created_at,
                     source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     supersedes, superseded_by, times_reinforced, is_active,
                     times_accessed, last_accessed, is_protected, strength,
                     context, context_tags, processed,
                     belief_scope, source_domain, cross_domain_applications, abstraction_level,
                     epoch_id,
                     local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        belief.id,
                        self.stack_id,
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
                        belief.strength,
                        belief.context,
                        self._to_json(belief.context_tags),
                        1 if belief.processed else 0,
                        getattr(belief, "belief_scope", "world"),
                        getattr(belief, "source_domain", None),
                        self._to_json(getattr(belief, "cross_domain_applications", None)),
                        getattr(belief, "abstraction_level", "specific"),
                        belief.epoch_id,
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
        sync_beliefs_to_file(
            self._beliefs_file, self.get_beliefs(limit=500, include_inactive=False), self._now()
        )

    def get_beliefs(
        self,
        limit: int = 100,
        include_inactive: bool = False,
        requesting_entity: Optional[str] = None,
    ) -> List[Belief]:
        """Get beliefs.

        Args:
            limit: Maximum number of beliefs to return
            include_inactive: If True, include superseded/archived beliefs
            requesting_entity: If provided, filter by access_grants. None = self-access (see all).
        """
        access_filter, access_params = self._build_access_filter(requesting_entity)
        with self._connect() as conn:
            if include_inactive:
                rows = conn.execute(
                    f"SELECT * FROM beliefs WHERE stack_id = ? AND deleted = 0{access_filter} ORDER BY created_at DESC LIMIT ?",
                    [self.stack_id] + access_params + [limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM beliefs WHERE stack_id = ? AND deleted = 0 AND (is_active = 1 OR is_active IS NULL){access_filter} ORDER BY created_at DESC LIMIT ?",
                    [self.stack_id] + access_params + [limit],
                ).fetchall()

        return [self._row_to_belief(row) for row in rows]

    def find_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE stack_id = ? AND statement = ? AND deleted = 0",
                (self.stack_id, statement),
            ).fetchone()

        return self._row_to_belief(row) if row else None

    def get_belief(
        self, belief_id: str, requesting_entity: Optional[str] = None
    ) -> Optional[Belief]:
        """Get a specific belief by ID."""
        query = "SELECT * FROM beliefs WHERE id = ? AND stack_id = ?"
        params: List[Any] = [belief_id, self.stack_id]

        # Apply privacy filter
        access_filter, access_params = self._build_access_filter(requesting_entity)
        query += access_filter
        params.extend(access_params)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        return self._row_to_belief(row) if row else None

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        """Convert row to Belief. Delegates to memory_crud._row_to_belief()."""
        return _mc_row_to_belief(row)

    # === Values ===

    def save_value(self, value: Value) -> str:
        """Save a value."""
        if not value.id:
            value.id = str(uuid.uuid4())

        if value.derived_from:
            check_derived_from_cycle(self, "value", value.id, value.derived_from)

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_values
                (id, stack_id, name, statement, priority, created_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 strength,
                 context, context_tags,
                 subject_ids, access_grants, consent_grants,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    value.id,
                    self.stack_id,
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
                    value.strength,
                    value.context,
                    self._to_json(value.context_tags),
                    self._to_json(getattr(value, "subject_ids", None)),
                    self._to_json(getattr(value, "access_grants", None)),
                    self._to_json(getattr(value, "consent_grants", None)),
                    value.epoch_id,
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
        sync_values_to_file(self._values_file, self.get_values(limit=100), self._now())

    def get_values(self, limit: int = 100, requesting_entity: Optional[str] = None) -> List[Value]:
        """Get values ordered by priority."""
        access_filter, access_params = self._build_access_filter(requesting_entity)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM agent_values WHERE stack_id = ? AND deleted = 0{access_filter} ORDER BY priority DESC LIMIT ?",
                [self.stack_id] + access_params + [limit],
            ).fetchall()

        return [self._row_to_value(row) for row in rows]

    def _row_to_value(self, row: sqlite3.Row) -> Value:
        """Convert row to Value. Delegates to memory_crud._row_to_value()."""
        return _mc_row_to_value(row)

    # === Goals ===

    def save_goal(self, goal: Goal) -> str:
        """Save a goal."""
        if not goal.id:
            goal.id = str(uuid.uuid4())

        if goal.derived_from:
            check_derived_from_cycle(self, "goal", goal.id, goal.derived_from)

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO goals
                (id, stack_id, title, description, goal_type, priority, status, created_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 strength,
                 context, context_tags,
                 subject_ids, access_grants, consent_grants,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    goal.id,
                    self.stack_id,
                    goal.title,
                    goal.description,
                    goal.goal_type,
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
                    goal.strength,
                    goal.context,
                    self._to_json(goal.context_tags),
                    self._to_json(getattr(goal, "subject_ids", None)),
                    self._to_json(getattr(goal, "access_grants", None)),
                    self._to_json(getattr(goal, "consent_grants", None)),
                    goal.epoch_id,
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
        sync_goals_to_file(self._goals_file, self.get_goals(status=None, limit=100), self._now())

    def get_goals(
        self,
        status: Optional[str] = "active",
        limit: int = 100,
        requesting_entity: Optional[str] = None,
    ) -> List[Goal]:
        """Get goals."""
        query = "SELECT * FROM goals WHERE stack_id = ? AND deleted = 0"
        params: List[Any] = [self.stack_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        access_filter, access_params = self._build_access_filter(requesting_entity)
        query += access_filter

        query += " ORDER BY created_at DESC LIMIT ?"
        params.extend(access_params)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_goal(row) for row in rows]

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        """Convert row to Goal. Delegates to memory_crud._row_to_goal()."""
        return _mc_row_to_goal(row)

    # === Notes ===

    def save_note(self, note: Note) -> str:
        """Save a note."""
        if not note.id:
            note.id = str(uuid.uuid4())

        if note.derived_from:
            check_derived_from_cycle(self, "note", note.id, note.derived_from)

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notes
                (id, stack_id, content, note_type, speaker, reason, tags, created_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 strength,
                 context, context_tags, source_entity,
                 subject_ids, access_grants, consent_grants,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    note.id,
                    self.stack_id,
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
                    note.strength,
                    note.context,
                    self._to_json(note.context_tags),
                    getattr(note, "source_entity", None),
                    self._to_json(getattr(note, "subject_ids", None)),
                    self._to_json(getattr(note, "access_grants", None)),
                    self._to_json(getattr(note, "consent_grants", None)),
                    note.epoch_id,
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
                    (id, stack_id, content, note_type, speaker, reason, tags, created_at,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     times_accessed, last_accessed, is_protected, strength,
                     processed, context, context_tags,
                     epoch_id,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        note.id,
                        self.stack_id,
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
                        note.strength,
                        1 if note.processed else 0,
                        note.context,
                        self._to_json(note.context_tags),
                        note.epoch_id,
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
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        note_type: Optional[str] = None,
        requesting_entity: Optional[str] = None,
    ) -> List[Note]:
        """Get notes."""
        query = "SELECT * FROM notes WHERE stack_id = ? AND deleted = 0"
        params: List[Any] = [self.stack_id]

        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        if note_type:
            query += " AND note_type = ?"
            params.append(note_type)

        access_filter, access_params = self._build_access_filter(requesting_entity)
        query += access_filter

        query += " ORDER BY created_at DESC LIMIT ?"
        params.extend(access_params)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_note(row) for row in rows]

    def _row_to_note(self, row: sqlite3.Row) -> Note:
        """Convert row to Note. Delegates to memory_crud._row_to_note()."""
        return _mc_row_to_note(row)

    # === Drives ===

    def save_drive(self, drive: Drive) -> str:
        """Save or update a drive."""
        if not drive.id:
            drive.id = str(uuid.uuid4())

        if drive.derived_from:
            check_derived_from_cycle(self, "drive", drive.id, drive.derived_from)

        now = self._now()

        with self._connect() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM drives WHERE stack_id = ? AND drive_type = ?",
                (self.stack_id, drive.drive_type),
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
                        subject_ids = ?, access_grants = ?, consent_grants = ?,
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
                        self._to_json(getattr(drive, "subject_ids", None)),
                        self._to_json(getattr(drive, "access_grants", None)),
                        self._to_json(getattr(drive, "consent_grants", None)),
                        now,
                        drive.id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO drives
                    (id, stack_id, drive_type, intensity, focus_areas, created_at, updated_at,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     strength,
                     context, context_tags,
                     subject_ids, access_grants, consent_grants,
                     epoch_id,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        drive.id,
                        self.stack_id,
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
                        drive.strength,
                        drive.context,
                        self._to_json(drive.context_tags),
                        self._to_json(getattr(drive, "subject_ids", None)),
                        self._to_json(getattr(drive, "access_grants", None)),
                        self._to_json(getattr(drive, "consent_grants", None)),
                        drive.epoch_id,
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

    def get_drives(self, requesting_entity: Optional[str] = None) -> List[Drive]:
        """Get all drives."""
        access_filter, access_params = self._build_access_filter(requesting_entity)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM drives WHERE stack_id = ? AND deleted = 0{access_filter}",
                [self.stack_id] + access_params,
            ).fetchall()

        return [self._row_to_drive(row) for row in rows]

    def get_drive(self, drive_type: str) -> Optional[Drive]:
        """Get a specific drive."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM drives WHERE stack_id = ? AND drive_type = ? AND deleted = 0",
                (self.stack_id, drive_type),
            ).fetchone()

        return self._row_to_drive(row) if row else None

    def _row_to_drive(self, row: sqlite3.Row) -> Drive:
        """Convert row to Drive. Delegates to memory_crud._row_to_drive()."""
        return _mc_row_to_drive(row)

    # === Relationships ===

    def save_relationship(self, relationship: Relationship) -> str:
        """Save or update a relationship. Logs history on changes."""
        if not relationship.id:
            relationship.id = str(uuid.uuid4())

        if relationship.derived_from:
            check_derived_from_cycle(
                self, "relationship", relationship.id, relationship.derived_from
            )

        now = self._now()

        with self._connect() as conn:
            # Check if exists - fetch full row for change detection
            existing = conn.execute(
                "SELECT * FROM relationships WHERE stack_id = ? AND entity_name = ?",
                (self.stack_id, relationship.entity_name),
            ).fetchone()

            if existing:
                relationship.id = existing["id"]

                # Detect changes and log history
                self._log_relationship_changes(conn, existing, relationship, now)

                conn.execute(
                    """
                    UPDATE relationships SET
                        entity_type = ?, relationship_type = ?, notes = ?,
                        sentiment = ?, interaction_count = ?, last_interaction = ?,
                        confidence = ?, source_type = ?, source_episodes = ?,
                        derived_from = ?, last_verified = ?, verification_count = ?,
                        confidence_history = ?, context = ?, context_tags = ?,
                        subject_ids = ?, access_grants = ?, consent_grants = ?,
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
                        self._to_json(getattr(relationship, "subject_ids", None)),
                        self._to_json(getattr(relationship, "access_grants", None)),
                        self._to_json(getattr(relationship, "consent_grants", None)),
                        now,
                        relationship.id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO relationships
                    (id, stack_id, entity_name, entity_type, relationship_type, notes,
                     sentiment, interaction_count, last_interaction, created_at,
                     confidence, source_type, source_episodes, derived_from,
                     last_verified, verification_count, confidence_history,
                     strength,
                     context, context_tags,
                     subject_ids, access_grants, consent_grants,
                     epoch_id,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        relationship.id,
                        self.stack_id,
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
                        relationship.strength,
                        relationship.context,
                        self._to_json(relationship.context_tags),
                        self._to_json(getattr(relationship, "subject_ids", None)),
                        self._to_json(getattr(relationship, "access_grants", None)),
                        self._to_json(getattr(relationship, "consent_grants", None)),
                        relationship.epoch_id,
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
        sync_relationships_to_file(self._relationships_file, self.get_relationships(), self._now())

    def get_relationships(
        self, entity_type: Optional[str] = None, requesting_entity: Optional[str] = None
    ) -> List[Relationship]:
        """Get relationships."""
        query = "SELECT * FROM relationships WHERE stack_id = ? AND deleted = 0"
        params: List[Any] = [self.stack_id]

        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)

        access_filter, access_params = self._build_access_filter(requesting_entity)
        query += access_filter
        params.extend(access_params)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_relationship(row) for row in rows]

    def get_relationship(self, entity_name: str) -> Optional[Relationship]:
        """Get a specific relationship."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationships WHERE stack_id = ? AND entity_name = ? AND deleted = 0",
                (self.stack_id, entity_name),
            ).fetchone()

        return self._row_to_relationship(row) if row else None

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        """Convert row to Relationship. Delegates to memory_crud._row_to_relationship()."""
        return _mc_row_to_relationship(row)

    # === Epochs (KEP v3 temporal eras) ===

    def save_epoch(self, epoch: Epoch) -> str:
        """Save an epoch. Returns the epoch ID."""

        if not epoch.id:
            epoch.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO epochs
                (id, stack_id, epoch_number, name, started_at, ended_at,
                 trigger_type, trigger_description, summary,
                 key_belief_ids, key_relationship_ids,
                 key_goal_ids, dominant_drive_ids,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    epoch.id,
                    self.stack_id,
                    epoch.epoch_number,
                    epoch.name,
                    epoch.started_at.isoformat() if epoch.started_at else now,
                    epoch.ended_at.isoformat() if epoch.ended_at else None,
                    epoch.trigger_type,
                    epoch.trigger_description,
                    epoch.summary,
                    self._to_json(epoch.key_belief_ids),
                    self._to_json(epoch.key_relationship_ids),
                    self._to_json(epoch.key_goal_ids),
                    self._to_json(epoch.dominant_drive_ids),
                    now,
                    epoch.cloud_synced_at.isoformat() if epoch.cloud_synced_at else None,
                    epoch.version,
                    1 if epoch.deleted else 0,
                ),
            )
            conn.commit()

        return epoch.id

    def get_epoch(self, epoch_id: str) -> Optional[Epoch]:
        """Get a specific epoch by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM epochs WHERE id = ? AND stack_id = ? AND deleted = 0",
                (epoch_id, self.stack_id),
            ).fetchone()

        return self._row_to_epoch(row) if row else None

    def get_epochs(self, limit: int = 100) -> List[Epoch]:
        """Get all epochs, ordered by epoch_number DESC."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM epochs WHERE stack_id = ? AND deleted = 0 "
                "ORDER BY epoch_number DESC LIMIT ?",
                (self.stack_id, limit),
            ).fetchall()

        return [self._row_to_epoch(row) for row in rows]

    def get_current_epoch(self) -> Optional[Epoch]:
        """Get the currently active (open) epoch, if any."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM epochs WHERE stack_id = ? AND ended_at IS NULL AND deleted = 0 "
                "ORDER BY epoch_number DESC LIMIT 1",
                (self.stack_id,),
            ).fetchone()

        return self._row_to_epoch(row) if row else None

    def close_epoch(self, epoch_id: str, summary: Optional[str] = None) -> bool:
        """Close an epoch by setting ended_at. Returns True if closed."""
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE epochs SET ended_at = ?, summary = COALESCE(?, summary), "
                "local_updated_at = ?, version = version + 1 "
                "WHERE id = ? AND stack_id = ? AND ended_at IS NULL AND deleted = 0",
                (now, summary, now, epoch_id, self.stack_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def _row_to_epoch(self, row: sqlite3.Row) -> Epoch:
        """Convert row to Epoch. Delegates to memory_crud._row_to_epoch()."""
        return _mc_row_to_epoch(row)

    # === Summaries (Fractal Summarization) ===

    def save_summary(self, summary: Summary) -> str:
        """Save a summary. Returns the summary ID."""
        if not summary.id:
            summary.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO summaries
                (id, stack_id, scope, period_start, period_end, epoch_id,
                 content, key_themes, supersedes, is_protected,
                 created_at, updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    summary.id,
                    self.stack_id,
                    summary.scope,
                    summary.period_start,
                    summary.period_end,
                    summary.epoch_id,
                    summary.content,
                    self._to_json(summary.key_themes),
                    self._to_json(summary.supersedes),
                    1 if summary.is_protected else 0,
                    summary.created_at.isoformat() if summary.created_at else now,
                    now,
                    summary.cloud_synced_at.isoformat() if summary.cloud_synced_at else None,
                    summary.version,
                    1 if summary.deleted else 0,
                ),
            )
            conn.commit()

        return summary.id

    def get_summary(self, summary_id: str) -> Optional[Summary]:
        """Get a specific summary by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE id = ? AND stack_id = ? AND deleted = 0",
                (summary_id, self.stack_id),
            ).fetchone()

        return self._row_to_summary(row) if row else None

    def list_summaries(self, stack_id: str, scope: Optional[str] = None) -> List[Summary]:
        """Get summaries, optionally filtered by scope."""
        with self._connect() as conn:
            if scope:
                rows = conn.execute(
                    "SELECT * FROM summaries WHERE stack_id = ? AND scope = ? AND deleted = 0 "
                    "ORDER BY period_start DESC",
                    (self.stack_id, scope),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM summaries WHERE stack_id = ? AND deleted = 0 "
                    "ORDER BY period_start DESC",
                    (self.stack_id,),
                ).fetchall()

        return [self._row_to_summary(row) for row in rows]

    def _row_to_summary(self, row: sqlite3.Row) -> Summary:
        """Convert row to Summary. Delegates to memory_crud._row_to_summary()."""
        return _mc_row_to_summary(row)

    # === Self-Narratives (KEP v3) ===

    def save_self_narrative(self, narrative: SelfNarrative) -> str:
        """Save a self-narrative. Returns the narrative ID."""
        if not narrative.id:
            narrative.id = str(uuid.uuid4())

        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO self_narratives
                (id, stack_id, epoch_id, narrative_type, content,
                 key_themes, unresolved_tensions, is_active, supersedes,
                 created_at, updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    narrative.id,
                    self.stack_id,
                    narrative.epoch_id,
                    narrative.narrative_type,
                    narrative.content,
                    self._to_json(narrative.key_themes),
                    self._to_json(narrative.unresolved_tensions),
                    1 if narrative.is_active else 0,
                    narrative.supersedes,
                    narrative.created_at.isoformat() if narrative.created_at else now,
                    now,
                    narrative.cloud_synced_at.isoformat() if narrative.cloud_synced_at else None,
                    narrative.version,
                    1 if narrative.deleted else 0,
                ),
            )
            conn.commit()

        return narrative.id

    def get_self_narrative(self, narrative_id: str) -> Optional[SelfNarrative]:
        """Get a specific self-narrative by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM self_narratives WHERE id = ? AND stack_id = ? AND deleted = 0",
                (narrative_id, self.stack_id),
            ).fetchone()

        return self._row_to_self_narrative(row) if row else None

    def list_self_narratives(
        self,
        stack_id: str,
        narrative_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[SelfNarrative]:
        """Get self-narratives, optionally filtered."""
        with self._connect() as conn:
            conditions = ["stack_id = ?", "deleted = 0"]
            params: list = [self.stack_id]

            if narrative_type:
                conditions.append("narrative_type = ?")
                params.append(narrative_type)

            if active_only:
                conditions.append("is_active = 1")

            where = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT * FROM self_narratives WHERE {where} ORDER BY updated_at DESC",
                params,
            ).fetchall()

        return [self._row_to_self_narrative(row) for row in rows]

    def deactivate_self_narratives(self, stack_id: str, narrative_type: str) -> int:
        """Deactivate all active narratives of a given type."""
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE self_narratives SET is_active = 0, updated_at = ? "
                "WHERE stack_id = ? AND narrative_type = ? AND is_active = 1 AND deleted = 0",
                (now, self.stack_id, narrative_type),
            )
            conn.commit()
            return cursor.rowcount

    def _row_to_self_narrative(self, row: sqlite3.Row) -> SelfNarrative:
        """Convert row to SelfNarrative. Delegates to memory_crud._row_to_self_narrative()."""
        return _mc_row_to_self_narrative(row)

    # === Trust Assessments (KEP v3) ===

    def save_trust_assessment(self, assessment: TrustAssessment) -> str:
        """Save or update a trust assessment. Returns the assessment ID."""
        now = self._now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM trust_assessments "
                "WHERE stack_id = ? AND entity = ? AND deleted = 0",
                (self.stack_id, assessment.entity),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE trust_assessments SET dimensions = ?, authority = ?, "
                    "evidence_episode_ids = ?, last_updated = ?, local_updated_at = ?, "
                    "version = version + 1 WHERE id = ?",
                    (
                        json.dumps(assessment.dimensions),
                        json.dumps(assessment.authority or []),
                        json.dumps(assessment.evidence_episode_ids or []),
                        now,
                        now,
                        existing["id"],
                    ),
                )
                return existing["id"]
            else:
                conn.execute(
                    "INSERT INTO trust_assessments "
                    "(id, stack_id, entity, dimensions, authority, evidence_episode_ids, "
                    "last_updated, created_at, local_updated_at, cloud_synced_at, "
                    "version, deleted) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        assessment.id,
                        self.stack_id,
                        assessment.entity,
                        json.dumps(assessment.dimensions),
                        json.dumps(assessment.authority or []),
                        json.dumps(assessment.evidence_episode_ids or []),
                        now,
                        now,
                        now,
                        None,
                        1,
                        0,
                    ),
                )
                return assessment.id

    def get_trust_assessment(self, entity: str) -> Optional[TrustAssessment]:
        """Get a trust assessment for a specific entity."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trust_assessments "
                "WHERE stack_id = ? AND entity = ? AND deleted = 0",
                (self.stack_id, entity),
            ).fetchone()
            if not row:
                return None
            return TrustAssessment(
                id=row["id"],
                stack_id=row["stack_id"],
                entity=row["entity"],
                dimensions=json.loads(row["dimensions"]),
                authority=(json.loads(row["authority"]) if row["authority"] else []),
                evidence_episode_ids=(
                    json.loads(row["evidence_episode_ids"]) if row["evidence_episode_ids"] else []
                ),
                last_updated=(parse_datetime(row["last_updated"]) if row["last_updated"] else None),
                created_at=(parse_datetime(row["created_at"]) if row["created_at"] else None),
                local_updated_at=(
                    parse_datetime(row["local_updated_at"]) if row["local_updated_at"] else None
                ),
                cloud_synced_at=(
                    parse_datetime(row["cloud_synced_at"]) if row["cloud_synced_at"] else None
                ),
                version=row["version"],
                deleted=bool(row["deleted"]),
            )

    def get_trust_assessments(self) -> List[TrustAssessment]:
        """Get all trust assessments for the agent."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trust_assessments "
                "WHERE stack_id = ? AND deleted = 0 ORDER BY entity",
                (self.stack_id,),
            ).fetchall()
            return [
                TrustAssessment(
                    id=r["id"],
                    stack_id=r["stack_id"],
                    entity=r["entity"],
                    dimensions=json.loads(r["dimensions"]),
                    authority=(json.loads(r["authority"]) if r["authority"] else []),
                    evidence_episode_ids=(
                        json.loads(r["evidence_episode_ids"]) if r["evidence_episode_ids"] else []
                    ),
                    last_updated=(parse_datetime(r["last_updated"]) if r["last_updated"] else None),
                    created_at=(parse_datetime(r["created_at"]) if r["created_at"] else None),
                    local_updated_at=(
                        parse_datetime(r["local_updated_at"]) if r["local_updated_at"] else None
                    ),
                    cloud_synced_at=(
                        parse_datetime(r["cloud_synced_at"]) if r["cloud_synced_at"] else None
                    ),
                    version=r["version"],
                    deleted=bool(r["deleted"]),
                )
                for r in rows
            ]

    def delete_trust_assessment(self, entity: str) -> bool:
        """Delete a trust assessment (soft delete)."""
        now = self._now()
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE trust_assessments SET deleted = 1, "
                "local_updated_at = ? "
                "WHERE stack_id = ? AND entity = ? AND deleted = 0",
                (now, self.stack_id, entity),
            )
            return result.rowcount > 0

    # === Diagnostic Sessions & Reports ===

    def save_diagnostic_session(self, session: DiagnosticSession) -> str:
        """Save a diagnostic session. Returns the session ID."""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO diagnostic_sessions "
                "(id, stack_id, session_type, access_level, status, consent_given, "
                "started_at, completed_at, local_updated_at, cloud_synced_at, "
                "version, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    self.stack_id,
                    session.session_type,
                    session.access_level,
                    session.status,
                    1 if session.consent_given else 0,
                    (session.started_at.isoformat() if session.started_at else now),
                    (session.completed_at.isoformat() if session.completed_at else None),
                    now,
                    None,
                    session.version,
                    1 if session.deleted else 0,
                ),
            )
        return session.id

    def get_diagnostic_session(self, session_id: str) -> Optional[DiagnosticSession]:
        """Get a specific diagnostic session by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM diagnostic_sessions "
                "WHERE id = ? AND stack_id = ? AND deleted = 0",
                (session_id, self.stack_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_diagnostic_session(row)

    def get_diagnostic_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[DiagnosticSession]:
        """Get diagnostic sessions, optionally filtered by status."""
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM diagnostic_sessions "
                    "WHERE stack_id = ? AND status = ? AND deleted = 0 "
                    "ORDER BY started_at DESC LIMIT ?",
                    (self.stack_id, status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM diagnostic_sessions "
                    "WHERE stack_id = ? AND deleted = 0 "
                    "ORDER BY started_at DESC LIMIT ?",
                    (self.stack_id, limit),
                ).fetchall()
            return [self._row_to_diagnostic_session(r) for r in rows]

    def complete_diagnostic_session(self, session_id: str) -> bool:
        """Mark a diagnostic session as completed. Returns True if updated."""
        now = self._now()
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE diagnostic_sessions SET status = 'completed', "
                "completed_at = ?, local_updated_at = ?, version = version + 1 "
                "WHERE id = ? AND stack_id = ? AND deleted = 0 AND status = 'active'",
                (now, now, session_id, self.stack_id),
            )
            return result.rowcount > 0

    def save_diagnostic_report(self, report: DiagnosticReport) -> str:
        """Save a diagnostic report. Returns the report ID."""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO diagnostic_reports "
                "(id, stack_id, session_id, findings, summary, "
                "created_at, local_updated_at, cloud_synced_at, version, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.id,
                    self.stack_id,
                    report.session_id,
                    json.dumps(report.findings) if report.findings is not None else None,
                    report.summary,
                    (report.created_at.isoformat() if report.created_at else now),
                    now,
                    None,
                    report.version,
                    1 if report.deleted else 0,
                ),
            )
        return report.id

    def get_diagnostic_report(self, report_id: str) -> Optional[DiagnosticReport]:
        """Get a specific diagnostic report by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM diagnostic_reports " "WHERE id = ? AND stack_id = ? AND deleted = 0",
                (report_id, self.stack_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_diagnostic_report(row)

    def get_diagnostic_reports(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[DiagnosticReport]:
        """Get diagnostic reports, optionally filtered by session."""
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM diagnostic_reports "
                    "WHERE stack_id = ? AND session_id = ? AND deleted = 0 "
                    "ORDER BY created_at DESC LIMIT ?",
                    (self.stack_id, session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM diagnostic_reports "
                    "WHERE stack_id = ? AND deleted = 0 "
                    "ORDER BY created_at DESC LIMIT ?",
                    (self.stack_id, limit),
                ).fetchall()
            return [self._row_to_diagnostic_report(r) for r in rows]

    def _row_to_diagnostic_session(self, row: sqlite3.Row) -> DiagnosticSession:
        """Convert row to DiagnosticSession. Delegates to memory_crud._row_to_diagnostic_session()."""
        return _mc_row_to_diagnostic_session(row)

    def _row_to_diagnostic_report(self, row: sqlite3.Row) -> DiagnosticReport:
        """Convert row to DiagnosticReport. Delegates to memory_crud._row_to_diagnostic_report()."""
        return _mc_row_to_diagnostic_report(row)

    # === Relationship History & Entity Models ===

    def _log_relationship_changes(
        self,
        conn: Any,
        existing_row: sqlite3.Row,
        new_rel: Relationship,
        now: str,
    ) -> None:
        """Detect changes between existing and new relationship, log history entries."""
        rel_id = existing_row["id"]
        entity_name = existing_row["entity_name"]

        # Check sentiment/trust change
        old_sentiment = existing_row["sentiment"]
        if new_rel.sentiment != old_sentiment:
            entry_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO relationship_history
                (id, stack_id, relationship_id, entity_name, event_type,
                 old_value, new_value, notes, created_at,
                 local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry_id,
                    self.stack_id,
                    rel_id,
                    entity_name,
                    "trust_change",
                    json.dumps({"sentiment": old_sentiment}),
                    json.dumps({"sentiment": new_rel.sentiment}),
                    None,
                    now,
                    now,
                    1,
                    0,
                ),
            )

        # Check relationship_type change
        old_type = existing_row["relationship_type"]
        if new_rel.relationship_type != old_type:
            entry_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO relationship_history
                (id, stack_id, relationship_id, entity_name, event_type,
                 old_value, new_value, notes, created_at,
                 local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry_id,
                    self.stack_id,
                    rel_id,
                    entity_name,
                    "type_change",
                    json.dumps({"relationship_type": old_type}),
                    json.dumps({"relationship_type": new_rel.relationship_type}),
                    None,
                    now,
                    now,
                    1,
                    0,
                ),
            )

        # Check notes change
        old_notes = existing_row["notes"]
        if new_rel.notes != old_notes and new_rel.notes is not None:
            entry_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO relationship_history
                (id, stack_id, relationship_id, entity_name, event_type,
                 old_value, new_value, notes, created_at,
                 local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry_id,
                    self.stack_id,
                    rel_id,
                    entity_name,
                    "note",
                    json.dumps({"notes": old_notes}) if old_notes else None,
                    json.dumps({"notes": new_rel.notes}),
                    None,
                    now,
                    now,
                    1,
                    0,
                ),
            )

        # Check interaction count change (log interaction event)
        old_count = existing_row["interaction_count"]
        if new_rel.interaction_count > old_count:
            entry_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO relationship_history
                (id, stack_id, relationship_id, entity_name, event_type,
                 old_value, new_value, notes, created_at,
                 local_updated_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry_id,
                    self.stack_id,
                    rel_id,
                    entity_name,
                    "interaction",
                    json.dumps({"interaction_count": old_count}),
                    json.dumps({"interaction_count": new_rel.interaction_count}),
                    None,
                    now,
                    now,
                    1,
                    0,
                ),
            )

    # === Relationship History ===

    def save_relationship_history(self, entry: RelationshipHistoryEntry) -> str:
        """Save a relationship history entry."""
        if not entry.id:
            entry.id = str(uuid.uuid4())

        now = self._now()
        if not entry.created_at:
            entry.created_at = datetime.now(timezone.utc)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relationship_history
                (id, stack_id, relationship_id, entity_name, event_type,
                 old_value, new_value, episode_id, notes, created_at,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.id,
                    self.stack_id,
                    entry.relationship_id,
                    entry.entity_name,
                    entry.event_type,
                    entry.old_value,
                    entry.new_value,
                    entry.episode_id,
                    entry.notes,
                    entry.created_at.isoformat() if entry.created_at else now,
                    now,
                    None,
                    1,
                    0,
                ),
            )
            conn.commit()

        return entry.id

    def get_relationship_history(
        self,
        entity_name: str,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[RelationshipHistoryEntry]:
        """Get history entries for a relationship."""
        query = (
            "SELECT * FROM relationship_history "
            "WHERE stack_id = ? AND entity_name = ? AND deleted = 0"
        )
        params: List[Any] = [self.stack_id, entity_name]

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_relationship_history(row) for row in rows]

    def _row_to_relationship_history(self, row: sqlite3.Row) -> RelationshipHistoryEntry:
        """Convert row to RelationshipHistoryEntry. Delegates to memory_crud._row_to_relationship_history()."""
        return _mc_row_to_relationship_history(row)

    # === Entity Models ===

    def save_entity_model(self, model: EntityModel) -> str:
        """Save an entity model."""
        if not model.id:
            model.id = str(uuid.uuid4())

        now = self._now()

        # Auto-populate subject_ids from entity_name
        if not model.subject_ids:
            model.subject_ids = [model.entity_name]

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO entity_models
                (id, stack_id, entity_name, model_type, observation, confidence,
                 source_episodes, created_at, updated_at,
                 subject_ids, access_grants, consent_grants,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model.id,
                    self.stack_id,
                    model.entity_name,
                    model.model_type,
                    model.observation,
                    model.confidence,
                    self._to_json(model.source_episodes),
                    (model.created_at.isoformat() if model.created_at else now),
                    now,
                    self._to_json(model.subject_ids),
                    self._to_json(model.access_grants),
                    self._to_json(model.consent_grants),
                    now,
                    None,
                    model.version,
                    0,
                ),
            )
            conn.commit()

        return model.id

    def get_entity_models(
        self,
        entity_name: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[EntityModel]:
        """Get entity models, optionally filtered."""
        query = "SELECT * FROM entity_models WHERE stack_id = ? AND deleted = 0"
        params: List[Any] = [self.stack_id]

        if entity_name:
            query += " AND entity_name = ?"
            params.append(entity_name)
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_entity_model(row) for row in rows]

    def get_entity_model(self, model_id: str) -> Optional[EntityModel]:
        """Get a specific entity model by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM entity_models WHERE id = ? AND stack_id = ? AND deleted = 0",
                (model_id, self.stack_id),
            ).fetchone()

        return self._row_to_entity_model(row) if row else None

    def _row_to_entity_model(self, row: sqlite3.Row) -> EntityModel:
        """Convert row to EntityModel. Delegates to memory_crud._row_to_entity_model()."""
        return _mc_row_to_entity_model(row)

    # === Playbooks (Procedural Memory) ===

    def save_playbook(self, playbook: Playbook) -> str:
        """Save a playbook. Returns the playbook ID."""
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO playbooks
                (id, stack_id, name, description, trigger_conditions, steps, failure_modes,
                 recovery_steps, mastery_level, times_used, success_rate, source_episodes, tags,
                 confidence, last_used, created_at,
                 subject_ids, access_grants, consent_grants,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    playbook.id,
                    self.stack_id,
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
                    self._to_json(getattr(playbook, "subject_ids", None)),
                    self._to_json(getattr(playbook, "access_grants", None)),
                    self._to_json(getattr(playbook, "consent_grants", None)),
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
                "SELECT * FROM playbooks WHERE id = ? AND stack_id = ? AND deleted = 0",
                (playbook_id, self.stack_id),
            )
            row = cur.fetchone()

        return self._row_to_playbook(row) if row else None

    def list_playbooks(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        requesting_entity: Optional[str] = None,
    ) -> List[Playbook]:
        """Get playbooks, optionally filtered by tags."""
        access_filter, access_params = self._build_access_filter(requesting_entity)
        with self._connect() as conn:
            query = f"""
                SELECT * FROM playbooks
                WHERE stack_id = ? AND deleted = 0{access_filter}
                ORDER BY times_used DESC, created_at DESC
                LIMIT ?
            """
            cur = conn.execute(query, [self.stack_id] + access_params + [limit])
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

            # Support both new format (stack_id:playbooks:id) and legacy (playbooks:id)
            new_prefix = f"{self.stack_id}:playbooks:"
            legacy_prefix = "playbooks:"

            with self._connect() as conn:
                cur = conn.execute(
                    """
                    SELECT e.id, e.embedding, distance
                    FROM vec_embeddings e
                    WHERE (e.id LIKE ? OR e.id LIKE ?)
                    ORDER BY distance
                    LIMIT ?
                """.replace("distance", f"vec_distance_L2(e.embedding, X'{packed.hex()}')"),
                    (f"{new_prefix}%", f"{legacy_prefix}%", limit * 2),
                )

                vec_results = cur.fetchall()

            # Extract playbook IDs from both formats
            playbook_ids = []
            for r in vec_results:
                vec_id = r[0]
                if vec_id.startswith(new_prefix):
                    playbook_ids.append(vec_id[len(new_prefix) :])
                elif vec_id.startswith(legacy_prefix):
                    playbook_ids.append(vec_id[len(legacy_prefix) :])

            playbooks = []
            for pid in playbook_ids:
                playbook = self.get_playbook(pid)
                if playbook:
                    playbooks.append(playbook)
                if len(playbooks) >= limit:
                    break

            return playbooks
        else:
            # Fall back to tokenized text search
            tokens = self._tokenize_query(query)
            columns = ["name", "description", "trigger_conditions"]
            with self._connect() as conn:
                if tokens:
                    filt, filt_params = self._build_token_filter(tokens, columns)
                else:
                    # All words too short, use full-phrase match
                    search_pattern = f"%{query}%"
                    filt = "(name LIKE ? OR description LIKE ? OR trigger_conditions LIKE ?)"
                    filt_params = [search_pattern, search_pattern, search_pattern]
                cur = conn.execute(
                    f"""
                    SELECT * FROM playbooks
                    WHERE stack_id = ? AND deleted = 0
                    AND {filt}
                    ORDER BY times_used DESC
                    LIMIT ?
                """,
                    [self.stack_id] + filt_params + [limit],
                )
                rows = cur.fetchall()

            playbooks = [self._row_to_playbook(row) for row in rows]
            if tokens:
                # Sort by token match score
                def _score(pb: "Playbook") -> float:
                    triggers = " ".join(pb.trigger_conditions) if pb.trigger_conditions else ""
                    combined = f"{pb.name or ''} {pb.description or ''} {triggers}"
                    return self._token_match_score(combined, tokens)

                playbooks.sort(key=_score, reverse=True)
            return playbooks

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
                WHERE id = ? AND stack_id = ?
            """,
                (
                    new_times_used,
                    new_success_rate,
                    new_mastery,
                    now,
                    now,
                    playbook_id,
                    self.stack_id,
                ),
            )

            self._queue_sync(conn, "playbooks", playbook_id, "upsert")
            conn.commit()

        return True

    def _row_to_playbook(self, row: sqlite3.Row) -> Playbook:
        """Convert row to Playbook. Delegates to memory_crud._row_to_playbook()."""
        return _mc_row_to_playbook(row)

    # === Boot Config ===

    def boot_set(self, key: str, value: str) -> None:
        """Set a boot config value. Creates or updates."""
        if not key or not isinstance(key, str):
            raise ValueError("Boot config key must be a non-empty string")
        if not isinstance(value, str):
            raise ValueError("Boot config value must be a string")
        # Strip whitespace from key, preserve value
        key = key.strip()
        if not key:
            raise ValueError("Boot config key must be a non-empty string")

        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO boot_config (id, stack_id, key, value, created_at, updated_at)
                VALUES (lower(hex(randomblob(4))), ?, ?, ?, ?, ?)
                ON CONFLICT(stack_id, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (self.stack_id, key, value, now, now),
            )

    def boot_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a boot config value. Returns default if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM boot_config WHERE stack_id = ? AND key = ?",
                (self.stack_id, key),
            ).fetchone()
        return row["value"] if row else default

    def boot_list(self) -> Dict[str, str]:
        """List all boot config values as a dict."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM boot_config WHERE stack_id = ? ORDER BY key",
                (self.stack_id,),
            ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def boot_delete(self, key: str) -> bool:
        """Delete a boot config value. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM boot_config WHERE stack_id = ? AND key = ?",
                (self.stack_id, key),
            )
        return cursor.rowcount > 0

    def boot_clear(self) -> int:
        """Clear all boot config for this agent. Returns count deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM boot_config WHERE stack_id = ?",
                (self.stack_id,),
            )
        return cursor.rowcount

    # === Raw Entries ===
    # Delegated to storage/raw_entries.py

    def save_raw(self, blob: str, source: str = "unknown") -> str:
        """Save a raw entry. Delegates to raw_entries.save_raw()."""
        with self._connect() as conn:
            return _save_raw(
                conn=conn,
                stack_id=self.stack_id,
                blob=blob,
                source=source,
                raw_dir=self._raw_dir,
                queue_sync_fn=self._queue_sync,
                save_embedding_fn=self._save_embedding,
                should_sync_raw_fn=self._should_sync_raw,
            )

    def _should_sync_raw(self) -> bool:
        """Check if raw sync is enabled. Delegates to raw_entries.should_sync_raw()."""
        return should_sync_raw()

    def _update_raw_fts(self, conn, raw_id, blob):
        """Update FTS5 index. Delegates to raw_entries.update_raw_fts()."""
        update_raw_fts(conn, raw_id, blob)

    def _append_raw_to_file(self, raw_id, content, timestamp, source, tags=None):
        """Append to flat file. Delegates to raw_entries.append_raw_to_file()."""
        append_raw_to_file(self._raw_dir, raw_id, content, timestamp, source, tags)

    def get_raw_dir(self):
        """Get raw flat files directory path."""
        return self._raw_dir

    def get_raw_files(self):
        """Get list of raw flat files. Delegates to raw_entries.get_raw_files()."""
        return _get_raw_files(self._raw_dir)

    def sync_raw_from_files(self):
        """Sync raw entries from flat files. Delegates to raw_entries.sync_raw_from_files()."""
        with self._connect() as conn:
            return _sync_raw_from_files(
                conn=conn,
                stack_id=self.stack_id,
                raw_dir=self._raw_dir,
                save_embedding_fn=self._save_embedding,
            )

    def _import_raw_entry(self, entry, existing_ids, result):
        """Import a raw entry. Delegates to raw_entries.import_raw_entry()."""
        from .raw_entries import import_raw_entry

        with self._connect() as conn:
            import_raw_entry(
                conn=conn,
                stack_id=self.stack_id,
                entry=entry,
                existing_ids=existing_ids,
                result=result,
                save_embedding_fn=self._save_embedding,
            )

    def get_raw(self, raw_id):
        """Get a raw entry by ID. Delegates to raw_entries.get_raw()."""
        with self._connect() as conn:
            return _get_raw(conn, self.stack_id, raw_id)

    def list_raw(self, processed=None, limit=100):
        """List raw entries. Delegates to raw_entries.list_raw()."""
        with self._connect() as conn:
            return _list_raw(conn, self.stack_id, processed, limit)

    def search_raw_fts(self, query, limit=50):
        """Search raw entries via FTS5. Delegates to raw_entries.search_raw_fts()."""
        with self._connect() as conn:
            return _search_raw_fts(conn, self.stack_id, query, limit)

    def _escape_like_pattern(self, pattern):
        """Escape LIKE pattern. Delegates to raw_entries.escape_like_pattern()."""
        return escape_like_pattern(pattern)

    def mark_raw_processed(self, raw_id, processed_into):
        """Mark raw entry as processed. Delegates to raw_entries.mark_raw_processed()."""
        with self._connect() as conn:
            return _mark_raw_processed(
                conn, self.stack_id, raw_id, processed_into, self._queue_sync
            )

    def mark_episode_processed(self, episode_id):
        """Mark episode as processed. Delegates to raw_entries.mark_processed()."""
        with self._connect() as conn:
            return _mark_processed(conn, self.stack_id, "episodes", episode_id, self._queue_sync)

    def mark_note_processed(self, note_id):
        """Mark note as processed. Delegates to raw_entries.mark_processed()."""
        with self._connect() as conn:
            return _mark_processed(conn, self.stack_id, "notes", note_id, self._queue_sync)

    def mark_belief_processed(self, belief_id):
        """Mark belief as processed. Delegates to raw_entries.mark_processed()."""
        with self._connect() as conn:
            return _mark_processed(conn, self.stack_id, "beliefs", belief_id, self._queue_sync)

    def get_processing_config(self):
        """Get processing config. Delegates to raw_entries.get_processing_config()."""
        with self._connect() as conn:
            return _get_processing_config(conn)

    def set_processing_config(
        self,
        layer_transition,
        *,
        enabled=None,
        model_id=None,
        quantity_threshold=None,
        valence_threshold=None,
        time_threshold_hours=None,
        batch_size=None,
        max_sessions_per_day=None,
    ):
        """Set processing config. Delegates to raw_entries.set_processing_config()."""
        with self._connect() as conn:
            return _set_processing_config(
                conn,
                layer_transition,
                enabled=enabled,
                model_id=model_id,
                quantity_threshold=quantity_threshold,
                valence_threshold=valence_threshold,
                time_threshold_hours=time_threshold_hours,
                batch_size=batch_size,
                max_sessions_per_day=max_sessions_per_day,
            )

    def get_stack_setting(self, key):
        """Get a stack setting. Delegates to raw_entries.get_stack_setting()."""
        with self._connect() as conn:
            return _get_stack_setting(conn, self.stack_id, key)

    def set_stack_setting(self, key, value):
        """Set a stack setting. Delegates to raw_entries.set_stack_setting()."""
        with self._connect() as conn:
            _set_stack_setting(conn, self.stack_id, key, value)

    def get_all_stack_settings(self):
        """Get all stack settings. Delegates to raw_entries.get_all_stack_settings()."""
        with self._connect() as conn:
            return _get_all_stack_settings(conn, self.stack_id)

    def delete_raw(self, raw_id):
        """Delete a raw entry. Delegates to raw_entries.delete_raw()."""
        with self._connect() as conn:
            return _delete_raw(conn, self.stack_id, raw_id, self._queue_sync)

    def _row_to_raw_entry(self, row):
        """Convert row to RawEntry. Delegates to raw_entries.row_to_raw_entry()."""
        return row_to_raw_entry(row)

    # === Memory Suggestions ===

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        """Save a memory suggestion. Returns the suggestion ID."""
        suggestion_id = suggestion.id or str(uuid.uuid4())
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_suggestions
                (id, stack_id, memory_type, content, confidence, source_raw_ids,
                 status, created_at, resolved_at, resolution_reason, promoted_to,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    suggestion_id,
                    self.stack_id,
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
                "SELECT * FROM memory_suggestions WHERE id = ? AND stack_id = ? AND deleted = 0",
                (suggestion_id, self.stack_id),
            ).fetchone()

        return self._row_to_suggestion(row) if row else None

    def get_suggestions(
        self,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemorySuggestion]:
        """Get suggestions, optionally filtered."""
        query = "SELECT * FROM memory_suggestions WHERE stack_id = ? AND deleted = 0"
        params: List[Any] = [self.stack_id]

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
                WHERE id = ? AND stack_id = ? AND deleted = 0
            """,
                (status, now, resolution_reason, promoted_to, now, suggestion_id, self.stack_id),
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
                WHERE id = ? AND stack_id = ? AND deleted = 0
            """,
                (now, suggestion_id, self.stack_id),
            )
            if cursor.rowcount > 0:
                self._queue_sync(conn, "memory_suggestions", suggestion_id, "delete")
                conn.commit()
                return True
        return False

    def _row_to_suggestion(self, row: sqlite3.Row) -> MemorySuggestion:
        """Convert row to MemorySuggestion. Delegates to memory_crud._row_to_suggestion()."""
        return _mc_row_to_suggestion(row)

    # === Search ===

    def search(
        self,
        query: str,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        prefer_cloud: bool = True,
        requesting_entity: Optional[str] = None,
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
        return self._local_search(query, limit, types, requesting_entity=requesting_entity)

    def _local_search(
        self, query: str, limit: int, types: List[str], requesting_entity: Optional[str] = None
    ) -> List[SearchResult]:
        """Local search using sqlite-vec or text matching.

        Args:
            query: Search query
            limit: Maximum results
            types: Memory types to search
            requesting_entity: If provided, filter by access_grants.

        Returns:
            List of SearchResult
        """
        if self._has_vec:
            results = self._vector_search(query, limit, types)
        else:
            results = self._text_search(query, limit, types, requesting_entity=requesting_entity)

        # Apply privacy filtering for external entity access
        if requesting_entity is not None:
            filtered = []
            for r in results:
                grants = getattr(r.record, "access_grants", None)
                if grants and requesting_entity in grants:
                    filtered.append(r)
                # NULL or empty grants = private, skip for external access
            return filtered[:limit]
        return results

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
            # Security: filter by stack_id prefix first to prevent timing side-channel
            agent_prefix = f"{self.stack_id}:"
            for row in rows:
                vec_id = row["id"]
                distance = row["distance"]

                # Parse vec_id - supports both new and legacy formats
                # New format: stack_id:table:record_id
                # Legacy format: table:record_id
                if vec_id.startswith(agent_prefix):
                    # New format - stack_id verified
                    parts = vec_id.split(":", 2)
                    if len(parts) != 3:
                        continue
                    _, table_name, record_id = parts
                else:
                    # Legacy format - stack_id will be verified in _fetch_record
                    parts = vec_id.split(":", 1)
                    if len(parts) != 2:
                        continue
                    table_name, record_id = parts

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
            f"SELECT * FROM {table} WHERE id = ? AND stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0",
            (record_id, self.stack_id),
        ).fetchone()

        if row:
            return converter(row), record_type
        return None, None

    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        """Split a search query into meaningful tokens (words with 3+ chars)."""
        return [w for w in query.split() if len(w) >= 3]

    @staticmethod
    def _build_token_filter(tokens: List[str], columns: List[str]) -> tuple:
        """Build a tokenized OR filter for multiple columns.

        Returns (sql_fragment, params) where sql_fragment is a parenthesized
        OR expression matching any token in any column, and params is the
        list of LIKE pattern values.
        """
        clauses = []
        params: list = []
        for token in tokens:
            pattern = f"%{token}%"
            for col in columns:
                clauses.append(f"{col} LIKE ?")
                params.append(pattern)
        sql = f"({' OR '.join(clauses)})"
        return sql, params

    @staticmethod
    def _token_match_score(text: str, tokens: List[str]) -> float:
        """Score a text by fraction of query tokens it contains (case-insensitive)."""
        if not tokens:
            return 1.0
        lower = text.lower()
        hits = sum(1 for t in tokens if t.lower() in lower)
        return hits / len(tokens)

    def _text_search(
        self, query: str, limit: int, types: List[str], requesting_entity: Optional[str] = None
    ) -> List[SearchResult]:
        """Fallback text-based search using tokenized LIKE matching."""
        results = []
        tokens = self._tokenize_query(query)
        access_filter, access_params = self._build_access_filter(requesting_entity)

        # If no meaningful tokens, fall back to full-phrase match
        if not tokens:
            search_term = f"%{query}%"
        else:
            search_term = None

        with self._connect() as conn:
            if "episode" in types:
                columns = ["objective", "outcome", "lessons"]
                if tokens:
                    filt, filt_params = self._build_token_filter(tokens, columns)
                else:
                    filt = "(objective LIKE ? OR outcome LIKE ? OR lessons LIKE ?)"
                    filt_params = [search_term, search_term, search_term]
                rows = conn.execute(
                    f"""SELECT * FROM episodes
                       WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
                       AND {filt}{access_filter}
                       LIMIT ?""",
                    [self.stack_id] + filt_params + access_params + [limit],
                ).fetchall()
                for row in rows:
                    ep = self._row_to_episode(row)
                    combined = f"{ep.objective or ''} {ep.outcome or ''} {ep.lessons or ''}"
                    score = self._token_match_score(combined, tokens) if tokens else 1.0
                    results.append(SearchResult(record=ep, record_type="episode", score=score))

            if "note" in types:
                columns = ["content"]
                if tokens:
                    filt, filt_params = self._build_token_filter(tokens, columns)
                else:
                    filt = "content LIKE ?"
                    filt_params = [search_term]
                rows = conn.execute(
                    f"""SELECT * FROM notes
                       WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
                       AND {filt}{access_filter}
                       LIMIT ?""",
                    [self.stack_id] + filt_params + access_params + [limit],
                ).fetchall()
                for row in rows:
                    note = self._row_to_note(row)
                    score = self._token_match_score(note.content or "", tokens) if tokens else 1.0
                    results.append(SearchResult(record=note, record_type="note", score=score))

            if "belief" in types:
                columns = ["statement"]
                if tokens:
                    filt, filt_params = self._build_token_filter(tokens, columns)
                else:
                    filt = "statement LIKE ?"
                    filt_params = [search_term]
                rows = conn.execute(
                    f"""SELECT * FROM beliefs
                       WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
                       AND {filt}{access_filter}
                       LIMIT ?""",
                    [self.stack_id] + filt_params + access_params + [limit],
                ).fetchall()
                for row in rows:
                    belief = self._row_to_belief(row)
                    score = (
                        self._token_match_score(belief.statement or "", tokens) if tokens else 1.0
                    )
                    results.append(SearchResult(record=belief, record_type="belief", score=score))

            if "value" in types:
                columns = ["name", "statement"]
                if tokens:
                    filt, filt_params = self._build_token_filter(tokens, columns)
                else:
                    filt = "(name LIKE ? OR statement LIKE ?)"
                    filt_params = [search_term, search_term]
                rows = conn.execute(
                    f"""SELECT * FROM agent_values
                       WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
                       AND {filt}{access_filter}
                       LIMIT ?""",
                    [self.stack_id] + filt_params + access_params + [limit],
                ).fetchall()
                for row in rows:
                    val = self._row_to_value(row)
                    combined = f"{val.name or ''} {val.statement or ''}"
                    score = self._token_match_score(combined, tokens) if tokens else 1.0
                    results.append(SearchResult(record=val, record_type="value", score=score))

            if "goal" in types:
                columns = ["title", "description"]
                if tokens:
                    filt, filt_params = self._build_token_filter(tokens, columns)
                else:
                    filt = "(title LIKE ? OR description LIKE ?)"
                    filt_params = [search_term, search_term]
                rows = conn.execute(
                    f"""SELECT * FROM goals
                       WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
                       AND {filt}{access_filter}
                       LIMIT ?""",
                    [self.stack_id] + filt_params + access_params + [limit],
                ).fetchall()
                for row in rows:
                    goal = self._row_to_goal(row)
                    combined = f"{goal.title or ''} {goal.description or ''}"
                    score = self._token_match_score(combined, tokens) if tokens else 1.0
                    results.append(SearchResult(record=goal, record_type="goal", score=score))

        # Sort by token match score descending
        results.sort(key=lambda r: r.score, reverse=True)
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
                    f"SELECT COUNT(*) FROM {table} WHERE stack_id = ? AND deleted = 0",
                    (self.stack_id,),
                ).fetchone()[0]
                stats[key] = count

            # Add count of pending suggestions specifically
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM memory_suggestions WHERE stack_id = ? AND status = 'pending' AND deleted = 0",
                (self.stack_id,),
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
        epoch_id: Optional[str] = None,
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
            epoch_id: If set, filter candidates to this epoch only

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

        # Build epoch filter clause
        epoch_clause = ""
        epoch_params: tuple = ()
        if epoch_id:
            epoch_clause = " AND epoch_id = ?"
            epoch_params = (epoch_id,)

        with self._connect() as conn:
            # Values - ordered by priority, exclude forgotten
            rows = conn.execute(
                f"SELECT * FROM agent_values WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY priority DESC LIMIT ?",
                (self.stack_id, *epoch_params, _values_limit),
            ).fetchall()
            result["values"] = [self._row_to_value(row) for row in rows]

            # Beliefs - ordered by confidence, exclude forgotten
            rows = conn.execute(
                f"SELECT * FROM beliefs WHERE stack_id = ? AND deleted = 0 AND strength > 0.0 AND (is_active = 1 OR is_active IS NULL){epoch_clause} ORDER BY confidence DESC LIMIT ?",
                (self.stack_id, *epoch_params, _beliefs_limit),
            ).fetchall()
            result["beliefs"] = [self._row_to_belief(row) for row in rows]

            # Goals - filtered by status, exclude forgotten
            if goals_status and goals_status != "all":
                rows = conn.execute(
                    f"SELECT * FROM goals WHERE stack_id = ? AND deleted = 0 AND strength > 0.0 AND status = ?{epoch_clause} ORDER BY created_at DESC LIMIT ?",
                    (self.stack_id, goals_status, *epoch_params, _goals_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM goals WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY created_at DESC LIMIT ?",
                    (self.stack_id, *epoch_params, _goals_limit),
                ).fetchall()
            result["goals"] = [self._row_to_goal(row) for row in rows]

            # Drives - all for agent (or limited), exclude forgotten
            if drives_limit is not None:
                rows = conn.execute(
                    f"SELECT * FROM drives WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} LIMIT ?",
                    (self.stack_id, *epoch_params, drives_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM drives WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause}",
                    (self.stack_id, *epoch_params),
                ).fetchall()
            result["drives"] = [self._row_to_drive(row) for row in rows]

            # Episodes - most recent, exclude forgotten
            rows = conn.execute(
                f"SELECT * FROM episodes WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY created_at DESC LIMIT ?",
                (self.stack_id, *epoch_params, _episodes_limit),
            ).fetchall()
            result["episodes"] = [self._row_to_episode(row) for row in rows]

            # Notes - most recent, exclude forgotten
            rows = conn.execute(
                f"SELECT * FROM notes WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY created_at DESC LIMIT ?",
                (self.stack_id, *epoch_params, _notes_limit),
            ).fetchall()
            result["notes"] = [self._row_to_note(row) for row in rows]

            # Relationships - all for agent (or limited), exclude forgotten
            if relationships_limit is not None:
                rows = conn.execute(
                    f"SELECT * FROM relationships WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} LIMIT ?",
                    (self.stack_id, *epoch_params, relationships_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM relationships WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause}",
                    (self.stack_id, *epoch_params),
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
                "SELECT * FROM beliefs WHERE id = ? AND stack_id = ? AND deleted = 0",
                (belief_id, self.stack_id),
            ).fetchone()
        return self._row_to_belief(row) if row else None

    def _get_value_by_id(self, value_id: str) -> Optional[Value]:
        """Get a value by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ? AND stack_id = ? AND deleted = 0",
                (value_id, self.stack_id),
            ).fetchone()
        return self._row_to_value(row) if row else None

    def _get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ? AND stack_id = ? AND deleted = 0",
                (goal_id, self.stack_id),
            ).fetchone()
        return self._row_to_goal(row) if row else None

    def _get_note_by_id(self, note_id: str) -> Optional[Note]:
        """Get a note by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ? AND stack_id = ? AND deleted = 0",
                (note_id, self.stack_id),
            ).fetchone()
        return self._row_to_note(row) if row else None

    def _get_drive_by_id(self, drive_id: str) -> Optional[Drive]:
        """Get a drive by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM drives WHERE id = ? AND stack_id = ? AND deleted = 0",
                (drive_id, self.stack_id),
            ).fetchone()
        return self._row_to_drive(row) if row else None

    def _get_relationship_by_id(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationships WHERE id = ? AND stack_id = ? AND deleted = 0",
                (relationship_id, self.stack_id),
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
        validate_table_name(table)

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
            check_derived_from_cycle(self, memory_type, memory_id, derived_from)
            updates.append("derived_from = ?")
            params.append(self._to_json(derived_from))
        if last_verified is not None:
            updates.append("last_verified = ?")
            params.append(last_verified.isoformat())
        if verification_count is not None:
            updates.append("verification_count = ?")
            params.append(verification_count)
        if confidence_history is not None:
            # Cap confidence_history to prevent unbounded growth
            max_confidence_history = 100
            if len(confidence_history) > max_confidence_history:
                confidence_history = confidence_history[-max_confidence_history:]
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
        params.extend([memory_id, self.stack_id])

        query = (
            f"UPDATE {table} SET {', '.join(updates)} WHERE id = ? AND stack_id = ? AND deleted = 0"
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
                    WHERE stack_id = ? AND deleted = 0
                    AND confidence {op} ?
                    ORDER BY confidence {"ASC" if below else "DESC"}
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.stack_id, threshold, limit)).fetchall()
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
                    WHERE stack_id = ? AND deleted = 0
                    AND source_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """

                try:
                    rows = conn.execute(query, (self.stack_id, source_type, limit)).fetchall()
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

        # Boost strength slightly on access (diminishing returns)
        # boost = 0.02 / (1 + times_accessed / 10) — starts at 0.02, falls to ~0.002 at 100 accesses
        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET times_accessed = COALESCE(times_accessed, 0) + 1,
                       last_accessed = ?,
                       local_updated_at = ?,
                       strength = MIN(1.0,
                           COALESCE(strength, 1.0) + 0.02 / (1.0 + COALESCE(times_accessed, 0) / 10.0))
                   WHERE id = ? AND stack_id = ?""",
                (now, now, memory_id, self.stack_id),
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
                       WHERE id IN ({placeholders}) AND stack_id = ?""",
                    (now, now, *ids, self.stack_id),
                )
                total_updated += cursor.rowcount

            conn.commit()

        return total_updated

    def update_strength(self, memory_type: str, memory_id: str, strength: float) -> bool:
        """Update the strength field of a memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            strength: New strength value (clamped to 0.0-1.0)

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

        strength = max(0.0, min(1.0, strength))
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                f"""UPDATE {table}
                   SET strength = ?,
                       local_updated_at = ?
                   WHERE id = ? AND stack_id = ? AND deleted = 0""",
                (strength, now, memory_id, self.stack_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_strength_batch(self, updates: list[tuple[str, str, float]]) -> int:
        """Update strength for multiple memories in a single transaction.

        Args:
            updates: List of (memory_type, memory_id, new_strength) tuples

        Returns:
            Number of memories successfully updated
        """
        if not updates:
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

        now = self._now()
        total_updated = 0

        with self._connect() as conn:
            for memory_type, memory_id, strength in updates:
                table = table_map.get(memory_type)
                if not table:
                    continue
                strength = max(0.0, min(1.0, strength))
                cursor = conn.execute(
                    f"""UPDATE {table}
                       SET strength = ?,
                           local_updated_at = ?
                       WHERE id = ? AND stack_id = ? AND deleted = 0""",
                    (strength, now, memory_id, self.stack_id),
                )
                total_updated += cursor.rowcount
            conn.commit()

        return total_updated

    def get_all_active_memories(
        self, memory_types: Optional[list[str]] = None
    ) -> list[tuple[str, Any]]:
        """Get all active (non-deleted, non-forgotten) memories for strength decay.

        Args:
            memory_types: Types to include (default: all except raw)

        Returns:
            List of (memory_type, record) tuples
        """
        if memory_types is None:
            memory_types = ["episode", "belief", "goal", "note", "relationship"]

        table_map = {
            "episode": ("episodes", self._row_to_episode),
            "belief": ("beliefs", self._row_to_belief),
            "goal": ("goals", self._row_to_goal),
            "note": ("notes", self._row_to_note),
            "relationship": ("relationships", self._row_to_relationship),
        }

        results = []
        with self._connect() as conn:
            for mtype in memory_types:
                entry = table_map.get(mtype)
                if not entry:
                    continue
                table, converter = entry
                rows = conn.execute(
                    f"""SELECT * FROM {table}
                       WHERE stack_id = ?
                         AND deleted = 0
                         AND COALESCE(is_protected, 0) = 0
                         AND COALESCE(strength, 1.0) > 0.0
                       ORDER BY strength ASC
                       LIMIT 500""",
                    (self.stack_id,),
                ).fetchall()
                for row in rows:
                    record = converter(dict(row))
                    results.append((mtype, record))

        return results

    # === Memory Lifecycle Operations (delegated to MemoryOps) ===

    def forget_memory(self, memory_type: str, memory_id: str, reason: Optional[str] = None) -> bool:
        """Tombstone a memory (mark as forgotten, don't delete)."""
        return self._memory_ops.forget_memory(memory_type, memory_id, reason)

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory."""
        return self._memory_ops.recover_memory(memory_type, memory_id)

    def protect_memory(self, memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark a memory as protected from forgetting."""
        return self._memory_ops.protect_memory(memory_type, memory_id, protected)

    def log_audit(
        self,
        memory_type: str,
        memory_id: str,
        operation: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an audit entry for a memory operation."""
        return self._memory_ops.log_audit(memory_type, memory_id, operation, actor, details)

    def get_audit_log(
        self,
        *,
        memory_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return self._memory_ops.get_audit_log(
            memory_type=memory_type, memory_id=memory_id, operation=operation, limit=limit
        )

    def weaken_memory(self, memory_type: str, memory_id: str, amount: float) -> bool:
        """Reduce a memory's strength by a given amount."""
        return self._memory_ops.weaken_memory(memory_type, memory_id, amount)

    def verify_memory(self, memory_type: str, memory_id: str) -> bool:
        """Verify a memory: boost strength and increment verification count."""
        return self._memory_ops.verify_memory(memory_type, memory_id)

    def boost_memory_strength(self, memory_type: str, memory_id: str, amount: float) -> bool:
        """Boost a memory's strength by a given amount (capped at 1.0)."""
        return self._memory_ops.boost_memory_strength(memory_type, memory_id, amount)

    def get_memories_derived_from(self, memory_type: str, memory_id: str) -> List[tuple]:
        """Find all memories that cite 'type:id' in their derived_from."""
        return self._memory_ops.get_memories_derived_from(memory_type, memory_id)

    def get_ungrounded_memories(self, stack_id: str) -> List[tuple]:
        """Find memories where ALL source refs have strength 0.0 or don't exist."""
        return self._memory_ops.get_ungrounded_memories(stack_id)

    def get_pre_v09_memories(self, stack_id: str) -> List[tuple]:
        """Find memories annotated with kernle:pre-v0.9-migration."""
        return self._memory_ops.get_pre_v09_memories(stack_id)

    def get_forgetting_candidates(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
        threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Get memories that are candidates for forgetting."""
        return self._memory_ops.get_forgetting_candidates(
            self._row_converters(), memory_types, limit, threshold
        )

    def get_forgotten_memories(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get all forgotten (tombstoned) memories."""
        return self._memory_ops.get_forgotten_memories(self._row_converters(), memory_types, limit)

    def _row_converters(self) -> Dict[str, Any]:
        """Return a dict of memory_type -> row converter callable."""
        return {
            "episode": self._row_to_episode,
            "belief": self._row_to_belief,
            "value": self._row_to_value,
            "goal": self._row_to_goal,
            "note": self._row_to_note,
            "drive": self._row_to_drive,
            "relationship": self._row_to_relationship,
        }

    # === Sync Engine (delegated to SyncEngine) ===

    def queue_sync_operation(self, operation: str, table: str, record_id: str, data=None) -> int:
        """Queue a sync operation for later synchronization."""
        return self._sync_engine.queue_sync_operation(operation, table, record_id, data)

    def get_pending_sync_operations(self, limit: int = 100):
        """Get all unsynced operations from the queue."""
        return self._sync_engine.get_pending_sync_operations(limit)

    def mark_synced(self, ids) -> int:
        """Mark sync queue entries as synced."""
        return self._sync_engine.mark_synced(ids)

    def get_sync_status(self):
        """Get sync queue status with counts."""
        return self._sync_engine.get_sync_status()

    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        return self._sync_engine.get_pending_sync_count()

    def get_queued_changes(self, limit: int = 100, max_retries: int = 5):
        """Get queued changes for sync."""
        return self._sync_engine.get_queued_changes(limit, max_retries)

    def _clear_queued_change(self, conn, queue_id: int):
        """Mark a change as synced."""
        self._sync_engine._clear_queued_change(conn, queue_id)

    def _record_sync_failure(self, conn, queue_id: int, error: str) -> int:
        """Record a sync failure and increment retry count."""
        return self._sync_engine._record_sync_failure(conn, queue_id, error)

    def get_failed_sync_records(self, min_retries: int = 5):
        """Get sync records that have exceeded max retries."""
        return self._sync_engine.get_failed_sync_records(min_retries)

    def clear_failed_sync_records(self, older_than_days: int = 7) -> int:
        """Clear failed sync records older than the specified days."""
        return self._sync_engine.clear_failed_sync_records(older_than_days)

    def _get_sync_meta(self, key: str):
        """Get a sync metadata value."""
        return self._sync_engine._get_sync_meta(key)

    def _set_sync_meta(self, key: str, value: str):
        """Set a sync metadata value."""
        self._sync_engine._set_sync_meta(key, value)

    def get_last_sync_time(self):
        """Get the timestamp of the last successful sync."""
        return self._sync_engine.get_last_sync_time()

    def get_sync_conflicts(self, limit: int = 100):
        """Get recent sync conflict history."""
        return self._sync_engine.get_sync_conflicts(limit)

    def save_sync_conflict(self, conflict) -> str:
        """Save a sync conflict record."""
        return self._sync_engine.save_sync_conflict(conflict)

    def clear_sync_conflicts(self, before=None) -> int:
        """Clear sync conflict history."""
        return self._sync_engine.clear_sync_conflicts(before)

    def is_online(self) -> bool:
        """Check if cloud storage is reachable."""
        return self._sync_engine.is_online()

    def _mark_synced(self, conn, table: str, record_id: str):
        """Mark a record as synced with the cloud."""
        self._sync_engine._mark_synced(conn, table, record_id)

    def _get_record_for_push(self, table: str, record_id: str):
        """Get a record by table and ID for pushing to cloud."""
        return self._sync_engine._get_record_for_push(table, record_id)

    def _push_record(self, table: str, record) -> bool:
        """Push a single record to cloud storage."""
        return self._sync_engine._push_record(table, record)

    def sync(self):
        """Sync with cloud storage."""
        return self._sync_engine.sync()

    def pull_changes(self, since=None):
        """Pull changes from cloud since the given timestamp."""
        return self._sync_engine.pull_changes(since)

    def _merge_array_fields(self, table: str, winner, loser):
        """Merge array fields from loser into winner using set union."""
        return self._sync_engine._merge_array_fields(table, winner, loser)

    # === Health Check Compliance Tracking ===

    def log_health_check(
        self, anxiety_score: Optional[int] = None, source: str = "cli", triggered_by: str = "manual"
    ) -> str:
        """Log a health check event for compliance tracking."""
        return _log_health_check(
            self._connect, self.stack_id, self._now(), anxiety_score, source, triggered_by
        )

    def get_health_check_stats(self) -> Dict[str, Any]:
        """Get health check compliance statistics."""
        return _get_health_check_stats(self._connect, self.stack_id)
