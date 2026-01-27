"""SQLite storage backend for Kernle.

Local-first storage with:
- SQLite for structured data
- sqlite-vec for vector search (semantic search)
- Sync metadata for cloud synchronization
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import (
    Storage, SyncResult, SyncStatus,
    Episode, Belief, Value, Goal, Note, Drive, Relationship, SearchResult
)

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1

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
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_sync ON episodes(cloud_synced_at);

-- Beliefs
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    belief_type TEXT DEFAULT 'fact',
    confidence REAL DEFAULT 0.8,
    created_at TEXT NOT NULL,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_beliefs_agent ON beliefs(agent_id);

-- Values
CREATE TABLE IF NOT EXISTS agent_values (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    statement TEXT NOT NULL,
    priority INTEGER DEFAULT 50,
    created_at TEXT NOT NULL,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_values_agent ON agent_values(agent_id);

-- Goals
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_goals_agent ON goals(agent_id);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);

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
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notes_agent ON notes(agent_id);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at);

-- Drives
CREATE TABLE IF NOT EXISTS drives (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    drive_type TEXT NOT NULL,
    intensity REAL DEFAULT 0.5,
    focus_areas TEXT,  -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(agent_id, drive_type)
);
CREATE INDEX IF NOT EXISTS idx_drives_agent ON drives(agent_id);

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
    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,
    UNIQUE(agent_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_relationships_agent ON relationships(agent_id);

-- Sync queue for offline changes
CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    operation TEXT NOT NULL,  -- insert, update, delete
    queued_at TEXT NOT NULL
);

-- Vector embeddings (for semantic search)
-- This will be populated when sqlite-vec is available
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- To detect when re-embedding needed
    embedding BLOB,  -- Vector stored as blob
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_record ON embeddings(table_name, record_id);
"""


class SQLiteStorage:
    """SQLite-based local storage for Kernle.
    
    Features:
    - Zero-config local storage
    - Semantic search with sqlite-vec (when available)
    - Sync metadata for cloud synchronization
    """
    
    def __init__(
        self,
        agent_id: str,
        db_path: Optional[Path] = None,
        cloud_storage: Optional[Storage] = None,
    ):
        self.agent_id = agent_id
        self.db_path = db_path or Path.home() / ".kernle" / "memories.db"
        self.cloud_storage = cloud_storage  # For sync
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        # Check for sqlite-vec
        self._has_vec = self._check_sqlite_vec()
        if not self._has_vec:
            logger.info("sqlite-vec not available, semantic search will use basic text matching")
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.executescript(SCHEMA)
            
            # Check/set schema version
            cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cur.fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()
    
    def _check_sqlite_vec(self) -> bool:
        """Check if sqlite-vec extension is available."""
        try:
            with self._get_conn() as conn:
                conn.enable_load_extension(True)
                # Try to load sqlite-vec
                # This path may vary by installation
                conn.load_extension("vec0")
                return True
        except Exception as e:
            logger.debug(f"sqlite-vec not available: {e}")
            return False
    
    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat()
    
    def _parse_datetime(self, s: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace('Z', '+00:00'))
        except ValueError:
            return None
    
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
    
    def _queue_sync(self, conn: sqlite3.Connection, table: str, record_id: str, operation: str):
        """Queue a change for sync."""
        conn.execute(
            "INSERT INTO sync_queue (table_name, record_id, operation, queued_at) VALUES (?, ?, ?, ?)",
            (table, record_id, operation, self._now())
        )
    
    # === Episodes ===
    
    def save_episode(self, episode: Episode) -> str:
        """Save an episode."""
        if not episode.id:
            episode.id = str(uuid.uuid4())
        
        now = self._now()
        episode.local_updated_at = self._parse_datetime(now)
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO episodes 
                (id, agent_id, objective, outcome, outcome_type, lessons, tags, 
                 created_at, local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.id,
                self.agent_id,
                episode.objective,
                episode.outcome,
                episode.outcome_type,
                self._to_json(episode.lessons),
                self._to_json(episode.tags),
                episode.created_at.isoformat() if episode.created_at else now,
                now,
                episode.cloud_synced_at.isoformat() if episode.cloud_synced_at else None,
                episode.version,
                1 if episode.deleted else 0
            ))
            self._queue_sync(conn, "episodes", episode.id, "upsert")
            conn.commit()
        
        return episode.id
    
    def get_episodes(
        self, 
        limit: int = 100, 
        since: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> List[Episode]:
        """Get episodes."""
        query = "SELECT * FROM episodes WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]
        
        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        
        episodes = [self._row_to_episode(row) for row in rows]
        
        # Filter by tags in Python (SQLite JSON support is limited)
        if tags:
            episodes = [
                e for e in episodes 
                if e.tags and any(t in e.tags for t in tags)
            ]
        
        return episodes
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ? AND agent_id = ?",
                (episode_id, self.agent_id)
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
            local_updated_at=self._parse_datetime(row["local_updated_at"]),
            cloud_synced_at=self._parse_datetime(row["cloud_synced_at"]),
            version=row["version"],
            deleted=bool(row["deleted"])
        )
    
    # === Beliefs ===
    
    def save_belief(self, belief: Belief) -> str:
        """Save a belief."""
        if not belief.id:
            belief.id = str(uuid.uuid4())
        
        now = self._now()
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO beliefs
                (id, agent_id, statement, belief_type, confidence, created_at,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                belief.id,
                self.agent_id,
                belief.statement,
                belief.belief_type,
                belief.confidence,
                belief.created_at.isoformat() if belief.created_at else now,
                now,
                belief.cloud_synced_at.isoformat() if belief.cloud_synced_at else None,
                belief.version,
                1 if belief.deleted else 0
            ))
            self._queue_sync(conn, "beliefs", belief.id, "upsert")
            conn.commit()
        
        return belief.id
    
    def get_beliefs(self, limit: int = 100) -> List[Belief]:
        """Get beliefs."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM beliefs WHERE agent_id = ? AND deleted = 0 ORDER BY created_at DESC LIMIT ?",
                (self.agent_id, limit)
            ).fetchall()
        
        return [self._row_to_belief(row) for row in rows]
    
    def find_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE agent_id = ? AND statement = ? AND deleted = 0",
                (self.agent_id, statement)
            ).fetchone()
        
        return self._row_to_belief(row) if row else None
    
    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        """Convert a row to a Belief."""
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
            deleted=bool(row["deleted"])
        )
    
    # === Values ===
    
    def save_value(self, value: Value) -> str:
        """Save a value."""
        if not value.id:
            value.id = str(uuid.uuid4())
        
        now = self._now()
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_values
                (id, agent_id, name, statement, priority, created_at,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                value.id,
                self.agent_id,
                value.name,
                value.statement,
                value.priority,
                value.created_at.isoformat() if value.created_at else now,
                now,
                value.cloud_synced_at.isoformat() if value.cloud_synced_at else None,
                value.version,
                1 if value.deleted else 0
            ))
            self._queue_sync(conn, "agent_values", value.id, "upsert")
            conn.commit()
        
        return value.id
    
    def get_values(self, limit: int = 100) -> List[Value]:
        """Get values ordered by priority."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_values WHERE agent_id = ? AND deleted = 0 ORDER BY priority DESC LIMIT ?",
                (self.agent_id, limit)
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
            deleted=bool(row["deleted"])
        )
    
    # === Goals ===
    
    def save_goal(self, goal: Goal) -> str:
        """Save a goal."""
        if not goal.id:
            goal.id = str(uuid.uuid4())
        
        now = self._now()
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO goals
                (id, agent_id, title, description, priority, status, created_at,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                goal.id,
                self.agent_id,
                goal.title,
                goal.description,
                goal.priority,
                goal.status,
                goal.created_at.isoformat() if goal.created_at else now,
                now,
                goal.cloud_synced_at.isoformat() if goal.cloud_synced_at else None,
                goal.version,
                1 if goal.deleted else 0
            ))
            self._queue_sync(conn, "goals", goal.id, "upsert")
            conn.commit()
        
        return goal.id
    
    def get_goals(self, status: Optional[str] = "active", limit: int = 100) -> List[Goal]:
        """Get goals."""
        query = "SELECT * FROM goals WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_conn() as conn:
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
            deleted=bool(row["deleted"])
        )
    
    # === Notes ===
    
    def save_note(self, note: Note) -> str:
        """Save a note."""
        if not note.id:
            note.id = str(uuid.uuid4())
        
        now = self._now()
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO notes
                (id, agent_id, content, note_type, speaker, reason, tags, created_at,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                note.id,
                self.agent_id,
                note.content,
                note.note_type,
                note.speaker,
                note.reason,
                self._to_json(note.tags),
                note.created_at.isoformat() if note.created_at else now,
                now,
                note.cloud_synced_at.isoformat() if note.cloud_synced_at else None,
                note.version,
                1 if note.deleted else 0
            ))
            self._queue_sync(conn, "notes", note.id, "upsert")
            conn.commit()
        
        return note.id
    
    def get_notes(
        self, 
        limit: int = 100, 
        since: Optional[datetime] = None,
        note_type: Optional[str] = None
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
        
        with self._get_conn() as conn:
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
            deleted=bool(row["deleted"])
        )
    
    # === Drives ===
    
    def save_drive(self, drive: Drive) -> str:
        """Save or update a drive."""
        if not drive.id:
            drive.id = str(uuid.uuid4())
        
        now = self._now()
        
        with self._get_conn() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM drives WHERE agent_id = ? AND drive_type = ?",
                (self.agent_id, drive.drive_type)
            ).fetchone()
            
            if existing:
                drive.id = existing["id"]
                conn.execute("""
                    UPDATE drives SET
                        intensity = ?, focus_areas = ?, updated_at = ?,
                        local_updated_at = ?, version = version + 1
                    WHERE id = ?
                """, (
                    drive.intensity,
                    self._to_json(drive.focus_areas),
                    now,
                    now,
                    drive.id
                ))
            else:
                conn.execute("""
                    INSERT INTO drives
                    (id, agent_id, drive_type, intensity, focus_areas, created_at, updated_at,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    drive.id,
                    self.agent_id,
                    drive.drive_type,
                    drive.intensity,
                    self._to_json(drive.focus_areas),
                    now,
                    now,
                    now,
                    None,
                    1,
                    0
                ))
            
            self._queue_sync(conn, "drives", drive.id, "upsert")
            conn.commit()
        
        return drive.id
    
    def get_drives(self) -> List[Drive]:
        """Get all drives."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM drives WHERE agent_id = ? AND deleted = 0",
                (self.agent_id,)
            ).fetchall()
        
        return [self._row_to_drive(row) for row in rows]
    
    def get_drive(self, drive_type: str) -> Optional[Drive]:
        """Get a specific drive."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM drives WHERE agent_id = ? AND drive_type = ? AND deleted = 0",
                (self.agent_id, drive_type)
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
            deleted=bool(row["deleted"])
        )
    
    # === Relationships ===
    
    def save_relationship(self, relationship: Relationship) -> str:
        """Save or update a relationship."""
        if not relationship.id:
            relationship.id = str(uuid.uuid4())
        
        now = self._now()
        
        with self._get_conn() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM relationships WHERE agent_id = ? AND entity_name = ?",
                (self.agent_id, relationship.entity_name)
            ).fetchone()
            
            if existing:
                relationship.id = existing["id"]
                conn.execute("""
                    UPDATE relationships SET
                        entity_type = ?, relationship_type = ?, notes = ?,
                        sentiment = ?, interaction_count = ?, last_interaction = ?,
                        local_updated_at = ?, version = version + 1
                    WHERE id = ?
                """, (
                    relationship.entity_type,
                    relationship.relationship_type,
                    relationship.notes,
                    relationship.sentiment,
                    relationship.interaction_count,
                    relationship.last_interaction.isoformat() if relationship.last_interaction else None,
                    now,
                    relationship.id
                ))
            else:
                conn.execute("""
                    INSERT INTO relationships
                    (id, agent_id, entity_name, entity_type, relationship_type, notes,
                     sentiment, interaction_count, last_interaction, created_at,
                     local_updated_at, cloud_synced_at, version, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relationship.id,
                    self.agent_id,
                    relationship.entity_name,
                    relationship.entity_type,
                    relationship.relationship_type,
                    relationship.notes,
                    relationship.sentiment,
                    relationship.interaction_count,
                    relationship.last_interaction.isoformat() if relationship.last_interaction else None,
                    now,
                    now,
                    None,
                    1,
                    0
                ))
            
            self._queue_sync(conn, "relationships", relationship.id, "upsert")
            conn.commit()
        
        return relationship.id
    
    def get_relationships(self, entity_type: Optional[str] = None) -> List[Relationship]:
        """Get relationships."""
        query = "SELECT * FROM relationships WHERE agent_id = ? AND deleted = 0"
        params: List[Any] = [self.agent_id]
        
        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)
        
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [self._row_to_relationship(row) for row in rows]
    
    def get_relationship(self, entity_name: str) -> Optional[Relationship]:
        """Get a specific relationship."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM relationships WHERE agent_id = ? AND entity_name = ? AND deleted = 0",
                (self.agent_id, entity_name)
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
            deleted=bool(row["deleted"])
        )
    
    # === Search ===
    
    def search(
        self, 
        query: str, 
        limit: int = 10,
        record_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search across memories.
        
        If sqlite-vec is available, uses semantic search.
        Otherwise, falls back to basic text matching.
        """
        results = []
        types = record_types or ["episode", "note", "belief"]
        
        # For now, use basic text search
        # TODO: Implement semantic search with sqlite-vec
        search_term = f"%{query}%"
        
        with self._get_conn() as conn:
            if "episode" in types:
                rows = conn.execute(
                    """SELECT * FROM episodes 
                       WHERE agent_id = ? AND deleted = 0 
                       AND (objective LIKE ? OR outcome LIKE ? OR lessons LIKE ?)
                       LIMIT ?""",
                    (self.agent_id, search_term, search_term, search_term, limit)
                ).fetchall()
                for row in rows:
                    results.append(SearchResult(
                        record=self._row_to_episode(row),
                        record_type="episode",
                        score=1.0  # Basic match, no real score
                    ))
            
            if "note" in types:
                rows = conn.execute(
                    """SELECT * FROM notes
                       WHERE agent_id = ? AND deleted = 0
                       AND content LIKE ?
                       LIMIT ?""",
                    (self.agent_id, search_term, limit)
                ).fetchall()
                for row in rows:
                    results.append(SearchResult(
                        record=self._row_to_note(row),
                        record_type="note",
                        score=1.0
                    ))
            
            if "belief" in types:
                rows = conn.execute(
                    """SELECT * FROM beliefs
                       WHERE agent_id = ? AND deleted = 0
                       AND statement LIKE ?
                       LIMIT ?""",
                    (self.agent_id, search_term, limit)
                ).fetchall()
                for row in rows:
                    results.append(SearchResult(
                        record=self._row_to_belief(row),
                        record_type="belief",
                        score=1.0
                    ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    # === Stats ===
    
    def get_stats(self) -> Dict[str, int]:
        """Get counts of each record type."""
        with self._get_conn() as conn:
            stats = {}
            for table, key in [
                ("episodes", "episodes"),
                ("beliefs", "beliefs"),
                ("agent_values", "values"),
                ("goals", "goals"),
                ("notes", "notes"),
                ("drives", "drives"),
                ("relationships", "relationships"),
            ]:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE agent_id = ? AND deleted = 0",
                    (self.agent_id,)
                ).fetchone()[0]
                stats[key] = count
        
        return stats
    
    # === Sync ===
    
    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sync_queue").fetchone()[0]
        return count
    
    def sync(self) -> SyncResult:
        """Sync with cloud storage.
        
        For now, this is a stub. Full implementation will:
        1. Push queued changes to cloud
        2. Pull changes from cloud
        3. Handle conflicts
        """
        result = SyncResult()
        
        if not self.cloud_storage:
            logger.debug("No cloud storage configured, skipping sync")
            return result
        
        # TODO: Implement full sync logic
        # For now, just count pending
        result.pushed = 0
        result.pulled = 0
        
        return result
