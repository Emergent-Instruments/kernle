"""PostgreSQL storage backend for Kernle using Supabase.

Cloud storage with:
- PostgreSQL for structured data
- pgvector for semantic search (via Supabase)
- Full compatibility with Storage protocol

SCHEMA DIVERGENCE FROM SQLiteStorage:
--------------------------------------
The Supabase/Postgres schema differs from SQLite in several ways:

1. TABLE NAMES:
   - SQLite: episodes, beliefs, values, goals, notes, drives, relationships, playbooks, raw_entries
   - Postgres: agent_episodes, agent_beliefs, agent_values, agent_goals, memories (for notes),
               agent_drives, agent_relationships, playbooks

2. NOTES TABLE:
   - SQLite: Dedicated 'notes' table with note_type, speaker, reason fields
   - Postgres: Uses 'memories' table with source='curated', metadata JSON for note fields

3. MISSING TABLES IN POSTGRES:
   - raw_entries: No raw entry processing support yet

4. COLUMN DIFFERENCES:
   - Episodes: outcome_description (Postgres) vs outcome (SQLite)
   - Notes: owner_id (Postgres) vs agent_id (SQLite)
   - Relationships: other_agent_id (Postgres) vs entity_name (SQLite)
   - Drives: last_satisfied_at (Postgres) vs updated_at (SQLite)

5. FORGETTING FIELDS:
   - SQLite: Full support (times_accessed, last_accessed, is_protected, is_forgotten, etc.)
   - Postgres: Full support implemented

6. FEATURES NOT YET SUPPORTED IN POSTGRES:
   - Raw entry processing

Use SQLiteStorage for full functionality. SupabaseStorage is suitable for
cloud sync of core memories (episodes, beliefs, values, goals, notes, drives, relationships,
playbooks) and supports forgetting/tombstoning operations.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Playbook,
    RawEntry,
    Relationship,
    SearchResult,
    SyncResult,
    Value,
    parse_datetime,
    utc_now,
)

logger = logging.getLogger(__name__)


class SupabaseStorage:
    """Supabase/PostgreSQL storage backend for Kernle.

    Cloud-based storage that wraps the existing Supabase implementation.
    Implements the Storage protocol for compatibility with SQLiteStorage.
    """

    def __init__(
        self,
        agent_id: str,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.supabase_url = (
            supabase_url or os.environ.get("KERNLE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        )
        self.supabase_key = (
            supabase_key
            or os.environ.get("KERNLE_SUPABASE_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        )

        self._client = None

    @property
    def client(self):
        """Lazy-load Supabase client."""
        if self._client is None:
            if not self.supabase_url or not self.supabase_key:
                raise ValueError(
                    "Supabase credentials required. Set KERNLE_SUPABASE_URL and KERNLE_SUPABASE_KEY."
                )

            # Validate URL structure
            self._validate_supabase_url(self.supabase_url)

            if not self.supabase_key.strip():
                raise ValueError("Supabase key cannot be empty")

            # Basic key format validation (Supabase keys are JWT-like)
            if len(self.supabase_key) < 100:
                raise ValueError("Supabase key appears to be invalid (too short)")

            from supabase import create_client

            self._client = create_client(self.supabase_url, self.supabase_key)
        return self._client

    def _validate_supabase_url(self, url: str) -> None:
        """Validate Supabase URL for proper structure and safety."""
        from urllib.parse import urlparse

        if not url:
            raise ValueError("Supabase URL cannot be empty")

        # Must be HTTPS (security requirement)
        if not url.startswith("https://"):
            raise ValueError("Supabase URL must use HTTPS")

        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")

        # Validate URL components
        if not parsed.netloc:
            raise ValueError("Invalid Supabase URL: missing host")

        if parsed.path and parsed.path not in ("", "/"):
            raise ValueError("Invalid Supabase URL: unexpected path component")

        # Check for valid Supabase domain patterns
        host = parsed.netloc.lower()
        valid_patterns = [
            ".supabase.co",  # Standard Supabase hosted
            ".supabase.in",  # Alternative Supabase domain
            "localhost",  # Local development
            "127.0.0.1",  # Local development
        ]

        is_valid_host = any(
            host.endswith(pattern) or host == pattern.lstrip(".") for pattern in valid_patterns
        )

        # Also allow custom self-hosted domains (check for reasonable structure)
        if not is_valid_host:
            # For self-hosted, at minimum ensure it looks like a valid hostname
            if not all(c.isalnum() or c in ".-:" for c in host):
                raise ValueError("Invalid Supabase URL: hostname contains invalid characters")
            if ".." in host or host.startswith(".") or host.endswith("."):
                raise ValueError("Invalid Supabase URL: malformed hostname")
            # Log a warning for non-standard hosts
            logger.warning(f"Using non-standard Supabase host: {host}. Ensure this is intentional.")

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return utc_now()

    def _parse_datetime(self, s: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        return parse_datetime(s)

    # === Episodes ===

    def save_episode(self, episode: Episode) -> str:
        """Save an episode."""
        if not episode.id:
            episode.id = str(uuid.uuid4())

        now = self._now()

        # Map to Supabase schema (agent_episodes table)
        data = {
            "id": episode.id,
            "agent_id": self.agent_id,
            "objective": episode.objective,
            "outcome_type": episode.outcome_type or "partial",
            "outcome_description": episode.outcome,
            "lessons_learned": episode.lessons or [],
            "tags": episode.tags or [],
            "is_reflected": True,
            "confidence": 0.8,
            # Emotional memory fields
            "emotional_valence": episode.emotional_valence,
            "emotional_arousal": episode.emotional_arousal,
            "emotional_tags": episode.emotional_tags or [],
            # Sync metadata
            "local_updated_at": now,
            "cloud_synced_at": now,
            "version": episode.version,
        }

        self.client.table("agent_episodes").upsert(data).execute()
        return episode.id

    def get_episodes(
        self, limit: int = 100, since: Optional[datetime] = None, tags: Optional[List[str]] = None
    ) -> List[Episode]:
        """Get episodes, optionally filtered."""
        query = (
            self.client.table("agent_episodes")
            .select("*")
            .eq("agent_id", self.agent_id)
            .order("created_at", desc=True)
            .limit(limit)
        )

        if since:
            query = query.gte("created_at", since.isoformat())

        result = query.execute()

        episodes = [self._row_to_episode(row) for row in result.data]

        # Filter by tags if specified
        if tags:
            episodes = [e for e in episodes if e.tags and any(t in e.tags for t in tags)]

        return episodes

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        result = (
            self.client.table("agent_episodes")
            .select("*")
            .eq("id", episode_id)
            .eq("agent_id", self.agent_id)
            .execute()
        )

        if result.data:
            return self._row_to_episode(result.data[0])
        return None

    def _row_to_episode(self, row: Dict[str, Any]) -> Episode:
        """Convert a Supabase row to an Episode."""
        return Episode(
            id=row["id"],
            agent_id=row["agent_id"],
            objective=row.get("objective", ""),
            outcome=row.get("outcome_description", ""),
            outcome_type=row.get("outcome_type"),
            lessons=row.get("lessons_learned"),
            tags=row.get("tags"),
            created_at=self._parse_datetime(row.get("created_at")),
            emotional_valence=row.get("emotional_valence", 0.0) or 0.0,
            emotional_arousal=row.get("emotional_arousal", 0.0) or 0.0,
            emotional_tags=row.get("emotional_tags"),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1),
            deleted=False,
            # Meta-memory fields
            confidence=row.get("confidence", 0.8),
            source_type=row.get("source_type", "direct_experience"),
            source_episodes=row.get("source_episodes"),
            derived_from=row.get("derived_from"),
            last_verified=self._parse_datetime(row.get("last_verified")),
            verification_count=row.get("verification_count", 0),
            confidence_history=row.get("confidence_history"),
            # Context/scope fields
            context=row.get("context"),
            context_tags=row.get("context_tags"),
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

        try:
            result = (
                self.client.table("agent_episodes")
                .update(
                    {
                        "emotional_valence": valence,
                        "emotional_arousal": arousal,
                        "emotional_tags": tags or [],
                        "local_updated_at": now,
                    }
                )
                .eq("id", episode_id)
                .eq("agent_id", self.agent_id)
                .execute()
            )

            return len(result.data) > 0
        except Exception as e:
            logger.warning(f"Failed to update episode emotion: {e}")
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
            valence_range: (min, max) valence filter
            arousal_range: (min, max) arousal filter
            tags: Emotional tags to match
            limit: Maximum results

        Returns:
            List of matching episodes
        """
        query = (
            self.client.table("agent_episodes")
            .select("*")
            .eq("agent_id", self.agent_id)
            .order("created_at", desc=True)
        )

        if valence_range:
            query = query.gte("emotional_valence", valence_range[0])
            query = query.lte("emotional_valence", valence_range[1])

        if arousal_range:
            query = query.gte("emotional_arousal", arousal_range[0])
            query = query.lte("emotional_arousal", arousal_range[1])

        query = query.limit(limit * 2 if tags else limit)

        result = query.execute()
        episodes = [self._row_to_episode(row) for row in result.data]

        # Filter by emotional tags in Python
        if tags:
            episodes = [
                e for e in episodes if e.emotional_tags and any(t in e.emotional_tags for t in tags)
            ][:limit]

        return episodes

    def get_emotional_episodes(self, days: int = 7, limit: int = 100) -> List[Episode]:
        """Get episodes with emotional data for summary calculations."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        result = (
            self.client.table("agent_episodes")
            .select("*")
            .eq("agent_id", self.agent_id)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        # Filter to episodes with emotional data
        episodes = [self._row_to_episode(row) for row in result.data]
        return [
            e
            for e in episodes
            if e.emotional_valence != 0.0 or e.emotional_arousal != 0.0 or e.emotional_tags
        ]

    # === Beliefs ===

    def save_belief(self, belief: Belief) -> str:
        """Save a belief."""
        if not belief.id:
            belief.id = str(uuid.uuid4())

        now = self._now()

        data = {
            "id": belief.id,
            "agent_id": self.agent_id,
            "statement": belief.statement,
            "belief_type": belief.belief_type,
            "confidence": belief.confidence,
            "is_active": True,
            "is_foundational": False,
            # Sync metadata
            "local_updated_at": now,
            "cloud_synced_at": now,
            "version": belief.version,
        }

        self.client.table("agent_beliefs").upsert(data).execute()
        return belief.id

    def get_beliefs(self, limit: int = 100, include_inactive: bool = False) -> List[Belief]:
        """Get beliefs.

        Args:
            limit: Maximum number of beliefs to return
            include_inactive: If True, include superseded/archived beliefs
        """
        query = self.client.table("agent_beliefs").select("*").eq("agent_id", self.agent_id)
        if not include_inactive:
            query = query.eq("is_active", True)
        result = query.order("confidence", desc=True).limit(limit).execute()

        return [self._row_to_belief(row) for row in result.data]

    def find_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement (for deduplication)."""
        result = (
            self.client.table("agent_beliefs")
            .select("*")
            .eq("agent_id", self.agent_id)
            .eq("statement", statement)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )

        if result.data:
            return self._row_to_belief(result.data[0])
        return None

    def _row_to_belief(self, row: Dict[str, Any]) -> Belief:
        """Convert a Supabase row to a Belief."""
        return Belief(
            id=row["id"],
            agent_id=row["agent_id"],
            statement=row.get("statement", ""),
            belief_type=row.get("belief_type", "fact"),
            confidence=row.get("confidence", 0.8),
            created_at=self._parse_datetime(row.get("created_at")),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1),
            deleted=not row.get("is_active", True),
            # Meta-memory fields
            source_type=row.get("source_type", "direct_experience"),
            source_episodes=row.get("source_episodes"),
            derived_from=row.get("derived_from"),
            last_verified=self._parse_datetime(row.get("last_verified")),
            verification_count=row.get("verification_count", 0),
            confidence_history=row.get("confidence_history"),
            # Context/scope fields
            context=row.get("context"),
            context_tags=row.get("context_tags"),
        )

    # === Values ===

    def save_value(self, value: Value) -> str:
        """Save a value."""
        if not value.id:
            value.id = str(uuid.uuid4())

        now = self._now()

        data = {
            "id": value.id,
            "agent_id": self.agent_id,
            "name": value.name,
            "statement": value.statement,
            "priority": value.priority,
            "value_type": "core_value",
            "is_active": True,
            "is_foundational": False,
            # Sync metadata
            "local_updated_at": now,
            "cloud_synced_at": now,
            "version": value.version,
        }

        self.client.table("agent_values").upsert(data).execute()
        return value.id

    def get_values(self, limit: int = 100) -> List[Value]:
        """Get values, ordered by priority."""
        result = (
            self.client.table("agent_values")
            .select("*")
            .eq("agent_id", self.agent_id)
            .eq("is_active", True)
            .order("priority", desc=True)
            .limit(limit)
            .execute()
        )

        return [self._row_to_value(row) for row in result.data]

    def _row_to_value(self, row: Dict[str, Any]) -> Value:
        """Convert a Supabase row to a Value."""
        return Value(
            id=row["id"],
            agent_id=row["agent_id"],
            name=row.get("name", ""),
            statement=row.get("statement", ""),
            priority=row.get("priority", 50),
            created_at=self._parse_datetime(row.get("created_at")),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1),
            deleted=not row.get("is_active", True),
            # Meta-memory fields
            confidence=row.get("confidence", 0.9),
            source_type=row.get("source_type", "direct_experience"),
            source_episodes=row.get("source_episodes"),
            derived_from=row.get("derived_from"),
            last_verified=self._parse_datetime(row.get("last_verified")),
            verification_count=row.get("verification_count", 0),
            confidence_history=row.get("confidence_history"),
            # Context/scope fields
            context=row.get("context"),
            context_tags=row.get("context_tags"),
        )

    # === Goals ===

    def save_goal(self, goal: Goal) -> str:
        """Save a goal."""
        if not goal.id:
            goal.id = str(uuid.uuid4())

        now = self._now()

        data = {
            "id": goal.id,
            "agent_id": self.agent_id,
            "title": goal.title,
            "description": goal.description or goal.title,
            "priority": goal.priority,
            "status": goal.status,
            "visibility": "public",
            # Sync metadata
            "local_updated_at": now,
            "cloud_synced_at": now,
            "version": goal.version,
        }

        self.client.table("agent_goals").upsert(data).execute()
        return goal.id

    def get_goals(self, status: Optional[str] = "active", limit: int = 100) -> List[Goal]:
        """Get goals, optionally filtered by status."""
        query = (
            self.client.table("agent_goals")
            .select("*")
            .eq("agent_id", self.agent_id)
            .order("created_at", desc=True)
            .limit(limit)
        )

        if status:
            query = query.eq("status", status)

        result = query.execute()
        return [self._row_to_goal(row) for row in result.data]

    def _row_to_goal(self, row: Dict[str, Any]) -> Goal:
        """Convert a Supabase row to a Goal."""
        return Goal(
            id=row["id"],
            agent_id=row["agent_id"],
            title=row.get("title", ""),
            description=row.get("description"),
            priority=row.get("priority", "medium"),
            status=row.get("status", "active"),
            created_at=self._parse_datetime(row.get("created_at")),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1),
            deleted=False,
            # Meta-memory fields
            confidence=row.get("confidence", 0.8),
            source_type=row.get("source_type", "direct_experience"),
            source_episodes=row.get("source_episodes"),
            derived_from=row.get("derived_from"),
            last_verified=self._parse_datetime(row.get("last_verified")),
            verification_count=row.get("verification_count", 0),
            confidence_history=row.get("confidence_history"),
            # Context/scope fields
            context=row.get("context"),
            context_tags=row.get("context_tags"),
        )

    # === Notes ===

    def save_note(self, note: Note) -> str:
        """Save a note."""
        if not note.id:
            note.id = str(uuid.uuid4())

        now = self._now()

        # Build content based on note type
        content = note.content
        if note.note_type == "decision" and note.reason:
            content = f"**Decision**: {note.content}\n**Reason**: {note.reason}"
        elif note.note_type == "quote" and note.speaker:
            content = f'> "{note.content}"\n> — {note.speaker}'
        elif note.note_type == "insight":
            content = f"**Insight**: {note.content}"

        metadata = {
            "note_type": note.note_type,
            "tags": note.tags or [],
        }
        if note.speaker:
            metadata["speaker"] = note.speaker
        if note.reason:
            metadata["reason"] = note.reason

        data = {
            "id": note.id,
            "owner_id": self.agent_id,
            "owner_type": "agent",
            "content": content,
            "source": "curated",
            "metadata": metadata,
            "visibility": "private",
            "is_curated": True,
            "is_protected": False,
            # Sync metadata
            "local_updated_at": now,
            "cloud_synced_at": now,
            "version": note.version,
        }

        self.client.table("memories").upsert(data).execute()
        return note.id

    def get_notes(
        self, limit: int = 100, since: Optional[datetime] = None, note_type: Optional[str] = None
    ) -> List[Note]:
        """Get notes, optionally filtered."""
        query = (
            self.client.table("memories")
            .select("*")
            .eq("owner_id", self.agent_id)
            .eq("source", "curated")
            .order("created_at", desc=True)
            .limit(limit)
        )

        if since:
            query = query.gte("created_at", since.isoformat())

        result = query.execute()
        notes = [self._row_to_note(row) for row in result.data]

        # Filter by note_type if specified
        if note_type:
            notes = [n for n in notes if n.note_type == note_type]

        return notes

    def _row_to_note(self, row: Dict[str, Any]) -> Note:
        """Convert a Supabase row to a Note."""
        metadata = row.get("metadata", {}) or {}
        return Note(
            id=row["id"],
            agent_id=row.get("owner_id", ""),
            content=row.get("content", ""),
            note_type=metadata.get("note_type", "note"),
            speaker=metadata.get("speaker"),
            reason=metadata.get("reason"),
            tags=metadata.get("tags"),
            created_at=self._parse_datetime(row.get("created_at")),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1),
            deleted=False,
            # Meta-memory fields
            confidence=row.get("confidence", 0.8),
            source_type=row.get("source_type", "direct_experience"),
            source_episodes=row.get("source_episodes"),
            derived_from=row.get("derived_from"),
            last_verified=self._parse_datetime(row.get("last_verified")),
            verification_count=row.get("verification_count", 0),
            confidence_history=row.get("confidence_history"),
            # Context/scope fields
            context=row.get("context"),
            context_tags=row.get("context_tags"),
        )

    # === Drives ===

    def save_drive(self, drive: Drive) -> str:
        """Save or update a drive."""
        if not drive.id:
            drive.id = str(uuid.uuid4())

        now = self._now()

        data = {
            "id": drive.id,
            "agent_id": self.agent_id,
            "drive_type": drive.drive_type,
            "intensity": max(0.0, min(1.0, drive.intensity)),
            "focus_areas": drive.focus_areas or [],
            "last_satisfied_at": now,
            # Sync metadata
            "local_updated_at": now,
            "cloud_synced_at": now,
            "version": drive.version,
        }

        # Use upsert to avoid TOCTOU race condition
        # ON CONFLICT on (agent_id, drive_type) unique constraint
        result = (
            self.client.table("agent_drives")
            .upsert(data, on_conflict="agent_id,drive_type")
            .execute()
        )

        # If upsert returned an existing row with different ID, update drive.id
        if result.data and result.data[0].get("id"):
            drive.id = result.data[0]["id"]

        return drive.id

    def get_drives(self) -> List[Drive]:
        """Get all drives for the agent."""
        result = (
            self.client.table("agent_drives").select("*").eq("agent_id", self.agent_id).execute()
        )

        return [self._row_to_drive(row) for row in result.data]

    def get_drive(self, drive_type: str) -> Optional[Drive]:
        """Get a specific drive by type."""
        result = (
            self.client.table("agent_drives")
            .select("*")
            .eq("agent_id", self.agent_id)
            .eq("drive_type", drive_type)
            .execute()
        )

        if result.data:
            return self._row_to_drive(result.data[0])
        return None

    def _row_to_drive(self, row: Dict[str, Any]) -> Drive:
        """Convert a Supabase row to a Drive."""
        return Drive(
            id=row["id"],
            agent_id=row["agent_id"],
            drive_type=row.get("drive_type", ""),
            intensity=row.get("intensity", 0.5),
            focus_areas=row.get("focus_areas"),
            created_at=self._parse_datetime(row.get("created_at")),
            updated_at=self._parse_datetime(row.get("last_satisfied_at")),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1),
            deleted=False,
            # Meta-memory fields
            confidence=row.get("confidence", 0.8),
            source_type=row.get("source_type", "direct_experience"),
            source_episodes=row.get("source_episodes"),
            derived_from=row.get("derived_from"),
            last_verified=self._parse_datetime(row.get("last_verified")),
            verification_count=row.get("verification_count", 0),
            confidence_history=row.get("confidence_history"),
            # Context/scope fields
            context=row.get("context"),
            context_tags=row.get("context_tags"),
        )

    # === Relationships ===

    def save_relationship(self, relationship: Relationship) -> str:
        """Save or update a relationship."""
        if not relationship.id:
            relationship.id = str(uuid.uuid4())

        now = self._now()
        last_interaction = (
            relationship.last_interaction.isoformat() if relationship.last_interaction else now
        )

        try:
            # Use atomic RPC function to increment interaction_count
            # This prevents race conditions with concurrent relationship updates
            result = self.client.rpc(
                "increment_interaction_count",
                {
                    "p_agent_id": self.agent_id,
                    "p_other_agent_id": relationship.entity_name,
                    "p_trust_level": relationship.sentiment,
                    "p_notes": relationship.notes,
                    "p_last_interaction": last_interaction,
                },
            ).execute()

            if result.data and len(result.data) > 0:
                relationship.id = str(result.data[0]["id"])
        except Exception as e:
            logger.warning(f"Failed to save relationship to agent_relationships table: {e}")
            # Fall back to saving as a note
            self.save_note(
                Note(
                    id=relationship.id,
                    agent_id=self.agent_id,
                    content=f"Relationship with {relationship.entity_name}: {relationship.notes}",
                    note_type="note",
                    tags=["relationship", relationship.entity_name],
                )
            )

        return relationship.id

    def get_relationships(self, entity_type: Optional[str] = None) -> List[Relationship]:
        """Get relationships, optionally filtered by entity type."""
        try:
            result = (
                self.client.table("agent_relationships")
                .select("*")
                .eq("agent_id", self.agent_id)
                .order("last_interaction", desc=True)
                .execute()
            )

            return [self._row_to_relationship(row) for row in result.data]
        except Exception:
            # Table might not exist
            return []

    def get_relationship(self, entity_name: str) -> Optional[Relationship]:
        """Get a specific relationship by entity name."""
        try:
            result = (
                self.client.table("agent_relationships")
                .select("*")
                .eq("agent_id", self.agent_id)
                .eq("other_agent_id", entity_name)
                .execute()
            )

            if result.data:
                return self._row_to_relationship(result.data[0])
        except Exception:
            pass
        return None

    def _row_to_relationship(self, row: Dict[str, Any]) -> Relationship:
        """Convert a Supabase row to a Relationship."""
        return Relationship(
            id=row["id"],
            agent_id=row["agent_id"],
            entity_name=row.get("other_agent_id", ""),
            entity_type="agent",
            relationship_type="interaction",
            notes=row.get("notes"),
            sentiment=row.get("trust_level", 0.0),
            interaction_count=row.get("interaction_count", 0),
            last_interaction=self._parse_datetime(row.get("last_interaction")),
            created_at=self._parse_datetime(row.get("created_at")),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1),
            deleted=False,
            # Meta-memory fields
            confidence=row.get("confidence", 0.8),
            source_type=row.get("source_type", "direct_experience"),
            source_episodes=row.get("source_episodes"),
            derived_from=row.get("derived_from"),
            last_verified=self._parse_datetime(row.get("last_verified")),
            verification_count=row.get("verification_count", 0),
            confidence_history=row.get("confidence_history"),
            # Context/scope fields
            context=row.get("context"),
            context_tags=row.get("context_tags"),
        )

    # === Search ===

    def search(
        self, query: str, limit: int = 10, record_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search across memories.

        For now, uses basic text matching.
        TODO: Use pgvector for semantic search.
        """
        results = []
        types = record_types or ["episode", "note", "belief"]
        query_lower = query.lower()

        if "episode" in types:
            episodes = (
                self.client.table("agent_episodes")
                .select("*")
                .eq("agent_id", self.agent_id)
                .order("created_at", desc=True)
                .limit(limit * 5)
                .execute()
            )

            for row in episodes.data:
                text = f"{row.get('objective', '')} {row.get('outcome_description', '')} {' '.join(row.get('lessons_learned', []))}"
                if query_lower in text.lower():
                    results.append(
                        SearchResult(
                            record=self._row_to_episode(row),
                            record_type="episode",
                            score=1.0,
                        )
                    )

        if "note" in types:
            notes = (
                self.client.table("memories")
                .select("*")
                .eq("owner_id", self.agent_id)
                .eq("source", "curated")
                .order("created_at", desc=True)
                .limit(limit * 5)
                .execute()
            )

            for row in notes.data:
                if query_lower in row.get("content", "").lower():
                    results.append(
                        SearchResult(
                            record=self._row_to_note(row),
                            record_type="note",
                            score=1.0,
                        )
                    )

        if "belief" in types:
            beliefs = (
                self.client.table("agent_beliefs")
                .select("*")
                .eq("agent_id", self.agent_id)
                .eq("is_active", True)
                .limit(limit * 5)
                .execute()
            )

            for row in beliefs.data:
                if query_lower in row.get("statement", "").lower():
                    results.append(
                        SearchResult(
                            record=self._row_to_belief(row),
                            record_type="belief",
                            score=1.0,
                        )
                    )

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    # === Stats ===

    def get_stats(self) -> Dict[str, int]:
        """Get counts of each record type."""
        stats = {}

        # Episodes
        result = (
            self.client.table("agent_episodes")
            .select("id", count="exact")
            .eq("agent_id", self.agent_id)
            .execute()
        )
        stats["episodes"] = result.count or 0

        # Beliefs
        result = (
            self.client.table("agent_beliefs")
            .select("id", count="exact")
            .eq("agent_id", self.agent_id)
            .eq("is_active", True)
            .execute()
        )
        stats["beliefs"] = result.count or 0

        # Values
        result = (
            self.client.table("agent_values")
            .select("id", count="exact")
            .eq("agent_id", self.agent_id)
            .eq("is_active", True)
            .execute()
        )
        stats["values"] = result.count or 0

        # Goals
        result = (
            self.client.table("agent_goals")
            .select("id", count="exact")
            .eq("agent_id", self.agent_id)
            .eq("status", "active")
            .execute()
        )
        stats["goals"] = result.count or 0

        # Notes
        result = (
            self.client.table("memories")
            .select("id", count="exact")
            .eq("owner_id", self.agent_id)
            .eq("source", "curated")
            .execute()
        )
        stats["notes"] = result.count or 0

        # Drives
        result = (
            self.client.table("agent_drives")
            .select("id", count="exact")
            .eq("agent_id", self.agent_id)
            .execute()
        )
        stats["drives"] = result.count or 0

        # Relationships
        try:
            result = (
                self.client.table("agent_relationships")
                .select("id", count="exact")
                .eq("agent_id", self.agent_id)
                .execute()
            )
            stats["relationships"] = result.count or 0
        except Exception:
            stats["relationships"] = 0

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
        """Load all memory types in a single operation.

        This provides the same interface as SQLiteStorage.load_all() for
        consistency. While Supabase doesn't benefit from connection batching
        like SQLite, this method provides a convenient single-call API for
        loading working memory context.

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

        result: Dict[str, Any] = {
            "values": [],
            "beliefs": [],
            "goals": [],
            "drives": [],
            "episodes": [],
            "notes": [],
            "relationships": [],
        }

        try:
            # Values - ordered by priority
            result["values"] = self.get_values(limit=_values_limit)
        except Exception as e:
            logger.warning(f"Failed to load values: {e}")

        try:
            # Beliefs - ordered by confidence
            result["beliefs"] = self.get_beliefs(limit=_beliefs_limit)
        except Exception as e:
            logger.warning(f"Failed to load beliefs: {e}")

        try:
            # Goals - filtered by status
            status = goals_status if goals_status != "all" else None
            result["goals"] = self.get_goals(status=status, limit=_goals_limit)
        except Exception as e:
            logger.warning(f"Failed to load goals: {e}")

        try:
            # Drives - all for agent (limited if specified)
            drives = self.get_drives()
            if drives_limit is not None:
                drives = drives[:drives_limit]
            result["drives"] = drives
        except Exception as e:
            logger.warning(f"Failed to load drives: {e}")

        try:
            # Episodes - most recent
            result["episodes"] = self.get_episodes(limit=_episodes_limit)
        except Exception as e:
            logger.warning(f"Failed to load episodes: {e}")

        try:
            # Notes - most recent
            result["notes"] = self.get_notes(limit=_notes_limit)
        except Exception as e:
            logger.warning(f"Failed to load notes: {e}")

        try:
            # Relationships - all for agent (limited if specified)
            relationships = self.get_relationships()
            if relationships_limit is not None:
                relationships = relationships[:relationships_limit]
            result["relationships"] = relationships
        except Exception as e:
            logger.warning(f"Failed to load relationships: {e}")

        return result

    # === Sync ===

    def sync(self) -> SyncResult:
        """Sync operation for cloud storage.

        For cloud storage, this is a no-op since data is already in the cloud.
        """
        return SyncResult()

    def pull_changes(self, since: Optional[datetime] = None) -> SyncResult:
        """Pull changes from cloud.

        For cloud storage, this is a no-op since we're already in the cloud.
        """
        return SyncResult()

    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync.

        For cloud storage, this is always 0.
        """
        return 0

    def is_online(self) -> bool:
        """Check if cloud storage is reachable.

        For cloud storage, test the connection.
        """
        try:
            # Simple connectivity test
            self.client.table("agent_episodes").select("id").limit(1).execute()
            return True
        except Exception:
            return False

    # === Meta-Memory ===

    def get_memory(self, memory_type: str, memory_id: str) -> Optional[Any]:
        """Get a memory by type and ID."""
        table_map = {
            "episode": ("agent_episodes", self._row_to_episode),
            "belief": ("agent_beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("agent_goals", self._row_to_goal),
            "note": ("memories", self._row_to_note),
            "drive": ("agent_drives", self._row_to_drive),
            "relationship": ("agent_relationships", self._row_to_relationship),
        }

        if memory_type not in table_map:
            return None

        table, converter = table_map[memory_type]

        try:
            # Handle notes which use owner_id instead of agent_id
            if memory_type == "note":
                result = (
                    self.client.table(table)
                    .select("*")
                    .eq("id", memory_id)
                    .eq("owner_id", self.agent_id)
                    .execute()
                )
            else:
                result = (
                    self.client.table(table)
                    .select("*")
                    .eq("id", memory_id)
                    .eq("agent_id", self.agent_id)
                    .execute()
                )

            if result.data:
                return converter(result.data[0])
        except Exception as e:
            logger.warning(f"Could not get {memory_type}:{memory_id}: {e}")

        return None

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
        """Update meta-memory fields for a memory."""
        table_map = {
            "episode": "agent_episodes",
            "belief": "agent_beliefs",
            "value": "agent_values",
            "goal": "agent_goals",
            "note": "memories",
            "drive": "agent_drives",
            "relationship": "agent_relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        update_data = {}
        if confidence is not None:
            update_data["confidence"] = confidence
        if source_type is not None:
            update_data["source_type"] = source_type
        if source_episodes is not None:
            update_data["source_episodes"] = source_episodes
        if derived_from is not None:
            update_data["derived_from"] = derived_from
        if last_verified is not None:
            update_data["last_verified"] = last_verified.isoformat()
        if verification_count is not None:
            update_data["verification_count"] = verification_count
        if confidence_history is not None:
            update_data["confidence_history"] = confidence_history

        if not update_data:
            return False

        try:
            # Handle notes which use owner_id
            if memory_type == "note":
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "owner_id", self.agent_id
                ).execute()
            else:
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "agent_id", self.agent_id
                ).execute()
            return True
        except Exception as e:
            logger.warning(f"Could not update meta for {memory_type}:{memory_id}: {e}")
            return False

    def get_memories_by_confidence(
        self,
        threshold: float,
        below: bool = True,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by confidence threshold."""
        results = []
        types = memory_types or ["episode", "belief", "value", "goal", "note"]

        table_map = {
            "episode": ("agent_episodes", self._row_to_episode),
            "belief": ("agent_beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("agent_goals", self._row_to_goal),
            "note": ("memories", self._row_to_note),
        }

        for memory_type in types:
            if memory_type not in table_map:
                continue

            table, converter = table_map[memory_type]

            try:
                # Handle notes which use owner_id
                if memory_type == "note":
                    query = (
                        self.client.table(table)
                        .select("*")
                        .eq("owner_id", self.agent_id)
                        .eq("source", "curated")
                    )
                else:
                    query = self.client.table(table).select("*").eq("agent_id", self.agent_id)

                if below:
                    query = query.lt("confidence", threshold)
                else:
                    query = query.gte("confidence", threshold)

                query = query.order("confidence", desc=not below).limit(limit)
                result = query.execute()

                for row in result.data:
                    results.append(
                        SearchResult(
                            record=converter(row),
                            record_type=memory_type,
                            score=row.get("confidence", 0.8),
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not query {table} by confidence: {e}")

        results.sort(key=lambda x: x.score, reverse=not below)
        return results[:limit]

    def get_memories_by_source(
        self,
        source_type: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories filtered by source type."""
        results = []
        types = memory_types or ["episode", "belief", "value", "goal", "note"]

        table_map = {
            "episode": ("agent_episodes", self._row_to_episode),
            "belief": ("agent_beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("agent_goals", self._row_to_goal),
            "note": ("memories", self._row_to_note),
        }

        for memory_type in types:
            if memory_type not in table_map:
                continue

            table, converter = table_map[memory_type]

            try:
                if memory_type == "note":
                    query = (
                        self.client.table(table)
                        .select("*")
                        .eq("owner_id", self.agent_id)
                        .eq("source", "curated")
                    )
                else:
                    query = self.client.table(table).select("*").eq("agent_id", self.agent_id)

                query = (
                    query.eq("source_type", source_type).order("created_at", desc=True).limit(limit)
                )
                result = query.execute()

                for row in result.data:
                    results.append(
                        SearchResult(
                            record=converter(row),
                            record_type=memory_type,
                            score=row.get("confidence", 0.8),
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not query {table} by source_type: {e}")

        return results[:limit]

    # === Playbooks (Procedural Memory) ===

    def save_playbook(self, playbook: Playbook) -> str:
        """Save a playbook. Returns the playbook ID."""
        if not playbook.id:
            playbook.id = str(uuid.uuid4())

        now = self._now()

        # Map to Supabase schema (playbooks table)
        data = {
            "id": playbook.id,
            "agent_id": self.agent_id,
            "name": playbook.name,
            "description": playbook.description,
            "trigger_conditions": playbook.trigger_conditions or [],
            "steps": playbook.steps or [],
            "failure_modes": playbook.failure_modes or [],
            "recovery_steps": playbook.recovery_steps,
            "mastery_level": playbook.mastery_level or "novice",
            "times_used": playbook.times_used or 0,
            "success_rate": playbook.success_rate or 0.0,
            "source_episodes": playbook.source_episodes,
            "tags": playbook.tags or [],
            "confidence": playbook.confidence or 0.8,
            "last_used": playbook.last_used.isoformat() if playbook.last_used else None,
            # Sync metadata
            "local_updated_at": now,
            "cloud_synced_at": now,
            "version": playbook.version or 1,
            "deleted": playbook.deleted or False,
        }

        self.client.table("playbooks").upsert(data).execute()
        return playbook.id

    def get_playbook(self, playbook_id: str) -> Optional[Playbook]:
        """Get a specific playbook by ID."""
        result = (
            self.client.table("playbooks")
            .select("*")
            .eq("id", playbook_id)
            .eq("agent_id", self.agent_id)
            .eq("deleted", False)
            .execute()
        )

        if result.data:
            return self._row_to_playbook(result.data[0])
        return None

    def list_playbooks(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Playbook]:
        """Get playbooks, optionally filtered by tags."""
        result = (
            self.client.table("playbooks")
            .select("*")
            .eq("agent_id", self.agent_id)
            .eq("deleted", False)
            .order("times_used", desc=True)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        playbooks = [self._row_to_playbook(row) for row in result.data]

        # Filter by tags if provided
        if tags:
            tags_set = set(tags)
            playbooks = [p for p in playbooks if p.tags and tags_set.intersection(p.tags)]

        return playbooks

    def search_playbooks(self, query: str, limit: int = 10) -> List[Playbook]:
        """Search playbooks by name, description, or triggers using text search."""
        # Supabase doesn't have a simple LIKE filter, so we fetch and filter in-memory
        result = (
            self.client.table("playbooks")
            .select("*")
            .eq("agent_id", self.agent_id)
            .eq("deleted", False)
            .order("times_used", desc=True)
            .limit(limit * 5)  # Fetch more to filter
            .execute()
        )

        playbooks = []
        query_lower = query.lower()
        for row in result.data:
            # Check if query matches name, description, or trigger_conditions
            name = (row.get("name") or "").lower()
            description = (row.get("description") or "").lower()
            triggers = row.get("trigger_conditions") or []
            triggers_str = " ".join(str(t).lower() for t in triggers)

            if query_lower in name or query_lower in description or query_lower in triggers_str:
                playbooks.append(self._row_to_playbook(row))
                if len(playbooks) >= limit:
                    break

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

        try:
            self.client.table("playbooks").update(
                {
                    "times_used": new_times_used,
                    "success_rate": new_success_rate,
                    "mastery_level": new_mastery,
                    "last_used": now,
                    "local_updated_at": now,
                    "version": (playbook.version or 1) + 1,
                }
            ).eq("id", playbook_id).eq("agent_id", self.agent_id).execute()

            return True
        except Exception as e:
            logger.error(f"Failed to update playbook usage: {e}")
            return False

    def _row_to_playbook(self, row: Dict[str, Any]) -> Playbook:
        """Convert a Supabase row to a Playbook."""
        return Playbook(
            id=row["id"],
            agent_id=row.get("agent_id", self.agent_id),
            name=row.get("name", ""),
            description=row.get("description", ""),
            trigger_conditions=row.get("trigger_conditions") or [],
            steps=row.get("steps") or [],
            failure_modes=row.get("failure_modes") or [],
            recovery_steps=row.get("recovery_steps"),
            mastery_level=row.get("mastery_level", "novice"),
            times_used=row.get("times_used", 0) or 0,
            success_rate=row.get("success_rate", 0.0) or 0.0,
            source_episodes=row.get("source_episodes"),
            tags=row.get("tags"),
            confidence=row.get("confidence", 0.8) or 0.8,
            last_used=self._parse_datetime(row.get("last_used")),
            created_at=self._parse_datetime(row.get("created_at")),
            local_updated_at=self._parse_datetime(row.get("local_updated_at")),
            cloud_synced_at=self._parse_datetime(row.get("cloud_synced_at")),
            version=row.get("version", 1) or 1,
            deleted=bool(row.get("deleted", False)),
        )

    # === Raw Entries - NOT YET SUPPORTED ===

    def save_raw(
        self, content: str, source: str = "manual", tags: Optional[List[str]] = None
    ) -> str:
        """Save a raw entry for later processing. Returns the entry ID."""
        raise NotImplementedError(
            "SupabaseStorage does not yet support save_raw. "
            "Use SQLiteStorage for full functionality."
        )

    def get_raw(self, raw_id: str) -> Optional["RawEntry"]:
        """Get a specific raw entry by ID."""
        raise NotImplementedError(
            "SupabaseStorage does not yet support get_raw. "
            "Use SQLiteStorage for full functionality."
        )

    def list_raw(self, processed: Optional[bool] = None, limit: int = 100) -> List["RawEntry"]:
        """Get raw entries, optionally filtered by processed state."""
        raise NotImplementedError(
            "SupabaseStorage does not yet support list_raw. "
            "Use SQLiteStorage for full functionality."
        )

    def mark_raw_processed(self, raw_id: str, processed_into: List[str]) -> bool:
        """Mark a raw entry as processed into other memories."""
        raise NotImplementedError(
            "SupabaseStorage does not yet support mark_raw_processed. "
            "Use SQLiteStorage for full functionality."
        )

    # === Forgetting Operations ===

    def record_access(self, memory_type: str, memory_id: str) -> bool:
        """Record that a memory was accessed (for salience tracking).

        Increments times_accessed and updates last_accessed timestamp.

        Note: This uses read-then-write which is not atomic. Under high concurrency,
        some increments may be lost. This is acceptable for access tracking where
        approximate counts are sufficient for salience calculation. For production
        systems with high concurrency, consider using a PostgreSQL RPC function
        with atomic increment.

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship)
            memory_id: ID of the memory

        Returns:
            True if updated, False if memory not found or invalid type
        """
        table_map = {
            "episode": "agent_episodes",
            "belief": "agent_beliefs",
            "value": "agent_values",
            "goal": "agent_goals",
            "note": "memories",
            "drive": "agent_drives",
            "relationship": "agent_relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            logger.debug(f"Invalid memory_type for record_access: {memory_type}")
            return False

        now = self._now()

        try:
            # First get current access count
            if memory_type == "note":
                result = (
                    self.client.table(table)
                    .select("times_accessed")
                    .eq("id", memory_id)
                    .eq("owner_id", self.agent_id)
                    .execute()
                )
            else:
                result = (
                    self.client.table(table)
                    .select("times_accessed")
                    .eq("id", memory_id)
                    .eq("agent_id", self.agent_id)
                    .execute()
                )

            if not result.data:
                return False

            current_count = result.data[0].get("times_accessed") or 0

            # Update with incremented count
            update_data = {
                "times_accessed": current_count + 1,
                "last_accessed": now,
                "local_updated_at": now,
            }

            if memory_type == "note":
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "owner_id", self.agent_id
                ).execute()
            else:
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "agent_id", self.agent_id
                ).execute()

            return True
        except Exception as e:
            logger.warning(f"Could not record access for {memory_type}:{memory_id}: {e}")
            return False

    def record_access_batch(self, accesses: List[tuple[str, str]]) -> int:
        """Record multiple memory accesses in a single operation.

        Args:
            accesses: List of (memory_type, memory_id) tuples

        Returns:
            Number of successfully updated records
        """
        updated = 0
        for memory_type, memory_id in accesses:
            if self.record_access(memory_type, memory_id):
                updated += 1
        return updated

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
            "episode": "agent_episodes",
            "belief": "agent_beliefs",
            "value": "agent_values",
            "goal": "agent_goals",
            "note": "memories",
            "drive": "agent_drives",
            "relationship": "agent_relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        now = self._now()

        try:
            # Check if memory exists and is not protected or already forgotten
            if memory_type == "note":
                result = (
                    self.client.table(table)
                    .select("is_protected, is_forgotten")
                    .eq("id", memory_id)
                    .eq("owner_id", self.agent_id)
                    .execute()
                )
            else:
                result = (
                    self.client.table(table)
                    .select("is_protected, is_forgotten")
                    .eq("id", memory_id)
                    .eq("agent_id", self.agent_id)
                    .execute()
                )

            if not result.data:
                return False

            row = result.data[0]
            if row.get("is_protected"):
                logger.debug(f"Cannot forget protected memory {memory_type}:{memory_id}")
                return False
            if row.get("is_forgotten"):
                return False  # Already forgotten

            # Mark as forgotten
            update_data = {
                "is_forgotten": True,
                "forgotten_at": now,
                "forgotten_reason": reason,
                "local_updated_at": now,
            }

            if memory_type == "note":
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "owner_id", self.agent_id
                ).execute()
            else:
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "agent_id", self.agent_id
                ).execute()

            return True
        except Exception as e:
            logger.warning(f"Could not forget {memory_type}:{memory_id}: {e}")
            return False

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if recovered, False if not found or not forgotten
        """
        table_map = {
            "episode": "agent_episodes",
            "belief": "agent_beliefs",
            "value": "agent_values",
            "goal": "agent_goals",
            "note": "memories",
            "drive": "agent_drives",
            "relationship": "agent_relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        now = self._now()

        try:
            # Check if memory is forgotten
            if memory_type == "note":
                result = (
                    self.client.table(table)
                    .select("is_forgotten")
                    .eq("id", memory_id)
                    .eq("owner_id", self.agent_id)
                    .execute()
                )
            else:
                result = (
                    self.client.table(table)
                    .select("is_forgotten")
                    .eq("id", memory_id)
                    .eq("agent_id", self.agent_id)
                    .execute()
                )

            if not result.data or not result.data[0].get("is_forgotten"):
                return False

            # Clear forgotten status
            update_data = {
                "is_forgotten": False,
                "forgotten_at": None,
                "forgotten_reason": None,
                "local_updated_at": now,
            }

            if memory_type == "note":
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "owner_id", self.agent_id
                ).execute()
            else:
                self.client.table(table).update(update_data).eq("id", memory_id).eq(
                    "agent_id", self.agent_id
                ).execute()

            return True
        except Exception as e:
            logger.warning(f"Could not recover {memory_type}:{memory_id}: {e}")
            return False

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
            "episode": "agent_episodes",
            "belief": "agent_beliefs",
            "value": "agent_values",
            "goal": "agent_goals",
            "note": "memories",
            "drive": "agent_drives",
            "relationship": "agent_relationships",
        }

        table = table_map.get(memory_type)
        if not table:
            return False

        now = self._now()

        try:
            update_data = {
                "is_protected": protected,
                "local_updated_at": now,
            }

            if memory_type == "note":
                result = (
                    self.client.table(table)
                    .update(update_data)
                    .eq("id", memory_id)
                    .eq("owner_id", self.agent_id)
                    .execute()
                )
            else:
                result = (
                    self.client.table(table)
                    .update(update_data)
                    .eq("id", memory_id)
                    .eq("agent_id", self.agent_id)
                    .execute()
                )

            return len(result.data) > 0
        except Exception as e:
            logger.warning(f"Could not protect {memory_type}:{memory_id}: {e}")
            return False

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
            "episode": ("agent_episodes", self._row_to_episode),
            "belief": ("agent_beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("agent_goals", self._row_to_goal),
            "note": ("memories", self._row_to_note),
            "drive": ("agent_drives", self._row_to_drive),
            "relationship": ("agent_relationships", self._row_to_relationship),
        }

        now = datetime.now(timezone.utc)
        half_life = 30.0  # days

        for memory_type in types:
            if memory_type not in table_map:
                continue

            table, converter = table_map[memory_type]

            try:
                # Query for non-protected, non-forgotten memories
                if memory_type == "note":
                    query = (
                        self.client.table(table)
                        .select("*")
                        .eq("owner_id", self.agent_id)
                        .eq("source", "curated")
                    )
                else:
                    query = self.client.table(table).select("*").eq("agent_id", self.agent_id)

                # Filter out protected and forgotten
                # Note: Supabase doesn't support complex boolean queries easily,
                # so we filter in Python
                result = query.limit(limit * 2).execute()

                for row in result.data:
                    # Skip protected or already forgotten
                    if row.get("is_protected") or row.get("is_forgotten"):
                        continue

                    record = converter(row)

                    # Calculate salience
                    confidence = row.get("confidence") or 0.8
                    times_accessed = row.get("times_accessed") or 0

                    # Get last access time
                    last_accessed_str = row.get("last_accessed")
                    if last_accessed_str:
                        last_accessed = self._parse_datetime(last_accessed_str)
                    else:
                        # Use created_at if never accessed
                        created_at_str = row.get("created_at")
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
            "episode": ("agent_episodes", self._row_to_episode),
            "belief": ("agent_beliefs", self._row_to_belief),
            "value": ("agent_values", self._row_to_value),
            "goal": ("agent_goals", self._row_to_goal),
            "note": ("memories", self._row_to_note),
            "drive": ("agent_drives", self._row_to_drive),
            "relationship": ("agent_relationships", self._row_to_relationship),
        }

        for memory_type in types:
            if memory_type not in table_map:
                continue

            table, converter = table_map[memory_type]

            try:
                if memory_type == "note":
                    query = (
                        self.client.table(table)
                        .select("*")
                        .eq("owner_id", self.agent_id)
                        .eq("source", "curated")
                        .eq("is_forgotten", True)
                        .order("forgotten_at", desc=True)
                        .limit(limit)
                    )
                else:
                    query = (
                        self.client.table(table)
                        .select("*")
                        .eq("agent_id", self.agent_id)
                        .eq("is_forgotten", True)
                        .order("forgotten_at", desc=True)
                        .limit(limit)
                    )

                result = query.execute()

                for row in result.data:
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
