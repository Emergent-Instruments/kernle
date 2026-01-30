"""Storage protocol for Kernle backends.

This defines the interface that all storage backends must implement.
Currently supported:
- SQLiteStorage: Local-first storage with sqlite-vec for semantic search
- SupabaseStorage: Cloud storage with pgvector (future: extracted from core.py)
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class VersionConflictError(Exception):
    """Raised when a record's version doesn't match the expected version.

    This indicates a concurrent modification - another process updated the
    record between when we read it and when we tried to save our changes.
    """

    def __init__(self, table: str, record_id: str, expected_version: int, actual_version: int):
        self.table = table
        self.record_id = record_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Version conflict on {table}/{record_id}: "
            f"expected version {expected_version}, found {actual_version}"
        )


# === Shared Utility Functions ===


def utc_now() -> str:
    """Get current timestamp as ISO string in UTC."""
    return datetime.now(timezone.utc).isoformat()


def parse_datetime(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


class SourceType(Enum):
    """How a memory was created/acquired."""

    DIRECT_EXPERIENCE = "direct_experience"  # Directly observed/experienced
    INFERENCE = "inference"  # Inferred from other memories
    TOLD_BY_AGENT = "told_by_agent"  # Told by another agent/user
    CONSOLIDATION = "consolidation"  # Created during consolidation
    UNKNOWN = "unknown"  # Legacy or untracked


class SyncStatus(Enum):
    """Sync status for a record."""

    LOCAL_ONLY = "local_only"  # Not yet synced to cloud
    SYNCED = "synced"  # In sync with cloud
    PENDING_PUSH = "pending_push"  # Local changes need to be pushed
    PENDING_PULL = "pending_pull"  # Cloud has newer version
    CONFLICT = "conflict"  # Conflicting changes


@dataclass
class SyncConflict:
    """Details of a sync conflict that was resolved.

    When local and cloud versions of a record differ, the sync engine
    resolves the conflict using last-write-wins and records the details
    here for user visibility.
    """

    id: str  # Unique ID for this conflict record
    table: str  # Table name (episodes, notes, beliefs, etc.)
    record_id: str  # ID of the record that had a conflict
    local_version: Dict[str, Any]  # Snapshot of local version before resolution
    cloud_version: Dict[str, Any]  # Snapshot of cloud version
    resolution: str  # "local_wins" or "cloud_wins"
    resolved_at: datetime  # When the conflict was resolved
    local_summary: Optional[str] = None  # Human-readable summary of local content
    cloud_summary: Optional[str] = None  # Human-readable summary of cloud content


@dataclass
class SyncResult:
    """Result of a sync operation."""

    pushed: int = 0  # Records pushed to cloud
    pulled: int = 0  # Records pulled from cloud
    conflicts: List[SyncConflict] = field(default_factory=list)  # Detailed conflict records
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def conflict_count(self) -> int:
        """Number of conflicts encountered."""
        return len(self.conflicts)


@dataclass
class QueuedChange:
    """A change queued for sync."""

    id: int
    table_name: str
    record_id: str
    operation: str  # 'insert', 'update', 'delete'
    payload: Optional[str] = None  # JSON payload for the change
    queued_at: Optional[datetime] = None


@dataclass
class ConfidenceChange:
    """A record of confidence change for tracking history."""

    timestamp: datetime
    old_confidence: float
    new_confidence: float
    reason: Optional[str] = None


@dataclass
class RawEntry:
    """A raw memory entry - unstructured blob capture for later processing.

    The raw layer is designed for zero-friction brain dumps. The agent dumps
    whatever they want into the blob field; the system only tracks housekeeping
    metadata like when it was captured and whether it's been processed.

    Note: For backward compatibility, blob/captured_at can be None if the legacy
    content/timestamp fields are provided. New code should always use blob/captured_at.
    """

    id: str
    agent_id: str
    # Primary fields (new schema)
    blob: Optional[str] = None  # The unstructured brain dump - no validation, no length limits
    captured_at: Optional[datetime] = None  # When the entry was captured
    source: str = "unknown"  # Auto-populated: cli|mcp|sdk|import|unknown
    processed: bool = False
    processed_into: Optional[List[str]] = None  # Audit trail: ["episode:abc", "note:xyz"]
    # Sync fields
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False

    # DEPRECATED fields - kept for backward compatibility during migration
    # These will be removed in a future version
    content: Optional[str] = None  # Use blob instead
    timestamp: Optional[datetime] = None  # Use captured_at instead
    tags: Optional[List[str]] = None  # Include in blob text instead
    confidence: float = 1.0  # Not meaningful for raw dumps
    source_type: str = "direct_experience"  # Meta-memory concept, not for raw

    def __post_init__(self):
        """Handle backward compatibility for blob/captured_at."""
        # If blob is not set but content is, use content as blob
        if self.blob is None and self.content is not None:
            self.blob = self.content
        # If captured_at is not set but timestamp is, use timestamp as captured_at
        if self.captured_at is None and self.timestamp is not None:
            self.captured_at = self.timestamp
        # For backward compatibility, also populate legacy fields from new fields
        if self.content is None and self.blob is not None:
            self.content = self.blob
        if self.timestamp is None and self.captured_at is not None:
            self.timestamp = self.captured_at


@dataclass
class MemoryLineage:
    """Provenance chain for a memory."""

    source_type: SourceType
    source_episodes: List[str]  # Episode IDs that support this memory
    derived_from: List[str]  # Memory IDs this was derived from (format: type:id)
    confidence_history: List[ConfidenceChange]


@dataclass
class Episode:
    """An episode/experience record."""

    id: str
    agent_id: str
    objective: str
    outcome: str
    outcome_type: Optional[str] = None
    lessons: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    # Emotional memory fields
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    emotional_arousal: float = 0.0  # 0.0 (calm) to 1.0 (intense)
    emotional_tags: Optional[List[str]] = None  # ["joy", "frustration", "curiosity"]
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: str = "direct_experience"  # SourceType value
    source_episodes: Optional[List[str]] = None  # IDs of related episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Forgetting fields
    times_accessed: int = 0  # Number of times this memory was retrieved
    last_accessed: Optional[datetime] = None  # When last accessed/retrieved
    is_protected: bool = False  # Never decay (core identity memories)
    is_forgotten: bool = False  # Tombstoned, not deleted
    forgotten_at: Optional[datetime] = None  # When it was forgotten
    forgotten_reason: Optional[str] = None  # Why it was forgotten
    # Context/scope fields for project-specific memories
    context: Optional[str] = None  # e.g., "project:api-service", "repo:myorg/myrepo"
    context_tags: Optional[List[str]] = None  # Additional context tags for filtering


@dataclass
class Belief:
    """A belief record."""

    id: str
    agent_id: str
    statement: str
    belief_type: str = "fact"
    confidence: float = 0.8
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    source_type: str = "direct_experience"  # SourceType value
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Belief revision fields
    supersedes: Optional[str] = None  # ID of belief this replaced
    superseded_by: Optional[str] = None  # ID of belief that replaced this
    times_reinforced: int = 0  # How many times confirmed
    is_active: bool = True  # False if superseded/archived
    # Forgetting fields
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    is_forgotten: bool = False
    forgotten_at: Optional[datetime] = None
    forgotten_reason: Optional[str] = None
    # Context/scope fields for project-specific memories
    context: Optional[str] = None  # e.g., "project:api-service", "repo:myorg/myrepo"
    context_tags: Optional[List[str]] = None  # Additional context tags for filtering


@dataclass
class Value:
    """A value record."""

    id: str
    agent_id: str
    name: str
    statement: str
    priority: int = 50
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.9  # Values tend to be high-confidence
    source_type: str = "direct_experience"  # SourceType value
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Forgetting fields
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = True  # Values are protected by default
    is_forgotten: bool = False
    forgotten_at: Optional[datetime] = None
    forgotten_reason: Optional[str] = None
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None


@dataclass
class Goal:
    """A goal record."""

    id: str
    agent_id: str
    title: str
    description: Optional[str] = None
    priority: str = "medium"
    status: str = "active"
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: str = "direct_experience"  # SourceType value
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Forgetting fields
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    is_forgotten: bool = False
    forgotten_at: Optional[datetime] = None
    forgotten_reason: Optional[str] = None
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None


@dataclass
class Note:
    """A note/memory record."""

    id: str
    agent_id: str
    content: str
    note_type: str = "note"
    speaker: Optional[str] = None
    reason: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: str = "direct_experience"  # SourceType value
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Forgetting fields
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    is_forgotten: bool = False
    forgotten_at: Optional[datetime] = None
    forgotten_reason: Optional[str] = None
    # Context/scope fields for project-specific memories
    context: Optional[str] = None  # e.g., "project:api-service", "repo:myorg/myrepo"
    context_tags: Optional[List[str]] = None  # Additional context tags for filtering


@dataclass
class Drive:
    """A drive/motivation record."""

    id: str
    agent_id: str
    drive_type: str
    intensity: float = 0.5
    focus_areas: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: str = "direct_experience"  # SourceType value
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Forgetting fields
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = True  # Drives are protected by default
    is_forgotten: bool = False
    forgotten_at: Optional[datetime] = None
    forgotten_reason: Optional[str] = None
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None


@dataclass
class Relationship:
    """A relationship record."""

    id: str
    agent_id: str
    entity_name: str
    entity_type: str
    relationship_type: str
    notes: Optional[str] = None
    sentiment: float = 0.0
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False
    # Meta-memory fields
    confidence: float = 0.8
    source_type: str = "direct_experience"  # SourceType value
    source_episodes: Optional[List[str]] = None  # IDs of supporting episodes
    derived_from: Optional[List[str]] = None  # Memory IDs this was derived from
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    confidence_history: Optional[List[Dict[str, Any]]] = None
    # Forgetting fields
    times_accessed: int = 0
    last_accessed: Optional[datetime] = None
    is_protected: bool = False
    is_forgotten: bool = False
    forgotten_at: Optional[datetime] = None
    forgotten_reason: Optional[str] = None
    # Context/scope fields for project-specific memories
    context: Optional[str] = None
    context_tags: Optional[List[str]] = None


@dataclass
class Playbook:
    """A playbook/procedural memory record.

    Playbooks are "how I do things" memory - executable procedures
    learned from experience. They encode successful workflows as
    reusable step sequences with applicability conditions and failure modes.
    """

    id: str
    agent_id: str
    name: str  # "Deploy to production"
    description: str  # What this playbook does
    trigger_conditions: List[str]  # When to use this
    steps: List[Dict[str, Any]]  # [{action, details, adaptations}]
    failure_modes: List[str]  # What can go wrong
    recovery_steps: Optional[List[str]] = None  # How to recover
    mastery_level: str = "novice"  # novice/competent/proficient/expert
    times_used: int = 0
    success_rate: float = 0.0
    source_episodes: Optional[List[str]] = None  # Where this was learned
    tags: Optional[List[str]] = None
    # Meta-memory fields
    confidence: float = 0.8
    last_used: Optional[datetime] = None
    created_at: Optional[datetime] = None
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class MemorySuggestion:
    """A suggested memory extracted from raw entries.

    MemorySuggestions are auto-extracted patterns from raw entries that
    require agent review before being promoted to structured memories.
    This enables auto-extraction while keeping the agent in control.

    Workflow:
    1. Raw entry captured (manual or auto-capture)
    2. System extracts suggestions based on patterns
    3. Agent reviews: approve (promote to memory), modify, or reject
    4. Approved suggestions become Episode, Belief, or Note records

    Status values:
    - pending: Awaiting review
    - promoted: Accepted and converted to structured memory
    - modified: Accepted with modifications
    - rejected: Declined (with optional reason)
    """

    id: str
    agent_id: str
    memory_type: str  # "episode", "belief", "note"
    content: Dict[str, Any]  # Structured data for the suggested memory
    confidence: float  # System confidence in this suggestion (0.0-1.0)
    source_raw_ids: List[str]  # Which raw entries this came from
    status: str = "pending"  # pending, promoted, modified, rejected
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_reason: Optional[str] = None
    # Link to promoted memory (if status is promoted/modified)
    promoted_to: Optional[str] = None  # Format: "type:id", e.g., "episode:abc123"
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


@dataclass
class SearchResult:
    """A search result with relevance score."""

    record: Any  # Episode, Note, Belief, Playbook, etc.
    record_type: str
    score: float


@runtime_checkable
class Storage(Protocol):
    """Protocol defining the storage interface for Kernle.

    All storage backends (SQLite, Supabase, etc.) must implement this interface.
    """

    agent_id: str

    # === Episodes ===

    @abstractmethod
    def save_episode(self, episode: Episode) -> str:
        """Save an episode. Returns the episode ID."""
        ...

    @abstractmethod
    def get_episodes(
        self, limit: int = 100, since: Optional[datetime] = None, tags: Optional[List[str]] = None
    ) -> List[Episode]:
        """Get episodes, optionally filtered."""
        ...

    @abstractmethod
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        ...

    # === Emotional Memory ===

    @abstractmethod
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
        ...

    @abstractmethod
    def get_emotional_episodes(self, days: int = 7, limit: int = 100) -> List[Episode]:
        """Get episodes with emotional data for summary calculations.

        Args:
            days: Number of days to look back
            limit: Maximum episodes to retrieve

        Returns:
            Episodes with non-zero emotional data
        """
        ...

    @abstractmethod
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
        ...

    # === Beliefs ===

    @abstractmethod
    def save_belief(self, belief: Belief) -> str:
        """Save a belief. Returns the belief ID."""
        ...

    @abstractmethod
    def get_beliefs(self, limit: int = 100, include_inactive: bool = False) -> List[Belief]:
        """Get beliefs.

        Args:
            limit: Maximum number of beliefs to return
            include_inactive: If True, include superseded/archived beliefs
        """
        ...

    @abstractmethod
    def find_belief(self, statement: str) -> Optional[Belief]:
        """Find a belief by statement (for deduplication)."""
        ...

    # === Values ===

    @abstractmethod
    def save_value(self, value: Value) -> str:
        """Save a value. Returns the value ID."""
        ...

    @abstractmethod
    def get_values(self, limit: int = 100) -> List[Value]:
        """Get values, ordered by priority."""
        ...

    # === Goals ===

    @abstractmethod
    def save_goal(self, goal: Goal) -> str:
        """Save a goal. Returns the goal ID."""
        ...

    @abstractmethod
    def get_goals(self, status: Optional[str] = "active", limit: int = 100) -> List[Goal]:
        """Get goals, optionally filtered by status."""
        ...

    # === Notes ===

    @abstractmethod
    def save_note(self, note: Note) -> str:
        """Save a note. Returns the note ID."""
        ...

    @abstractmethod
    def get_notes(
        self, limit: int = 100, since: Optional[datetime] = None, note_type: Optional[str] = None
    ) -> List[Note]:
        """Get notes, optionally filtered."""
        ...

    # === Drives ===

    @abstractmethod
    def save_drive(self, drive: Drive) -> str:
        """Save or update a drive. Returns the drive ID."""
        ...

    @abstractmethod
    def get_drives(self) -> List[Drive]:
        """Get all drives for the agent."""
        ...

    @abstractmethod
    def get_drive(self, drive_type: str) -> Optional[Drive]:
        """Get a specific drive by type."""
        ...

    # === Relationships ===

    @abstractmethod
    def save_relationship(self, relationship: Relationship) -> str:
        """Save or update a relationship. Returns the relationship ID."""
        ...

    @abstractmethod
    def get_relationships(self, entity_type: Optional[str] = None) -> List[Relationship]:
        """Get relationships, optionally filtered by entity type."""
        ...

    @abstractmethod
    def get_relationship(self, entity_name: str) -> Optional[Relationship]:
        """Get a specific relationship by entity name."""
        ...

    # === Playbooks (Procedural Memory) ===

    @abstractmethod
    def save_playbook(self, playbook: "Playbook") -> str:
        """Save a playbook. Returns the playbook ID."""
        ...

    @abstractmethod
    def get_playbook(self, playbook_id: str) -> Optional["Playbook"]:
        """Get a specific playbook by ID."""
        ...

    @abstractmethod
    def list_playbooks(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List["Playbook"]:
        """Get playbooks, optionally filtered by tags."""
        ...

    @abstractmethod
    def search_playbooks(self, query: str, limit: int = 10) -> List["Playbook"]:
        """Search playbooks by name, description, or triggers."""
        ...

    @abstractmethod
    def update_playbook_usage(self, playbook_id: str, success: bool) -> bool:
        """Update playbook usage statistics.

        Args:
            playbook_id: ID of the playbook
            success: Whether the usage was successful

        Returns:
            True if updated, False if playbook not found
        """
        ...

    # === Raw Entries ===

    @abstractmethod
    def save_raw(
        self, content: str, source: str = "manual", tags: Optional[List[str]] = None
    ) -> str:
        """Save a raw entry for later processing. Returns the entry ID."""
        ...

    @abstractmethod
    def get_raw(self, raw_id: str) -> Optional[RawEntry]:
        """Get a specific raw entry by ID."""
        ...

    @abstractmethod
    def list_raw(self, processed: Optional[bool] = None, limit: int = 100) -> List[RawEntry]:
        """Get raw entries, optionally filtered by processed state."""
        ...

    @abstractmethod
    def mark_raw_processed(self, raw_id: str, processed_into: List[str]) -> bool:
        """Mark a raw entry as processed into other memories.

        Args:
            raw_id: ID of the raw entry
            processed_into: List of memory refs (format: type:id)

        Returns:
            True if updated, False if not found
        """
        ...

    # === Memory Suggestions ===

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        """Save a memory suggestion. Returns the suggestion ID.

        Args:
            suggestion: The suggestion to save

        Returns:
            The suggestion ID
        """
        return suggestion.id  # Default: just return ID (no-op)

    def get_suggestion(self, suggestion_id: str) -> Optional[MemorySuggestion]:
        """Get a specific suggestion by ID.

        Args:
            suggestion_id: ID of the suggestion

        Returns:
            The suggestion or None if not found
        """
        return None

    def get_suggestions(
        self,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemorySuggestion]:
        """Get suggestions, optionally filtered.

        Args:
            status: Filter by status (pending, promoted, modified, rejected)
            memory_type: Filter by suggested memory type (episode, belief, note)
            limit: Maximum suggestions to return

        Returns:
            List of suggestions matching the filters
        """
        return []

    def update_suggestion_status(
        self,
        suggestion_id: str,
        status: str,
        resolution_reason: Optional[str] = None,
        promoted_to: Optional[str] = None,
    ) -> bool:
        """Update the status of a suggestion.

        Args:
            suggestion_id: ID of the suggestion to update
            status: New status (pending, promoted, modified, rejected)
            resolution_reason: Optional reason for the resolution
            promoted_to: Reference to promoted memory (format: type:id)

        Returns:
            True if updated, False if suggestion not found
        """
        return False

    def delete_suggestion(self, suggestion_id: str) -> bool:
        """Delete a suggestion (soft delete by marking deleted=1).

        Args:
            suggestion_id: ID of the suggestion to delete

        Returns:
            True if deleted, False if not found
        """
        return False

    # === Search ===

    @abstractmethod
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
           try cloud search first with timeout
        2. On cloud failure or no credentials, fall back to local search
        3. Local search uses semantic vectors (if available) or text matching

        Args:
            query: Search query
            limit: Maximum results
            record_types: Filter by type (episode, note, belief, etc.)
            prefer_cloud: If True, try cloud search first (default True)

        Returns:
            List of SearchResult objects
        """
        ...

    # === Cloud Search ===

    def has_cloud_credentials(self) -> bool:
        """Check if cloud credentials are available for hybrid search.

        Returns:
            True if backend_url and auth_token are configured.
        """
        return False  # Default: no cloud credentials

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
        return {
            "healthy": False,
            "error": "Cloud search not supported by this storage backend",
        }

    # === Stats ===

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Get counts of each record type."""
        ...

    # === Batch Insertion ===

    def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        """Save multiple episodes in a single transaction.

        This is an optional optimization that storage backends can implement
        to batch multiple database writes into a single transaction, improving
        performance when processing large codebases or bulk imports.

        Default implementation falls back to individual saves.

        Args:
            episodes: List of Episode objects to save

        Returns:
            List of episode IDs (in the same order as input)
        """
        return [self.save_episode(ep) for ep in episodes]

    def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]:
        """Save multiple beliefs in a single transaction.

        This is an optional optimization that storage backends can implement
        to batch multiple database writes into a single transaction, improving
        performance when processing large codebases or bulk imports.

        Default implementation falls back to individual saves.

        Args:
            beliefs: List of Belief objects to save

        Returns:
            List of belief IDs (in the same order as input)
        """
        return [self.save_belief(b) for b in beliefs]

    def save_notes_batch(self, notes: List[Note]) -> List[str]:
        """Save multiple notes in a single transaction.

        This is an optional optimization that storage backends can implement
        to batch multiple database writes into a single transaction, improving
        performance when processing large codebases or bulk imports.

        Default implementation falls back to individual saves.

        Args:
            notes: List of Note objects to save

        Returns:
            List of note IDs (in the same order as input)
        """
        return [self.save_note(n) for n in notes]

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
    ) -> Optional[Dict[str, Any]]:
        """Load all memory types in a single operation (optional optimization).

        This is an optional method that storage backends can implement to
        batch multiple queries into a single database connection, avoiding
        N+1 query patterns.

        Default implementation returns None, indicating the caller should
        fall back to individual get_* methods.

        Args:
            values_limit: Max values to load (None = use high limit for budget loading)
            beliefs_limit: Max beliefs to load (None = use high limit for budget loading)
            goals_limit: Max goals to load (None = use high limit for budget loading)
            goals_status: Goal status filter
            episodes_limit: Max episodes to load (None = use high limit for budget loading)
            notes_limit: Max notes to load (None = use high limit for budget loading)
            drives_limit: Max drives to load (None = all drives)
            relationships_limit: Max relationships to load (None = all relationships)

        Returns:
            Dict with keys: values, beliefs, goals, drives, episodes, notes, relationships
            Or None if not implemented (caller should use individual methods)
        """
        return None  # Default: not implemented, use individual methods

    # === Sync ===

    @abstractmethod
    def sync(self) -> SyncResult:
        """Sync local changes with cloud.

        For cloud-only storage, this is a no-op.
        For local storage, this pushes/pulls changes.
        """
        ...

    @abstractmethod
    def pull_changes(self, since: Optional[datetime] = None) -> SyncResult:
        """Pull changes from cloud since the given timestamp.

        Args:
            since: Pull changes since this time. If None, uses last sync time.

        Returns:
            SyncResult with pulled count and any conflicts.
        """
        ...

    @abstractmethod
    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        ...

    @abstractmethod
    def is_online(self) -> bool:
        """Check if cloud storage is reachable.

        Returns True if connected, False if offline.
        """
        ...

    # === Meta-Memory ===

    @abstractmethod
    def get_memory(self, memory_type: str, memory_id: str) -> Optional[Any]:
        """Get a memory by type and ID.

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship)
            memory_id: ID of the memory

        Returns:
            The memory record or None if not found
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    # === Forgetting ===

    @abstractmethod
    def record_access(self, memory_type: str, memory_id: str) -> bool:
        """Record that a memory was accessed (for salience tracking).

        Increments times_accessed and updates last_accessed timestamp.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if updated, False if memory not found
        """
        ...

    def record_access_batch(self, accesses: List[tuple[str, str]]) -> int:
        """Record multiple memory accesses in a single operation.

        This is an optimization for bulk access tracking, such as when
        loading working memory or returning search results.

        Args:
            accesses: List of (memory_type, memory_id) tuples

        Returns:
            Number of memories successfully updated
        """
        # Default implementation: call record_access for each item
        # Storage backends can override for better performance
        count = 0
        for memory_type, memory_id in accesses:
            if self.record_access(memory_type, memory_id):
                count += 1
        return count

    @abstractmethod
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
            True if forgotten, False if not found or already forgotten
        """
        ...

    @abstractmethod
    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if recovered, False if not found or not forgotten
        """
        ...

    @abstractmethod
    def protect_memory(self, memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark a memory as protected from forgetting.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            protected: True to protect, False to unprotect

        Returns:
            True if updated, False if memory not found
        """
        ...

    @abstractmethod
    def get_forgetting_candidates(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Get memories that are candidates for forgetting.

        Returns memories that are:
        - Not protected
        - Not already forgotten
        - Sorted by salience (lowest first)

        Args:
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of candidate memories with computed salience scores
        """
        ...

    @abstractmethod
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
        ...

    # === Sync Queue Methods (for CLI sync commands) ===

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return utc_now()

    def _clear_queued_change(self, conn: Any, change_id: str) -> None:
        """Clear a queued sync change after successful sync."""
        pass

    def _mark_synced(self, conn: Any, table: str, record_id: str) -> None:
        """Mark a record as synced."""
        pass

    def _set_sync_meta(self, key: str, value: str) -> None:
        """Set sync metadata."""
        pass

    def get_queued_changes(self, limit: int = 100) -> List[Any]:
        """Get pending sync queue changes."""
        return []

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get timestamp of last successful sync."""
        return None

    def get_sync_conflicts(self, limit: int = 100) -> List[SyncConflict]:
        """Get recent sync conflict history.

        Args:
            limit: Maximum number of conflicts to return

        Returns:
            List of SyncConflict records, most recent first
        """
        return []

    def save_sync_conflict(self, conflict: SyncConflict) -> str:
        """Save a sync conflict record.

        Args:
            conflict: The conflict to save

        Returns:
            The conflict ID
        """
        return conflict.id

    def clear_sync_conflicts(self, before: Optional[datetime] = None) -> int:
        """Clear sync conflict history.

        Args:
            before: If provided, only clear conflicts before this timestamp.
                    If None, clear all conflicts.

        Returns:
            Number of conflicts cleared
        """
        return 0

    def _connect(self) -> Any:
        """Get database connection (for sync operations)."""
        raise NotImplementedError("Subclass must implement _connect")

    def _get_record_for_push(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a record formatted for push sync."""
        return None
