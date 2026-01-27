"""Storage protocol for Kernle backends.

This defines the interface that all storage backends must implement.
Currently supported:
- SQLiteStorage: Local-first storage with sqlite-vec for semantic search
- SupabaseStorage: Cloud storage with pgvector (future: extracted from core.py)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable
from enum import Enum


class SourceType(Enum):
    """How a memory was created/acquired."""
    DIRECT_EXPERIENCE = "direct_experience"  # Directly observed/experienced
    INFERENCE = "inference"                   # Inferred from other memories
    TOLD_BY_AGENT = "told_by_agent"          # Told by another agent/user
    CONSOLIDATION = "consolidation"           # Created during consolidation
    UNKNOWN = "unknown"                       # Legacy or untracked


class SyncStatus(Enum):
    """Sync status for a record."""
    LOCAL_ONLY = "local_only"      # Not yet synced to cloud
    SYNCED = "synced"              # In sync with cloud
    PENDING_PUSH = "pending_push"  # Local changes need to be pushed
    PENDING_PULL = "pending_pull"  # Cloud has newer version
    CONFLICT = "conflict"          # Conflicting changes


@dataclass
class SyncResult:
    """Result of a sync operation."""
    pushed: int = 0           # Records pushed to cloud
    pulled: int = 0           # Records pulled from cloud
    conflicts: int = 0        # Conflicts encountered (resolved with last-write-wins)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0


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
class MemoryLineage:
    """Provenance chain for a memory."""
    source_type: SourceType
    source_episodes: List[str]  # Episode IDs that support this memory
    derived_from: List[str]      # Memory IDs this was derived from (format: type:id)
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


@dataclass
class SearchResult:
    """A search result with relevance score."""
    record: Any  # Episode, Note, Belief, etc.
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
        self, 
        limit: int = 100, 
        since: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> List[Episode]:
        """Get episodes, optionally filtered."""
        ...
    
    @abstractmethod
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID."""
        ...
    
    # === Beliefs ===
    
    @abstractmethod
    def save_belief(self, belief: Belief) -> str:
        """Save a belief. Returns the belief ID."""
        ...
    
    @abstractmethod
    def get_beliefs(self, limit: int = 100) -> List[Belief]:
        """Get beliefs."""
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
        self, 
        limit: int = 100, 
        since: Optional[datetime] = None,
        note_type: Optional[str] = None
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
    
    # === Search ===
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        limit: int = 10,
        record_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Semantic search across memories.
        
        Args:
            query: Search query
            limit: Maximum results
            record_types: Filter by type (episode, note, belief, etc.)
        """
        ...
    
    # === Stats ===
    
    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Get counts of each record type."""
        ...
    
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
