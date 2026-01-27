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
    conflicts: int = 0        # Conflicts encountered
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0


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
    # Sync metadata
    local_updated_at: Optional[datetime] = None
    cloud_synced_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False


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
    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        ...
