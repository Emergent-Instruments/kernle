"""Kernle storage backends.

This module provides the storage abstraction layer for Kernle.
Local-first storage using SQLite.
"""

from .base import (
    # Dynamic trust constants
    DEFAULT_TRUST,
    SEED_TRUST,
    SELF_TRUST_FLOOR,
    TRUST_DECAY_RATE,
    TRUST_DEPTH_DECAY,
    TRUST_THRESHOLDS,
    Belief,
    ConfidenceChange,
    DiagnosticReport,
    DiagnosticSession,
    Drive,
    EntityModel,
    Episode,
    Epoch,
    Goal,
    MemoryLineage,
    MemorySuggestion,
    Note,
    Playbook,
    QueuedChange,
    RawEntry,
    Relationship,
    RelationshipHistoryEntry,
    SearchResult,
    SelfNarrative,
    SourceType,
    Storage,
    Summary,
    SyncConflict,
    SyncResult,
    SyncStatus,
    TrustAssessment,
    Value,
    VersionConflictError,
    parse_datetime,
    utc_now,
)
from .embeddings import (
    EmbeddingProvider,
    HashEmbedder,
    OpenAIEmbedder,
    get_default_embedder,
)
from .sqlite import SQLiteStorage

__all__ = [
    # Protocol and types
    "Storage",
    "VersionConflictError",
    "SyncConflict",
    "SyncResult",
    "SyncStatus",
    "QueuedChange",
    # Data classes
    "Episode",
    "Belief",
    "Value",
    "Goal",
    "Note",
    "Drive",
    "Relationship",
    "RelationshipHistoryEntry",
    "EntityModel",
    "Playbook",
    "SearchResult",
    "SelfNarrative",
    "RawEntry",
    "MemorySuggestion",
    "TrustAssessment",
    "Epoch",
    "Summary",
    "DiagnosticSession",
    "DiagnosticReport",
    # Dynamic trust constants
    "DEFAULT_TRUST",
    "SEED_TRUST",
    "TRUST_DECAY_RATE",
    "TRUST_DEPTH_DECAY",
    "TRUST_THRESHOLDS",
    "SELF_TRUST_FLOOR",
    # Meta-memory types
    "SourceType",
    "ConfidenceChange",
    "MemoryLineage",
    # Utilities
    "utc_now",
    "parse_datetime",
    # Implementations
    "SQLiteStorage",
    # Embeddings
    "EmbeddingProvider",
    "HashEmbedder",
    "OpenAIEmbedder",
    "get_default_embedder",
]
