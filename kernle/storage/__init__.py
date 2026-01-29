"""Kernle storage backends.

This module provides the storage abstraction layer for Kernle.
Supports multiple backends:
- SQLiteStorage: Local-first storage (default)
- SupabaseStorage: Cloud storage via Supabase/PostgreSQL

Use get_storage() factory function to create the appropriate backend.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from .base import (
    Belief,
    ConfidenceChange,
    Drive,
    Episode,
    Goal,
    MemoryLineage,
    MemorySuggestion,
    Note,
    Playbook,
    QueuedChange,
    RawEntry,
    Relationship,
    SearchResult,
    SourceType,
    Storage,
    SyncConflict,
    SyncResult,
    SyncStatus,
    Value,
    parse_datetime,
    utc_now,
)
from .embeddings import (
    EmbeddingProvider,
    HashEmbedder,
    OpenAIEmbedder,
    get_default_embedder,
)
from .postgres import SupabaseStorage
from .sqlite import SQLiteStorage

StorageType = Literal["sqlite", "postgres", "supabase", "auto"]


def get_storage(
    agent_id: str,
    storage_type: StorageType = "auto",
    *,
    # SQLite options
    db_path: Optional[Path] = None,
    # Postgres/Supabase options
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    # Sync options
    cloud_storage: Optional["Storage"] = None,
) -> Storage:
    """Factory function to create the appropriate storage backend.

    Args:
        agent_id: Unique identifier for the agent
        storage_type: Type of storage to use:
            - "sqlite": Local SQLite storage (default)
            - "postgres" or "supabase": Cloud Supabase/PostgreSQL storage
            - "auto": Auto-detect based on environment variables
        db_path: Path to SQLite database file (sqlite only)
        supabase_url: Supabase project URL (postgres only)
        supabase_key: Supabase API key (postgres only)
        cloud_storage: Optional cloud storage for sync (sqlite only)

    Returns:
        Storage backend implementing the Storage protocol

    Examples:
        # Auto-detect (uses SQLite if no Supabase credentials)
        storage = get_storage("my_agent")

        # Explicit SQLite
        storage = get_storage("my_agent", "sqlite", db_path=Path("~/.kernle/test.db"))

        # Explicit Supabase
        storage = get_storage("my_agent", "postgres",
                              supabase_url="https://xxx.supabase.co",
                              supabase_key="my_key")

        # Local with cloud sync
        cloud = get_storage("my_agent", "postgres")
        local = get_storage("my_agent", "sqlite", cloud_storage=cloud)
    """
    # Resolve environment variables for Supabase
    supabase_url = (
        supabase_url or os.environ.get("KERNLE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    )
    supabase_key = (
        supabase_key
        or os.environ.get("KERNLE_SUPABASE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )

    # Auto-detect storage type
    if storage_type == "auto":
        # Use Supabase if credentials are available, otherwise SQLite
        if supabase_url and supabase_key:
            storage_type = "postgres"
        else:
            storage_type = "sqlite"

    # Create appropriate backend
    if storage_type == "sqlite":
        return SQLiteStorage(
            agent_id=agent_id,
            db_path=db_path,
            cloud_storage=cloud_storage,
        )
    elif storage_type in ("postgres", "supabase"):
        return SupabaseStorage(
            agent_id=agent_id,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


__all__ = [
    # Protocol and types
    "Storage",
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
    "Playbook",
    "SearchResult",
    "RawEntry",
    "MemorySuggestion",
    # Meta-memory types
    "SourceType",
    "ConfidenceChange",
    "MemoryLineage",
    # Utilities
    "utc_now",
    "parse_datetime",
    # Implementations
    "SQLiteStorage",
    "SupabaseStorage",
    # Embeddings
    "EmbeddingProvider",
    "HashEmbedder",
    "OpenAIEmbedder",
    "get_default_embedder",
    # Factory
    "get_storage",
]
