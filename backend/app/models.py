"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Auth Models
# =============================================================================

class AgentRegister(BaseModel):
    """Request to register a new agent."""
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")
    display_name: str | None = None
    email: str | None = None  # Optional, for account recovery


class AgentLogin(BaseModel):
    """Request for agent token."""
    agent_id: str
    secret: str  # Agent's secret key


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str | None = None  # Stable user identifier for namespacing
    secret: str | None = None  # Only included on registration (one-time)


class AgentInfo(BaseModel):
    """Agent information."""
    agent_id: str
    display_name: str | None
    created_at: datetime
    last_sync_at: datetime | None
    user_id: str | None = None


# =============================================================================
# Sync Models
# =============================================================================

class SyncOperation(BaseModel):
    """A single sync operation from local to cloud."""
    operation: Literal["insert", "update", "delete"]
    table: str
    record_id: str
    data: dict[str, Any] | None = None  # None for delete
    local_updated_at: datetime
    version: int = 1


class SyncPushRequest(BaseModel):
    """Request to push local changes to cloud."""
    operations: list[SyncOperation]
    last_sync_at: datetime | None = None  # For conflict detection


class SyncPushResponse(BaseModel):
    """Response from sync push."""
    synced: int
    conflicts: list[dict[str, Any]] = []
    server_time: datetime


class SyncPullRequest(BaseModel):
    """Request to pull changes from cloud."""
    since: datetime | None = None  # Pull changes since this time


class SyncPullResponse(BaseModel):
    """Response with changes from cloud."""
    operations: list[SyncOperation]
    server_time: datetime
    has_more: bool = False


# =============================================================================
# Memory Models
# =============================================================================

class MemorySearchRequest(BaseModel):
    """Request to search memories."""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    memory_types: list[str] | None = None  # Filter by type


class MemorySearchResult(BaseModel):
    """A single search result."""
    id: str
    memory_type: str
    content: str
    score: float
    created_at: datetime
    metadata: dict[str, Any] = {}


class MemorySearchResponse(BaseModel):
    """Response from memory search."""
    results: list[MemorySearchResult]
    query: str
    total: int
