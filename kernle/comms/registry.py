"""
Agent Registry for Kernle Comms.

Provides agent discovery and profile management:
- Register agents with capabilities
- Discover agents by capability
- Manage trust levels and reputation
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Base exception for registry errors."""

    pass


class AgentNotFoundError(RegistryError):
    """Agent not found in registry."""

    pass


class AgentAlreadyExistsError(RegistryError):
    """Agent already exists in registry."""

    pass


@dataclass
class AgentProfile:
    """Profile of a registered agent.

    Attributes:
        agent_id: Unique agent identifier
        user_id: Owner's user ID
        display_name: Human-readable name
        capabilities: List of capabilities (e.g., ["code_review", "research"])
        public_key: Ed25519 public key for message verification
        endpoints: Communication endpoints (webhook URLs, etc.)
        trust_level: Trust status (unverified, verified, trusted)
        reputation_score: Reputation from 0.0 to 1.0
        is_public: Whether agent is discoverable
        registered_at: When agent was registered
        last_seen_at: Last activity timestamp
    """

    agent_id: str
    user_id: str
    display_name: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    public_key: Optional[str] = None
    endpoints: Dict[str, str] = field(default_factory=dict)
    trust_level: str = "unverified"
    reputation_score: float = 0.0
    is_public: bool = False
    registered_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "display_name": self.display_name,
            "capabilities": self.capabilities,
            "public_key": self.public_key,
            "endpoints": self.endpoints,
            "trust_level": self.trust_level,
            "reputation_score": self.reputation_score,
            "is_public": self.is_public,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentProfile":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            user_id=data["user_id"],
            display_name=data.get("display_name"),
            capabilities=data.get("capabilities") or [],
            public_key=data.get("public_key"),
            endpoints=data.get("endpoints") or {},
            trust_level=data.get("trust_level", "unverified"),
            reputation_score=float(data.get("reputation_score", 0.0)),
            is_public=bool(data.get("is_public", False)),
            registered_at=cls._parse_datetime(data.get("registered_at")),
            last_seen_at=cls._parse_datetime(data.get("last_seen_at")),
        )

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None


class AgentRegistry:
    """Registry for agent discovery and profile management.

    Provides local SQLite storage for agent profiles with optional
    sync to Kernle Cloud for cross-agent discovery.
    """

    def __init__(self, storage: Any):
        """Initialize registry with storage backend.

        Args:
            storage: SQLiteStorage or PostgresStorage instance
        """
        self._storage = storage
        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure registry table exists."""
        # Schema is created by storage backend migration
        pass

    def register(
        self,
        agent_id: str,
        user_id: str,
        display_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        public_key: Optional[str] = None,
        endpoints: Optional[Dict[str, str]] = None,
        is_public: bool = False,
    ) -> AgentProfile:
        """Register a new agent in the registry.

        Args:
            agent_id: Unique agent identifier
            user_id: Owner's user ID
            display_name: Human-readable name
            capabilities: List of capabilities
            public_key: Ed25519 public key
            endpoints: Communication endpoints
            is_public: Whether agent is discoverable

        Returns:
            The created AgentProfile

        Raises:
            AgentAlreadyExistsError: If agent already registered
        """
        # Check if already exists
        existing = self.get_profile(agent_id)
        if existing:
            raise AgentAlreadyExistsError(f"Agent '{agent_id}' already registered")

        now = datetime.now(timezone.utc)
        profile = AgentProfile(
            agent_id=agent_id,
            user_id=user_id,
            display_name=display_name,
            capabilities=capabilities or [],
            public_key=public_key,
            endpoints=endpoints or {},
            trust_level="unverified",
            reputation_score=0.0,
            is_public=is_public,
            registered_at=now,
            last_seen_at=now,
        )

        self._save_profile(profile)
        logger.info(f"Registered agent: {agent_id}")
        return profile

    def get_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent's profile.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentProfile if found, None otherwise
        """
        return self._load_profile(agent_id)

    def update_profile(
        self,
        agent_id: str,
        display_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        public_key: Optional[str] = None,
        endpoints: Optional[Dict[str, str]] = None,
        is_public: Optional[bool] = None,
    ) -> AgentProfile:
        """Update an agent's profile.

        Args:
            agent_id: Agent identifier
            display_name: New display name (None = no change)
            capabilities: New capabilities (None = no change)
            public_key: New public key (None = no change)
            endpoints: New endpoints (None = no change)
            is_public: New visibility (None = no change)

        Returns:
            Updated AgentProfile

        Raises:
            AgentNotFoundError: If agent not found
        """
        profile = self.get_profile(agent_id)
        if not profile:
            raise AgentNotFoundError(f"Agent '{agent_id}' not found")

        if display_name is not None:
            profile.display_name = display_name
        if capabilities is not None:
            profile.capabilities = capabilities
        if public_key is not None:
            profile.public_key = public_key
        if endpoints is not None:
            profile.endpoints = endpoints
        if is_public is not None:
            profile.is_public = is_public

        profile.last_seen_at = datetime.now(timezone.utc)
        self._save_profile(profile)
        logger.info(f"Updated agent profile: {agent_id}")
        return profile

    def update_last_seen(self, agent_id: str) -> None:
        """Update agent's last_seen_at timestamp.

        Args:
            agent_id: Agent identifier
        """
        profile = self.get_profile(agent_id)
        if profile:
            profile.last_seen_at = datetime.now(timezone.utc)
            self._save_profile(profile)

    def delete_profile(self, agent_id: str) -> bool:
        """Delete an agent's profile.

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted, False if not found
        """
        return self._delete_profile(agent_id)

    def discover(
        self,
        capabilities: Optional[List[str]] = None,
        trust_level: Optional[str] = None,
        limit: int = 50,
    ) -> List[AgentProfile]:
        """Discover agents by criteria.

        Args:
            capabilities: Filter by capabilities (any match)
            trust_level: Filter by minimum trust level
            limit: Maximum results

        Returns:
            List of matching AgentProfiles (public only)
        """
        return self._search_profiles(
            capabilities=capabilities,
            trust_level=trust_level,
            is_public=True,
            limit=limit,
        )

    def list_all(self, limit: int = 100) -> List[AgentProfile]:
        """List all registered agents (for admin/debugging).

        Args:
            limit: Maximum results

        Returns:
            List of all AgentProfiles
        """
        return self._search_profiles(limit=limit)

    # === Storage Operations ===

    def _save_profile(self, profile: AgentProfile) -> None:
        """Save profile to storage."""
        conn = self._storage._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO agent_registry
                   (agent_id, user_id, display_name, capabilities, public_key,
                    endpoints, trust_level, reputation_score, is_public,
                    registered_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile.agent_id,
                    profile.user_id,
                    profile.display_name,
                    json.dumps(profile.capabilities),
                    profile.public_key,
                    json.dumps(profile.endpoints),
                    profile.trust_level,
                    profile.reputation_score,
                    1 if profile.is_public else 0,
                    profile.registered_at.isoformat() if profile.registered_at else None,
                    profile.last_seen_at.isoformat() if profile.last_seen_at else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _load_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Load profile from storage."""
        conn = self._storage._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM agent_registry WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()

            if not row:
                return None

            return self._row_to_profile(row)
        finally:
            conn.close()

    def _delete_profile(self, agent_id: str) -> bool:
        """Delete profile from storage."""
        conn = self._storage._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM agent_registry WHERE agent_id = ?",
                (agent_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def _search_profiles(
        self,
        capabilities: Optional[List[str]] = None,
        trust_level: Optional[str] = None,
        is_public: Optional[bool] = None,
        limit: int = 50,
    ) -> List[AgentProfile]:
        """Search profiles with filters."""
        conn = self._storage._get_conn()
        try:
            query = "SELECT * FROM agent_registry WHERE 1=1"
            params: List[Any] = []

            if is_public is not None:
                query += " AND is_public = ?"
                params.append(1 if is_public else 0)

            if trust_level:
                # Trust levels: unverified < verified < trusted
                trust_order = {"unverified": 0, "verified": 1, "trusted": 2}
                min_trust = trust_order.get(trust_level, 0)
                query += " AND trust_level IN (?)"
                # Get levels >= min_trust
                valid_levels = [k for k, v in trust_order.items() if v >= min_trust]
                query = query.replace("(?)", f"({','.join('?' * len(valid_levels))})")
                params.extend(valid_levels)

            query += " ORDER BY reputation_score DESC, last_seen_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            profiles = [self._row_to_profile(row) for row in rows]

            # Filter by capabilities if specified (post-query for SQLite)
            if capabilities:
                profiles = [
                    p for p in profiles if any(cap in p.capabilities for cap in capabilities)
                ]

            return profiles[:limit]
        finally:
            conn.close()

    def _row_to_profile(self, row) -> AgentProfile:
        """Convert database row to AgentProfile."""
        return AgentProfile(
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            display_name=row["display_name"],
            capabilities=json.loads(row["capabilities"] or "[]"),
            public_key=row["public_key"],
            endpoints=json.loads(row["endpoints"] or "{}"),
            trust_level=row["trust_level"],
            reputation_score=float(row["reputation_score"] or 0),
            is_public=bool(row["is_public"]),
            registered_at=AgentProfile._parse_datetime(row["registered_at"]),
            last_seen_at=AgentProfile._parse_datetime(row["last_seen_at"]),
        )
