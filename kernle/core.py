"""
Kernle Core - Stratified memory for synthetic intelligences.

This module provides the main Kernle class, which is the primary interface
for memory operations. It uses the storage abstraction layer to support
both local SQLite storage and cloud Supabase storage.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# Import feature mixins
from kernle.features import (
    AnxietyMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,
)

# Import logging utilities
from kernle.logging_config import (
    log_checkpoint,
    log_load,
    log_save,
)

# Import storage abstraction
from kernle.storage import Belief, Drive, Episode, Goal, Note, Relationship, Value, get_storage

if TYPE_CHECKING:
    from kernle.storage import Storage as StorageProtocol

# Set up logging
logger = logging.getLogger(__name__)

# Default token budget for memory loading
DEFAULT_TOKEN_BUDGET = 8000

# Maximum token budget allowed (consistent across CLI, MCP, and core)
MAX_TOKEN_BUDGET = 50000

# Minimum token budget allowed
MIN_TOKEN_BUDGET = 100

# Maximum characters per memory item (for truncation)
DEFAULT_MAX_ITEM_CHARS = 500

# Token estimation safety margin (actual JSON output is larger than text estimation)
TOKEN_ESTIMATION_SAFETY_MARGIN = 1.3

# Priority scores for each memory type (higher = more important)
MEMORY_TYPE_PRIORITIES = {
    "checkpoint": 1.00,  # Always loaded first
    "value": 0.90,
    "belief": 0.70,
    "goal": 0.65,
    "drive": 0.60,
    "episode": 0.40,
    "note": 0.35,
    "relationship": 0.30,
}


def estimate_tokens(text: str, include_safety_margin: bool = True) -> int:
    """Estimate token count from text.

    Uses the simple heuristic of ~4 characters per token, with a safety
    margin to account for JSON serialization overhead.

    Args:
        text: The text to estimate tokens for
        include_safety_margin: If True, multiply by safety margin (default: True)

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    base_estimate = len(text) // 4
    if include_safety_margin:
        return int(base_estimate * TOKEN_ESTIMATION_SAFETY_MARGIN)
    return base_estimate


def truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary with ellipsis.

    Args:
        text: Text to truncate
        max_chars: Maximum characters (including ellipsis)

    Returns:
        Truncated text with "..." if truncated
    """
    if not text or len(text) <= max_chars:
        return text

    # Leave room for ellipsis
    target = max_chars - 3
    if target <= 0:
        return "..."

    # Find last space before target
    truncated = text[:target]
    last_space = truncated.rfind(" ")

    if last_space > target // 2:  # Only use word boundary if reasonable
        truncated = truncated[:last_space]

    return truncated + "..."


def compute_priority_score(memory_type: str, record: Any) -> float:
    """Compute priority score for a memory record.

    The score combines the base type priority with record-specific factors:
    - Values: priority field (0-100 -> 0.0-1.0)
    - Beliefs: confidence (0.0-1.0)
    - Goals: recency (newer = higher)
    - Drives: intensity (0.0-1.0)
    - Episodes: recency (newer = higher)
    - Notes: recency (newer = higher)
    - Relationships: last_interaction recency

    Args:
        memory_type: Type of memory (value, belief, etc.)
        record: The memory record (dataclass or dict)

    Returns:
        Priority score (0.0-1.0)
    """
    base_priority = MEMORY_TYPE_PRIORITIES.get(memory_type, 0.5)

    # Get record value based on type
    if memory_type == "value":
        # priority is 0-100, normalize to 0-1
        priority = (
            getattr(record, "priority", 50)
            if hasattr(record, "priority")
            else record.get("priority", 50)
        )
        type_factor = priority / 100.0
    elif memory_type == "belief":
        type_factor = (
            getattr(record, "confidence", 0.8)
            if hasattr(record, "confidence")
            else record.get("confidence", 0.8)
        )
    elif memory_type == "drive":
        type_factor = (
            getattr(record, "intensity", 0.5)
            if hasattr(record, "intensity")
            else record.get("intensity", 0.5)
        )
    elif memory_type in ("goal", "episode", "note"):
        # For time-based priority, we'd need to compute recency
        # For now, use a default factor (records are already sorted by recency)
        type_factor = 0.7
    elif memory_type == "relationship":
        # Use sentiment as a factor
        sentiment = (
            getattr(record, "sentiment", 0.0)
            if hasattr(record, "sentiment")
            else record.get("sentiment", 0.0)
        )
        type_factor = (sentiment + 1) / 2  # Normalize -1..1 to 0..1
    else:
        type_factor = 0.5

    # Combine base priority with type-specific factor
    # Weight: 60% type priority, 40% record-specific factor
    return base_priority * 0.6 + type_factor * 0.4


class Kernle(
    AnxietyMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,
):
    """Main interface for Kernle memory operations.

    Supports both local SQLite storage and cloud Supabase storage.
    Storage backend is auto-detected based on environment variables,
    or can be explicitly provided.

    Examples:
        # Auto-detect storage (SQLite if no Supabase creds, else Supabase)
        k = Kernle(agent_id="my_agent")

        # Explicit SQLite
        from kernle.storage import SQLiteStorage
        storage = SQLiteStorage(agent_id="my_agent")
        k = Kernle(agent_id="my_agent", storage=storage)

        # Explicit Supabase (backwards compatible)
        k = Kernle(
            agent_id="my_agent",
            supabase_url="https://xxx.supabase.co",
            supabase_key="my_key"
        )
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        storage: Optional["StorageProtocol"] = None,
        # Keep supabase_url/key for backwards compatibility
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize Kernle.

        Args:
            agent_id: Unique identifier for the agent
            storage: Optional storage backend. If None, auto-detects.
            supabase_url: Supabase project URL (deprecated, use storage param)
            supabase_key: Supabase API key (deprecated, use storage param)
            checkpoint_dir: Directory for local checkpoints
        """
        self.agent_id = self._validate_agent_id(
            agent_id or os.environ.get("KERNLE_AGENT_ID", "default")
        )
        self.checkpoint_dir = self._validate_checkpoint_dir(
            checkpoint_dir or Path.home() / ".kernle" / "checkpoints"
        )

        # Store credentials for backwards compatibility
        self._supabase_url = (
            supabase_url or os.environ.get("KERNLE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        )
        self._supabase_key = (
            supabase_key
            or os.environ.get("KERNLE_SUPABASE_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        )

        # Initialize storage
        if storage is not None:
            self._storage = storage
        else:
            # Auto-detect storage based on environment
            self._storage = get_storage(
                agent_id=self.agent_id,
                supabase_url=self._supabase_url,
                supabase_key=self._supabase_key,
            )

        # Auto-sync configuration: enabled by default if sync is available
        # Can be disabled via KERNLE_AUTO_SYNC=false
        auto_sync_env = os.environ.get("KERNLE_AUTO_SYNC", "").lower()
        if auto_sync_env in ("false", "0", "no", "off"):
            self._auto_sync = False
        elif auto_sync_env in ("true", "1", "yes", "on"):
            self._auto_sync = True
        else:
            # Default: enabled if storage supports sync (has cloud_storage or is cloud-based)
            self._auto_sync = (
                self._storage.is_online() or self._storage.get_pending_sync_count() > 0
            )

        logger.debug(
            f"Kernle initialized with storage: {type(self._storage).__name__}, auto_sync: {self._auto_sync}"
        )

    @property
    def storage(self) -> "StorageProtocol":
        """Get the storage backend."""
        return self._storage

    @property
    def client(self):
        """Backwards-compatible access to Supabase client.

        DEPRECATED: Use storage abstraction methods instead.

        Raises:
            ValueError: If using SQLite storage (no Supabase client available)
        """
        from kernle.storage import SupabaseStorage

        if isinstance(self._storage, SupabaseStorage):
            return self._storage.client
        raise ValueError(
            "Direct Supabase client access not available with SQLite storage. "
            "Use storage abstraction methods instead, or configure Supabase credentials."
        )

    @property
    def auto_sync(self) -> bool:
        """Whether auto-sync is enabled.

        When enabled:
        - load() will pull remote changes first
        - checkpoint() will push local changes after saving
        """
        return self._auto_sync

    @auto_sync.setter
    def auto_sync(self, value: bool):
        """Enable or disable auto-sync."""
        self._auto_sync = value

    def _validate_agent_id(self, agent_id: str) -> str:
        """Validate and sanitize agent ID."""
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")

        # Remove potentially dangerous characters
        sanitized = "".join(c for c in agent_id.strip() if c.isalnum() or c in "-_.")

        if not sanitized:
            raise ValueError("Agent ID must contain alphanumeric characters")

        if len(sanitized) > 100:
            raise ValueError("Agent ID too long (max 100 characters)")

        return sanitized

    def _validate_checkpoint_dir(self, checkpoint_dir: Path) -> Path:
        """Validate checkpoint directory path."""
        import tempfile

        try:
            # Resolve to absolute path to prevent directory traversal
            resolved_path = checkpoint_dir.resolve()

            # Ensure it's within a safe directory (user's home, system temp, or /tmp)
            home_path = Path.home().resolve()
            tmp_path = Path("/tmp").resolve()
            system_temp = Path(tempfile.gettempdir()).resolve()

            # Use is_relative_to() for secure path validation (Python 3.9+)
            # This properly handles edge cases like /home/user/../etc that startswith() misses
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
                raise ValueError("Checkpoint directory must be within user home or temp directory")

            return resolved_path

        except (OSError, ValueError) as e:
            logger.error(f"Invalid checkpoint directory: {e}")
            raise ValueError(f"Invalid checkpoint directory: {e}")

    def _validate_string_input(
        self, value: str, field_name: str, max_length: Optional[int] = 1000
    ) -> str:
        """Validate and sanitize string inputs.

        Args:
            value: String to validate
            field_name: Name of the field (for error messages)
            max_length: Maximum length, or None to skip length check

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        if max_length is not None and len(value) > max_length:
            raise ValueError(f"{field_name} too long (max {max_length} characters)")

        # Basic sanitization - remove null bytes and control characters
        sanitized = value.replace("\x00", "").replace("\r\n", "\n")

        return sanitized

    # =========================================================================
    # LOAD
    # =========================================================================

    def load(
        self,
        budget: int = DEFAULT_TOKEN_BUDGET,
        truncate: bool = True,
        max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
        sync: Optional[bool] = None,
        track_access: bool = True,
    ) -> Dict[str, Any]:
        """Load working memory context with budget-aware selection.

        Memories are loaded by priority across all types until the budget
        is exhausted, preventing context overflow. Higher priority items
        are loaded first.

        Priority order (highest first):
        - Checkpoint: Always loaded (task continuity)
        - Values: 0.90 base, sorted by priority DESC
        - Beliefs: 0.70 base, sorted by confidence DESC
        - Goals: 0.65 base, sorted by recency
        - Drives: 0.60 base, sorted by intensity DESC
        - Episodes: 0.40 base, sorted by recency
        - Notes: 0.35 base, sorted by recency
        - Relationships: 0.30 base, sorted by last_interaction

        Args:
            budget: Token budget for memory (default: 8000, range: 100-50000)
            truncate: If True, truncate long items to fit more in budget
            max_item_chars: Max characters per item when truncating (default: 500)
            sync: Override auto_sync setting. If None, uses self.auto_sync.
            track_access: If True (default), record access for salience tracking.
                Set to False for internal operations (like sync) that should not
                affect salience decay.

        Returns:
            Dict containing all memory layers
        """
        # Validate budget parameter (defense in depth - also validated at MCP layer)
        if not isinstance(budget, int) or budget < MIN_TOKEN_BUDGET:
            budget = MIN_TOKEN_BUDGET
        elif budget > MAX_TOKEN_BUDGET:
            budget = MAX_TOKEN_BUDGET

        # Validate max_item_chars parameter
        if not isinstance(max_item_chars, int) or max_item_chars < 10:
            max_item_chars = 10
        elif max_item_chars > 10000:
            max_item_chars = 10000

        # Sync before load if enabled
        should_sync = sync if sync is not None else self._auto_sync
        if should_sync:
            self._sync_before_load()

        # Load checkpoint first - always included
        checkpoint = self.load_checkpoint()
        remaining_budget = budget

        # Estimate checkpoint tokens
        if checkpoint:
            checkpoint_text = json.dumps(checkpoint, default=str)
            remaining_budget -= estimate_tokens(checkpoint_text)

        # Fetch candidates from all types with high limits for budget selection
        batched = self._storage.load_all(
            values_limit=None,  # Use high limit (1000)
            beliefs_limit=None,
            goals_limit=None,
            goals_status="active",
            episodes_limit=None,
            notes_limit=None,
            drives_limit=None,
            relationships_limit=None,
        )

        if batched is not None:
            # Build candidate list with priority scores
            candidates = []

            # Values - sorted by priority DESC
            for v in batched.get("values", []):
                candidates.append((compute_priority_score("value", v), "value", v))

            # Beliefs - sorted by confidence DESC
            for b in batched.get("beliefs", []):
                candidates.append((compute_priority_score("belief", b), "belief", b))

            # Goals - recency already handled by storage
            for g in batched.get("goals", []):
                candidates.append((compute_priority_score("goal", g), "goal", g))

            # Drives - sorted by intensity DESC
            for d in batched.get("drives", []):
                candidates.append((compute_priority_score("drive", d), "drive", d))

            # Episodes - recency already handled by storage
            for e in batched.get("episodes", []):
                candidates.append((compute_priority_score("episode", e), "episode", e))

            # Notes - recency already handled by storage
            for n in batched.get("notes", []):
                candidates.append((compute_priority_score("note", n), "note", n))

            # Relationships - sorted by last_interaction
            for r in batched.get("relationships", []):
                candidates.append((compute_priority_score("relationship", r), "relationship", r))

            # Sort by priority descending
            candidates.sort(key=lambda x: x[0], reverse=True)

            # Track total candidates for metadata
            total_candidates = len(candidates)
            selected_count = 0

            # Fill budget with highest priority items
            selected = {
                "values": [],
                "beliefs": [],
                "goals": [],
                "drives": [],
                "episodes": [],
                "notes": [],
                "relationships": [],
            }

            for priority, memory_type, record in candidates:
                # Format the record for token estimation
                if memory_type == "value":
                    text = f"{record.name}: {record.statement}"
                elif memory_type == "belief":
                    text = record.statement
                elif memory_type == "goal":
                    text = f"{record.title} {record.description or ''}"
                elif memory_type == "drive":
                    text = f"{record.drive_type}: {record.focus_areas or ''}"
                elif memory_type == "episode":
                    text = f"{record.objective} {record.outcome}"
                elif memory_type == "note":
                    text = record.content
                elif memory_type == "relationship":
                    text = f"{record.entity_name}: {record.notes or ''}"
                else:
                    text = str(record)

                # Truncate if enabled and text exceeds limit
                if truncate and len(text) > max_item_chars:
                    text = truncate_at_word_boundary(text, max_item_chars)

                # Estimate tokens for this item
                tokens = estimate_tokens(text)

                # Check if it fits in budget
                if tokens <= remaining_budget:
                    if memory_type == "value":
                        selected["values"].append(record)
                    elif memory_type == "belief":
                        selected["beliefs"].append(record)
                    elif memory_type == "goal":
                        selected["goals"].append(record)
                    elif memory_type == "drive":
                        selected["drives"].append(record)
                    elif memory_type == "episode":
                        selected["episodes"].append(record)
                    elif memory_type == "note":
                        selected["notes"].append(record)
                    elif memory_type == "relationship":
                        selected["relationships"].append(record)

                    remaining_budget -= tokens
                    selected_count += 1

                # Stop if budget exhausted
                if remaining_budget <= 0:
                    break

            # Extract lessons from selected episodes
            lessons = []
            for ep in selected["episodes"]:
                if ep.lessons:
                    lessons.extend(ep.lessons[:2])

            # Filter recent work (non-checkpoint episodes)
            recent_work = [
                {
                    "objective": e.objective,
                    "outcome_type": e.outcome_type,
                    "tags": e.tags,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in selected["episodes"]
                if not e.tags or "checkpoint" not in e.tags
            ][:5]

            # Format selected items for API compatibility
            batched_result = {
                "checkpoint": checkpoint,
                "values": [
                    {
                        "id": v.id,
                        "name": v.name,
                        "statement": (
                            truncate_at_word_boundary(v.statement, max_item_chars)
                            if truncate
                            else v.statement
                        ),
                        "priority": v.priority,
                        "value_type": "core_value",
                    }
                    for v in selected["values"]
                ],
                "beliefs": [
                    {
                        "id": b.id,
                        "statement": (
                            truncate_at_word_boundary(b.statement, max_item_chars)
                            if truncate
                            else b.statement
                        ),
                        "belief_type": b.belief_type,
                        "confidence": b.confidence,
                    }
                    for b in sorted(selected["beliefs"], key=lambda x: x.confidence, reverse=True)
                ],
                "goals": [
                    {
                        "id": g.id,
                        "title": g.title,
                        "description": (
                            truncate_at_word_boundary(g.description, max_item_chars)
                            if truncate and g.description
                            else g.description
                        ),
                        "priority": g.priority,
                        "status": g.status,
                    }
                    for g in selected["goals"]
                ],
                "drives": [
                    {
                        "id": d.id,
                        "drive_type": d.drive_type,
                        "intensity": d.intensity,
                        "last_satisfied_at": d.updated_at.isoformat() if d.updated_at else None,
                        "focus_areas": d.focus_areas,
                    }
                    for d in selected["drives"]
                ],
                "lessons": lessons,
                "recent_work": recent_work,
                "recent_notes": [
                    {
                        "content": (
                            truncate_at_word_boundary(n.content, max_item_chars)
                            if truncate
                            else n.content
                        ),
                        "metadata": {
                            "note_type": n.note_type,
                            "tags": n.tags,
                            "speaker": n.speaker,
                            "reason": n.reason,
                        },
                        "created_at": n.created_at.isoformat() if n.created_at else None,
                    }
                    for n in selected["notes"]
                ],
                "relationships": [
                    {
                        "other_agent_id": r.entity_name,
                        "entity_name": r.entity_name,
                        "trust_level": (r.sentiment + 1) / 2,
                        "sentiment": r.sentiment,
                        "interaction_count": r.interaction_count,
                        "last_interaction": (
                            r.last_interaction.isoformat() if r.last_interaction else None
                        ),
                        "notes": (
                            truncate_at_word_boundary(r.notes, max_item_chars)
                            if truncate and r.notes
                            else r.notes
                        ),
                    }
                    for r in sorted(
                        selected["relationships"],
                        key=lambda x: x.last_interaction
                        or datetime.min.replace(tzinfo=timezone.utc),
                        reverse=True,
                    )
                ],
                "_meta": {
                    "budget_used": budget - remaining_budget,
                    "budget_total": budget,
                    "excluded_count": total_candidates - selected_count,
                },
            }

            # Track access for all loaded memories (for salience-based forgetting)
            if track_access:
                accesses = []
                for v in selected["values"]:
                    accesses.append(("value", v.id))
                for b in selected["beliefs"]:
                    accesses.append(("belief", b.id))
                for g in selected["goals"]:
                    accesses.append(("goal", g.id))
                for d in selected["drives"]:
                    accesses.append(("drive", d.id))
                for e in selected["episodes"]:
                    accesses.append(("episode", e.id))
                for n in selected["notes"]:
                    accesses.append(("note", n.id))
                for r in selected["relationships"]:
                    accesses.append(("relationship", r.id))

                if accesses:
                    self._storage.record_access_batch(accesses)

            # Log the load operation (batched path)
            log_load(
                self.agent_id,
                values=len(selected["values"]),
                beliefs=len(selected["beliefs"]),
                episodes=len(selected["episodes"]),
                checkpoint=checkpoint is not None,
            )

            return batched_result

        # Fallback to individual queries (for backends without load_all)
        # Note: This path doesn't do budget-aware selection, so we report
        # the budget as fully used and no exclusions (legacy behavior)
        result = {
            "checkpoint": self.load_checkpoint(),
            "values": self.load_values(),
            "beliefs": self.load_beliefs(),
            "goals": self.load_goals(),
            "drives": self.load_drives(),
            "lessons": self.load_lessons(),
            "recent_work": self.load_recent_work(),
            "recent_notes": self.load_recent_notes(),
            "relationships": self.load_relationships(),
            "_meta": {
                "budget_used": budget,
                "budget_total": budget,
                "excluded_count": 0,
            },
        }

        # Log the load operation
        log_load(
            self.agent_id,
            values=len(result.get("values", [])),
            beliefs=len(result.get("beliefs", [])),
            episodes=len(result.get("recent_work", [])),
            checkpoint=result.get("checkpoint") is not None,
        )

        return result

    def load_values(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Load normative values (highest authority)."""
        values = self._storage.get_values(limit=limit)
        return [
            {
                "id": v.id,
                "name": v.name,
                "statement": v.statement,
                "priority": v.priority,
                "value_type": "core_value",  # Default for backwards compatibility
            }
            for v in values
        ]

    def load_beliefs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Load semantic beliefs."""
        beliefs = self._storage.get_beliefs(limit=limit)
        # Sort by confidence descending
        beliefs = sorted(beliefs, key=lambda b: b.confidence, reverse=True)
        return [
            {
                "id": b.id,
                "statement": b.statement,
                "belief_type": b.belief_type,
                "confidence": b.confidence,
            }
            for b in beliefs[:limit]
        ]

    def load_goals(self, limit: int = 10, status: str = "active") -> List[Dict[str, Any]]:
        """Load goals filtered by status.

        Args:
            limit: Maximum number of goals to return
            status: Filter by status - "active", "completed", "paused", or "all"
        """
        goals = self._storage.get_goals(status=None if status == "all" else status, limit=limit)
        return [
            {
                "id": g.id,
                "title": g.title,
                "description": g.description,
                "priority": g.priority,
                "status": g.status,
            }
            for g in goals
        ]

    def load_lessons(self, limit: int = 20) -> List[str]:
        """Load lessons from reflected episodes."""
        episodes = self._storage.get_episodes(limit=limit)

        lessons = []
        for ep in episodes:
            if ep.lessons:
                lessons.extend(ep.lessons[:2])
        return lessons

    def load_recent_work(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Load recent episodes."""
        episodes = self._storage.get_episodes(limit=limit * 2)

        # Filter out checkpoints
        non_checkpoint = [e for e in episodes if not e.tags or "checkpoint" not in e.tags]

        return [
            {
                "objective": e.objective,
                "outcome_type": e.outcome_type,
                "tags": e.tags,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in non_checkpoint[:limit]
        ]

    def load_recent_notes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Load recent curated notes."""
        notes = self._storage.get_notes(limit=limit)
        return [
            {
                "content": n.content,
                "metadata": {
                    "note_type": n.note_type,
                    "tags": n.tags,
                    "speaker": n.speaker,
                    "reason": n.reason,
                },
                "created_at": n.created_at.isoformat() if n.created_at else None,
            }
            for n in notes
        ]

    # =========================================================================
    # CHECKPOINT
    # =========================================================================

    def checkpoint(
        self,
        task: str,
        pending: Optional[list[str]] = None,
        context: Optional[str] = None,
        sync: Optional[bool] = None,
    ) -> dict:
        """Save current working state.

        If auto_sync is enabled (or sync=True), pushes local changes to remote
        after saving the checkpoint locally.

        Args:
            task: Description of the current task/state
            pending: List of pending items to continue later
            context: Additional context about the state
            sync: Override auto_sync setting. If None, uses self.auto_sync.

        Returns:
            Dict containing the checkpoint data
        """
        checkpoint_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "current_task": task,
            "pending": pending or [],
            "context": context,
        }

        # Save locally with proper error handling
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot create checkpoint directory: {e}")
            raise ValueError(f"Cannot create checkpoint directory: {e}")

        checkpoint_file = self.checkpoint_dir / f"{self.agent_id}.json"

        existing = []
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except (json.JSONDecodeError, OSError, PermissionError) as e:
                logger.warning(f"Could not load existing checkpoint: {e}")
                existing = []

        existing.append(checkpoint_data)
        existing = existing[-10:]  # Keep last 10

        try:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot save checkpoint: {e}")
            raise ValueError(f"Cannot save checkpoint: {e}")

        # Also save as episode
        try:
            episode = Episode(
                id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                objective=f"[CHECKPOINT] {self._validate_string_input(task, 'task', 500)}",
                outcome=self._validate_string_input(
                    context or "Working state checkpoint", "context", 1000
                ),
                outcome_type="partial",
                lessons=pending or [],
                tags=["checkpoint", "working_state"],
                created_at=datetime.now(timezone.utc),
            )
            self._storage.save_episode(episode)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint to database: {e}")
            # Local save is sufficient, continue

        # Sync after checkpoint if enabled
        should_sync = sync if sync is not None else self._auto_sync
        if should_sync:
            sync_result = self._sync_after_checkpoint()
            checkpoint_data["_sync"] = sync_result

        # Log the checkpoint save
        log_checkpoint(
            self.agent_id,
            task=task,
            context_len=len(context or ""),
        )

        return checkpoint_data

    # Maximum checkpoint file size (10MB) to prevent DoS via large files
    MAX_CHECKPOINT_SIZE = 10 * 1024 * 1024

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{self.agent_id}.json"
        if checkpoint_file.exists():
            try:
                # Check file size before loading to prevent DoS
                file_size = checkpoint_file.stat().st_size
                if file_size > self.MAX_CHECKPOINT_SIZE:
                    logger.error(
                        f"Checkpoint file too large ({file_size} bytes, max {self.MAX_CHECKPOINT_SIZE})"
                    )
                    raise ValueError(f"Checkpoint file too large ({file_size} bytes)")

                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoints = json.load(f)
                    if isinstance(checkpoints, list) and checkpoints:
                        return checkpoints[-1]
                    elif isinstance(checkpoints, dict):
                        return checkpoints
            except (json.JSONDecodeError, OSError, PermissionError) as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return None

    def clear_checkpoint(self) -> bool:
        """Clear local checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{self.agent_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False

    # =========================================================================
    # EPISODES
    # =========================================================================

    def episode(
        self,
        objective: str,
        outcome: str,
        lessons: Optional[List[str]] = None,
        repeat: Optional[List[str]] = None,
        avoid: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        relates_to: Optional[List[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Record an episodic experience.

        Args:
            relates_to: List of memory IDs this episode relates to (for linking)
            source: Source context (e.g., 'session with Sean', 'heartbeat check')
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
        """
        # Validate inputs
        objective = self._validate_string_input(objective, "objective", 1000)
        outcome = self._validate_string_input(outcome, "outcome", 1000)

        if lessons:
            lessons = [self._validate_string_input(lesson, "lesson", 500) for lesson in lessons]
        if repeat:
            repeat = [self._validate_string_input(r, "repeat pattern", 500) for r in repeat]
        if avoid:
            avoid = [self._validate_string_input(a, "avoid pattern", 500) for a in avoid]
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]

        episode_id = str(uuid.uuid4())

        # Determine outcome type using substring matching for flexibility
        outcome_lower = outcome.lower().strip()
        if any(
            word in outcome_lower
            for word in ("success", "done", "completed", "finished", "accomplished")
        ):
            outcome_type = "success"
        elif any(
            word in outcome_lower for word in ("fail", "error", "broke", "unable", "couldn't")
        ):
            outcome_type = "failure"
        else:
            outcome_type = "partial"

        # Combine lessons with repeat/avoid patterns
        all_lessons = lessons or []
        if repeat:
            all_lessons.extend([f"Repeat: {r}" for r in repeat])
        if avoid:
            all_lessons.extend([f"Avoid: {a}" for a in avoid])

        # Determine source_type from source context
        source_type = "direct_experience"
        if source:
            source_lower = source.lower()
            if any(x in source_lower for x in ["told", "said", "heard", "learned from"]):
                source_type = "told_by_agent"
            elif any(x in source_lower for x in ["infer", "deduce", "conclude"]):
                source_type = "inference"

        episode = Episode(
            id=episode_id,
            agent_id=self.agent_id,
            objective=objective,
            outcome=outcome,
            outcome_type=outcome_type,
            lessons=all_lessons if all_lessons else None,
            tags=tags or ["manual"],
            created_at=datetime.now(timezone.utc),
            confidence=0.8,
            source_type=source_type,
            source_episodes=relates_to,  # Link to related memories
            # Store source context in derived_from for now (as free text marker)
            derived_from=[f"context:{source}"] if source else None,
            # Context/scope fields
            context=context,
            context_tags=context_tags,
        )

        self._storage.save_episode(episode)

        # Log the episode save
        log_save(
            self.agent_id,
            memory_type="episode",
            memory_id=episode_id,
            summary=objective[:50],
        )

        return episode_id

    def update_episode(
        self,
        episode_id: str,
        outcome: Optional[str] = None,
        lessons: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Update an existing episode."""
        # Validate inputs
        episode_id = self._validate_string_input(episode_id, "episode_id", 100)

        # Get the existing episode
        existing = self._storage.get_episode(episode_id)

        if not existing:
            return False

        if outcome is not None:
            outcome = self._validate_string_input(outcome, "outcome", 1000)
            existing.outcome = outcome
            # Update outcome_type based on new outcome using substring matching
            outcome_lower = outcome.lower().strip()
            if any(
                word in outcome_lower
                for word in ("success", "done", "completed", "finished", "accomplished")
            ):
                outcome_type = "success"
            elif any(
                word in outcome_lower for word in ("fail", "error", "broke", "unable", "couldn't")
            ):
                outcome_type = "failure"
            else:
                outcome_type = "partial"
            existing.outcome_type = outcome_type

        if lessons:
            lessons = [self._validate_string_input(lesson, "lesson", 500) for lesson in lessons]
            # Merge with existing lessons
            existing_lessons = existing.lessons or []
            existing.lessons = list(set(existing_lessons + lessons))

        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]
            # Merge with existing tags
            existing_tags = existing.tags or []
            existing.tags = list(set(existing_tags + tags))

        # Use atomic update with optimistic concurrency control
        # This prevents race conditions where concurrent updates could overwrite each other
        self._storage.update_episode_atomic(existing)
        return True

    # =========================================================================
    # NOTES
    # =========================================================================

    def note(
        self,
        content: str,
        type: str = "note",
        speaker: Optional[str] = None,
        reason: Optional[str] = None,
        tags: Optional[List[str]] = None,
        protect: bool = False,
        relates_to: Optional[List[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Capture a quick note (decision, insight, quote).

        Args:
            relates_to: List of memory IDs this note relates to (for linking)
            source: Source context (e.g., 'conversation with X', 'reading Y')
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
        """
        # Validate inputs
        content = self._validate_string_input(content, "content", 2000)

        if type not in ("note", "decision", "insight", "quote"):
            raise ValueError("Invalid note type. Must be one of: note, decision, insight, quote")

        if speaker:
            speaker = self._validate_string_input(speaker, "speaker", 200)
        if reason:
            reason = self._validate_string_input(reason, "reason", 1000)
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]

        note_id = str(uuid.uuid4())

        # Format content based on type
        if type == "decision":
            formatted = f"**Decision**: {content}"
            if reason:
                formatted += f"\n**Reason**: {reason}"
        elif type == "quote":
            speaker_name = speaker or "Unknown"
            formatted = f'> "{content}"\n> — {speaker_name}'
        elif type == "insight":
            formatted = f"**Insight**: {content}"
        else:
            formatted = content

        # Determine source_type from source context
        source_type = "direct_experience"
        if source:
            source_lower = source.lower()
            if any(x in source_lower for x in ["told", "said", "heard", "learned from"]):
                source_type = "told_by_agent"
            elif any(x in source_lower for x in ["infer", "deduce", "conclude"]):
                source_type = "inference"
            elif type == "quote":
                source_type = "told_by_agent"

        note = Note(
            id=note_id,
            agent_id=self.agent_id,
            content=formatted,
            note_type=type,
            speaker=speaker,
            reason=reason,
            tags=tags or [],
            created_at=datetime.now(timezone.utc),
            source_type=source_type,
            source_episodes=relates_to,  # Link to related memories
            derived_from=[f"context:{source}"] if source else None,
            is_protected=protect,
            # Context/scope fields
            context=context,
            context_tags=context_tags,
        )

        self._storage.save_note(note)
        return note_id

    # =========================================================================
    # RAW ENTRIES (Zero-friction capture)
    # =========================================================================

    def raw(
        self,
        blob: Optional[str] = None,
        source: str = "unknown",
        # DEPRECATED parameters
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Quick capture of unstructured brain dump for later processing.

        The raw layer is designed for zero-friction capture. Dump whatever you
        want into the blob field; the system only tracks housekeeping metadata.

        Args:
            blob: The raw brain dump content (no validation, no length limits).
            source: Auto-populated source identifier (cli|mcp|sdk|import|unknown).

        Deprecated Args:
            content: Use blob instead. Will be removed in future version.
            tags: Include tags in blob text instead. Will be removed in future version.

        Returns:
            Raw entry ID
        """
        import warnings

        # Handle deprecated parameters
        if content is not None:
            warnings.warn(
                "The 'content' parameter is deprecated. Use 'blob' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if blob is None:
                blob = content

        if tags is not None:
            warnings.warn(
                "The 'tags' parameter is deprecated. Include tags in blob text instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if blob is None:
            raise ValueError("blob parameter is required")

        # Basic validation - no length limit, but sanitize control chars
        blob = self._validate_string_input(blob, "blob", max_length=None)

        return self._storage.save_raw(blob=blob, source=source, tags=tags)

    def list_raw(self, processed: Optional[bool] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List raw entries, optionally filtered by processed state.

        Args:
            processed: Filter by processed state (None = all, True = processed, False = unprocessed)
            limit: Maximum entries to return

        Returns:
            List of raw entry dicts with blob as primary content field
        """
        entries = self._storage.list_raw(processed=processed, limit=limit)
        return [
            {
                "id": e.id,
                "blob": e.blob,  # Primary content field
                "captured_at": e.captured_at.isoformat() if e.captured_at else None,
                "source": e.source,
                "processed": e.processed,
                "processed_into": e.processed_into,
                # Legacy fields for backward compatibility
                "content": e.blob,  # Alias for blob
                "timestamp": e.captured_at.isoformat() if e.captured_at else None,  # Alias
                "tags": e.tags,  # Deprecated but included for compatibility
            }
            for e in entries
        ]

    def get_raw(self, raw_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific raw entry by ID.

        Args:
            raw_id: ID of the raw entry

        Returns:
            Raw entry dict with blob as primary content field, or None if not found
        """
        entry = self._storage.get_raw(raw_id)
        if entry:
            return {
                "id": entry.id,
                "blob": entry.blob,  # Primary content field
                "captured_at": entry.captured_at.isoformat() if entry.captured_at else None,
                "source": entry.source,
                "processed": entry.processed,
                "processed_into": entry.processed_into,
                # Legacy fields for backward compatibility
                "content": entry.blob,  # Alias for blob
                "timestamp": entry.captured_at.isoformat() if entry.captured_at else None,
                "tags": entry.tags,  # Deprecated
            }
        return None

    def search_raw(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search raw entries using keyword search (FTS5).

        This is a safety net for when backlogs accumulate. For semantic search
        across all memory types, use the regular search() method instead.

        Args:
            query: FTS5 search query (supports AND, OR, NOT, phrases in quotes)
            limit: Maximum number of results

        Returns:
            List of matching raw entry dicts, ordered by relevance.
        """
        entries = self._storage.search_raw_fts(query, limit=limit)
        return [
            {
                "id": e.id,
                "blob": e.blob,
                "captured_at": e.captured_at.isoformat() if e.captured_at else None,
                "source": e.source,
                "processed": e.processed,
                "processed_into": e.processed_into,
                # Legacy fields
                "content": e.blob,
                "timestamp": e.captured_at.isoformat() if e.captured_at else None,
                "tags": e.tags,
            }
            for e in entries
        ]

    def process_raw(
        self,
        raw_id: str,
        as_type: str,
        **kwargs,
    ) -> str:
        """Convert a raw entry into a structured memory.

        Args:
            raw_id: ID of the raw entry to process
            as_type: Type to convert to (episode, note, belief)
            **kwargs: Additional arguments for the target type

        Returns:
            ID of the created memory

        Raises:
            ValueError: If raw entry not found or invalid as_type
        """
        entry = self._storage.get_raw(raw_id)
        if not entry:
            raise ValueError(f"Raw entry {raw_id} not found")

        if entry.processed:
            raise ValueError(f"Raw entry {raw_id} already processed")

        # Create the appropriate memory type
        memory_id = None
        memory_ref = None

        if as_type == "episode":
            # Extract or use provided objective/outcome
            # Use blob (preferred) or content (deprecated) for backwards compatibility
            content = entry.blob or entry.content or ""
            objective = kwargs.get("objective") or content[:100]
            outcome = kwargs.get("outcome", "completed")
            lessons = kwargs.get("lessons") or ([content] if len(content) > 100 else None)
            tags = kwargs.get("tags") or entry.tags or []
            if "raw" not in tags:
                tags.append("raw")

            memory_id = self.episode(
                objective=objective,
                outcome=outcome,
                lessons=lessons,
                tags=tags,
            )
            memory_ref = f"episode:{memory_id}"

        elif as_type == "note":
            note_type = kwargs.get("type", "note")
            tags = kwargs.get("tags") or entry.tags or []
            if "raw" not in tags:
                tags.append("raw")

            memory_id = self.note(
                content=entry.content,
                type=note_type,
                speaker=kwargs.get("speaker"),
                reason=kwargs.get("reason"),
                tags=tags,
            )
            memory_ref = f"note:{memory_id}"

        elif as_type == "belief":
            confidence = kwargs.get("confidence", 0.7)
            belief_type = kwargs.get("type", "observation")

            memory_id = self.belief(
                statement=entry.content,
                type=belief_type,
                confidence=confidence,
            )
            memory_ref = f"belief:{memory_id}"

        else:
            raise ValueError(f"Invalid as_type: {as_type}. Must be one of: episode, note, belief")

        # Mark the raw entry as processed
        self._storage.mark_raw_processed(raw_id, [memory_ref])

        return memory_id

    # =========================================================================
    # BATCH INSERTION
    # =========================================================================

    def episodes_batch(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Save multiple episodes in a single transaction for bulk imports.

        This method optimizes performance when saving many episodes at once,
        such as when importing from external sources or processing large codebases.
        All episodes are saved in a single database transaction.

        Args:
            episodes: List of episode dicts with keys:
                - objective (str, required): What you were trying to accomplish
                - outcome (str, required): What actually happened
                - outcome_type (str, optional): "success", "failure", or "partial"
                - lessons (List[str], optional): Lessons learned
                - tags (List[str], optional): Tags for categorization
                - confidence (float, optional): Confidence level 0.0-1.0

        Returns:
            List of episode IDs (in the same order as input)

        Example:
            ids = k.episodes_batch([
                {"objective": "Fix login bug", "outcome": "Successfully fixed"},
                {"objective": "Add tests", "outcome": "Added 10 unit tests"},
            ])
        """
        episode_objects = []
        for ep_data in episodes:
            objective = self._validate_string_input(ep_data.get("objective", ""), "objective", 1000)
            outcome = self._validate_string_input(ep_data.get("outcome", ""), "outcome", 1000)

            episode = Episode(
                id=ep_data.get("id", str(uuid.uuid4())),
                agent_id=self.agent_id,
                objective=objective,
                outcome=outcome,
                outcome_type=ep_data.get("outcome_type", "partial"),
                lessons=ep_data.get("lessons"),
                tags=ep_data.get("tags", ["batch"]),
                created_at=datetime.now(timezone.utc),
                confidence=ep_data.get("confidence", 0.8),
                source_type=ep_data.get("source_type", "direct_experience"),
            )
            episode_objects.append(episode)

        # Use batch method if available, otherwise fall back to individual saves
        if hasattr(self._storage, "save_episodes_batch"):
            return self._storage.save_episodes_batch(episode_objects)
        else:
            return [self._storage.save_episode(ep) for ep in episode_objects]

    def beliefs_batch(self, beliefs: List[Dict[str, Any]]) -> List[str]:
        """Save multiple beliefs in a single transaction for bulk imports.

        This method optimizes performance when saving many beliefs at once,
        such as when importing knowledge from external sources.
        All beliefs are saved in a single database transaction.

        Args:
            beliefs: List of belief dicts with keys:
                - statement (str, required): The belief statement
                - type (str, optional): "fact", "opinion", "principle", "strategy", or "model"
                - confidence (float, optional): Confidence level 0.0-1.0

        Returns:
            List of belief IDs (in the same order as input)

        Example:
            ids = k.beliefs_batch([
                {"statement": "Python uses indentation for blocks", "confidence": 1.0},
                {"statement": "Type hints improve code quality", "confidence": 0.9},
            ])
        """
        belief_objects = []
        for b_data in beliefs:
            statement = self._validate_string_input(b_data.get("statement", ""), "statement", 1000)

            belief = Belief(
                id=b_data.get("id", str(uuid.uuid4())),
                agent_id=self.agent_id,
                statement=statement,
                belief_type=b_data.get("type", "fact"),
                confidence=b_data.get("confidence", 0.8),
                created_at=datetime.now(timezone.utc),
                source_type=b_data.get("source_type", "direct_experience"),
            )
            belief_objects.append(belief)

        # Use batch method if available, otherwise fall back to individual saves
        if hasattr(self._storage, "save_beliefs_batch"):
            return self._storage.save_beliefs_batch(belief_objects)
        else:
            return [self._storage.save_belief(b) for b in belief_objects]

    def notes_batch(self, notes: List[Dict[str, Any]]) -> List[str]:
        """Save multiple notes in a single transaction for bulk imports.

        This method optimizes performance when saving many notes at once,
        such as when importing from external sources or ingesting documents.
        All notes are saved in a single database transaction.

        Args:
            notes: List of note dicts with keys:
                - content (str, required): The note content
                - type (str, optional): "note", "decision", "insight", or "quote"
                - speaker (str, optional): Who said this (for quotes)
                - reason (str, optional): Why this note matters
                - tags (List[str], optional): Tags for categorization

        Returns:
            List of note IDs (in the same order as input)

        Example:
            ids = k.notes_batch([
                {"content": "Users prefer dark mode", "type": "insight"},
                {"content": "Use TypeScript for new services", "type": "decision"},
            ])
        """
        note_objects = []
        for n_data in notes:
            content = self._validate_string_input(n_data.get("content", ""), "content", 2000)

            note = Note(
                id=n_data.get("id", str(uuid.uuid4())),
                agent_id=self.agent_id,
                content=content,
                note_type=n_data.get("type", "note"),
                speaker=n_data.get("speaker"),
                reason=n_data.get("reason"),
                tags=n_data.get("tags", []),
                created_at=datetime.now(timezone.utc),
                source_type=n_data.get("source_type", "direct_experience"),
            )
            note_objects.append(note)

        # Use batch method if available, otherwise fall back to individual saves
        if hasattr(self._storage, "save_notes_batch"):
            return self._storage.save_notes_batch(note_objects)
        else:
            return [self._storage.save_note(n) for n in note_objects]

    # =========================================================================
    # DUMP / EXPORT
    # =========================================================================

    def dump(self, include_raw: bool = True, format: str = "markdown") -> str:
        """Export all memory to a readable format.

        Args:
            include_raw: Include raw entries in the dump
            format: Output format ("markdown" or "json")

        Returns:
            Formatted string of all memory
        """
        if format == "json":
            return self._dump_json(include_raw)
        else:
            return self._dump_markdown(include_raw)

    def _dump_markdown(self, include_raw: bool) -> str:
        """Export memory as markdown."""
        lines = []
        lines.append(f"# Memory Dump for {self.agent_id}")
        lines.append(f"_Exported at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
        lines.append("")

        # Values
        values = self._storage.get_values(limit=100)
        if values:
            lines.append("## Values")
            for v in sorted(values, key=lambda x: x.priority, reverse=True):
                lines.append(f"- **{v.name}** (priority {v.priority}): {v.statement}")
            lines.append("")

        # Beliefs
        beliefs = self._storage.get_beliefs(limit=100)
        if beliefs:
            lines.append("## Beliefs")
            for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True):
                lines.append(f"- [{b.confidence:.0%}] {b.statement}")
            lines.append("")

        # Goals
        goals = self._storage.get_goals(status=None, limit=100)
        if goals:
            lines.append("## Goals")
            for g in goals:
                status_icon = (
                    "✓" if g.status == "completed" else "○" if g.status == "active" else "⏸"
                )
                lines.append(f"- {status_icon} [{g.priority}] {g.title}")
                if g.description and g.description != g.title:
                    lines.append(f"  {g.description}")
            lines.append("")

        # Episodes
        episodes = self._storage.get_episodes(limit=100)
        if episodes:
            lines.append("## Episodes")
            for e in episodes:
                date_str = e.created_at.strftime("%Y-%m-%d") if e.created_at else "unknown"
                outcome_icon = (
                    "✓"
                    if e.outcome_type == "success"
                    else "✗" if e.outcome_type == "failure" else "○"
                )
                lines.append(f"### {outcome_icon} {e.objective}")
                lines.append(f"*{date_str}* | {e.outcome}")
                if e.lessons:
                    lines.append("**Lessons:**")
                    for lesson in e.lessons:
                        lines.append(f"  - {lesson}")
                if e.tags:
                    lines.append(f"Tags: {', '.join(e.tags)}")
                lines.append("")

        # Notes
        notes = self._storage.get_notes(limit=100)
        if notes:
            lines.append("## Notes")
            for n in notes:
                date_str = n.created_at.strftime("%Y-%m-%d") if n.created_at else "unknown"
                lines.append(f"### [{n.note_type}] {date_str}")
                lines.append(n.content)
                if n.tags:
                    lines.append(f"Tags: {', '.join(n.tags)}")
                lines.append("")

        # Drives
        drives = self._storage.get_drives()
        if drives:
            lines.append("## Drives")
            for d in drives:
                bar = "█" * int(d.intensity * 10) + "░" * (10 - int(d.intensity * 10))
                focus = f" → {', '.join(d.focus_areas)}" if d.focus_areas else ""
                lines.append(f"- {d.drive_type}: [{bar}] {d.intensity:.0%}{focus}")
            lines.append("")

        # Relationships
        relationships = self._storage.get_relationships()
        if relationships:
            lines.append("## Relationships")
            for r in relationships:
                sentiment_str = f"{r.sentiment:+.2f}" if r.sentiment else "neutral"
                lines.append(f"- **{r.entity_name}** ({r.entity_type}): {sentiment_str}")
                if r.notes:
                    lines.append(f"  {r.notes}")
            lines.append("")

        # Raw entries
        if include_raw:
            raw_entries = self._storage.list_raw(limit=100)
            if raw_entries:
                lines.append("## Raw Entries")
                for raw in raw_entries:
                    date_str = (
                        raw.timestamp.strftime("%Y-%m-%d %H:%M") if raw.timestamp else "unknown"
                    )
                    status = "✓" if raw.processed else "○"
                    lines.append(f"### {status} {date_str}")
                    lines.append(raw.content)
                    if raw.tags:
                        lines.append(f"Tags: {', '.join(raw.tags)}")
                    if raw.processed and raw.processed_into:
                        lines.append(f"Processed into: {', '.join(raw.processed_into)}")
                    lines.append("")

        return "\n".join(lines)

    def _dump_json(self, include_raw: bool) -> str:
        """Export memory as JSON with full meta-memory fields."""

        def _dt(dt: Optional[datetime]) -> Optional[str]:
            """Convert datetime to ISO string."""
            return dt.isoformat() if dt else None

        data = {
            "agent_id": self.agent_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "values": [
                {
                    "id": v.id,
                    "name": v.name,
                    "statement": v.statement,
                    "priority": v.priority,
                    "created_at": _dt(v.created_at),
                    "local_updated_at": _dt(v.local_updated_at),
                    "confidence": v.confidence,
                    "source_type": v.source_type,
                    "source_episodes": v.source_episodes,
                    "times_accessed": v.times_accessed,
                    "last_accessed": _dt(v.last_accessed),
                    "is_protected": v.is_protected,
                }
                for v in self._storage.get_values(limit=100)
            ],
            "beliefs": [
                {
                    "id": b.id,
                    "statement": b.statement,
                    "type": b.belief_type,
                    "confidence": b.confidence,
                    "created_at": _dt(b.created_at),
                    "local_updated_at": _dt(b.local_updated_at),
                    "source_type": b.source_type,
                    "source_episodes": b.source_episodes,
                    "derived_from": b.derived_from,
                    "times_accessed": b.times_accessed,
                    "last_accessed": _dt(b.last_accessed),
                    "is_protected": b.is_protected,
                    "supersedes": b.supersedes,
                    "superseded_by": b.superseded_by,
                    "times_reinforced": b.times_reinforced,
                    "is_active": b.is_active,
                }
                for b in self._storage.get_beliefs(limit=100)
            ],
            "goals": [
                {
                    "id": g.id,
                    "title": g.title,
                    "description": g.description,
                    "priority": g.priority,
                    "status": g.status,
                    "created_at": _dt(g.created_at),
                    "local_updated_at": _dt(g.local_updated_at),
                    "confidence": g.confidence,
                    "source_type": g.source_type,
                    "source_episodes": g.source_episodes,
                    "times_accessed": g.times_accessed,
                    "last_accessed": _dt(g.last_accessed),
                    "is_protected": g.is_protected,
                }
                for g in self._storage.get_goals(status=None, limit=100)
            ],
            "episodes": [
                {
                    "id": e.id,
                    "objective": e.objective,
                    "outcome": e.outcome,
                    "outcome_type": e.outcome_type,
                    "lessons": e.lessons,
                    "tags": e.tags,
                    "created_at": _dt(e.created_at),
                    "local_updated_at": _dt(e.local_updated_at),
                    "confidence": e.confidence,
                    "source_type": e.source_type,
                    "source_episodes": e.source_episodes,
                    "derived_from": e.derived_from,
                    "emotional_valence": e.emotional_valence,
                    "emotional_arousal": e.emotional_arousal,
                    "emotional_tags": e.emotional_tags,
                    "times_accessed": e.times_accessed,
                    "last_accessed": _dt(e.last_accessed),
                    "is_protected": e.is_protected,
                }
                for e in self._storage.get_episodes(limit=100)
            ],
            "notes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "type": n.note_type,
                    "speaker": n.speaker,
                    "reason": n.reason,
                    "tags": n.tags,
                    "created_at": _dt(n.created_at),
                    "local_updated_at": _dt(n.local_updated_at),
                    "confidence": n.confidence,
                    "source_type": n.source_type,
                    "source_episodes": n.source_episodes,
                    "times_accessed": n.times_accessed,
                    "last_accessed": _dt(n.last_accessed),
                    "is_protected": n.is_protected,
                }
                for n in self._storage.get_notes(limit=100)
            ],
            "drives": [
                {
                    "id": d.id,
                    "type": d.drive_type,
                    "intensity": d.intensity,
                    "focus_areas": d.focus_areas,
                    "created_at": _dt(d.created_at),
                    "updated_at": _dt(d.updated_at),
                    "local_updated_at": _dt(d.local_updated_at),
                    "confidence": d.confidence,
                    "source_type": d.source_type,
                    "times_accessed": d.times_accessed,
                    "last_accessed": _dt(d.last_accessed),
                    "is_protected": d.is_protected,
                }
                for d in self._storage.get_drives()
            ],
            "relationships": [
                {
                    "id": r.id,
                    "entity_name": r.entity_name,
                    "entity_type": r.entity_type,
                    "relationship_type": r.relationship_type,
                    "sentiment": r.sentiment,
                    "notes": r.notes,
                    "interaction_count": r.interaction_count,
                    "last_interaction": _dt(r.last_interaction),
                    "created_at": _dt(r.created_at),
                    "local_updated_at": _dt(r.local_updated_at),
                    "confidence": r.confidence,
                    "source_type": r.source_type,
                    "times_accessed": r.times_accessed,
                    "last_accessed": _dt(r.last_accessed),
                    "is_protected": r.is_protected,
                }
                for r in self._storage.get_relationships()
            ],
        }

        if include_raw:
            data["raw_entries"] = [
                {
                    "id": r.id,
                    "content": r.content,
                    "timestamp": _dt(r.timestamp),
                    "source": r.source,
                    "processed": r.processed,
                    "processed_into": r.processed_into,
                    "tags": r.tags,
                    "local_updated_at": _dt(r.local_updated_at),
                    "confidence": r.confidence,
                    "source_type": r.source_type,
                }
                for r in self._storage.list_raw(limit=100)
            ]

        return json.dumps(data, indent=2, default=str)

    def export(self, path: str, include_raw: bool = True, format: str = "markdown"):
        """Export memory to a file.

        Args:
            path: Path to export file
            include_raw: Include raw entries
            format: Output format ("markdown" or "json")
        """
        content = self.dump(include_raw=include_raw, format=format)

        # Determine format from extension if not specified
        if format == "markdown" and path.endswith(".json"):
            format = "json"
            content = self.dump(include_raw=include_raw, format="json")
        elif format == "json" and (path.endswith(".md") or path.endswith(".markdown")):
            format = "markdown"
            content = self.dump(include_raw=include_raw, format="markdown")

        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(content, encoding="utf-8")

    # =========================================================================
    # BELIEFS & VALUES
    # =========================================================================

    def belief(
        self,
        statement: str,
        type: str = "fact",
        confidence: float = 0.8,
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Add or update a belief.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
        """
        belief_id = str(uuid.uuid4())

        belief = Belief(
            id=belief_id,
            agent_id=self.agent_id,
            statement=statement,
            belief_type=type,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            context=context,
            context_tags=context_tags,
        )

        self._storage.save_belief(belief)
        return belief_id

    def value(
        self,
        name: str,
        statement: str,
        priority: int = 50,
        type: str = "core_value",
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Add or affirm a value.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
        """
        value_id = str(uuid.uuid4())

        value = Value(
            id=value_id,
            agent_id=self.agent_id,
            name=name,
            statement=statement,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            context=context,
            context_tags=context_tags,
        )

        self._storage.save_value(value)
        return value_id

    def goal(
        self,
        title: str,
        description: Optional[str] = None,
        priority: str = "medium",
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Add a goal.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
        """
        goal_id = str(uuid.uuid4())

        goal = Goal(
            id=goal_id,
            agent_id=self.agent_id,
            title=title,
            description=description or title,
            priority=priority,
            status="active",
            created_at=datetime.now(timezone.utc),
            context=context,
            context_tags=context_tags,
        )

        self._storage.save_goal(goal)
        return goal_id

    def update_goal(
        self,
        goal_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update a goal's status, priority, or description."""
        # Validate inputs
        goal_id = self._validate_string_input(goal_id, "goal_id", 100)

        # Get goals to find matching one
        goals = self._storage.get_goals(status=None, limit=1000)
        existing = None
        for g in goals:
            if g.id == goal_id:
                existing = g
                break

        if not existing:
            return False

        if status is not None:
            if status not in ("active", "completed", "paused"):
                raise ValueError("Invalid status. Must be one of: active, completed, paused")
            existing.status = status

        if priority is not None:
            if priority not in ("low", "medium", "high"):
                raise ValueError("Invalid priority. Must be one of: low, medium, high")
            existing.priority = priority

        if description is not None:
            description = self._validate_string_input(description, "description", 1000)
            existing.description = description

        # TODO: Add update_goal_atomic for optimistic concurrency control
        existing.version += 1
        self._storage.save_goal(existing)
        return True

    def update_belief(
        self,
        belief_id: str,
        confidence: Optional[float] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """Update a belief's confidence or deactivate it."""
        # Validate inputs
        belief_id = self._validate_string_input(belief_id, "belief_id", 100)

        # Get beliefs to find matching one (include inactive to allow reactivation)
        beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        existing = None
        for b in beliefs:
            if b.id == belief_id:
                existing = b
                break

        if not existing:
            return False

        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
            existing.confidence = confidence

        if is_active is not None:
            existing.is_active = is_active
            if not is_active:
                existing.deleted = True

        # Use atomic update with optimistic concurrency control
        self._storage.update_belief_atomic(existing)
        return True

    # =========================================================================
    # BELIEF REVISION
    # =========================================================================

    def find_contradictions(
        self,
        belief_statement: str,
        similarity_threshold: float = 0.6,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find beliefs that might contradict a statement.

        Uses semantic similarity to find related beliefs, then checks for
        potential contradictions using heuristic pattern matching.

        Args:
            belief_statement: The statement to check for contradictions
            similarity_threshold: Minimum similarity score (0-1) for related beliefs
            limit: Maximum number of potential contradictions to return

        Returns:
            List of dicts with belief info and contradiction analysis
        """
        # Search for semantically similar beliefs
        search_results = self._storage.search(
            belief_statement,
            limit=limit * 2,
            record_types=["belief"],  # Get more to filter
        )

        contradictions = []
        stmt_lower = belief_statement.lower().strip()

        for result in search_results:
            if result.record_type != "belief":
                continue

            belief = result.record
            belief_stmt_lower = belief.statement.lower().strip()

            # Skip exact matches
            if belief_stmt_lower == stmt_lower:
                continue

            # Check for contradiction patterns
            contradiction_type = None
            confidence = 0.0
            explanation = ""

            # Negation patterns
            negation_pairs = [
                ("never", "always"),
                ("should not", "should"),
                ("cannot", "can"),
                ("don't", "do"),
                ("avoid", "prefer"),
                ("reject", "accept"),
                ("false", "true"),
                ("dislike", "like"),
                ("hate", "love"),
                ("wrong", "right"),
                ("bad", "good"),
            ]

            for neg, pos in negation_pairs:
                if (neg in stmt_lower and pos in belief_stmt_lower) or (
                    pos in stmt_lower and neg in belief_stmt_lower
                ):
                    # Check word overlap for topic relevance
                    words_stmt = set(stmt_lower.split()) - {
                        "i",
                        "the",
                        "a",
                        "an",
                        "to",
                        "and",
                        "or",
                        "is",
                        "are",
                        "that",
                        "this",
                    }
                    words_belief = set(belief_stmt_lower.split()) - {
                        "i",
                        "the",
                        "a",
                        "an",
                        "to",
                        "and",
                        "or",
                        "is",
                        "are",
                        "that",
                        "this",
                    }
                    overlap = len(words_stmt & words_belief)

                    if overlap >= 2:
                        contradiction_type = "direct_negation"
                        confidence = min(0.5 + overlap * 0.1 + result.score * 0.2, 0.95)
                        explanation = f"Negation conflict: '{neg}' vs '{pos}' with {overlap} overlapping terms"
                        break

            # Comparative opposition (more/less, better/worse, etc.)
            if not contradiction_type:
                comparative_pairs = [
                    ("more", "less"),
                    ("better", "worse"),
                    ("faster", "slower"),
                    ("higher", "lower"),
                    ("greater", "lesser"),
                    ("stronger", "weaker"),
                    ("easier", "harder"),
                    ("simpler", "more complex"),
                    ("safer", "riskier"),
                    ("cheaper", "more expensive"),
                    ("larger", "smaller"),
                    ("longer", "shorter"),
                    ("increase", "decrease"),
                    ("improve", "worsen"),
                    ("enhance", "diminish"),
                ]
                for comp_a, comp_b in comparative_pairs:
                    if (comp_a in stmt_lower and comp_b in belief_stmt_lower) or (
                        comp_b in stmt_lower and comp_a in belief_stmt_lower
                    ):
                        # Check word overlap for topic relevance (need high overlap for comparatives)
                        words_stmt = set(stmt_lower.split()) - {
                            "i",
                            "the",
                            "a",
                            "an",
                            "to",
                            "and",
                            "or",
                            "is",
                            "are",
                            "that",
                            "this",
                            "than",
                            comp_a,
                            comp_b,
                        }
                        words_belief = set(belief_stmt_lower.split()) - {
                            "i",
                            "the",
                            "a",
                            "an",
                            "to",
                            "and",
                            "or",
                            "is",
                            "are",
                            "that",
                            "this",
                            "than",
                            comp_a,
                            comp_b,
                        }
                        overlap = len(words_stmt & words_belief)

                        if overlap >= 2:
                            contradiction_type = "comparative_opposition"
                            # Higher confidence for comparative oppositions with strong topic overlap
                            confidence = min(0.6 + overlap * 0.08 + result.score * 0.2, 0.95)
                            explanation = f"Comparative opposition: '{comp_a}' vs '{comp_b}' with {overlap} overlapping terms"
                            break

            # Preference conflicts
            if not contradiction_type:
                preference_pairs = [
                    ("prefer", "avoid"),
                    ("like", "dislike"),
                    ("enjoy", "hate"),
                    ("favor", "oppose"),
                    ("support", "reject"),
                    ("want", "don't want"),
                ]
                for pref, anti in preference_pairs:
                    if (pref in stmt_lower and anti in belief_stmt_lower) or (
                        anti in stmt_lower and pref in belief_stmt_lower
                    ):
                        words_stmt = set(stmt_lower.split()) - {
                            "i",
                            "the",
                            "a",
                            "an",
                            "to",
                            "and",
                            "or",
                        }
                        words_belief = set(belief_stmt_lower.split()) - {
                            "i",
                            "the",
                            "a",
                            "an",
                            "to",
                            "and",
                            "or",
                        }
                        overlap = len(words_stmt & words_belief)

                        if overlap >= 2:
                            contradiction_type = "preference_conflict"
                            confidence = min(0.4 + overlap * 0.1 + result.score * 0.2, 0.85)
                            explanation = f"Preference conflict: '{pref}' vs '{anti}'"
                            break

            if contradiction_type:
                contradictions.append(
                    {
                        "belief_id": belief.id,
                        "statement": belief.statement,
                        "confidence": belief.confidence,
                        "times_reinforced": belief.times_reinforced,
                        "is_active": belief.is_active,
                        "contradiction_type": contradiction_type,
                        "contradiction_confidence": round(confidence, 2),
                        "explanation": explanation,
                        "semantic_similarity": round(result.score, 2),
                    }
                )

        # Sort by contradiction confidence
        contradictions.sort(key=lambda x: x["contradiction_confidence"], reverse=True)
        return contradictions[:limit]

    # Opposition word pairs for semantic contradiction detection
    # Format: (word, opposite) - both directions are checked
    _OPPOSITION_PAIRS = [
        # Frequency/Certainty
        ("always", "never"),
        ("sometimes", "never"),
        ("often", "rarely"),
        ("frequently", "seldom"),
        ("constantly", "occasionally"),
        # Modal verbs and necessity
        ("should", "shouldn't"),
        ("must", "mustn't"),
        ("can", "cannot"),
        ("will", "won't"),
        ("would", "wouldn't"),
        ("could", "couldn't"),
        # Preferences and attitudes
        ("like", "dislike"),
        ("love", "hate"),
        ("prefer", "avoid"),
        ("enjoy", "despise"),
        ("favor", "oppose"),
        ("want", "reject"),
        ("appreciate", "resent"),
        ("embrace", "shun"),
        # Value judgments
        ("good", "bad"),
        ("best", "worst"),
        ("important", "unnecessary"),
        ("essential", "optional"),
        ("critical", "trivial"),
        ("valuable", "worthless"),
        ("beneficial", "harmful"),
        ("helpful", "unhelpful"),
        ("useful", "useless"),
        # Comparatives
        ("more", "less"),
        ("better", "worse"),
        ("faster", "slower"),
        ("higher", "lower"),
        ("greater", "lesser"),
        ("stronger", "weaker"),
        ("easier", "harder"),
        ("simpler", "complex"),
        ("safer", "riskier"),
        ("cheaper", "expensive"),
        ("larger", "smaller"),
        ("longer", "shorter"),
        # Actions and states
        ("increase", "decrease"),
        ("improve", "worsen"),
        ("enhance", "diminish"),
        ("enable", "disable"),
        ("allow", "prevent"),
        ("support", "block"),
        ("accept", "reject"),
        ("approve", "disapprove"),
        ("agree", "disagree"),
        ("include", "exclude"),
        ("add", "remove"),
        ("create", "destroy"),
        # Truth values
        ("true", "false"),
        ("right", "wrong"),
        ("correct", "incorrect"),
        ("accurate", "inaccurate"),
        ("valid", "invalid"),
        # Quality descriptors
        ("efficient", "inefficient"),
        ("effective", "ineffective"),
        ("reliable", "unreliable"),
        ("stable", "unstable"),
        ("secure", "insecure"),
        ("safe", "dangerous"),
        # Recommendations
        ("recommended", "discouraged"),
        ("advisable", "inadvisable"),
        ("encouraged", "forbidden"),
        ("suggested", "prohibited"),
    ]

    # Negation prefixes that can flip meaning
    _NEGATION_PREFIXES = ["not", "no", "non", "un", "in", "dis", "anti", "counter"]

    # Stop words to exclude from topic overlap calculations
    _STOP_WORDS = frozenset(
        [
            "i",
            "the",
            "a",
            "an",
            "to",
            "and",
            "or",
            "is",
            "are",
            "that",
            "this",
            "it",
            "be",
            "was",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "for",
            "of",
            "in",
            "on",
            "at",
            "by",
            "with",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "but",
            "if",
            "then",
            "because",
            "while",
            "although",
            "though",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "me",
            "you",
            "him",
            "she",
            "we",
            "they",
            "who",
            "which",
            "what",
            "when",
            "where",
            "why",
            "how",
        ]
    )

    def find_semantic_contradictions(
        self,
        belief: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find beliefs that are semantically similar but may contradict.

        This method uses embedding-based similarity search to find beliefs
        that discuss the same topic, then applies opposition detection to
        identify potential contradictions. Unlike find_contradictions() which
        requires explicit opposition words, this can detect semantic opposition
        like "Testing is important" vs "Testing slows me down".

        Args:
            belief: The belief statement to check for contradictions
            similarity_threshold: Minimum similarity score (0-1) for related beliefs.
                Higher values (0.7-0.9) find more topically related beliefs.
            limit: Maximum number of potential contradictions to return

        Returns:
            List of dicts containing:
                - belief_id: ID of the potentially contradicting belief
                - statement: The belief statement
                - confidence: Belief's confidence level
                - similarity_score: Semantic similarity (0-1)
                - opposition_score: Strength of detected opposition (0-1)
                - opposition_type: Type of opposition detected
                - explanation: Human-readable explanation of the potential contradiction

        Example:
            >>> k = Kernle("my-agent")
            >>> k.belief("Testing is essential for code quality")
            >>> contradictions = k.find_semantic_contradictions(
            ...     "Testing slows down development"
            ... )
            >>> for c in contradictions:
            ...     print(f"{c['statement']}: {c['explanation']}")
        """
        belief = self._validate_string_input(belief, "belief", 2000)

        # Search for semantically similar beliefs
        search_results = self._storage.search(
            belief,
            limit=limit * 3,
            record_types=["belief"],  # Get more to filter by threshold
        )

        contradictions = []
        belief_lower = belief.lower().strip()

        for result in search_results:
            if result.record_type != "belief":
                continue

            # Filter by similarity threshold
            if result.score < similarity_threshold:
                continue

            existing_belief = result.record
            existing_lower = existing_belief.statement.lower().strip()

            # Skip exact matches
            if existing_lower == belief_lower:
                continue

            # Skip inactive beliefs by default
            if not existing_belief.is_active:
                continue

            # Detect opposition
            opposition = self._detect_opposition(belief_lower, existing_lower)

            if opposition["score"] > 0:
                contradictions.append(
                    {
                        "belief_id": existing_belief.id,
                        "statement": existing_belief.statement,
                        "confidence": existing_belief.confidence,
                        "times_reinforced": existing_belief.times_reinforced,
                        "is_active": existing_belief.is_active,
                        "similarity_score": round(result.score, 3),
                        "opposition_score": round(opposition["score"], 3),
                        "opposition_type": opposition["type"],
                        "explanation": opposition["explanation"],
                    }
                )

        # Sort by combined score (similarity * opposition)
        contradictions.sort(
            key=lambda x: x["similarity_score"] * x["opposition_score"], reverse=True
        )
        return contradictions[:limit]

    def _detect_opposition(
        self,
        stmt1: str,
        stmt2: str,
    ) -> Dict[str, Any]:
        """Detect if two similar statements have opposing meanings.

        Uses multiple heuristics:
        1. Direct opposition words (always/never, good/bad, etc.)
        2. Negation patterns (is vs is not, should vs shouldn't)
        3. Sentiment/valence indicators

        Args:
            stmt1: First statement (lowercase)
            stmt2: Second statement (lowercase)

        Returns:
            Dict with:
                - score: Opposition strength (0-1), 0 means no opposition detected
                - type: Type of opposition detected
                - explanation: Human-readable explanation
        """
        result = {"score": 0.0, "type": "none", "explanation": ""}

        words1 = set(stmt1.split())
        words2 = set(stmt2.split())

        # Calculate topic overlap (excluding stop words and opposition words)
        content_words1 = words1 - self._STOP_WORDS
        content_words2 = words2 - self._STOP_WORDS
        overlap = content_words1 & content_words2
        overlap_count = len(overlap)

        # Need some topic overlap to be a meaningful contradiction
        if overlap_count < 1:
            return result

        # 1. Check for direct opposition word pairs
        for word_a, word_b in self._OPPOSITION_PAIRS:
            # Check both directions
            if (word_a in stmt1 and word_b in stmt2) or (word_b in stmt1 and word_a in stmt2):
                # Verify words are used in meaningful context (not just substrings)
                a_in_1 = word_a in words1
                b_in_2 = word_b in words2
                b_in_1 = word_b in words1
                a_in_2 = word_a in words2

                if (a_in_1 and b_in_2) or (b_in_1 and a_in_2):
                    score = min(0.5 + overlap_count * 0.1, 0.95)
                    return {
                        "score": score,
                        "type": "opposition_words",
                        "explanation": f"Opposing terms '{word_a}' vs '{word_b}' with {overlap_count} shared topic words: {', '.join(list(overlap)[:3])}",
                    }

        # 2. Check for negation patterns
        negation_found = self._check_negation_pattern(stmt1, stmt2)
        if negation_found:
            score = min(0.4 + overlap_count * 0.1, 0.85)
            return {
                "score": score,
                "type": "negation",
                "explanation": f"Negation pattern detected with {overlap_count} shared topic words: {', '.join(list(overlap)[:3])}",
            }

        # 3. Check for sentiment opposition using positive/negative indicator words
        sentiment_opposition = self._check_sentiment_opposition(stmt1, stmt2)
        if sentiment_opposition["detected"]:
            score = min(0.3 + overlap_count * 0.1, 0.75)
            return {
                "score": score,
                "type": "sentiment_opposition",
                "explanation": f"Sentiment opposition: '{sentiment_opposition['word1']}' vs '{sentiment_opposition['word2']}' with topic overlap",
            }

        return result

    def _check_negation_pattern(self, stmt1: str, stmt2: str) -> bool:
        """Check if one statement negates the other.

        Looks for patterns like:
        - "X is good" vs "X is not good"
        - "should use X" vs "should not use X"
        - "I like X" vs "I don't like X"
        """
        # Common negation patterns
        negation_patterns = [
            ("is not", "is"),
            ("is", "is not"),
            ("are not", "are"),
            ("are", "are not"),
            ("do not", "do"),
            ("do", "do not"),
            ("does not", "does"),
            ("does", "does not"),
            ("should not", "should"),
            ("should", "should not"),
            ("shouldn't", "should"),
            ("should", "shouldn't"),
            ("can not", "can"),
            ("can", "can not"),
            ("cannot", "can"),
            ("can", "cannot"),
            ("can't", "can"),
            ("can", "can't"),
            ("won't", "will"),
            ("will", "won't"),
            ("don't", "do"),
            ("do", "don't"),
            ("doesn't", "does"),
            ("does", "doesn't"),
            ("isn't", "is"),
            ("is", "isn't"),
            ("aren't", "are"),
            ("are", "aren't"),
            ("wasn't", "was"),
            ("was", "wasn't"),
            ("weren't", "were"),
            ("were", "weren't"),
            ("not recommended", "recommended"),
            ("recommended", "not recommended"),
            ("not important", "important"),
            ("important", "not important"),
            ("no need", "need"),
            ("need", "no need"),
        ]

        for pattern_a, pattern_b in negation_patterns:
            if pattern_a in stmt1 and pattern_b in stmt2:
                # Make sure pattern_a is not a substring of pattern_b in stmt1
                if pattern_b not in stmt1 or stmt1.index(pattern_a) != stmt1.find(pattern_b):
                    return True
            if pattern_b in stmt1 and pattern_a in stmt2:
                if pattern_a not in stmt1 or stmt1.index(pattern_b) != stmt1.find(pattern_a):
                    return True

        return False

    def _check_sentiment_opposition(
        self,
        stmt1: str,
        stmt2: str,
    ) -> Dict[str, Any]:
        """Check for sentiment/valence opposition between statements.

        Looks for one statement having positive sentiment words and
        the other having negative sentiment words about the same topic.
        """
        positive_words = {
            "good",
            "great",
            "excellent",
            "important",
            "essential",
            "valuable",
            "helpful",
            "useful",
            "beneficial",
            "necessary",
            "crucial",
            "vital",
            "effective",
            "efficient",
            "reliable",
            "fast",
            "quick",
            "easy",
            "simple",
            "clear",
            "clean",
            "safe",
            "secure",
            "stable",
            "robust",
            "powerful",
            "flexible",
            "scalable",
            "maintainable",
            "readable",
            "elegant",
            "beautiful",
            "brilliant",
            "amazing",
            "wonderful",
            "love",
            "like",
            "enjoy",
            "prefer",
            "appreciate",
            "recommend",
            "success",
            "win",
            "gain",
            "improve",
            "enhance",
            "boost",
        }

        negative_words = {
            "bad",
            "poor",
            "terrible",
            "unimportant",
            "unnecessary",
            "worthless",
            "unhelpful",
            "useless",
            "harmful",
            "optional",
            "trivial",
            "minor",
            "ineffective",
            "inefficient",
            "unreliable",
            "slow",
            "sluggish",
            "hard",
            "complex",
            "confusing",
            "messy",
            "dangerous",
            "insecure",
            "unstable",
            "fragile",
            "weak",
            "rigid",
            "limited",
            "unmaintainable",
            "unreadable",
            "ugly",
            "awful",
            "horrible",
            "terrible",
            "disaster",
            "hate",
            "dislike",
            "avoid",
            "reject",
            "despise",
            "discourage",
            "failure",
            "loss",
            "degrade",
            "diminish",
            "reduce",
            "slows",
            "slow",
            "slowdown",
            "overhead",
            "bloat",
            "bloated",
            "waste",
            "wasted",
            "wastes",
            "wasting",
        }

        words1 = set(stmt1.split())
        words2 = set(stmt2.split())

        pos1 = words1 & positive_words
        neg1 = words1 & negative_words
        pos2 = words2 & positive_words
        neg2 = words2 & negative_words

        # Check for cross-sentiment: positive in one, negative in other
        if pos1 and neg2:
            return {
                "detected": True,
                "word1": list(pos1)[0],
                "word2": list(neg2)[0],
            }
        if neg1 and pos2:
            return {
                "detected": True,
                "word1": list(neg1)[0],
                "word2": list(pos2)[0],
            }

        return {"detected": False, "word1": "", "word2": ""}

    def reinforce_belief(self, belief_id: str) -> bool:
        """Increase reinforcement count when a belief is confirmed.

        Also slightly increases confidence (with diminishing returns).

        Args:
            belief_id: ID of the belief to reinforce

        Returns:
            True if reinforced, False if belief not found
        """
        belief_id = self._validate_string_input(belief_id, "belief_id", 100)

        # Get the belief (include inactive to allow reinforcing superseded beliefs back)
        beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        existing = None
        for b in beliefs:
            if b.id == belief_id:
                existing = b
                break

        if not existing:
            return False

        # Store old confidence BEFORE modification for accurate history tracking
        old_confidence = existing.confidence

        # Increment reinforcement count first
        existing.times_reinforced += 1

        # Slightly increase confidence (diminishing returns)
        # Each reinforcement adds less confidence, capped at 0.99
        # Use (times_reinforced) which is already incremented, so first reinforcement uses 1
        confidence_boost = 0.05 * (1.0 / (1 + existing.times_reinforced * 0.1))
        room_to_grow = 0.99 - existing.confidence
        existing.confidence = min(0.99, existing.confidence + room_to_grow * confidence_boost)

        # Update confidence history with accurate old/new values
        history = existing.confidence_history or []
        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old": round(old_confidence, 3),
                "new": round(existing.confidence, 3),
                "reason": f"Reinforced (count: {existing.times_reinforced})",
            }
        )
        existing.confidence_history = history[-20:]  # Keep last 20 entries

        existing.last_verified = datetime.now(timezone.utc)
        existing.verification_count += 1

        # Use atomic update with optimistic concurrency control
        self._storage.update_belief_atomic(existing)
        return True

    def supersede_belief(
        self,
        old_id: str,
        new_statement: str,
        confidence: float = 0.8,
        reason: Optional[str] = None,
    ) -> str:
        """Replace an old belief with a new one, maintaining the revision chain.

        Args:
            old_id: ID of the belief being superseded
            new_statement: The new belief statement
            confidence: Confidence in the new belief
            reason: Optional reason for the supersession

        Returns:
            ID of the new belief

        Raises:
            ValueError: If old belief not found
        """
        old_id = self._validate_string_input(old_id, "old_id", 100)
        new_statement = self._validate_string_input(new_statement, "new_statement", 2000)

        # Get the old belief
        beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        old_belief = None
        for b in beliefs:
            if b.id == old_id:
                old_belief = b
                break

        if not old_belief:
            raise ValueError(f"Belief {old_id} not found")

        # Create the new belief
        new_id = str(uuid.uuid4())
        new_belief = Belief(
            id=new_id,
            agent_id=self.agent_id,
            statement=new_statement,
            belief_type=old_belief.belief_type,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            source_type="inference",
            supersedes=old_id,
            superseded_by=None,
            times_reinforced=0,
            is_active=True,
            # Inherit source episodes from old belief
            source_episodes=old_belief.source_episodes,
            derived_from=[f"belief:{old_id}"],
            confidence_history=[
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "old": 0.0,
                    "new": confidence,
                    "reason": reason or f"Superseded belief {old_id[:8]}",
                }
            ],
        )
        self._storage.save_belief(new_belief)

        # Update the old belief
        old_belief.superseded_by = new_id
        old_belief.is_active = False

        # Add to confidence history
        history = old_belief.confidence_history or []
        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old": old_belief.confidence,
                "new": old_belief.confidence,
                "reason": f"Superseded by belief {new_id[:8]}: {reason or 'no reason given'}",
            }
        )
        old_belief.confidence_history = history[-20:]
        # Use atomic update with optimistic concurrency control
        self._storage.update_belief_atomic(old_belief)

        return new_id

    def revise_beliefs_from_episode(self, episode_id: str) -> Dict[str, Any]:
        """Analyze an episode and update relevant beliefs.

        Extracts lessons and patterns from the episode, then:
        1. Reinforces beliefs that were confirmed
        2. Identifies beliefs that may be contradicted
        3. Suggests new beliefs based on lessons

        Args:
            episode_id: ID of the episode to analyze

        Returns:
            Dict with keys: reinforced, contradicted, suggested_new
        """
        episode_id = self._validate_string_input(episode_id, "episode_id", 100)

        # Get the episode
        episode = self._storage.get_episode(episode_id)
        if not episode:
            return {
                "error": "Episode not found",
                "reinforced": [],
                "contradicted": [],
                "suggested_new": [],
            }

        result = {
            "episode_id": episode_id,
            "reinforced": [],
            "contradicted": [],
            "suggested_new": [],
        }

        # Build evidence text from episode
        evidence_parts = []
        if episode.outcome_type == "success":
            evidence_parts.append(f"Successfully: {episode.objective}")
        elif episode.outcome_type == "failure":
            evidence_parts.append(f"Failed: {episode.objective}")

        evidence_parts.append(episode.outcome)

        if episode.lessons:
            evidence_parts.extend(episode.lessons)

        evidence_text = " ".join(evidence_parts)

        # Get all active beliefs
        beliefs = self._storage.get_beliefs(limit=500)

        for belief in beliefs:
            belief_stmt_lower = belief.statement.lower()
            evidence_lower = evidence_text.lower()

            # Check for word overlap
            belief_words = set(belief_stmt_lower.split()) - {
                "i",
                "the",
                "a",
                "an",
                "to",
                "and",
                "or",
                "is",
                "are",
                "should",
                "can",
            }
            evidence_words = set(evidence_lower.split()) - {
                "i",
                "the",
                "a",
                "an",
                "to",
                "and",
                "or",
                "is",
                "are",
                "should",
                "can",
            }
            overlap = belief_words & evidence_words

            if len(overlap) < 2:
                continue  # Not related enough

            # Determine if evidence supports or contradicts
            is_supporting = False
            is_contradicting = False

            if episode.outcome_type == "success":
                # Success supports "should" beliefs about what worked
                if any(
                    word in belief_stmt_lower
                    for word in ["should", "prefer", "good", "important", "effective"]
                ):
                    is_supporting = True
                # Success contradicts "avoid" beliefs about what worked
                elif any(word in belief_stmt_lower for word in ["avoid", "never", "don't", "bad"]):
                    is_contradicting = True

            elif episode.outcome_type == "failure":
                # Failure contradicts "should" beliefs about what failed
                if any(
                    word in belief_stmt_lower
                    for word in ["should", "prefer", "good", "important", "effective"]
                ):
                    is_contradicting = True
                # Failure supports "avoid" beliefs
                elif any(word in belief_stmt_lower for word in ["avoid", "never", "don't", "bad"]):
                    is_supporting = True

            if is_supporting:
                # Reinforce the belief
                self.reinforce_belief(belief.id)
                result["reinforced"].append(
                    {
                        "belief_id": belief.id,
                        "statement": belief.statement,
                        "overlap": list(overlap),
                    }
                )

            elif is_contradicting:
                # Flag as potentially contradicted
                result["contradicted"].append(
                    {
                        "belief_id": belief.id,
                        "statement": belief.statement,
                        "overlap": list(overlap),
                        "evidence": evidence_text[:200],
                    }
                )

        # Suggest new beliefs from lessons
        if episode.lessons:
            for lesson in episode.lessons:
                # Check if a similar belief already exists
                existing = self._storage.find_belief(lesson)
                if not existing:
                    # Check for similar beliefs via search
                    similar = self._storage.search(lesson, limit=3, record_types=["belief"])
                    if not any(r.score > 0.9 for r in similar):
                        result["suggested_new"].append(
                            {
                                "statement": lesson,
                                "source_episode": episode_id,
                                "suggested_confidence": (
                                    0.7 if episode.outcome_type == "success" else 0.6
                                ),
                            }
                        )

        # Link episode to affected beliefs
        for reinforced in result["reinforced"]:
            belief = next((b for b in beliefs if b.id == reinforced["belief_id"]), None)
            if belief:
                source_eps = belief.source_episodes or []
                if episode_id not in source_eps:
                    belief.source_episodes = source_eps + [episode_id]
                    self._storage.save_belief(belief)

        return result

    def get_belief_history(self, belief_id: str) -> List[Dict[str, Any]]:
        """Get the supersession chain for a belief.

        Walks both backwards (what this belief superseded) and forwards
        (what superseded this belief) to build the full revision history.

        Args:
            belief_id: ID of the belief to trace

        Returns:
            List of beliefs in chronological order, with revision metadata
        """
        belief_id = self._validate_string_input(belief_id, "belief_id", 100)

        # Get all beliefs including inactive ones
        all_beliefs = self._storage.get_beliefs(limit=1000, include_inactive=True)
        belief_map = {b.id: b for b in all_beliefs}

        if belief_id not in belief_map:
            return []

        history = []
        visited = set()

        # Walk backwards to find the original belief
        def walk_back(bid: str) -> Optional[str]:
            if bid in visited or bid not in belief_map:
                return None
            belief = belief_map[bid]
            if belief.supersedes and belief.supersedes in belief_map:
                return belief.supersedes
            return None

        # Find the root
        root_id = belief_id
        while True:
            prev = walk_back(root_id)
            if prev:
                root_id = prev
            else:
                break

        # Walk forward from root
        current_id = root_id
        while current_id and current_id not in visited and current_id in belief_map:
            visited.add(current_id)
            belief = belief_map[current_id]

            entry = {
                "id": belief.id,
                "statement": belief.statement,
                "confidence": belief.confidence,
                "times_reinforced": belief.times_reinforced,
                "is_active": belief.is_active,
                "is_current": belief.id == belief_id,
                "created_at": belief.created_at.isoformat() if belief.created_at else None,
                "supersedes": belief.supersedes,
                "superseded_by": belief.superseded_by,
            }

            # Add supersession reason if available from confidence history
            if belief.confidence_history:
                for h in reversed(belief.confidence_history):
                    reason = h.get("reason", "")
                    if "Superseded" in reason:
                        entry["supersession_reason"] = reason
                        break

            history.append(entry)
            current_id = belief.superseded_by

        return history

    # =========================================================================
    # SEARCH
    # =========================================================================

    def search(
        self, query: str, limit: int = 10, min_score: float = None, track_access: bool = True
    ) -> List[Dict[str, Any]]:
        """Search across episodes, notes, and beliefs.

        Args:
            query: Search query string
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0-1.0) to include in results.
                       If None, returns all results up to limit.
            track_access: If True (default), record access for salience tracking.
        """
        # Request more results if filtering by score
        fetch_limit = limit * 3 if min_score else limit
        results = self._storage.search(query, limit=fetch_limit)

        # Filter by minimum score if specified
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Track access for returned results
        if track_access and results:
            accesses = [(r.record_type, r.record.id) for r in results[:limit]]
            self._storage.record_access_batch(accesses)

        formatted = []
        for r in results:
            record = r.record
            record_type = r.record_type

            if record_type == "episode":
                formatted.append(
                    {
                        "type": "episode",
                        "title": record.objective[:60] if record.objective else "",
                        "content": record.outcome,
                        "lessons": (record.lessons or [])[:2],
                        "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
                    }
                )
            elif record_type == "note":
                formatted.append(
                    {
                        "type": record.note_type or "note",
                        "title": record.content[:60] if record.content else "",
                        "content": record.content,
                        "tags": record.tags or [],
                        "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
                    }
                )
            elif record_type == "belief":
                formatted.append(
                    {
                        "type": "belief",
                        "title": record.statement[:60] if record.statement else "",
                        "content": record.statement,
                        "confidence": record.confidence,
                        "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
                    }
                )

        return formatted[:limit]

    # =========================================================================
    # STATUS
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = self._storage.get_stats()

        return {
            "agent_id": self.agent_id,
            "values": stats.get("values", 0),
            "beliefs": stats.get("beliefs", 0),
            "goals": stats.get("goals", 0),
            "episodes": stats.get("episodes", 0),
            "raw": stats.get("raw", 0),
            "checkpoint": self.load_checkpoint() is not None,
        }

    # =========================================================================
    # FORMATTING
    # =========================================================================

    def format_memory(self, memory: Optional[Dict[str, Any]] = None) -> str:
        """Format memory for injection into context."""
        if memory is None:
            memory = self.load()

        lines = [
            f"# Working Memory ({self.agent_id})",
            f"_Loaded at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
        ]

        # Checkpoint - prominently displayed at top
        if memory.get("checkpoint"):
            cp = memory["checkpoint"]

            # Calculate checkpoint age
            age_warning = ""
            try:
                ts = cp.get("timestamp", "")
                if ts:
                    cp_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    age = now - cp_time
                    if age.total_seconds() > 24 * 3600:
                        age_warning = f"\n⚠ _Checkpoint is {age.days}+ days old - may be stale_"
                    elif age.total_seconds() > 6 * 3600:
                        age_warning = f"\n⚠ _Checkpoint is {age.seconds // 3600}+ hours old_"
            except Exception:
                pass

            lines.append("## Working State")
            lines.append(f"**Task**: {cp.get('current_task', 'unknown')}")
            if cp.get("context"):
                lines.append(f"**Context**: {cp['context']}")
            if cp.get("pending"):
                lines.append("**Pending**:")
                for p in cp["pending"]:
                    lines.append(f"  - {p}")
            if age_warning:
                lines.append(age_warning)
            lines.append("")

        # Values
        if memory.get("values"):
            lines.append("## Values")
            for v in memory["values"]:
                lines.append(f"- **{v['name']}**: {v['statement']}")
            lines.append("")

        # Goals
        if memory.get("goals"):
            lines.append("## Goals")
            for g in memory["goals"]:
                priority = f" [{g['priority']}]" if g.get("priority") else ""
                lines.append(f"- {g['title']}{priority}")
            lines.append("")

        # Beliefs
        if memory.get("beliefs"):
            lines.append("## Beliefs")
            for b in memory["beliefs"]:
                conf = f" ({b['confidence']})" if b.get("confidence") else ""
                lines.append(f"- {b['statement']}{conf}")
            lines.append("")

        # Lessons
        if memory.get("lessons"):
            lines.append("## Lessons")
            for lesson in memory["lessons"][:10]:
                lines.append(f"- {lesson}")
            lines.append("")

        # Recent work
        if memory.get("recent_work"):
            lines.append("## Recent Work")
            for w in memory["recent_work"][:3]:
                lines.append(f"- {w['objective']} [{w.get('outcome_type', '?')}]")
            lines.append("")

        # Drives
        if memory.get("drives"):
            lines.append("## Drives")
            for d in memory["drives"]:
                lines.append(f"- **{d['drive_type']}**: {d['intensity']:.0%}")
            lines.append("")

        # Relationships
        if memory.get("relationships"):
            lines.append("## Key Relationships")
            for r in memory["relationships"][:5]:
                lines.append(f"- {r['entity_name']}: sentiment {r.get('sentiment', 0):.0%}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # DRIVES (Motivation System)
    # =========================================================================

    DRIVE_TYPES = ["existence", "growth", "curiosity", "connection", "reproduction"]

    def load_drives(self) -> List[Dict[str, Any]]:
        """Load current drive states."""
        drives = self._storage.get_drives()
        return [
            {
                "id": d.id,
                "drive_type": d.drive_type,
                "intensity": d.intensity,
                "last_satisfied_at": d.updated_at.isoformat() if d.updated_at else None,
                "focus_areas": d.focus_areas,
            }
            for d in drives
        ]

    def drive(
        self,
        drive_type: str,
        intensity: float = 0.5,
        focus_areas: Optional[List[str]] = None,
        decay_hours: int = 24,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Set or update a drive.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
        """
        if drive_type not in self.DRIVE_TYPES:
            raise ValueError(f"Invalid drive type. Must be one of: {self.DRIVE_TYPES}")

        # Check if drive exists
        existing = self._storage.get_drive(drive_type)

        now = datetime.now(timezone.utc)

        if existing:
            existing.intensity = max(0.0, min(1.0, intensity))
            existing.focus_areas = focus_areas or []
            existing.updated_at = now
            # TODO: Add update_drive_atomic for optimistic concurrency control
            existing.version += 1
            if context is not None:
                existing.context = context
            if context_tags is not None:
                existing.context_tags = context_tags
            self._storage.save_drive(existing)
            return existing.id
        else:
            drive_id = str(uuid.uuid4())
            drive = Drive(
                id=drive_id,
                agent_id=self.agent_id,
                drive_type=drive_type,
                intensity=max(0.0, min(1.0, intensity)),
                focus_areas=focus_areas or [],
                created_at=now,
                updated_at=now,
                context=context,
                context_tags=context_tags,
            )
            self._storage.save_drive(drive)
            return drive_id

    def satisfy_drive(self, drive_type: str, amount: float = 0.2) -> bool:
        """Record satisfaction of a drive (reduces intensity toward baseline)."""
        existing = self._storage.get_drive(drive_type)

        if existing:
            new_intensity = max(0.1, existing.intensity - amount)
            existing.intensity = new_intensity
            existing.updated_at = datetime.now(timezone.utc)
            # TODO: Add update_drive_atomic for optimistic concurrency control
            existing.version += 1
            self._storage.save_drive(existing)
            return True
        return False

    # =========================================================================
    # RELATIONAL MEMORY (Models of Other Agents)
    # =========================================================================

    def load_relationships(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Load relationship models for other agents."""
        relationships = self._storage.get_relationships()

        # Sort by last interaction, descending
        relationships = sorted(
            relationships,
            key=lambda r: r.last_interaction or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        return [
            {
                "other_agent_id": r.entity_name,  # backwards compat
                "entity_name": r.entity_name,
                "entity_type": r.entity_type,
                "trust_level": (r.sentiment + 1) / 2,  # Convert sentiment to trust
                "sentiment": r.sentiment,
                "interaction_count": r.interaction_count,
                "last_interaction": r.last_interaction.isoformat() if r.last_interaction else None,
                "notes": r.notes,
            }
            for r in relationships[:limit]
        ]

    def relationship(
        self,
        other_agent_id: str,
        trust_level: Optional[float] = None,
        notes: Optional[str] = None,
        interaction_type: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> str:
        """Update relationship model for another entity.

        Args:
            other_agent_id: Name/identifier of the other entity
            trust_level: Trust level 0.0-1.0 (converted to sentiment -1 to 1)
            notes: Notes about the relationship
            interaction_type: Type of interaction being logged
            entity_type: Type of entity (person, agent, organization, system)
        """
        # Check existing
        existing = self._storage.get_relationship(other_agent_id)

        now = datetime.now(timezone.utc)

        if existing:
            if trust_level is not None:
                # Convert trust_level (0-1) to sentiment (-1 to 1)
                existing.sentiment = max(-1.0, min(1.0, (trust_level * 2) - 1))
            if notes:
                existing.notes = notes
            if entity_type:
                existing.entity_type = entity_type
            existing.interaction_count += 1
            existing.last_interaction = now
            existing.version += 1
            self._storage.save_relationship(existing)
            return existing.id
        else:
            rel_id = str(uuid.uuid4())
            relationship = Relationship(
                id=rel_id,
                agent_id=self.agent_id,
                entity_name=other_agent_id,
                entity_type=entity_type or "person",
                relationship_type=interaction_type or "interaction",
                notes=notes,
                sentiment=((trust_level * 2) - 1) if trust_level is not None else 0.0,
                interaction_count=1,
                last_interaction=now,
                created_at=now,
            )
            self._storage.save_relationship(relationship)
            return rel_id

    # =========================================================================
    # PLAYBOOKS (Procedural Memory)
    # =========================================================================

    MASTERY_LEVELS = ["novice", "competent", "proficient", "expert"]

    def playbook(
        self,
        name: str,
        description: str,
        steps: Union[List[Dict[str, Any]], List[str]],
        triggers: Optional[List[str]] = None,
        failure_modes: Optional[List[str]] = None,
        recovery_steps: Optional[List[str]] = None,
        source_episodes: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 0.8,
    ) -> str:
        """Create a new playbook (procedural memory).

        Args:
            name: Short name for the playbook (e.g., "Deploy to production")
            description: What this playbook does
            steps: List of steps - can be dicts with {action, details, adaptations}
                   or simple strings
            triggers: When to use this playbook (situation descriptions)
            failure_modes: What can go wrong
            recovery_steps: How to recover from failures
            source_episodes: Episode IDs this was learned from
            tags: Tags for categorization
            confidence: Initial confidence (0.0-1.0)

        Returns:
            Playbook ID
        """
        from kernle.storage import Playbook

        # Validate inputs
        name = self._validate_string_input(name, "name", 200)
        description = self._validate_string_input(description, "description", 2000)

        # Normalize steps to dict format
        normalized_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, str):
                normalized_steps.append(
                    {
                        "action": step,
                        "details": None,
                        "adaptations": None,
                    }
                )
            elif isinstance(step, dict):
                normalized_steps.append(
                    {
                        "action": step.get("action", f"Step {i + 1}"),
                        "details": step.get("details"),
                        "adaptations": step.get("adaptations"),
                    }
                )
            else:
                raise ValueError(f"Invalid step format at index {i}")

        # Validate optional lists
        if triggers:
            triggers = [self._validate_string_input(t, "trigger", 500) for t in triggers]
        if failure_modes:
            failure_modes = [
                self._validate_string_input(f, "failure_mode", 500) for f in failure_modes
            ]
        if recovery_steps:
            recovery_steps = [
                self._validate_string_input(r, "recovery_step", 500) for r in recovery_steps
            ]
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]

        playbook_id = str(uuid.uuid4())

        playbook = Playbook(
            id=playbook_id,
            agent_id=self.agent_id,
            name=name,
            description=description,
            trigger_conditions=triggers or [],
            steps=normalized_steps,
            failure_modes=failure_modes or [],
            recovery_steps=recovery_steps,
            mastery_level="novice",
            times_used=0,
            success_rate=0.0,
            source_episodes=source_episodes,
            tags=tags,
            confidence=max(0.0, min(1.0, confidence)),
            last_used=None,
            created_at=datetime.now(timezone.utc),
        )

        self._storage.save_playbook(playbook)
        return playbook_id

    def load_playbooks(
        self, limit: int = 10, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load playbooks (procedural memories).

        Args:
            limit: Maximum number of playbooks to return
            tags: Filter by tags

        Returns:
            List of playbook dicts
        """
        playbooks = self._storage.list_playbooks(tags=tags, limit=limit)

        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "triggers": p.trigger_conditions,
                "steps": p.steps,
                "failure_modes": p.failure_modes,
                "recovery_steps": p.recovery_steps,
                "mastery_level": p.mastery_level,
                "times_used": p.times_used,
                "success_rate": p.success_rate,
                "confidence": p.confidence,
                "tags": p.tags,
                "last_used": p.last_used.isoformat() if p.last_used else None,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in playbooks
        ]

    def find_playbook(self, situation: str) -> Optional[Dict[str, Any]]:
        """Find the most relevant playbook for a given situation.

        Uses semantic search to match the situation against playbook
        triggers and descriptions.

        Args:
            situation: Description of the current situation/task

        Returns:
            Best matching playbook dict, or None if no good match
        """
        # Search for relevant playbooks
        playbooks = self._storage.search_playbooks(situation, limit=5)

        if not playbooks:
            return None

        # Return the best match (first result from search)
        p = playbooks[0]
        return {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "triggers": p.trigger_conditions,
            "steps": p.steps,
            "failure_modes": p.failure_modes,
            "recovery_steps": p.recovery_steps,
            "mastery_level": p.mastery_level,
            "times_used": p.times_used,
            "success_rate": p.success_rate,
            "confidence": p.confidence,
            "tags": p.tags,
        }

    def record_playbook_use(self, playbook_id: str, success: bool) -> bool:
        """Record a playbook usage and update statistics.

        Call this after executing a playbook to track its effectiveness.

        Args:
            playbook_id: ID of the playbook that was used
            success: Whether the execution was successful

        Returns:
            True if updated, False if playbook not found
        """
        return self._storage.update_playbook_usage(playbook_id, success)

    def get_playbook(self, playbook_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific playbook by ID.

        Args:
            playbook_id: ID of the playbook

        Returns:
            Playbook dict or None if not found
        """
        p = self._storage.get_playbook(playbook_id)
        if not p:
            return None

        return {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "triggers": p.trigger_conditions,
            "steps": p.steps,
            "failure_modes": p.failure_modes,
            "recovery_steps": p.recovery_steps,
            "mastery_level": p.mastery_level,
            "times_used": p.times_used,
            "success_rate": p.success_rate,
            "source_episodes": p.source_episodes,
            "confidence": p.confidence,
            "tags": p.tags,
            "last_used": p.last_used.isoformat() if p.last_used else None,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }

    def search_playbooks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search playbooks by query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching playbook dicts
        """
        playbooks = self._storage.search_playbooks(query, limit=limit)

        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "triggers": p.trigger_conditions,
                "mastery_level": p.mastery_level,
                "times_used": p.times_used,
                "success_rate": p.success_rate,
                "tags": p.tags,
            }
            for p in playbooks
        ]

    # =========================================================================
    # TEMPORAL MEMORY (Time-Aware Retrieval)
    # =========================================================================

    def load_temporal(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Load memories within a time range."""
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end.replace(hour=0, minute=0, second=0, microsecond=0)

        # Get episodes in range
        episodes = self._storage.get_episodes(limit=limit, since=start)
        episodes = [e for e in episodes if e.created_at and e.created_at <= end]

        # Get notes in range
        notes = self._storage.get_notes(limit=limit, since=start)
        notes = [n for n in notes if n.created_at and n.created_at <= end]

        return {
            "range": {"start": start.isoformat(), "end": end.isoformat()},
            "episodes": [
                {
                    "objective": e.objective,
                    "outcome_type": e.outcome_type,
                    "lessons_learned": e.lessons,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in episodes
            ],
            "notes": [
                {
                    "content": n.content,
                    "metadata": {"note_type": n.note_type, "tags": n.tags},
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in notes
            ],
        }

    def what_happened(self, when: str = "today") -> Dict[str, Any]:
        """Natural language time query."""
        now = datetime.now(timezone.utc)

        if when == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif when == "yesterday":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return self.load_temporal(start, end)
        elif when == "this week":
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif when == "last hour":
            start = now - timedelta(hours=1)
        else:
            # Default to today
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        return self.load_temporal(start, now)

    # =========================================================================
    # SIGNAL DETECTION (Auto-Capture Significance)
    # =========================================================================

    SIGNAL_PATTERNS = {
        "success": {
            "keywords": ["completed", "done", "finished", "succeeded", "works", "fixed", "solved"],
            "weight": 0.7,
            "type": "positive",
        },
        "failure": {
            "keywords": ["failed", "error", "broken", "doesn't work", "bug", "issue"],
            "weight": 0.7,
            "type": "negative",
        },
        "decision": {
            "keywords": ["decided", "chose", "going with", "will use", "picked"],
            "weight": 0.8,
            "type": "decision",
        },
        "lesson": {
            "keywords": ["learned", "realized", "insight", "discovered", "understood"],
            "weight": 0.9,
            "type": "lesson",
        },
        "feedback": {
            "keywords": ["great", "thanks", "helpful", "perfect", "exactly", "wrong", "not what"],
            "weight": 0.6,
            "type": "feedback",
        },
    }

    def detect_significance(self, text: str) -> Dict[str, Any]:
        """Detect if text contains significant signals worth capturing."""
        text_lower = text.lower()
        signals = []
        total_weight = 0.0

        for signal_name, pattern in self.SIGNAL_PATTERNS.items():
            for keyword in pattern["keywords"]:
                if keyword in text_lower:
                    signals.append(
                        {
                            "signal": signal_name,
                            "type": pattern["type"],
                            "weight": pattern["weight"],
                        }
                    )
                    total_weight = max(total_weight, pattern["weight"])
                    break  # One match per pattern is enough

        return {
            "significant": total_weight >= 0.6,
            "score": total_weight,
            "signals": signals,
        }

    def auto_capture(self, text: str, context: Optional[str] = None) -> Optional[str]:
        """Automatically capture text if it's significant."""
        detection = self.detect_significance(text)

        if detection["significant"]:
            # Determine what type of capture
            primary_signal = detection["signals"][0] if detection["signals"] else None

            if primary_signal:
                if primary_signal["type"] == "decision":
                    return self.note(text, type="decision", tags=["auto-captured"])
                elif primary_signal["type"] == "lesson":
                    return self.note(text, type="insight", tags=["auto-captured"])
                elif primary_signal["type"] in ("positive", "negative"):
                    # Could be an episode outcome
                    outcome = "success" if primary_signal["type"] == "positive" else "partial"
                    return self.episode(
                        objective=context or "Auto-captured event",
                        outcome=outcome,
                        lessons=[text] if "learn" in text.lower() else None,
                        tags=["auto-captured"],
                    )
                else:
                    return self.note(text, type="note", tags=["auto-captured"])

        return None

    # CONSOLIDATION
    # =========================================================================

    def consolidate(self, min_episodes: int = 3) -> Dict[str, Any]:
        """Run memory consolidation.

        Analyzes recent episodes to extract patterns, lessons, and beliefs.

        Args:
            min_episodes: Minimum episodes required to consolidate

        Returns:
            Consolidation results
        """
        episodes = self._storage.get_episodes(limit=50)

        if len(episodes) < min_episodes:
            return {
                "consolidated": 0,
                "new_beliefs": 0,
                "lessons_found": 0,
                "message": f"Need at least {min_episodes} episodes to consolidate",
            }

        # Simple consolidation: extract lessons from recent episodes
        all_lessons = []
        for ep in episodes:
            if ep.lessons:
                all_lessons.extend(ep.lessons)

        # Count unique lessons
        from collections import Counter

        lesson_counts = Counter(all_lessons)
        common_lessons = [lesson for lesson, count in lesson_counts.items() if count >= 2]

        return {
            "consolidated": len(episodes),
            "new_beliefs": 0,  # Would need LLM integration for belief extraction
            "lessons_found": len(common_lessons),
            "common_lessons": common_lessons[:5],
        }

    # =========================================================================
    # IDENTITY SYNTHESIS
    # =========================================================================

    def synthesize_identity(self) -> Dict[str, Any]:
        """Synthesize identity from memory.

        Combines values, beliefs, goals, and experiences into a coherent
        identity narrative.

        Returns:
            Identity synthesis including narrative and key components
        """
        values = self._storage.get_values(limit=10)
        beliefs = self._storage.get_beliefs(limit=20)
        goals = self._storage.get_goals(status="active", limit=10)
        episodes = self._storage.get_episodes(limit=20)
        drives = self._storage.get_drives()

        # Build narrative from components
        narrative_parts = []

        if values:
            top_value = max(values, key=lambda v: v.priority)
            narrative_parts.append(
                f"I value {top_value.name.lower()} highly: {top_value.statement}"
            )

        if beliefs:
            high_conf = [b for b in beliefs if b.confidence >= 0.8]
            if high_conf:
                narrative_parts.append(f"I believe: {high_conf[0].statement}")

        if goals:
            narrative_parts.append(f"I'm currently working on: {goals[0].title}")

        narrative = " ".join(narrative_parts) if narrative_parts else "Identity still forming."

        # Calculate confidence using the comprehensive scoring method
        confidence = self.get_identity_confidence()

        return {
            "narrative": narrative,
            "core_values": [
                {"name": v.name, "statement": v.statement, "priority": v.priority}
                for v in sorted(values, key=lambda v: v.priority, reverse=True)[:5]
            ],
            "key_beliefs": [
                {"statement": b.statement, "confidence": b.confidence, "foundational": False}
                for b in sorted(beliefs, key=lambda b: b.confidence, reverse=True)[:5]
            ],
            "active_goals": [{"title": g.title, "priority": g.priority} for g in goals[:5]],
            "drives": {d.drive_type: d.intensity for d in drives},
            "significant_episodes": [
                {
                    "objective": e.objective,
                    "outcome": e.outcome_type,
                    "lessons": e.lessons,
                }
                for e in episodes[:5]
            ],
            "confidence": confidence,
        }

    def get_identity_confidence(self) -> float:
        """Get overall identity confidence score.

        Calculates identity coherence based on:
        - Core values (20%): Having defined principles
        - Beliefs (20%): Both count and confidence quality
        - Goals (15%): Having direction and purpose
        - Episodes (20%): Experience count and reflection (lessons) rate
        - Drives (15%): Understanding intrinsic motivations
        - Relationships (10%): Modeling connections to others

        Returns:
            Confidence score (0.0-1.0) based on identity completeness and quality
        """
        # Get identity data
        values = self._storage.get_values(limit=10)
        beliefs = self._storage.get_beliefs(limit=20)
        goals = self._storage.get_goals(status="active", limit=10)
        episodes = self._storage.get_episodes(limit=50)
        drives = self._storage.get_drives()
        relationships = self._storage.get_relationships()

        # Values (20%): quantity × quality (priority)
        # Ideal: 3-5 values with high priority
        if values and len(values) > 0:
            value_count_score = min(1.0, len(values) / 5)
            avg_priority = sum(v.priority / 100 for v in values) / len(values)
            value_score = (value_count_score * 0.6 + avg_priority * 0.4) * 0.20
        else:
            value_score = 0.0

        # Beliefs (20%): quantity × quality (confidence)
        # Ideal: 5-10 beliefs with high confidence
        if beliefs and len(beliefs) > 0:
            avg_belief_conf = sum(b.confidence for b in beliefs) / len(beliefs)
            belief_count_score = min(1.0, len(beliefs) / 10)
            belief_score = (belief_count_score * 0.5 + avg_belief_conf * 0.5) * 0.20
        else:
            belief_score = 0.0

        # Goals (15%): having active direction
        # Ideal: 2-5 active goals
        goal_score = min(1.0, len(goals) / 5) * 0.15

        # Episodes (20%): experience × reflection
        # Ideal: 10-20 episodes with lessons extracted
        if episodes and len(episodes) > 0:
            with_lessons = sum(1 for e in episodes if e.lessons)
            lesson_rate = with_lessons / len(episodes)
            episode_count_score = min(1.0, len(episodes) / 20)
            episode_score = (episode_count_score * 0.5 + lesson_rate * 0.5) * 0.20
        else:
            episode_score = 0.0

        # Drives (15%): understanding motivations
        # Ideal: 2-3 drives defined (curiosity, growth, connection, etc.)
        drive_score = min(1.0, len(drives) / 3) * 0.15

        # Relationships (10%): modeling connections
        # Ideal: 3-5 key relationships tracked
        relationship_score = min(1.0, len(relationships) / 5) * 0.10

        total = (
            value_score
            + belief_score
            + goal_score
            + episode_score
            + drive_score
            + relationship_score
        )

        return round(total, 3)

    def detect_identity_drift(self, days: int = 30) -> Dict[str, Any]:
        """Detect changes in identity over time.

        Args:
            days: Number of days to analyze

        Returns:
            Drift analysis including changed values and evolved beliefs
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Get recent additions
        recent_episodes = self._storage.get_episodes(limit=50, since=since)

        # Simple drift detection based on episode count and themes
        drift_score = min(1.0, len(recent_episodes) / 20) * 0.5

        return {
            "period_days": days,
            "drift_score": drift_score,
            "changed_values": [],  # Would need historical comparison
            "evolved_beliefs": [],
            "new_experiences": [
                {
                    "objective": e.objective,
                    "outcome": e.outcome_type,
                    "lessons": e.lessons,
                    "date": e.created_at.strftime("%Y-%m-%d") if e.created_at else "",
                }
                for e in recent_episodes[:5]
            ],
        }

    # =========================================================================
    # SYNC
    # =========================================================================

    def sync(self) -> Dict[str, Any]:
        """Sync local changes with cloud storage.

        Returns:
            Sync results including counts and any errors
        """
        result = self._storage.sync()
        return {
            "pushed": result.pushed,
            "pulled": result.pulled,
            "conflicts": result.conflicts,
            "errors": result.errors,
            "success": result.success,
        }

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status.

        Returns:
            Sync status including pending count and connectivity
        """
        return {
            "pending": self._storage.get_pending_sync_count(),
            "online": self._storage.is_online(),
        }

    def _sync_before_load(self) -> Dict[str, Any]:
        """Pull remote changes before loading local state.

        Called automatically by load() when auto_sync is enabled.
        Non-blocking: logs errors but doesn't fail the load.

        Returns:
            Dict with pull result or error info
        """
        result = {
            "attempted": False,
            "pulled": 0,
            "conflicts": 0,
            "errors": [],
        }

        try:
            # Check if sync is available
            if not self._storage.is_online():
                logger.debug("Sync before load: offline, skipping pull")
                return result

            result["attempted"] = True
            pull_result = self._storage.pull_changes()
            result["pulled"] = pull_result.pulled
            result["conflicts"] = pull_result.conflicts
            result["errors"] = pull_result.errors

            if pull_result.pulled > 0:
                logger.info(f"Sync before load: pulled {pull_result.pulled} changes")
            if pull_result.errors:
                logger.warning(
                    f"Sync before load: {len(pull_result.errors)} errors: {pull_result.errors[:3]}"
                )

        except Exception as e:
            # Don't fail the load on sync errors
            logger.warning(f"Sync before load failed (continuing with local data): {e}")
            result["errors"].append(str(e))

        return result

    def _sync_after_checkpoint(self) -> Dict[str, Any]:
        """Push local changes after saving a checkpoint.

        Called automatically by checkpoint() when auto_sync is enabled.
        Non-blocking: logs errors but doesn't fail the checkpoint save.

        Returns:
            Dict with push result or error info
        """
        result = {
            "attempted": False,
            "pushed": 0,
            "conflicts": 0,
            "errors": [],
        }

        try:
            # Check if sync is available
            if not self._storage.is_online():
                logger.debug("Sync after checkpoint: offline, changes queued for later")
                result["errors"].append("Offline - changes queued")
                return result

            result["attempted"] = True
            sync_result = self._storage.sync()
            result["pushed"] = sync_result.pushed
            result["conflicts"] = sync_result.conflicts
            result["errors"] = sync_result.errors

            if sync_result.pushed > 0:
                logger.info(f"Sync after checkpoint: pushed {sync_result.pushed} changes")
            if sync_result.errors:
                logger.warning(
                    f"Sync after checkpoint: {len(sync_result.errors)} errors: {sync_result.errors[:3]}"
                )

        except Exception as e:
            # Don't fail the checkpoint on sync errors
            logger.warning(f"Sync after checkpoint failed (local save succeeded): {e}")
            result["errors"].append(str(e))

        return result
