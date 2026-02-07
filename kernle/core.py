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

from kernle.features import (
    AnxietyMixin,
    ConsolidationMixin,
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
from kernle.storage import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Relationship,
    SelfNarrative,
    Summary,
    TrustAssessment,
    Value,
    get_storage,
)
from kernle.storage.base import (
    DEFAULT_TRUST,
    SEED_TRUST,
    SELF_TRUST_FLOOR,
    TRUST_DECAY_RATE,
    TRUST_DEPTH_DECAY,
    TRUST_THRESHOLDS,
)

# Import feature mixins
from kernle.utils import get_kernle_home

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
    "self_narrative": 0.90,  # Autobiographical identity — loads alongside values
    "summary_decade": 0.95,
    "summary_epoch": 0.85,
    "summary_year": 0.80,
    "belief": 0.70,
    "goal": 0.65,
    "drive": 0.60,
    "summary_quarter": 0.50,
    "episode": 0.40,
    "summary_month": 0.35,
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


def _get_memory_hint_text(memory_type: str, record: Any) -> str:
    """Get the primary text content of a memory record for echo hints."""
    if memory_type == "value":
        return f"{record.name}: {record.statement}"
    elif memory_type == "belief":
        return record.statement
    elif memory_type == "goal":
        return f"{record.title} {record.description or ''}"
    elif memory_type == "drive":
        return f"{record.drive_type}: {' '.join(record.focus_areas or [])}"
    elif memory_type == "episode":
        return f"{record.objective} {record.outcome}"
    elif memory_type == "note":
        return record.content
    elif memory_type == "relationship":
        return f"{record.entity_name}: {record.notes or ''}"
    return str(record)


def _truncate_to_words(text: str, max_words: int = 8) -> str:
    """Truncate text to approximately max_words words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _get_record_tags(memory_type: str, record: Any) -> List[str]:
    """Extract tags from a memory record."""
    tags = getattr(record, "tags", None) or []
    context_tags = getattr(record, "context_tags", None) or []
    focus_areas = []
    if memory_type == "drive":
        focus_areas = getattr(record, "focus_areas", None) or []
    return tags + context_tags + focus_areas


def _get_record_created_at(record: Any) -> Optional[datetime]:
    """Extract created_at datetime from a record."""
    return getattr(record, "created_at", None)


def _build_memory_echoes(
    excluded: list,
    max_echoes: int = 20,
) -> Dict[str, Any]:
    """Build memory echoes (peripheral awareness) from excluded candidates.

    After budget selection, this generates compact hints about memories that
    didn't fit in the token budget, giving the agent peripheral awareness
    of what else exists in memory.

    Args:
        excluded: Excluded candidate list [(priority, type, record), ...],
                  sorted by priority descending
        max_echoes: Maximum number of echo entries (default: 20)

    Returns:
        Dict with keys: echoes, temporal_summary, topic_clusters
    """
    if not excluded:
        return {
            "echoes": [],
            "temporal_summary": None,
            "topic_clusters": [],
        }

    echoes = []
    for priority, memory_type, record in excluded[:max_echoes]:
        hint_text = _get_memory_hint_text(memory_type, record)
        hint = _truncate_to_words(hint_text, max_words=8)
        echoes.append(
            {
                "type": memory_type,
                "id": record.id,
                "hint": hint,
                "salience": round(priority, 3),
            }
        )

    all_dates = []
    for _, memory_type, record in excluded:
        created = _get_record_created_at(record)
        if created:
            all_dates.append(created)

    temporal_summary = None
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        span_days = (max_date - min_date).days
        span_years = round(span_days / 365.25, 1)
        temporal_summary = (
            f"Memory spans {min_date.strftime('%Y-%m-%d')} to "
            f"{max_date.strftime('%Y-%m-%d')} ({span_years} years). "
            f"{len(excluded)} excluded memories."
        )

    tag_counts: Dict[str, int] = {}
    for _, memory_type, record in excluded:
        for tag in _get_record_tags(memory_type, record):
            tag_lower = tag.lower()
            tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    topic_clusters = [tag for tag, _count in sorted_tags[:6]]

    return {
        "echoes": echoes,
        "temporal_summary": temporal_summary,
        "topic_clusters": topic_clusters,
    }


def _get_record_attr(record: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute from a record, supporting both dataclass and dict."""
    if hasattr(record, attr):
        return getattr(record, attr, default)
    if isinstance(record, dict):
        return record.get(attr, default)
    return default


def compute_priority_score(memory_type: str, record: Any) -> float:
    """Compute priority score for a memory record.

    The score combines three weighted factors:
    - 55% type weight (base priority for memory type * type-specific factor)
    - 35% record factors (confidence, recency, etc.)
    - 10% emotional salience (abs(valence) * arousal * time-decay)

    Emotional salience uses a 90-day half-life decay so high-impact episodes
    remain cognitively available longer than standard 30-day salience.

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
        priority = _get_record_attr(record, "priority", 50)
        type_factor = priority / 100.0
    elif memory_type == "belief":
        type_factor = _get_record_attr(record, "confidence", 0.8)
    elif memory_type == "drive":
        type_factor = _get_record_attr(record, "intensity", 0.5)
    elif memory_type in ("goal", "episode", "note"):
        # For time-based priority, we'd need to compute recency
        # For now, use a default factor (records are already sorted by recency)
        type_factor = 0.7
    elif memory_type == "relationship":
        # Use sentiment as a factor
        sentiment = _get_record_attr(record, "sentiment", 0.0)
        type_factor = (sentiment + 1) / 2  # Normalize -1..1 to 0..1
    elif memory_type == "self_narrative":
        # Active narratives are always high priority
        type_factor = 0.9
    elif memory_type.startswith("summary_"):
        # Summaries use scope-based priority directly
        type_factor = 0.8
    else:
        type_factor = 0.5

    type_weight = base_priority  # base priority for this memory type
    record_factors = type_factor  # type-specific factor (confidence, priority, etc.)

    # Emotional salience: abs(valence) * arousal * time-decay(90-day half-life)
    valence = _get_record_attr(record, "emotional_valence", 0.0)
    arousal = _get_record_attr(record, "emotional_arousal", 0.0)

    emotional_salience = 0.0
    if abs(valence) > 0 or arousal > 0:
        half_life = 90.0  # 3x standard 30-day salience
        days_since = 0.0
        created_at = _get_record_attr(record, "created_at", None)
        if created_at is not None:
            try:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                delta = now - created_at
                days_since = max(0.0, delta.total_seconds() / 86400.0)
            except (ValueError, TypeError):
                pass
        emotional_salience = abs(valence) * arousal * (half_life / (days_since + half_life))

    # Weighted combination: 55% type weight, 35% record factors, 10% emotional salience
    score = 0.55 * type_weight + 0.35 * record_factors + 0.10 * emotional_salience

    # Belief scope boost: self-beliefs get +0.05 priority (KEP v3)
    if memory_type == "belief":
        belief_scope = _get_record_attr(record, "belief_scope", "world")
        if belief_scope == "self":
            score = min(1.0, score + 0.05)

    return score


class Kernle(
    AnxietyMixin,
    ConsolidationMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,
):
    """Main interface for Kernle memory operations.

    This is the **legacy compatibility API**. Write methods (episode, belief,
    value, goal, note, drive, relationship, raw) write directly to the storage
    backend and do NOT enforce provenance hierarchy or maintenance mode.

    For enforced memory operations, use :attr:`entity` which routes writes
    through the stack with full provenance validation.

    Use ``strict=True`` to route all writes through the stack, which enforces
    maintenance mode blocking, provenance hierarchy (when enabled), and stack
    component hooks.

    Examples:
        # Legacy mode (default) — no enforcement
        k = Kernle(stack_id="my_agent")

        # Strict mode — writes routed through stack enforcement
        k = Kernle(stack_id="my_agent", strict=True)

        # Recommended: use Entity directly for full enforcement
        from kernle import Entity
        e = Entity(core_id="my_agent")
    """

    def __init__(
        self,
        stack_id: Optional[str] = None,
        storage: Optional["StorageProtocol"] = None,
        # Keep supabase_url/key for backwards compatibility
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        strict: bool = False,
    ):
        """Initialize Kernle.

        Args:
            stack_id: Unique identifier for the agent
            storage: Optional storage backend. If None, auto-detects.
            supabase_url: Supabase project URL (deprecated, use storage param)
            supabase_key: Supabase API key (deprecated, use storage param)
            checkpoint_dir: Directory for local checkpoints
            strict: If True, route writes through stack enforcement layer
                (maintenance mode, provenance validation, component hooks).
                Requires SQLite-backed storage.
        """
        self.stack_id = self._validate_stack_id(
            stack_id or os.environ.get("KERNLE_STACK_ID", "default")
        )
        self.checkpoint_dir = self._validate_checkpoint_dir(
            checkpoint_dir or get_kernle_home() / "checkpoints"
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
                stack_id=self.stack_id,
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

        self._strict = strict

        logger.debug(
            f"Kernle initialized with storage: {type(self._storage).__name__}, "
            f"auto_sync: {self._auto_sync}, strict: {self._strict}"
        )

    @property
    def _write_backend(self):
        """Return the write target for memory operations.

        In strict mode, returns the stack (which enforces maintenance mode,
        provenance validation, and component hooks). In legacy mode, returns
        the storage backend directly (no enforcement).

        Raises:
            ValueError: If strict=True but no SQLite-backed stack is available.
        """
        if self._strict:
            stack = self.stack
            if stack is None:
                raise ValueError(
                    "strict=True requires SQLite-backed storage. "
                    "Use Entity for enforced writes with other storage backends."
                )
            return stack
        return self._storage

    @property
    def storage(self) -> "StorageProtocol":
        """Get the storage backend.

        .. deprecated:: 0.4.0
            Direct storage access will be deprecated in a future release.
            Prefer :attr:`entity` and :attr:`stack` for the new architecture.
        """
        return self._storage

    @property
    def entity(self):
        """Access the Entity (CoreProtocol) for new-style composition.

        The Entity is lazily created on first access. It provides the
        coordinator/bus for the new component architecture (v0.4.0+).

        Returns:
            Entity: The CoreProtocol implementation.
        """
        if not hasattr(self, "_entity"):
            from kernle.entity import Entity

            self._entity = Entity(core_id=self.stack_id)
        return self._entity

    @property
    def stack(self):
        """Access the SQLiteStack (StackProtocol) wrapper.

        The SQLiteStack is lazily created on first access. It wraps a
        *separate* SQLiteStorage pointing at the same database file,
        providing the StackProtocol interface.

        If the Entity has already been created, the stack is automatically
        attached as the active stack.

        Returns:
            SQLiteStack: The StackProtocol implementation, or None if the
            underlying storage is not SQLite-based.
        """
        if not hasattr(self, "_stack"):
            from kernle.storage.sqlite import SQLiteStorage as _SQLiteStorage

            if not isinstance(self._storage, _SQLiteStorage):
                return None

            from kernle.stack.sqlite_stack import SQLiteStack

            self._stack = SQLiteStack(
                stack_id=self.stack_id,
                db_path=self._storage.db_path,
            )
            if hasattr(self, "_entity"):
                self._entity.attach_stack(self._stack, alias="default", set_active=True)
        return self._stack

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

    def _validate_stack_id(self, stack_id: str) -> str:
        """Validate and sanitize agent ID.

        Rejects path traversal attempts before sanitizing.
        """
        if not stack_id or not stack_id.strip():
            raise ValueError("Stack ID cannot be empty")

        stripped = stack_id.strip()

        # Reject path traversal characters and patterns
        if "/" in stripped or "\\" in stripped:
            raise ValueError("Stack ID must not contain path separators")
        if stripped == "." or stripped == "..":
            raise ValueError("Stack ID must not be a relative path component")
        if ".." in stripped.split("."):
            raise ValueError("Stack ID must not contain path traversal sequences")

        # Remove potentially dangerous characters
        sanitized = "".join(c for c in stripped if c.isalnum() or c in "-_.")

        if not sanitized:
            raise ValueError("Stack ID must contain alphanumeric characters")

        if len(sanitized) > 100:
            raise ValueError("Stack ID too long (max 100 characters)")

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

    @staticmethod
    def _validate_derived_from(refs: Optional[List[str]]) -> Optional[List[str]]:
        """Validate derived_from references format.

        Accepts refs in format 'type:id' or 'context:description'.
        Filters out empty strings and validates basic structure.

        Args:
            refs: List of memory references

        Returns:
            Validated list, or None if empty
        """
        if not refs:
            return None

        valid_types = {
            "raw",
            "episode",
            "belief",
            "note",
            "value",
            "goal",
            "drive",
            "relationship",
            "context",
            "kernle",
        }
        validated = []
        for ref in refs:
            if not ref or not isinstance(ref, str):
                continue
            ref = ref.strip()
            if ":" not in ref:
                continue  # Skip malformed refs
            ref_type = ref.split(":", 1)[0]
            if ref_type not in valid_types:
                continue  # Skip unknown types
            validated.append(ref)

        return validated if validated else None

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
        epoch_id: Optional[str] = None,
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
            epoch_id: If set, filter candidates to this specific epoch before
                budget selection. NULL epoch_id memories are excluded.

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
            epoch_id=epoch_id,
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

            # Summaries - with supersession logic
            all_summaries = self._storage.list_summaries(self.stack_id)
            # Collect IDs superseded by higher-scope summaries
            superseded_ids = set()
            for s in all_summaries:
                if s.supersedes:
                    superseded_ids.update(s.supersedes)
            # Only include non-superseded summaries
            for s in all_summaries:
                if s.id not in superseded_ids:
                    scope_key = f"summary_{s.scope}"
                    candidates.append((compute_priority_score(scope_key, s), "summary", s))

            # Self-narratives - only active ones
            active_narratives = self._storage.list_self_narratives(self.stack_id, active_only=True)
            for n in active_narratives:
                candidates.append(
                    (compute_priority_score("self_narrative", n), "self_narrative", n)
                )

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
                "summaries": [],
                "self_narratives": [],
            }

            selected_indices = set()
            for idx, (priority, memory_type, record) in enumerate(candidates):
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
                elif memory_type == "summary":
                    text = f"[{record.scope}] {record.content}"
                elif memory_type == "self_narrative":
                    text = f"[{record.narrative_type}] {record.content}"
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
                    elif memory_type == "summary":
                        selected["summaries"].append(record)
                    elif memory_type == "self_narrative":
                        selected["self_narratives"].append(record)

                    remaining_budget -= tokens
                    selected_count += 1
                    selected_indices.add(idx)

                # Stop if budget exhausted
                if remaining_budget <= 0:
                    break

            # Build excluded list (preserves priority order) for memory echoes
            excluded_candidates = [c for i, c in enumerate(candidates) if i not in selected_indices]

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
                        "other_stack_id": r.entity_name,
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
                "summaries": [
                    {
                        "id": s.id,
                        "scope": s.scope,
                        "period_start": s.period_start,
                        "period_end": s.period_end,
                        "content": (
                            truncate_at_word_boundary(s.content, max_item_chars)
                            if truncate
                            else s.content
                        ),
                        "key_themes": s.key_themes,
                    }
                    for s in selected["summaries"]
                ],
                "self_narratives": [
                    {
                        "id": sn.id,
                        "narrative_type": sn.narrative_type,
                        "content": (
                            truncate_at_word_boundary(sn.content, max_item_chars)
                            if truncate
                            else sn.content
                        ),
                        "key_themes": sn.key_themes,
                        "unresolved_tensions": sn.unresolved_tensions,
                    }
                    for sn in selected["self_narratives"]
                ],
                "_meta": {
                    "budget_used": budget - remaining_budget,
                    "budget_total": budget,
                    "excluded_count": total_candidates - selected_count,
                    **_build_memory_echoes(excluded_candidates),
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
                self.stack_id,
                values=len(selected["values"]),
                beliefs=len(selected["beliefs"]),
                episodes=len(selected["episodes"]),
                checkpoint=checkpoint is not None,
            )

            # Include boot config (zero-cost, always available)
            boot = self.boot_list()
            if boot:
                batched_result["boot_config"] = boot

            # Include trust summary
            trust_summary = self._build_trust_summary()
            if trust_summary:
                batched_result["trust"] = trust_summary

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

        # Include boot config
        boot = self.boot_list()
        if boot:
            result["boot_config"] = boot

        # Log the load operation
        log_load(
            self.stack_id,
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
            "stack_id": self.stack_id,
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

        checkpoint_file = self.checkpoint_dir / f"{self.stack_id}.json"

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
                stack_id=self.stack_id,
                objective=f"[CHECKPOINT] {self._validate_string_input(task, 'task', 500)}",
                outcome=self._validate_string_input(
                    context or "Working state checkpoint", "context", 1000
                ),
                outcome_type="partial",
                lessons=pending or [],
                tags=["checkpoint", "working_state"],
                created_at=datetime.now(timezone.utc),
            )
            self._write_backend.save_episode(episode)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint to database: {e}")
            # Local save is sufficient, continue

        # Auto-export boot file on checkpoint (keeps boot.md in sync)
        try:
            self._export_boot_file()
        except Exception as e:
            logger.warning(f"Failed to export boot file on checkpoint: {e}")

        # Sync after checkpoint if enabled
        should_sync = sync if sync is not None else self._auto_sync
        if should_sync:
            sync_result = self._sync_after_checkpoint()
            checkpoint_data["_sync"] = sync_result

        # Log the checkpoint save
        log_checkpoint(
            self.stack_id,
            task=task,
            context_len=len(context or ""),
        )

        return checkpoint_data

    # Maximum checkpoint file size (10MB) to prevent DoS via large files
    MAX_CHECKPOINT_SIZE = 10 * 1024 * 1024

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{self.stack_id}.json"
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
        checkpoint_file = self.checkpoint_dir / f"{self.stack_id}.json"
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
        derived_from: Optional[List[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Record an episodic experience.

        Args:
            derived_from: List of memory IDs this episode was derived from (for linking)
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
                source_type = "external"
            elif any(x in source_lower for x in ["infer", "deduce", "conclude"]):
                source_type = "inference"

        # Build derived_from: explicit lineage + source context marker
        derived_from_value = list(derived_from) if derived_from else []
        if source:
            derived_from_value.append(f"context:{source}")

        episode = Episode(
            id=episode_id,
            stack_id=self.stack_id,
            objective=objective,
            outcome=outcome,
            outcome_type=outcome_type,
            lessons=all_lessons if all_lessons else None,
            tags=tags or ["manual"],
            created_at=datetime.now(timezone.utc),
            confidence=0.8,
            source_type=source_type,
            source_episodes=None,  # Reserved for supporting evidence (episode IDs)
            derived_from=derived_from_value if derived_from_value else None,
            # Context/scope fields
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_episode(episode)

        # Log the episode save
        log_save(
            self.stack_id,
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
        derived_from: Optional[List[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Capture a quick note (decision, insight, quote).

        Args:
            derived_from: List of memory IDs this note was derived from (for linking)
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
                source_type = "external"
            elif any(x in source_lower for x in ["infer", "deduce", "conclude"]):
                source_type = "inference"
            elif type == "quote":
                source_type = "external"

        # Build derived_from: explicit lineage + source context marker
        derived_from_value = list(derived_from) if derived_from else []
        if source:
            derived_from_value.append(f"context:{source}")

        note = Note(
            id=note_id,
            stack_id=self.stack_id,
            content=formatted,
            note_type=type,
            speaker=speaker,
            reason=reason,
            tags=tags or [],
            created_at=datetime.now(timezone.utc),
            source_type=source_type,
            source_episodes=None,  # Reserved for supporting evidence (episode IDs)
            derived_from=derived_from_value if derived_from_value else None,
            is_protected=protect,
            # Context/scope fields
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_note(note)
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

        if self._strict:
            from kernle.types import RawEntry

            raw_entry = RawEntry(
                id=str(uuid.uuid4()),
                stack_id=self.stack_id,
                blob=blob,
                source=source,
            )
            return self._write_backend.save_raw(raw_entry)
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

        raw_ref = f"raw:{raw_id}"

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
                source="raw-processing",
                derived_from=[raw_ref],
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
                source="raw-processing",
                derived_from=[raw_ref],
            )
            memory_ref = f"note:{memory_id}"

        elif as_type == "belief":
            confidence = kwargs.get("confidence", 0.7)
            belief_type = kwargs.get("type", "observation")

            memory_id = self.belief(
                statement=entry.content,
                type=belief_type,
                confidence=confidence,
                source="raw-processing",
                derived_from=[raw_ref],
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
                stack_id=self.stack_id,
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
        backend = self._write_backend
        if hasattr(backend, "save_episodes_batch"):
            return backend.save_episodes_batch(episode_objects)
        else:
            return [backend.save_episode(ep) for ep in episode_objects]

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
                stack_id=self.stack_id,
                statement=statement,
                belief_type=b_data.get("type", "fact"),
                confidence=b_data.get("confidence", 0.8),
                created_at=datetime.now(timezone.utc),
                source_type=b_data.get("source_type", "direct_experience"),
            )
            belief_objects.append(belief)

        # Use batch method if available, otherwise fall back to individual saves
        backend = self._write_backend
        if hasattr(backend, "save_beliefs_batch"):
            return backend.save_beliefs_batch(belief_objects)
        else:
            return [backend.save_belief(b) for b in belief_objects]

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
                stack_id=self.stack_id,
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
        backend = self._write_backend
        if hasattr(backend, "save_notes_batch"):
            return backend.save_notes_batch(note_objects)
        else:
            return [backend.save_note(n) for n in note_objects]

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
        lines.append(f"# Memory Dump for {self.stack_id}")
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
            "stack_id": self.stack_id,
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
    # BOOT CONFIG
    # =========================================================================

    def boot_set(self, key: str, value: str) -> None:
        """Set a boot config value."""
        self._storage.boot_set(key, value)
        self._export_boot_file()

    def boot_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a boot config value."""
        return self._storage.boot_get(key, default)

    def boot_list(self) -> Dict[str, str]:
        """List all boot config values."""
        return self._storage.boot_list()

    def boot_delete(self, key: str) -> bool:
        """Delete a boot config value."""
        result = self._storage.boot_delete(key)
        if result:
            self._export_boot_file()
        return result

    def boot_clear(self) -> int:
        """Clear all boot config."""
        count = self._storage.boot_clear()
        if count > 0:
            self._export_boot_file()
        return count

    def _export_boot_file(self) -> None:
        """Auto-export boot config to flat file.

        Writes to ~/.kernle/{stack_id}/boot.md with 0600 permissions.
        """
        config = self.boot_list()
        if not config:
            # Remove boot file if config is empty
            boot_path = get_kernle_home() / self.stack_id / "boot.md"
            if boot_path.exists():
                boot_path.unlink()
            return

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# Boot Config",
            "",
        ]
        for k, v in sorted(config.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append(f"<!-- Auto-generated by kernle at {now_str} -->")
        lines.append(
            f"<!-- Do not edit manually. Use: kernle -a {self.stack_id} boot set <key> <value> -->"
        )
        lines.append("")

        boot_path = get_kernle_home() / self.stack_id / "boot.md"
        boot_path.parent.mkdir(parents=True, exist_ok=True)
        boot_path.write_text("\n".join(lines), encoding="utf-8")

        # Set secure file permissions (owner read/write only)
        try:
            import os

            os.chmod(boot_path, 0o600)
        except OSError:
            pass  # Best effort

    def _format_boot_section(self) -> list[str]:
        """Format boot config as markdown lines for inclusion in load/export-cache."""
        config = self.boot_list()
        if not config:
            return []
        lines = ["## Boot Config"]
        for k, v in sorted(config.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        return lines

    # === Trust Layer (KEP v3) ===

    def seed_trust(self) -> int:
        """Apply seed trust templates. Returns number of assessments created."""
        created = 0
        for seed in SEED_TRUST:
            existing = self._storage.get_trust_assessment(seed["entity"])
            if existing is None:
                assessment = TrustAssessment(
                    id=str(uuid.uuid4()),
                    stack_id=self.stack_id,
                    entity=seed["entity"],
                    dimensions=seed["dimensions"],
                    authority=seed.get("authority", []),
                )
                self._storage.save_trust_assessment(assessment)
                created += 1
        return created

    def trust_list(self) -> List[Dict[str, Any]]:
        """List all trust assessments."""
        return [
            {
                "entity": a.entity,
                "dimensions": a.dimensions,
                "authority": a.authority or [],
                "evidence_count": len(a.evidence_episode_ids or []),
                "last_updated": a.last_updated.isoformat() if a.last_updated else None,
            }
            for a in self._storage.get_trust_assessments()
        ]

    def trust_show(self, entity: str) -> Optional[Dict[str, Any]]:
        """Show detailed trust assessment for an entity."""
        a = self._storage.get_trust_assessment(entity)
        if a is None:
            return None
        return {
            "id": a.id,
            "entity": a.entity,
            "dimensions": a.dimensions,
            "authority": a.authority or [],
            "evidence_episode_ids": a.evidence_episode_ids or [],
            "last_updated": a.last_updated.isoformat() if a.last_updated else None,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }

    def trust_set(
        self,
        entity: str,
        domain: str = "general",
        score: float = 0.5,
        authority: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Set or update trust for an entity in a specific domain."""
        score = max(0.0, min(1.0, score))
        existing = self._storage.get_trust_assessment(entity)
        if existing:
            existing.dimensions[domain] = {"score": score}
            if authority is not None:
                existing.authority = authority
            return self._storage.save_trust_assessment(existing)
        else:
            return self._storage.save_trust_assessment(
                TrustAssessment(
                    id=str(uuid.uuid4()),
                    stack_id=self.stack_id,
                    entity=entity,
                    dimensions={domain: {"score": score}},
                    authority=authority or [],
                )
            )

    def gate_memory_input(
        self,
        source_entity: str,
        action: str,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check whether a source entity has sufficient trust for an action.

        This is advisory -- the agent retains sovereignty over final decisions.
        """
        threshold = TRUST_THRESHOLDS.get(action)
        if threshold is None:
            return {
                "allowed": False,
                "trust_level": 0.0,
                "domain": "unknown",
                "reason": f"Unknown action type: {action}",
            }

        assessment = self._storage.get_trust_assessment(source_entity)
        if assessment is None:
            return {
                "allowed": False,
                "trust_level": 0.0,
                "domain": target or "general",
                "reason": f"No trust assessment for entity: {source_entity}",
            }

        authority = assessment.authority or []
        has_all_authority = any(a.get("scope") == "all" for a in authority)

        domain = target or "general"
        dims = assessment.dimensions or {}
        domain_trust = dims.get(domain, dims.get("general", {}))
        trust_score = domain_trust.get("score", 0.0) if isinstance(domain_trust, dict) else 0.0

        allowed = trust_score >= threshold or has_all_authority
        if allowed:
            reason = f"Trust {trust_score:.2f} >= threshold {threshold:.2f} for {action}"
        else:
            reason = f"Trust {trust_score:.2f} < threshold {threshold:.2f} for {action}"

        return {
            "allowed": allowed,
            "trust_level": trust_score,
            "domain": domain,
            "reason": reason,
        }

    def _build_trust_summary(self) -> Optional[Dict[str, Any]]:
        """Build trust summary for load() output."""
        assessments = self._storage.get_trust_assessments()
        if not assessments:
            return None
        summary = {}
        for a in assessments:
            general = a.dimensions.get("general", {})
            score = general.get("score", 0.0) if isinstance(general, dict) else 0.0
            summary[a.entity] = {
                "trust": round(score, 2),
                "authority": [auth.get("scope", "unknown") for auth in (a.authority or [])],
            }
        return summary

    # === Dynamic Trust (KEP v3 section 8.6-8.7) ===

    def compute_direct_trust(self, entity: str, domain: str = "general") -> Dict[str, Any]:
        """Compute trust score from episode history with recency weighting.

        Looks at episodes with the given source_entity, classifies outcomes
        as positive/negative, and computes a recency-weighted score.
        """
        episodes = self._storage.get_episodes_by_source_entity(entity)
        if not episodes:
            return {
                "entity": entity,
                "domain": domain,
                "score": DEFAULT_TRUST,
                "positive": 0,
                "negative": 0,
                "total": 0,
                "source": "default",
            }

        now = datetime.now(timezone.utc)
        positive_weight = 0.0
        negative_weight = 0.0

        for ep in episodes:
            # Recency weight: exponential decay, halving every 30 days
            if ep.created_at:
                days_ago = max(0, (now - ep.created_at).total_seconds() / 86400)
            else:
                days_ago = 30.0
            recency = 0.5 ** (days_ago / 30.0)

            # Classify outcome: positive if outcome_type is "success" or
            # emotional_valence > 0, negative otherwise
            is_positive = False
            if ep.outcome_type in ("success", "positive"):
                is_positive = True
            elif ep.outcome_type in ("failure", "negative"):
                is_positive = False
            elif ep.emotional_valence > 0:
                is_positive = True

            if is_positive:
                positive_weight += recency
            else:
                negative_weight += recency

        total_weight = positive_weight + negative_weight
        if total_weight == 0:
            score = DEFAULT_TRUST
        else:
            score = positive_weight / total_weight

        return {
            "entity": entity,
            "domain": domain,
            "score": round(score, 4),
            "positive": round(positive_weight, 4),
            "negative": round(negative_weight, 4),
            "total": len(episodes),
            "source": "computed",
        }

    def apply_trust_decay(self, entity: str, days_since_interaction: float) -> Dict[str, Any]:
        """Apply trust decay toward neutral (0.5) without reinforcement.

        Formula: current + (0.5 - current) * min(decay_factor, 1.0)
        where decay_factor = TRUST_DECAY_RATE * days_since_interaction
        """
        assessment = self._storage.get_trust_assessment(entity)
        if assessment is None:
            return {
                "entity": entity,
                "error": f"No trust assessment for entity: {entity}",
            }

        decay_factor = min(TRUST_DECAY_RATE * days_since_interaction, 1.0)
        updated_dims = {}

        for domain, dim_data in assessment.dimensions.items():
            if not isinstance(dim_data, dict):
                updated_dims[domain] = dim_data
                continue
            current = dim_data.get("score", DEFAULT_TRUST)
            # Self-trust has a floor
            if entity == "self":
                floor = SELF_TRUST_FLOOR
                decayed = current + (floor - current) * decay_factor
                decayed = max(floor, decayed)
            else:
                decayed = current + (DEFAULT_TRUST - current) * decay_factor
            updated_dims[domain] = {"score": round(decayed, 4)}

        assessment.dimensions = updated_dims
        self._storage.save_trust_assessment(assessment)

        return {
            "entity": entity,
            "days": days_since_interaction,
            "decay_factor": round(decay_factor, 4),
            "dimensions": updated_dims,
        }

    def compute_transitive_trust(
        self, target: str, chain: List[str], domain: str = "general"
    ) -> Dict[str, Any]:
        """Compute transitive trust through a chain of entities.

        Trust flows through entity chains with 15% decay per hop:
        trust = product of (direct_trust * depth_decay^i) for each entity in chain.
        """
        if not chain:
            return {
                "target": target,
                "chain": [],
                "domain": domain,
                "score": 0.0,
                "hops": [],
                "error": "Empty chain",
            }

        trust = 1.0
        hops = []

        for i, entity in enumerate(chain):
            assessment = self._storage.get_trust_assessment(entity)
            if assessment is None:
                direct = DEFAULT_TRUST
            else:
                dims = assessment.dimensions or {}
                domain_data = dims.get(domain, dims.get("general", {}))
                direct = (
                    domain_data.get("score", DEFAULT_TRUST)
                    if isinstance(domain_data, dict)
                    else DEFAULT_TRUST
                )

            hop_factor = direct * (TRUST_DEPTH_DECAY**i)
            trust *= hop_factor
            hops.append(
                {
                    "entity": entity,
                    "direct_trust": round(direct, 4),
                    "depth_decay": round(TRUST_DEPTH_DECAY**i, 4),
                    "cumulative": round(trust, 4),
                }
            )

        return {
            "target": target,
            "chain": chain,
            "domain": domain,
            "score": round(trust, 4),
            "hops": hops,
        }

    def compute_self_trust_floor(self) -> Dict[str, Any]:
        """Compute self-trust floor from historical accuracy.

        self_trust_floor = max(0.5, historical_accuracy_rate)
        where accuracy is based on episodes where source_entity is 'self'.
        """
        episodes = self._storage.get_episodes_by_source_entity("self")
        if not episodes:
            return {
                "floor": SELF_TRUST_FLOOR,
                "accuracy": None,
                "total_episodes": 0,
                "source": "default",
            }

        positive = sum(
            1
            for e in episodes
            if e.outcome_type in ("success", "positive") or e.emotional_valence > 0
        )
        total = len(episodes)
        accuracy = positive / total if total > 0 else 0.5
        floor = max(SELF_TRUST_FLOOR, accuracy)

        return {
            "floor": round(floor, 4),
            "accuracy": round(accuracy, 4),
            "total_episodes": total,
            "positive_episodes": positive,
            "source": "computed",
        }

    def trust_compute(self, entity: str, domain: str = "general") -> Dict[str, Any]:
        """Compute and optionally update trust for an entity from episode history.

        This is the main entry point for dynamic trust computation. It computes
        direct trust from episodes and returns the result without automatically
        updating the stored assessment (the caller can use trust_set to persist).
        """
        result = self.compute_direct_trust(entity, domain)

        # If entity is "self", also compute the floor
        if entity == "self":
            floor_result = self.compute_self_trust_floor()
            result["self_trust_floor"] = floor_result["floor"]
            result["score"] = max(result["score"], floor_result["floor"])

        return result

    def trust_chain(self, target: str, chain: List[str], domain: str = "general") -> Dict[str, Any]:
        """Compute transitive trust through a chain (CLI entry point)."""
        return self.compute_transitive_trust(target, chain, domain)

    def export_cache(
        self,
        path: Optional[str] = None,
        min_confidence: float = 0.4,
        max_beliefs: int = 50,
        include_checkpoint: bool = True,
    ) -> str:
        """Export a curated MEMORY.md cache from beliefs, values, and goals.

        This produces a read-only bootstrap cache for workspace injection.
        The output is designed to give an agent immediate context before
        `kernle load` runs. It is NOT a full memory dump — just the
        high-signal layers.

        Args:
            path: If provided, write to this file. Otherwise return string.
            min_confidence: Minimum belief confidence to include (default: 0.4)
            max_beliefs: Maximum number of beliefs to include (default: 50)
            include_checkpoint: Include last checkpoint if available

        Returns:
            The markdown content (also written to path if provided)
        """
        # Validate inputs
        min_confidence = max(0.0, min(1.0, min_confidence))
        max_beliefs = max(1, min(1000, max_beliefs))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# MEMORY.md — Long-Term Memory",
            "",
            f"<!-- AUTO-GENERATED by `kernle export-cache` at {now_str} -->",
            "<!-- Do not edit manually. Source of truth is Kernle. -->",
            f"<!-- Regenerate with: kernle -a {self.stack_id} export-cache -->",
            "",
        ]

        # Boot config (always first — this is the pre-load config)
        boot_lines = self._format_boot_section()
        if boot_lines:
            lines.extend(boot_lines)

        # Values (highest priority first)
        values = self._storage.get_values(limit=20)
        if values:
            lines.append("## Values")
            for v in sorted(
                values, key=lambda x: x.priority if x.priority is not None else 0, reverse=True
            ):
                safe_stmt = v.statement.replace("\n", " ").replace("\r", "") if v.statement else ""
                lines.append(f"- **{v.name}** (priority {v.priority or 0}): {safe_stmt}")
            lines.append("")

        # Goals (active only)
        goals = self._storage.get_goals(status="active", limit=20)
        if goals:
            lines.append("## Goals")
            for g in sorted(
                goals, key=lambda x: x.priority if x.priority is not None else 0, reverse=True
            ):
                desc = f" — {g.description}" if g.description and g.description != g.title else ""
                lines.append(f"- [{g.priority}] {g.title}{desc}")
            lines.append("")

        # Beliefs (filtered by confidence, sorted desc)
        # Fetch all beliefs since storage orders by created_at, not confidence
        beliefs = self._storage.get_beliefs(limit=max(max_beliefs * 3, 200))
        if beliefs:
            filtered = [b for b in beliefs if (b.confidence or 0) >= min_confidence]
            filtered.sort(
                key=lambda x: x.confidence if x.confidence is not None else 0.0, reverse=True
            )
            filtered = filtered[:max_beliefs]

            if filtered:
                lines.append("## Beliefs")
                for b in filtered:
                    # Strip newlines to prevent markdown structure injection
                    safe_statement = b.statement.replace("\n", " ").replace("\r", "")
                    lines.append(f"- [{b.confidence:.0%}] {safe_statement}")
                lines.append("")

        # Relationships (top by interaction count)
        relationships = self._storage.get_relationships()
        if relationships:
            # Sort by interaction count descending
            sorted_rels = sorted(
                relationships,
                key=lambda r: r.interaction_count,
                reverse=True,
            )[:10]
            if sorted_rels:
                lines.append("## Key Relationships")
                for r in sorted_rels:
                    notes_str = (
                        f" — {r.notes[:80]}..."
                        if r.notes and len(r.notes) > 80
                        else (f" — {r.notes}" if r.notes else "")
                    )
                    lines.append(f"- **{r.entity_name}** ({r.entity_type}){notes_str}")
                lines.append("")

        # Checkpoint (for session continuity)
        if include_checkpoint:
            cp = self.load_checkpoint()
            if cp:
                lines.append("## Last Checkpoint")
                lines.append(f"**Task**: {cp.get('current_task', 'unknown')}")
                if cp.get("context"):
                    lines.append(f"**Context**: {cp['context']}")
                if cp.get("pending"):
                    lines.append("**Pending**:")
                    for p in cp["pending"]:
                        lines.append(f"  - {p}")
                lines.append("")

        content = "\n".join(lines)

        if path:
            export_path = Path(path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(content, encoding="utf-8")

        return content

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
        source: Optional[str] = None,
        derived_from: Optional[List[str]] = None,
    ) -> str:
        """Add or update a belief.

        Args:
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
            source: Source context (e.g., 'raw-processing', 'consolidation', 'told by Claire')
            derived_from: List of memory refs this was derived from (format: type:id)
        """
        confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
        belief_id = str(uuid.uuid4())

        # Determine source_type from source context
        source_type = "direct_experience"
        if source:
            source_lower = source.lower()
            if any(x in source_lower for x in ["told", "said", "heard", "learned from"]):
                source_type = "external"
            elif any(x in source_lower for x in ["infer", "deduce", "conclude"]):
                source_type = "inference"
            elif "consolidat" in source_lower or "promot" in source_lower:
                source_type = "consolidation"
            elif "seed" in source_lower:
                source_type = "seed"

        # Build derived_from: explicit lineage + source context marker
        derived_from_value = list(derived_from) if derived_from else []
        if source:
            derived_from_value.append(f"context:{source}")
        derived_from_value = self._validate_derived_from(derived_from_value)

        belief = Belief(
            id=belief_id,
            stack_id=self.stack_id,
            statement=statement,
            belief_type=type,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            source_type=source_type,
            derived_from=derived_from_value,
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_belief(belief)
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
            stack_id=self.stack_id,
            name=name,
            statement=statement,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_value(value)
        return value_id

    def goal(
        self,
        title: str,
        description: Optional[str] = None,
        goal_type: str = "task",
        priority: str = "medium",
        context: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Add a goal.

        Args:
            goal_type: Type of goal (task, aspiration, commitment, exploration)
            context: Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')
            context_tags: Additional context tags for filtering
        """
        valid_goal_types = ("task", "aspiration", "commitment", "exploration")
        if goal_type not in valid_goal_types:
            raise ValueError(f"Invalid goal_type. Must be one of: {', '.join(valid_goal_types)}")

        goal_id = str(uuid.uuid4())

        # Set protection based on goal_type
        is_protected = goal_type in ("aspiration", "commitment")

        goal = Goal(
            id=goal_id,
            stack_id=self.stack_id,
            title=title,
            description=description or title,
            goal_type=goal_type,
            priority=priority,
            status="active",
            created_at=datetime.now(timezone.utc),
            is_protected=is_protected,
            context=context,
            context_tags=context_tags,
        )

        self._write_backend.save_goal(goal)

        # Protect aspiration/commitment goals from forgetting
        if is_protected:
            self._storage.protect_memory("goal", goal_id, protected=True)

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
        self._write_backend.save_goal(existing)
        return True

    # === Epoch Management ===

    def epoch_create(
        self,
        name: str,
        trigger_type: str = "declared",
        trigger_description: Optional[str] = None,
    ) -> str:
        """Create a new epoch (temporal era).

        Automatically closes any currently open epoch before creating a new one.

        Args:
            name: Name for this epoch (e.g., "onboarding", "production-v2")
            trigger_type: What triggered this epoch (declared, detected, system)
            trigger_description: Optional description of the trigger event

        Returns:
            The new epoch's ID
        """
        from kernle.storage.base import Epoch

        name = self._validate_string_input(name, "name", 200)
        if trigger_type not in ("declared", "detected", "system"):
            raise ValueError("trigger_type must be one of: declared, detected, system")

        if trigger_description is not None:
            trigger_description = self._validate_string_input(
                trigger_description, "trigger_description", 500
            )

        # Close any currently open epoch
        current = self._storage.get_current_epoch()
        if current:
            self._storage.close_epoch(current.id, summary=None)

        # Determine next epoch number
        epochs = self._storage.get_epochs(limit=1)
        next_number = (epochs[0].epoch_number + 1) if epochs else 1

        epoch = Epoch(
            id=str(uuid.uuid4()),
            stack_id=self.stack_id,
            epoch_number=next_number,
            name=name,
            started_at=datetime.now(timezone.utc),
            trigger_type=trigger_type,
            trigger_description=trigger_description,
        )

        return self._storage.save_epoch(epoch)

    def epoch_close(
        self,
        epoch_id: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> bool:
        """Close an epoch.

        Args:
            epoch_id: ID of epoch to close (defaults to current epoch)
            summary: Optional summary of the epoch

        Returns:
            True if closed, False if not found or already closed
        """
        if epoch_id is None:
            current = self._storage.get_current_epoch()
            if not current:
                return False
            epoch_id = current.id
        else:
            epoch_id = self._validate_string_input(epoch_id, "epoch_id", 100)

        if summary is not None:
            summary = self._validate_string_input(summary, "summary", 2000)

        return self._storage.close_epoch(epoch_id, summary=summary)

    def get_current_epoch(self):
        """Get the currently active epoch, if any."""
        return self._storage.get_current_epoch()

    def get_epochs(self, limit: int = 100):
        """Get all epochs, most recent first."""
        return self._storage.get_epochs(limit=limit)

    def get_epoch(self, epoch_id: str):
        """Get a specific epoch by ID."""
        epoch_id = self._validate_string_input(epoch_id, "epoch_id", 100)
        return self._storage.get_epoch(epoch_id)

    # === Summaries (Fractal Summarization) ===

    def summary_save(
        self,
        content: str,
        scope: str,
        period_start: str,
        period_end: str,
        key_themes: Optional[List[str]] = None,
        supersedes: Optional[List[str]] = None,
        epoch_id: Optional[str] = None,
    ) -> str:
        """Create or update a summary.

        Args:
            content: SI-written narrative compression
            scope: Temporal scope ('month', 'quarter', 'year', 'decade', 'epoch')
            period_start: Start of the period (ISO date)
            period_end: End of the period (ISO date)
            key_themes: Key themes/topics covered
            supersedes: IDs of lower-scope summaries this covers
            epoch_id: Associated epoch ID

        Returns:
            The summary ID
        """

        valid_scopes = ("month", "quarter", "year", "decade", "epoch")
        if scope not in valid_scopes:
            raise ValueError(f"scope must be one of: {', '.join(valid_scopes)}")

        content = self._validate_string_input(content, "content", 10000)
        period_start = self._validate_string_input(period_start, "period_start", 30)
        period_end = self._validate_string_input(period_end, "period_end", 30)

        summary = Summary(
            id=str(uuid.uuid4()),
            stack_id=self.stack_id,
            scope=scope,
            period_start=period_start,
            period_end=period_end,
            epoch_id=epoch_id,
            content=content,
            key_themes=key_themes,
            supersedes=supersedes,
            is_protected=True,
            created_at=datetime.now(timezone.utc),
        )

        return self._storage.save_summary(summary)

    def summary_get(self, summary_id: str):
        """Get a specific summary by ID."""
        summary_id = self._validate_string_input(summary_id, "summary_id", 100)
        return self._storage.get_summary(summary_id)

    def summary_list(self, scope: Optional[str] = None):
        """Get all summaries, optionally filtered by scope."""
        if scope:
            valid_scopes = ("month", "quarter", "year", "decade", "epoch")
            if scope not in valid_scopes:
                raise ValueError(f"scope must be one of: {', '.join(valid_scopes)}")
        return self._storage.list_summaries(self.stack_id, scope=scope)

    # === Self-Narrative API ===

    def narrative_save(
        self,
        content: str,
        narrative_type: str = "identity",
        key_themes: Optional[List[str]] = None,
        unresolved_tensions: Optional[List[str]] = None,
        epoch_id: Optional[str] = None,
    ) -> str:
        """Create or update a self-narrative.

        Deactivates existing active narratives of the same type first,
        then saves the new one as active.

        Args:
            content: The narrative content
            narrative_type: 'identity', 'developmental', or 'aspirational'
            key_themes: Key themes in the narrative
            unresolved_tensions: Unresolved tensions or contradictions
            epoch_id: Associated epoch ID

        Returns:
            The narrative ID
        """
        valid_types = ("identity", "developmental", "aspirational")
        if narrative_type not in valid_types:
            raise ValueError(f"narrative_type must be one of: {', '.join(valid_types)}")

        content = self._validate_string_input(content, "content", 10000)

        # Deactivate existing active narratives of the same type
        self._storage.deactivate_self_narratives(self.stack_id, narrative_type)

        narrative = SelfNarrative(
            id=str(uuid.uuid4()),
            stack_id=self.stack_id,
            content=content,
            narrative_type=narrative_type,
            epoch_id=epoch_id,
            key_themes=key_themes,
            unresolved_tensions=unresolved_tensions,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

        return self._storage.save_self_narrative(narrative)

    def narrative_get_active(self, narrative_type: str = "identity") -> Optional[SelfNarrative]:
        """Get the active self-narrative for a given type.

        Args:
            narrative_type: 'identity', 'developmental', or 'aspirational'

        Returns:
            The active narrative or None
        """
        narratives = self._storage.list_self_narratives(
            self.stack_id, narrative_type=narrative_type, active_only=True
        )
        return narratives[0] if narratives else None

    def narrative_list(
        self,
        narrative_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[SelfNarrative]:
        """List self-narratives, optionally filtered.

        Args:
            narrative_type: Filter by type (identity, developmental, aspirational)
            active_only: If True, only return active narratives

        Returns:
            List of matching narratives
        """
        if narrative_type:
            valid_types = ("identity", "developmental", "aspirational")
            if narrative_type not in valid_types:
                raise ValueError(f"narrative_type must be one of: {', '.join(valid_types)}")
        return self._storage.list_self_narratives(
            self.stack_id, narrative_type=narrative_type, active_only=active_only
        )

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

            # Filter by similarity threshold
            if result.score < similarity_threshold:
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

    def reinforce_belief(
        self,
        belief_id: str,
        evidence_source: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """Increase reinforcement count when a belief is confirmed.

        Also slightly increases confidence (with diminishing returns).

        Args:
            belief_id: ID of the belief to reinforce
            evidence_source: What triggered this reinforcement (e.g., "episode:abc123")
            reason: Human-readable reason for reinforcement

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
        room_to_grow = max(0.0, 0.99 - existing.confidence)  # Prevent negative when > 0.99
        existing.confidence = max(
            0.0, min(0.99, existing.confidence + room_to_grow * confidence_boost)
        )

        # Update confidence history with accurate old/new values
        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old": round(old_confidence, 3),
            "new": round(existing.confidence, 3),
            "reason": reason or f"Reinforced (count: {existing.times_reinforced})",
        }
        if evidence_source:
            history_entry["evidence_source"] = evidence_source

        history = existing.confidence_history or []
        history.append(history_entry)
        existing.confidence_history = history[-20:]  # Keep last 20 entries

        # Track supporting evidence in source_episodes
        if evidence_source and evidence_source.startswith("episode:"):
            existing.source_episodes = existing.source_episodes or []
            if evidence_source not in existing.source_episodes:
                existing.source_episodes.append(evidence_source)

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
            confidence: Confidence in the new belief (clamped to 0.0-1.0)
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
        confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
        new_id = str(uuid.uuid4())
        new_belief = Belief(
            id=new_id,
            stack_id=self.stack_id,
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
        self._write_backend.save_belief(new_belief)

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
                # Reinforce the belief with episode as evidence
                self.reinforce_belief(
                    belief.id,
                    evidence_source=f"episode:{episode_id}",
                    reason=f"Confirmed by episode: {episode.objective[:50]}",
                )
                result["reinforced"].append(
                    {
                        "belief_id": belief.id,
                        "statement": belief.statement,
                        "overlap": list(overlap),
                        "evidence_source": f"episode:{episode_id}",
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
                    self._write_backend.save_belief(belief)

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

        # Walk backwards to find the original belief (with cycle detection)
        back_visited = set()

        def walk_back(bid: str) -> Optional[str]:
            if bid in back_visited or bid not in belief_map:
                return None
            back_visited.add(bid)
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
            "stack_id": self.stack_id,
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
            f"# Working Memory ({self.stack_id})",
            f"_Loaded at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            "<!-- USAGE: This is your persistent memory. Resume work from 'Continue With' ",
            "section without announcing recovery. Save checkpoints with specific task ",
            "descriptions before breaks or when context pressure builds. -->",
            "",
        ]

        # Boot config - always first (pre-load config)
        if memory.get("boot_config"):
            lines.append("## Boot Config")
            for k, v in sorted(memory["boot_config"].items()):
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Checkpoint - prominently displayed at top with directive language
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
            except Exception as e:
                logger.debug(f"Failed to parse checkpoint age: {e}")

            lines.append("## Continue With")
            lines.append(f"**Current task**: {cp.get('current_task', 'unknown')}")
            if cp.get("context"):
                lines.append(f"**Context**: {cp['context']}")
            if cp.get("pending"):
                lines.append("**Next steps**:")
                for p in cp["pending"]:
                    lines.append(f"  - {p}")
            if age_warning:
                lines.append(age_warning)
            lines.append("")
            # Add directive for seamless continuation
            lines.append("_Resume this work naturally. Don't announce recovery or ask what to do._")
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

        # Footer with checkpoint guidance
        lines.append("---")
        lines.append(
            f'_Save state: `kernle -a {self.stack_id} checkpoint "<specific task>"` '
            "before breaks or context pressure._"
        )

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
            self._write_backend.save_drive(existing)
            return existing.id
        else:
            drive_id = str(uuid.uuid4())
            drive = Drive(
                id=drive_id,
                stack_id=self.stack_id,
                drive_type=drive_type,
                intensity=max(0.0, min(1.0, intensity)),
                focus_areas=focus_areas or [],
                created_at=now,
                updated_at=now,
                context=context,
                context_tags=context_tags,
            )
            self._write_backend.save_drive(drive)
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
            self._write_backend.save_drive(existing)
            return True
        return False

    # =========================================================================
    # RELATIONAL MEMORY (Models of Other Entities)
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
                "other_stack_id": r.entity_name,  # backwards compat
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
        other_stack_id: str,
        trust_level: Optional[float] = None,
        notes: Optional[str] = None,
        interaction_type: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> str:
        """Update relationship model for another entity.

        Args:
            other_stack_id: Name/identifier of the other entity
            trust_level: Trust level 0.0-1.0 (converted to sentiment -1 to 1)
            notes: Notes about the relationship
            interaction_type: Type of interaction being logged
            entity_type: Type of entity (person, agent, organization, system)
        """
        # Check existing
        existing = self._storage.get_relationship(other_stack_id)

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
            self._write_backend.save_relationship(existing)
            return existing.id
        else:
            rel_id = str(uuid.uuid4())
            relationship = Relationship(
                id=rel_id,
                stack_id=self.stack_id,
                entity_name=other_stack_id,
                entity_type=entity_type or "person",
                relationship_type=interaction_type or "interaction",
                notes=notes,
                sentiment=((trust_level * 2) - 1) if trust_level is not None else 0.0,
                interaction_count=1,
                last_interaction=now,
                created_at=now,
            )
            self._write_backend.save_relationship(relationship)
            return rel_id

    # =========================================================================
    # RELATIONSHIP HISTORY
    # =========================================================================

    def get_relationship_history(
        self,
        entity_name: str,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get the history of changes for a relationship.

        Args:
            entity_name: Name of the entity
            event_type: Filter by event type (interaction, trust_change, type_change, note)
            limit: Maximum entries to return

        Returns:
            List of history entries as dicts
        """
        entries = self._storage.get_relationship_history(
            entity_name, event_type=event_type, limit=limit
        )
        return [
            {
                "id": e.id,
                "relationship_id": e.relationship_id,
                "entity_name": e.entity_name,
                "event_type": e.event_type,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "episode_id": e.episode_id,
                "notes": e.notes,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ]

    # =========================================================================
    # ENTITY MODELS (Mental Models of Other Entities)
    # =========================================================================

    def add_entity_model(
        self,
        entity_name: str,
        model_type: str,
        observation: str,
        confidence: float = 0.7,
        source_episodes: Optional[List[str]] = None,
    ) -> str:
        """Add a mental model observation about an entity.

        Args:
            entity_name: The entity this model is about
            model_type: Type of model (behavioral, preference, capability)
            observation: The observation/model content
            confidence: Confidence in the observation (0.0-1.0)
            source_episodes: Episode IDs supporting this observation

        Returns:
            The entity model ID
        """
        from kernle.storage.base import EntityModel

        entity_name = self._validate_string_input(entity_name, "entity_name", 200)
        observation = self._validate_string_input(observation, "observation", 2000)

        if model_type not in ("behavioral", "preference", "capability"):
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                "Must be one of: behavioral, preference, capability"
            )

        confidence = max(0.0, min(1.0, confidence))

        model_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        model = EntityModel(
            id=model_id,
            stack_id=self.stack_id,
            entity_name=entity_name,
            model_type=model_type,
            observation=observation,
            confidence=confidence,
            source_episodes=source_episodes,
            created_at=now,
            updated_at=now,
            subject_ids=[entity_name],  # Auto-populate
        )

        self._storage.save_entity_model(model)
        return model_id

    def get_entity_models(
        self,
        entity_name: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get entity models, optionally filtered.

        Args:
            entity_name: Filter by entity name
            model_type: Filter by model type
            limit: Maximum models to return

        Returns:
            List of entity models as dicts
        """
        models = self._storage.get_entity_models(
            entity_name=entity_name, model_type=model_type, limit=limit
        )
        return [
            {
                "id": m.id,
                "entity_name": m.entity_name,
                "model_type": m.model_type,
                "observation": m.observation,
                "confidence": m.confidence,
                "source_episodes": m.source_episodes,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m in models
        ]

    def get_entity_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific entity model by ID.

        Args:
            model_id: ID of the entity model

        Returns:
            Entity model as dict or None if not found
        """
        m = self._storage.get_entity_model(model_id)
        if not m:
            return None
        return {
            "id": m.id,
            "entity_name": m.entity_name,
            "model_type": m.model_type,
            "observation": m.observation,
            "confidence": m.confidence,
            "source_episodes": m.source_episodes,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        }

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
            stack_id=self.stack_id,
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

    def consolidate_epoch_closing(self, epoch_id: str) -> Dict[str, Any]:
        """Orchestrate full epoch-closing consolidation.

        A deeper consolidation sequence triggered when closing an epoch.
        Produces scaffold prompts for six steps of reflection.

        Args:
            epoch_id: ID of the epoch being closed

        Returns:
            Structured scaffold with all six epoch-closing steps
        """
        from kernle.features.consolidation import build_epoch_closing_scaffold

        epoch_id = self._validate_string_input(epoch_id, "epoch_id", 100)
        return build_epoch_closing_scaffold(self, epoch_id)

    def consolidate(self, min_episodes: int = 3) -> Dict[str, Any]:
        """Deprecated: use promote() instead.

        Run memory consolidation (legacy). Analyzes recent episodes to
        extract patterns, lessons, and beliefs.

        Args:
            min_episodes: Minimum episodes required to consolidate

        Returns:
            Consolidation results
        """
        import warnings

        warnings.warn(
            "Kernle.consolidate() is deprecated, use Kernle.promote() instead",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def promote(
        self,
        auto: bool = False,
        min_occurrences: int = 2,
        min_episodes: int = 3,
        confidence: float = 0.7,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Promote recurring patterns from episodes into beliefs.

        Scans recent episodes for recurring lessons and patterns. In auto
        mode, creates beliefs directly. In default mode, returns suggestions
        for the agent to review.

        This is the episodes → beliefs promotion step. The agent controls
        when and whether promotion happens (SI autonomy principle).

        Args:
            auto: If True, create beliefs automatically. If False, return
                suggestions only (default: False).
            min_occurrences: Minimum times a lesson must appear across
                episodes to be considered for promotion (default: 2).
            min_episodes: Minimum episodes required to run promotion
                (default: 3).
            confidence: Initial confidence for auto-created beliefs
                (default: 0.7). Clamped to 0.1-0.95.
            limit: Maximum episodes to scan (default: 50).

        Returns:
            Dict with promotion results:
            - episodes_scanned: number of episodes analyzed
            - patterns_found: number of recurring patterns detected
            - suggestions: list of {lesson, count, source_episodes, promoted, belief_id}
            - beliefs_created: number of beliefs created (auto mode only)
        """

        confidence = max(0.1, min(0.95, confidence))
        limit = max(1, min(200, limit))
        min_occurrences = max(2, min_occurrences)

        episodes = self._storage.get_episodes(limit=limit)
        episodes = [ep for ep in episodes if ep.strength > 0.0]

        if len(episodes) < min_episodes:
            return {
                "episodes_scanned": len(episodes),
                "patterns_found": 0,
                "suggestions": [],
                "beliefs_created": 0,
                "message": f"Need at least {min_episodes} episodes (found {len(episodes)})",
            }

        # Map lessons to their source episodes
        lesson_sources: Dict[str, List[str]] = {}
        for ep in episodes:
            if ep.lessons:
                for lesson in ep.lessons:
                    # Normalize: strip whitespace, lowercase for matching
                    normalized = lesson.strip()
                    if not normalized:
                        continue
                    if normalized not in lesson_sources:
                        lesson_sources[normalized] = []
                    lesson_sources[normalized].append(ep.id)

        # Find recurring patterns
        recurring = [
            (lesson, ep_ids)
            for lesson, ep_ids in lesson_sources.items()
            if len(ep_ids) >= min_occurrences
        ]
        # Sort by frequency (most common first)
        recurring.sort(key=lambda x: -len(x[1]))

        # Check existing beliefs to avoid duplicates
        existing_beliefs = self._storage.get_beliefs(limit=200)
        existing_statements = {b.statement.strip().lower() for b in existing_beliefs if b.is_active}

        suggestions = []
        beliefs_created = 0

        for lesson, source_ep_ids in recurring:
            # Skip if a very similar belief already exists
            if lesson.strip().lower() in existing_statements:
                suggestions.append(
                    {
                        "lesson": lesson,
                        "count": len(source_ep_ids),
                        "source_episodes": source_ep_ids[:5],
                        "promoted": False,
                        "skipped": "similar_belief_exists",
                    }
                )
                continue

            suggestion = {
                "lesson": lesson,
                "count": len(source_ep_ids),
                "source_episodes": source_ep_ids[:5],
                "promoted": False,
                "belief_id": None,
            }

            if auto:
                # Create belief with proper provenance
                derived_from = [f"episode:{eid}" for eid in source_ep_ids[:10]]
                belief_id = self.belief(
                    statement=lesson,
                    type="pattern",
                    confidence=confidence,
                    source="promotion",
                    derived_from=derived_from,
                )
                suggestion["promoted"] = True
                suggestion["belief_id"] = belief_id
                beliefs_created += 1
                # Add to existing set to prevent duplicates within same run
                existing_statements.add(lesson.strip().lower())

            suggestions.append(suggestion)

        return {
            "episodes_scanned": len(episodes),
            "patterns_found": len(recurring),
            "suggestions": suggestions[:20],  # Cap output
            "beliefs_created": beliefs_created,
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
