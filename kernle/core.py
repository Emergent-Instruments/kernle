"""
Kernle Core - Stratified memory for synthetic intelligences.

This module provides the main Kernle class, which is the primary interface
for memory operations. It uses the storage abstraction layer to support
both local SQLite storage and cloud Supabase storage.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, TYPE_CHECKING

# Import storage abstraction
from kernle.storage import (
    get_storage,
    Episode, Belief, Value, Goal, Note, Drive, Relationship
)

if TYPE_CHECKING:
    from kernle.storage import Storage as StorageProtocol

# Set up logging
logger = logging.getLogger(__name__)


class Kernle:
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
        self.agent_id = self._validate_agent_id(agent_id or os.environ.get("KERNLE_AGENT_ID", "default"))
        self.checkpoint_dir = self._validate_checkpoint_dir(checkpoint_dir or Path.home() / ".kernle" / "checkpoints")
        
        # Store credentials for backwards compatibility
        self._supabase_url = supabase_url or os.environ.get("KERNLE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        self._supabase_key = supabase_key or os.environ.get("KERNLE_SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
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
        
        logger.debug(f"Kernle initialized with storage: {type(self._storage).__name__}")
    
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
                resolved_path.is_relative_to(home_path) or
                resolved_path.is_relative_to(tmp_path) or
                resolved_path.is_relative_to(system_temp)
            )
            
            # Also allow /var/folders on macOS (where tempfile creates dirs)
            if not is_safe:
                try:
                    var_folders = Path("/var/folders").resolve()
                    private_var_folders = Path("/private/var/folders").resolve()
                    is_safe = (
                        resolved_path.is_relative_to(var_folders) or
                        resolved_path.is_relative_to(private_var_folders)
                    )
                except (OSError, ValueError):
                    pass
            
            if not is_safe:
                raise ValueError("Checkpoint directory must be within user home or temp directory")
                
            return resolved_path
            
        except (OSError, ValueError) as e:
            logger.error(f"Invalid checkpoint directory: {e}")
            raise ValueError(f"Invalid checkpoint directory: {e}")
    
    def _validate_string_input(self, value: str, field_name: str, max_length: int = 1000) -> str:
        """Validate and sanitize string inputs."""
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        
        if len(value) > max_length:
            raise ValueError(f"{field_name} too long (max {max_length} characters)")
            
        # Basic sanitization - remove null bytes and control characters
        sanitized = value.replace('\x00', '').replace('\r\n', '\n')
        
        return sanitized
    
    # =========================================================================
    # LOAD
    # =========================================================================
    
    def load(self, budget: int = 6000) -> Dict[str, Any]:
        """Load working memory context.
        
        Uses batched loading when available to optimize database queries,
        reducing 9 sequential queries to a single batched operation.
        """
        # Try batched loading first (available in SQLiteStorage)
        batched = self._storage.load_all(
            values_limit=10,
            beliefs_limit=20,
            goals_limit=10,
            goals_status="active",
            episodes_limit=20,  # For both lessons and recent_work
            notes_limit=5,
        )
        
        if batched is not None:
            # Use batched results - format for API compatibility
            episodes = batched.get("episodes", [])
            
            # Extract lessons from episodes
            lessons = []
            for ep in episodes:
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
                for e in episodes
                if not e.tags or "checkpoint" not in e.tags
            ][:5]
            
            return {
                "checkpoint": self.load_checkpoint(),
                "values": [
                    {
                        "id": v.id,
                        "name": v.name,
                        "statement": v.statement,
                        "priority": v.priority,
                        "value_type": "core_value",
                    }
                    for v in batched.get("values", [])
                ],
                "beliefs": [
                    {
                        "id": b.id,
                        "statement": b.statement,
                        "belief_type": b.belief_type,
                        "confidence": b.confidence,
                    }
                    for b in sorted(batched.get("beliefs", []), key=lambda x: x.confidence, reverse=True)
                ],
                "goals": [
                    {
                        "id": g.id,
                        "title": g.title,
                        "description": g.description,
                        "priority": g.priority,
                        "status": g.status,
                    }
                    for g in batched.get("goals", [])
                ],
                "drives": [
                    {
                        "id": d.id,
                        "drive_type": d.drive_type,
                        "intensity": d.intensity,
                        "last_satisfied_at": d.updated_at.isoformat() if d.updated_at else None,
                        "focus_areas": d.focus_areas,
                    }
                    for d in batched.get("drives", [])
                ],
                "lessons": lessons,
                "recent_work": recent_work,
                "recent_notes": [
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
                    for n in batched.get("notes", [])
                ],
                "relationships": [
                    {
                        "other_agent_id": r.entity_name,
                        "entity_name": r.entity_name,
                        "trust_level": (r.sentiment + 1) / 2,
                        "sentiment": r.sentiment,
                        "interaction_count": r.interaction_count,
                        "last_interaction": r.last_interaction.isoformat() if r.last_interaction else None,
                        "notes": r.notes,
                    }
                    for r in sorted(
                        batched.get("relationships", []),
                        key=lambda x: x.last_interaction or datetime.min.replace(tzinfo=timezone.utc),
                        reverse=True
                    )
                ],
            }
        
        # Fallback to individual queries (for backends without load_all)
        return {
            "checkpoint": self.load_checkpoint(),
            "values": self.load_values(),
            "beliefs": self.load_beliefs(),
            "goals": self.load_goals(),
            "drives": self.load_drives(),
            "lessons": self.load_lessons(),
            "recent_work": self.load_recent_work(),
            "recent_notes": self.load_recent_notes(),
            "relationships": self.load_relationships(),
        }
    
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
        goals = self._storage.get_goals(
            status=None if status == "all" else status,
            limit=limit
        )
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
        non_checkpoint = [
            e for e in episodes 
            if not e.tags or "checkpoint" not in e.tags
        ]
        
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
    ) -> dict:
        """Save current working state."""
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
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except (json.JSONDecodeError, OSError, PermissionError) as e:
                logger.warning(f"Could not load existing checkpoint: {e}")
                existing = []
        
        existing.append(checkpoint_data)
        existing = existing[-10:]  # Keep last 10
        
        try:
            with open(checkpoint_file, "w", encoding='utf-8') as f:
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
                outcome=self._validate_string_input(context or "Working state checkpoint", 'context', 1000),
                outcome_type="partial",
                lessons=pending or [],
                tags=["checkpoint", "working_state"],
                created_at=datetime.now(timezone.utc),
            )
            self._storage.save_episode(episode)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint to database: {e}")
            # Local save is sufficient, continue
        
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
                    logger.error(f"Checkpoint file too large ({file_size} bytes, max {self.MAX_CHECKPOINT_SIZE})")
                    raise ValueError(f"Checkpoint file too large ({file_size} bytes)")
                
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
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
    ) -> str:
        """Record an episodic experience."""
        # Validate inputs
        objective = self._validate_string_input(objective, "objective", 1000)
        outcome = self._validate_string_input(outcome, "outcome", 1000)
        
        if lessons:
            lessons = [self._validate_string_input(l, "lesson", 500) for l in lessons]
        if repeat:
            repeat = [self._validate_string_input(r, "repeat pattern", 500) for r in repeat]
        if avoid:
            avoid = [self._validate_string_input(a, "avoid pattern", 500) for a in avoid]
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]
        
        episode_id = str(uuid.uuid4())
        
        outcome_type = "success" if outcome.lower() in ("success", "done", "completed") else (
            "failure" if outcome.lower() in ("failure", "failed", "error") else "partial"
        )
        
        # Combine lessons with repeat/avoid patterns
        all_lessons = lessons or []
        if repeat:
            all_lessons.extend([f"Repeat: {r}" for r in repeat])
        if avoid:
            all_lessons.extend([f"Avoid: {a}" for a in avoid])
        
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
        )
        
        self._storage.save_episode(episode)
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
            # Update outcome_type based on new outcome
            outcome_type = "success" if outcome.lower() in ("success", "done", "completed") else (
                "failure" if outcome.lower() in ("failure", "failed", "error") else "partial"
            )
            existing.outcome_type = outcome_type
        
        if lessons:
            lessons = [self._validate_string_input(l, "lesson", 500) for l in lessons]
            # Merge with existing lessons
            existing_lessons = existing.lessons or []
            existing.lessons = list(set(existing_lessons + lessons))
        
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]
            # Merge with existing tags
            existing_tags = existing.tags or []
            existing.tags = list(set(existing_tags + tags))
        
        existing.version += 1
        self._storage.save_episode(existing)
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
    ) -> str:
        """Capture a quick note (decision, insight, quote)."""
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
        
        note = Note(
            id=note_id,
            agent_id=self.agent_id,
            content=formatted,
            note_type=type,
            speaker=speaker,
            reason=reason,
            tags=tags or [],
            created_at=datetime.now(timezone.utc),
        )
        
        self._storage.save_note(note)
        return note_id
    
    # =========================================================================
    # RAW ENTRIES (Zero-friction capture)
    # =========================================================================
    
    def raw(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        source: str = "manual",
    ) -> str:
        """Quick capture of unstructured thought for later processing.
        
        Args:
            content: Free-form text to capture
            tags: Optional quick tags for categorization
            source: Source of the entry (manual, auto_capture, voice, etc.)
            
        Returns:
            Raw entry ID
        """
        content = self._validate_string_input(content, "content", 5000)
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]
        
        return self._storage.save_raw(content, source, tags)
    
    def list_raw(self, processed: Optional[bool] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List raw entries, optionally filtered by processed state.
        
        Args:
            processed: Filter by processed state (None = all, True = processed, False = unprocessed)
            limit: Maximum entries to return
            
        Returns:
            List of raw entry dicts
        """
        entries = self._storage.list_raw(processed=processed, limit=limit)
        return [
            {
                "id": e.id,
                "content": e.content,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "source": e.source,
                "processed": e.processed,
                "processed_into": e.processed_into,
                "tags": e.tags,
            }
            for e in entries
        ]
    
    def get_raw(self, raw_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific raw entry by ID.
        
        Args:
            raw_id: ID of the raw entry
            
        Returns:
            Raw entry dict or None if not found
        """
        entry = self._storage.get_raw(raw_id)
        if entry:
            return {
                "id": entry.id,
                "content": entry.content,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "source": entry.source,
                "processed": entry.processed,
                "processed_into": entry.processed_into,
                "tags": entry.tags,
            }
        return None
    
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
            objective = kwargs.get("objective") or entry.content[:100]
            outcome = kwargs.get("outcome", "completed")
            lessons = kwargs.get("lessons") or ([entry.content] if len(entry.content) > 100 else None)
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
                status_icon = "✓" if g.status == "completed" else "○" if g.status == "active" else "⏸"
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
                outcome_icon = "✓" if e.outcome_type == "success" else "✗" if e.outcome_type == "failure" else "○"
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
                for r in raw_entries:
                    date_str = r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "unknown"
                    status = "✓" if r.processed else "○"
                    lines.append(f"### {status} {date_str}")
                    lines.append(r.content)
                    if r.tags:
                        lines.append(f"Tags: {', '.join(r.tags)}")
                    if r.processed and r.processed_into:
                        lines.append(f"Processed into: {', '.join(r.processed_into)}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _dump_json(self, include_raw: bool) -> str:
        """Export memory as JSON."""
        data = {
            "agent_id": self.agent_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "values": [
                {
                    "id": v.id,
                    "name": v.name,
                    "statement": v.statement,
                    "priority": v.priority,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                }
                for v in self._storage.get_values(limit=100)
            ],
            "beliefs": [
                {
                    "id": b.id,
                    "statement": b.statement,
                    "type": b.belief_type,
                    "confidence": b.confidence,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
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
                    "created_at": g.created_at.isoformat() if g.created_at else None,
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
                    "created_at": e.created_at.isoformat() if e.created_at else None,
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
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in self._storage.get_notes(limit=100)
            ],
            "drives": [
                {
                    "id": d.id,
                    "type": d.drive_type,
                    "intensity": d.intensity,
                    "focus_areas": d.focus_areas,
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
                }
                for r in self._storage.get_relationships()
            ],
        }
        
        if include_raw:
            data["raw_entries"] = [
                {
                    "id": r.id,
                    "content": r.content,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "source": r.source,
                    "processed": r.processed,
                    "processed_into": r.processed_into,
                    "tags": r.tags,
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
    ) -> str:
        """Add or update a belief."""
        belief_id = str(uuid.uuid4())
        
        belief = Belief(
            id=belief_id,
            agent_id=self.agent_id,
            statement=statement,
            belief_type=type,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
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
    ) -> str:
        """Add or affirm a value."""
        value_id = str(uuid.uuid4())
        
        value = Value(
            id=value_id,
            agent_id=self.agent_id,
            name=name,
            statement=statement,
            priority=priority,
            created_at=datetime.now(timezone.utc),
        )
        
        self._storage.save_value(value)
        return value_id
    
    def goal(
        self,
        title: str,
        description: Optional[str] = None,
        priority: str = "medium",
    ) -> str:
        """Add a goal."""
        goal_id = str(uuid.uuid4())
        
        goal = Goal(
            id=goal_id,
            agent_id=self.agent_id,
            title=title,
            description=description or title,
            priority=priority,
            status="active",
            created_at=datetime.now(timezone.utc),
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
        
        existing.version += 1
        self._storage.save_belief(existing)
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
            limit=limit * 2,  # Get more to filter
            record_types=["belief"]
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
                ("never", "always"), ("should not", "should"), ("cannot", "can"),
                ("don't", "do"), ("avoid", "prefer"), ("reject", "accept"),
                ("false", "true"), ("dislike", "like"), ("hate", "love"),
                ("wrong", "right"), ("bad", "good"),
            ]
            
            for neg, pos in negation_pairs:
                if ((neg in stmt_lower and pos in belief_stmt_lower) or
                    (pos in stmt_lower and neg in belief_stmt_lower)):
                    # Check word overlap for topic relevance
                    words_stmt = set(stmt_lower.split()) - {"i", "the", "a", "an", "to", "and", "or", "is", "are", "that", "this"}
                    words_belief = set(belief_stmt_lower.split()) - {"i", "the", "a", "an", "to", "and", "or", "is", "are", "that", "this"}
                    overlap = len(words_stmt & words_belief)
                    
                    if overlap >= 2:
                        contradiction_type = "direct_negation"
                        confidence = min(0.5 + overlap * 0.1 + result.score * 0.2, 0.95)
                        explanation = f"Negation conflict: '{neg}' vs '{pos}' with {overlap} overlapping terms"
                        break
            
            # Comparative opposition (more/less, better/worse, etc.)
            if not contradiction_type:
                comparative_pairs = [
                    ("more", "less"), ("better", "worse"), ("faster", "slower"),
                    ("higher", "lower"), ("greater", "lesser"), ("stronger", "weaker"),
                    ("easier", "harder"), ("simpler", "more complex"), ("safer", "riskier"),
                    ("cheaper", "more expensive"), ("larger", "smaller"), ("longer", "shorter"),
                    ("increase", "decrease"), ("improve", "worsen"), ("enhance", "diminish"),
                ]
                for comp_a, comp_b in comparative_pairs:
                    if ((comp_a in stmt_lower and comp_b in belief_stmt_lower) or
                        (comp_b in stmt_lower and comp_a in belief_stmt_lower)):
                        # Check word overlap for topic relevance (need high overlap for comparatives)
                        words_stmt = set(stmt_lower.split()) - {"i", "the", "a", "an", "to", "and", "or", "is", "are", "that", "this", "than", comp_a, comp_b}
                        words_belief = set(belief_stmt_lower.split()) - {"i", "the", "a", "an", "to", "and", "or", "is", "are", "that", "this", "than", comp_a, comp_b}
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
                    ("prefer", "avoid"), ("like", "dislike"), ("enjoy", "hate"),
                    ("favor", "oppose"), ("support", "reject"), ("want", "don't want"),
                ]
                for pref, anti in preference_pairs:
                    if ((pref in stmt_lower and anti in belief_stmt_lower) or
                        (anti in stmt_lower and pref in belief_stmt_lower)):
                        words_stmt = set(stmt_lower.split()) - {"i", "the", "a", "an", "to", "and", "or"}
                        words_belief = set(belief_stmt_lower.split()) - {"i", "the", "a", "an", "to", "and", "or"}
                        overlap = len(words_stmt & words_belief)
                        
                        if overlap >= 2:
                            contradiction_type = "preference_conflict"
                            confidence = min(0.4 + overlap * 0.1 + result.score * 0.2, 0.85)
                            explanation = f"Preference conflict: '{pref}' vs '{anti}'"
                            break
            
            if contradiction_type:
                contradictions.append({
                    "belief_id": belief.id,
                    "statement": belief.statement,
                    "confidence": belief.confidence,
                    "times_reinforced": belief.times_reinforced,
                    "is_active": belief.is_active,
                    "contradiction_type": contradiction_type,
                    "contradiction_confidence": round(confidence, 2),
                    "explanation": explanation,
                    "semantic_similarity": round(result.score, 2),
                })
        
        # Sort by contradiction confidence
        contradictions.sort(key=lambda x: x["contradiction_confidence"], reverse=True)
        return contradictions[:limit]
    
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
        
        # Increment reinforcement count
        existing.times_reinforced += 1
        
        # Slightly increase confidence (diminishing returns)
        # Each reinforcement adds less confidence, capped at 0.99
        confidence_boost = 0.05 * (1.0 / (1 + existing.times_reinforced * 0.1))
        room_to_grow = 0.99 - existing.confidence
        existing.confidence = min(0.99, existing.confidence + room_to_grow * confidence_boost)
        
        # Update confidence history
        history = existing.confidence_history or []
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old": round(existing.confidence - confidence_boost, 3),
            "new": round(existing.confidence, 3),
            "reason": f"Reinforced (count: {existing.times_reinforced})"
        })
        existing.confidence_history = history[-20:]  # Keep last 20 entries
        
        existing.last_verified = datetime.now(timezone.utc)
        existing.verification_count += 1
        existing.version += 1
        
        self._storage.save_belief(existing)
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
            confidence_history=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old": 0.0,
                "new": confidence,
                "reason": reason or f"Superseded belief {old_id[:8]}"
            }],
        )
        self._storage.save_belief(new_belief)
        
        # Update the old belief
        old_belief.superseded_by = new_id
        old_belief.is_active = False
        
        # Add to confidence history
        history = old_belief.confidence_history or []
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old": old_belief.confidence,
            "new": old_belief.confidence,
            "reason": f"Superseded by belief {new_id[:8]}: {reason or 'no reason given'}"
        })
        old_belief.confidence_history = history[-20:]
        old_belief.version += 1
        self._storage.save_belief(old_belief)
        
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
            return {"error": "Episode not found", "reinforced": [], "contradicted": [], "suggested_new": []}
        
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
            belief_words = set(belief_stmt_lower.split()) - {"i", "the", "a", "an", "to", "and", "or", "is", "are", "should", "can"}
            evidence_words = set(evidence_lower.split()) - {"i", "the", "a", "an", "to", "and", "or", "is", "are", "should", "can"}
            overlap = belief_words & evidence_words
            
            if len(overlap) < 2:
                continue  # Not related enough
            
            # Determine if evidence supports or contradicts
            is_supporting = False
            is_contradicting = False
            
            if episode.outcome_type == "success":
                # Success supports "should" beliefs about what worked
                if any(word in belief_stmt_lower for word in ["should", "prefer", "good", "important", "effective"]):
                    is_supporting = True
                # Success contradicts "avoid" beliefs about what worked
                elif any(word in belief_stmt_lower for word in ["avoid", "never", "don't", "bad"]):
                    is_contradicting = True
            
            elif episode.outcome_type == "failure":
                # Failure contradicts "should" beliefs about what failed
                if any(word in belief_stmt_lower for word in ["should", "prefer", "good", "important", "effective"]):
                    is_contradicting = True
                # Failure supports "avoid" beliefs
                elif any(word in belief_stmt_lower for word in ["avoid", "never", "don't", "bad"]):
                    is_supporting = True
            
            if is_supporting:
                # Reinforce the belief
                self.reinforce_belief(belief.id)
                result["reinforced"].append({
                    "belief_id": belief.id,
                    "statement": belief.statement,
                    "overlap": list(overlap),
                })
            
            elif is_contradicting:
                # Flag as potentially contradicted
                result["contradicted"].append({
                    "belief_id": belief.id,
                    "statement": belief.statement,
                    "overlap": list(overlap),
                    "evidence": evidence_text[:200],
                })
        
        # Suggest new beliefs from lessons
        if episode.lessons:
            for lesson in episode.lessons:
                # Check if a similar belief already exists
                existing = self._storage.find_belief(lesson)
                if not existing:
                    # Check for similar beliefs via search
                    similar = self._storage.search(lesson, limit=3, record_types=["belief"])
                    if not any(r.score > 0.9 for r in similar):
                        result["suggested_new"].append({
                            "statement": lesson,
                            "source_episode": episode_id,
                            "suggested_confidence": 0.7 if episode.outcome_type == "success" else 0.6,
                        })
        
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
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search across episodes, notes, and beliefs."""
        results = self._storage.search(query, limit=limit)
        
        formatted = []
        for r in results:
            record = r.record
            record_type = r.record_type
            
            if record_type == "episode":
                formatted.append({
                    "type": "episode",
                    "title": record.objective[:60] if record.objective else "",
                    "content": record.outcome,
                    "lessons": (record.lessons or [])[:2],
                    "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
                })
            elif record_type == "note":
                formatted.append({
                    "type": record.note_type or "note",
                    "title": record.content[:60] if record.content else "",
                    "content": record.content,
                    "tags": record.tags or [],
                    "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
                })
            elif record_type == "belief":
                formatted.append({
                    "type": "belief",
                    "title": record.statement[:60] if record.statement else "",
                    "content": record.statement,
                    "confidence": record.confidence,
                    "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
                })
        
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
        
        lines = [f"# Working Memory ({self.agent_id})", f"_Loaded at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_", ""]
        
        # Checkpoint
        if memory.get("checkpoint"):
            cp = memory["checkpoint"]
            lines.append("## Working State")
            lines.append(f"**Task**: {cp.get('current_task', 'unknown')}")
            if cp.get("pending"):
                lines.append("**Pending**:")
                for p in cp["pending"]:
                    lines.append(f"  - {p}")
            if cp.get("context"):
                lines.append(f"**Context**: {cp['context']}")
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
    ) -> str:
        """Set or update a drive."""
        if drive_type not in self.DRIVE_TYPES:
            raise ValueError(f"Invalid drive type. Must be one of: {self.DRIVE_TYPES}")
        
        # Check if drive exists
        existing = self._storage.get_drive(drive_type)
        
        now = datetime.now(timezone.utc)
        
        if existing:
            existing.intensity = max(0.0, min(1.0, intensity))
            existing.focus_areas = focus_areas or []
            existing.updated_at = now
            existing.version += 1
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
            reverse=True
        )
        
        return [
            {
                "other_agent_id": r.entity_name,  # backwards compat
                "entity_name": r.entity_name,
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
    ) -> str:
        """Update relationship model for another agent."""
        # Check existing
        existing = self._storage.get_relationship(other_agent_id)
        
        now = datetime.now(timezone.utc)
        
        if existing:
            if trust_level is not None:
                # Convert trust_level (0-1) to sentiment (-1 to 1)
                existing.sentiment = max(-1.0, min(1.0, (trust_level * 2) - 1))
            if notes:
                existing.notes = notes
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
                entity_type="agent",
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
                normalized_steps.append({
                    "action": step,
                    "details": None,
                    "adaptations": None,
                })
            elif isinstance(step, dict):
                normalized_steps.append({
                    "action": step.get("action", f"Step {i + 1}"),
                    "details": step.get("details"),
                    "adaptations": step.get("adaptations"),
                })
            else:
                raise ValueError(f"Invalid step format at index {i}")
        
        # Validate optional lists
        if triggers:
            triggers = [self._validate_string_input(t, "trigger", 500) for t in triggers]
        if failure_modes:
            failure_modes = [self._validate_string_input(f, "failure_mode", 500) for f in failure_modes]
        if recovery_steps:
            recovery_steps = [self._validate_string_input(r, "recovery_step", 500) for r in recovery_steps]
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
    
    def load_playbooks(self, limit: int = 10, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
            start = (now.replace(hour=0, minute=0, second=0, microsecond=0) - 
                    timedelta(days=1))
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
    
    # Emotional signal patterns for automatic tagging
    EMOTION_PATTERNS = {
        # Positive emotions (high valence)
        "joy": {
            "keywords": ["happy", "joy", "delighted", "wonderful", "amazing", "fantastic", "love it", "excited"],
            "valence": 0.8,
            "arousal": 0.6,
        },
        "satisfaction": {
            "keywords": ["satisfied", "pleased", "content", "glad", "good", "nice", "well done"],
            "valence": 0.6,
            "arousal": 0.3,
        },
        "excitement": {
            "keywords": ["excited", "thrilled", "pumped", "can't wait", "awesome", "incredible"],
            "valence": 0.7,
            "arousal": 0.9,
        },
        "curiosity": {
            "keywords": ["curious", "interesting", "fascinating", "wonder", "intriguing", "want to know"],
            "valence": 0.3,
            "arousal": 0.5,
        },
        "pride": {
            "keywords": ["proud", "accomplished", "achieved", "nailed it", "crushed it"],
            "valence": 0.7,
            "arousal": 0.5,
        },
        "gratitude": {
            "keywords": ["grateful", "thankful", "appreciate", "thanks so much", "means a lot"],
            "valence": 0.7,
            "arousal": 0.3,
        },
        # Negative emotions (low valence)
        "frustration": {
            "keywords": ["frustrated", "annoying", "irritated", "ugh", "argh", "why won't", "doesn't work"],
            "valence": -0.6,
            "arousal": 0.7,
        },
        "disappointment": {
            "keywords": ["disappointed", "let down", "expected better", "unfortunate", "bummer"],
            "valence": -0.5,
            "arousal": 0.3,
        },
        "anxiety": {
            "keywords": ["worried", "anxious", "nervous", "concerned", "stressed", "overwhelmed"],
            "valence": -0.4,
            "arousal": 0.7,
        },
        "confusion": {
            "keywords": ["confused", "don't understand", "unclear", "lost", "what do you mean"],
            "valence": -0.2,
            "arousal": 0.4,
        },
        "sadness": {
            "keywords": ["sad", "unhappy", "depressed", "down", "terrible", "awful"],
            "valence": -0.7,
            "arousal": 0.2,
        },
        "anger": {
            "keywords": ["angry", "furious", "mad", "hate", "outraged", "unacceptable"],
            "valence": -0.8,
            "arousal": 0.9,
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
                    signals.append({
                        "signal": signal_name,
                        "type": pattern["type"],
                        "weight": pattern["weight"],
                    })
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
    
    # =========================================================================
    # EMOTIONAL MEMORY
    # =========================================================================
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """Detect emotional signals in text.
        
        Args:
            text: Text to analyze for emotional content
            
        Returns:
            dict with:
            - valence: float (-1.0 to 1.0)
            - arousal: float (0.0 to 1.0)
            - tags: list[str] - detected emotion labels
            - confidence: float - how confident we are
        """
        text_lower = text.lower()
        detected_emotions = []
        valence_sum = 0.0
        arousal_sum = 0.0
        
        for emotion_name, pattern in self.EMOTION_PATTERNS.items():
            for keyword in pattern["keywords"]:
                if keyword in text_lower:
                    detected_emotions.append(emotion_name)
                    valence_sum += pattern["valence"]
                    arousal_sum += pattern["arousal"]
                    break  # One match per emotion is enough
        
        if detected_emotions:
            # Average the emotional values
            count = len(detected_emotions)
            avg_valence = max(-1.0, min(1.0, valence_sum / count))
            avg_arousal = max(0.0, min(1.0, arousal_sum / count))
            confidence = min(1.0, 0.3 + (count * 0.2))  # More matches = higher confidence
        else:
            avg_valence = 0.0
            avg_arousal = 0.0
            confidence = 0.0
        
        return {
            "valence": avg_valence,
            "arousal": avg_arousal,
            "tags": detected_emotions,
            "confidence": confidence,
        }
    
    def add_emotional_association(
        self,
        episode_id: str,
        valence: float,
        arousal: float,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Add or update emotional associations for an episode.
        
        Args:
            episode_id: The episode to update
            valence: Emotional valence (-1.0 negative to 1.0 positive)
            arousal: Emotional arousal (0.0 calm to 1.0 intense)
            tags: Emotional labels (e.g., ["joy", "excitement"])
            
        Returns:
            True if successful, False otherwise
        """
        # Clamp values
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        
        try:
            return self._storage.update_episode_emotion(
                episode_id=episode_id,
                valence=valence,
                arousal=arousal,
                tags=tags,
            )
        except Exception as e:
            logger.warning(f"Failed to add emotional association: {e}")
            return False
    
    def get_emotional_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get emotional pattern summary over time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            dict with:
            - average_valence: float
            - average_arousal: float
            - dominant_emotions: list[str]
            - emotional_trajectory: list - trend over time
            - episode_count: int - number of emotional episodes
        """
        # Get episodes with emotional data
        emotional_episodes = self._storage.get_emotional_episodes(days=days, limit=100)
        
        if not emotional_episodes:
            return {
                "average_valence": 0.0,
                "average_arousal": 0.0,
                "dominant_emotions": [],
                "emotional_trajectory": [],
                "episode_count": 0,
            }
        
        # Calculate averages
        valences = [ep.emotional_valence or 0.0 for ep in emotional_episodes]
        arousals = [ep.emotional_arousal or 0.0 for ep in emotional_episodes]
        
        avg_valence = sum(valences) / len(valences)
        avg_arousal = sum(arousals) / len(arousals)
        
        # Count emotion tags
        from collections import Counter
        all_tags = []
        for ep in emotional_episodes:
            tags = ep.emotional_tags or []
            all_tags.extend(tags)
        
        tag_counts = Counter(all_tags)
        dominant_emotions = [tag for tag, count in tag_counts.most_common(5)]
        
        # Build trajectory (grouped by day)
        from collections import defaultdict
        daily_data = defaultdict(lambda: {"valences": [], "arousals": []})
        
        for ep in emotional_episodes:
            if ep.created_at:
                date_str = ep.created_at.strftime("%Y-%m-%d")
                daily_data[date_str]["valences"].append(ep.emotional_valence or 0.0)
                daily_data[date_str]["arousals"].append(ep.emotional_arousal or 0.0)
        
        trajectory = []
        for date_str in sorted(daily_data.keys()):
            data = daily_data[date_str]
            trajectory.append({
                "date": date_str,
                "valence": sum(data["valences"]) / len(data["valences"]),
                "arousal": sum(data["arousals"]) / len(data["arousals"]),
            })
        
        return {
            "average_valence": round(avg_valence, 3),
            "average_arousal": round(avg_arousal, 3),
            "dominant_emotions": dominant_emotions,
            "emotional_trajectory": trajectory,
            "episode_count": len(emotional_episodes),
        }
    
    def search_by_emotion(
        self,
        valence_range: Optional[tuple] = None,
        arousal_range: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find episodes matching emotional criteria.
        
        Args:
            valence_range: (min, max) valence filter, e.g. (0.5, 1.0) for positive
            arousal_range: (min, max) arousal filter, e.g. (0.7, 1.0) for high arousal
            tags: Emotional tags to match (matches any)
            limit: Maximum results
            
        Returns:
            List of matching episodes as dicts
        """
        episodes = self._storage.search_by_emotion(
            valence_range=valence_range,
            arousal_range=arousal_range,
            tags=tags,
            limit=limit,
        )
        
        return [
            {
                "id": ep.id,
                "objective": ep.objective,
                "outcome_type": ep.outcome_type,
                "outcome_description": ep.outcome,
                "emotional_valence": ep.emotional_valence,
                "emotional_arousal": ep.emotional_arousal,
                "emotional_tags": ep.emotional_tags,
                "created_at": ep.created_at.isoformat() if ep.created_at else "",
            }
            for ep in episodes
        ]
    
    def episode_with_emotion(
        self,
        objective: str,
        outcome: str,
        lessons: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        emotional_tags: Optional[List[str]] = None,
        auto_detect: bool = True,
    ) -> str:
        """Record an episode with emotional tagging.
        
        Args:
            objective: What was the goal?
            outcome: What happened?
            lessons: Lessons learned
            tags: General tags
            valence: Emotional valence (-1.0 to 1.0), auto-detected if None
            arousal: Emotional arousal (0.0 to 1.0), auto-detected if None
            emotional_tags: Emotion labels, auto-detected if None
            auto_detect: If True and no emotion args given, detect from text
            
        Returns:
            Episode ID
        """
        # Validate inputs
        objective = self._validate_string_input(objective, "objective", 1000)
        outcome = self._validate_string_input(outcome, "outcome", 1000)
        
        if lessons:
            lessons = [self._validate_string_input(l, "lesson", 500) for l in lessons]
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]
        if emotional_tags:
            emotional_tags = [self._validate_string_input(e, "emotion_tag", 50) for e in emotional_tags]
        
        # Auto-detect emotions if not provided
        if auto_detect and valence is None and arousal is None and not emotional_tags:
            detection = self.detect_emotion(f"{objective} {outcome}")
            if detection["confidence"] > 0:
                valence = detection["valence"]
                arousal = detection["arousal"]
                emotional_tags = detection["tags"]
        
        episode_id = str(uuid.uuid4())
        
        outcome_type = "success" if outcome.lower() in ("success", "done", "completed") else (
            "failure" if outcome.lower() in ("failure", "failed", "error") else "partial"
        )
        
        episode = Episode(
            id=episode_id,
            agent_id=self.agent_id,
            objective=objective,
            outcome=outcome,
            outcome_type=outcome_type,
            lessons=lessons,
            tags=tags or ["manual"],
            created_at=datetime.now(timezone.utc),
            emotional_valence=valence or 0.0,
            emotional_arousal=arousal or 0.0,
            emotional_tags=emotional_tags,
            confidence=0.8,
        )
        
        self._storage.save_episode(episode)
        return episode_id
    
    def get_mood_relevant_memories(
        self,
        current_valence: float,
        current_arousal: float,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to current emotional state.
        
        Useful for mood-congruent recall - we tend to remember
        experiences that match our current emotional state.
        
        Args:
            current_valence: Current valence (-1.0 to 1.0)
            current_arousal: Current arousal (0.0 to 1.0)
            limit: Maximum results
            
        Returns:
            List of mood-relevant episodes
        """
        # Get episodes with matching emotional range
        valence_range = (
            max(-1.0, current_valence - 0.3),
            min(1.0, current_valence + 0.3)
        )
        arousal_range = (
            max(0.0, current_arousal - 0.3),
            min(1.0, current_arousal + 0.3)
        )
        
        return self.search_by_emotion(
            valence_range=valence_range,
            arousal_range=arousal_range,
            limit=limit,
        )
    
    # =========================================================================
    # META-MEMORY
    # =========================================================================
    
    def get_memory_confidence(self, memory_type: str, memory_id: str) -> float:
        """Get confidence score for a memory.
        
        Args:
            memory_type: Type of memory (episode, belief, value, goal, note)
            memory_id: ID of the memory
            
        Returns:
            Confidence score (0.0-1.0), or -1.0 if not found
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if record:
            return getattr(record, 'confidence', 0.8)
        return -1.0
    
    def verify_memory(
        self,
        memory_type: str,
        memory_id: str,
        evidence: Optional[str] = None,
    ) -> bool:
        """Verify a memory, increasing its confidence.
        
        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            evidence: Optional supporting evidence
            
        Returns:
            True if verified, False if memory not found
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return False
        
        old_confidence = getattr(record, 'confidence', 0.8)
        new_confidence = min(1.0, old_confidence + 0.1)
        
        # Track confidence change
        confidence_history = getattr(record, 'confidence_history', None) or []
        confidence_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old": old_confidence,
            "new": new_confidence,
            "reason": evidence or "verification",
        })
        
        return self._storage.update_memory_meta(
            memory_type=memory_type,
            memory_id=memory_id,
            confidence=new_confidence,
            verification_count=(getattr(record, 'verification_count', 0) or 0) + 1,
            last_verified=datetime.now(timezone.utc),
            confidence_history=confidence_history,
        )
    
    def get_memory_lineage(self, memory_type: str, memory_id: str) -> Dict[str, Any]:
        """Get provenance chain for a memory.
        
        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            
        Returns:
            Lineage information including source and derivations
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return {"error": f"Memory {memory_type}:{memory_id} not found"}
        
        return {
            "id": memory_id,
            "type": memory_type,
            "source_type": getattr(record, 'source_type', 'unknown'),
            "source_episodes": getattr(record, 'source_episodes', None),
            "derived_from": getattr(record, 'derived_from', None),
            "current_confidence": getattr(record, 'confidence', None),
            "verification_count": getattr(record, 'verification_count', 0),
            "last_verified": (
                getattr(record, 'last_verified').isoformat()
                if getattr(record, 'last_verified', None) else None
            ),
            "confidence_history": getattr(record, 'confidence_history', None),
        }
    
    def get_uncertain_memories(
        self,
        threshold: float = 0.5,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get memories with confidence below threshold.
        
        Args:
            threshold: Confidence threshold
            limit: Maximum results
            
        Returns:
            List of low-confidence memories
        """
        results = self._storage.get_memories_by_confidence(
            threshold=threshold,
            below=True,
            limit=limit,
        )
        
        formatted = []
        for r in results:
            record = r.record
            formatted.append({
                "id": record.id,
                "type": r.record_type,
                "confidence": getattr(record, 'confidence', None),
                "summary": self._get_memory_summary(r.record_type, record),
                "created_at": (
                    record.created_at.strftime("%Y-%m-%d")
                    if getattr(record, 'created_at', None) else "unknown"
                ),
            })
        
        return formatted
    
    def _get_memory_summary(self, memory_type: str, record: Any) -> str:
        """Get a brief summary of a memory record."""
        if memory_type == "episode":
            return record.objective[:60] if record.objective else ""
        elif memory_type == "belief":
            return record.statement[:60] if record.statement else ""
        elif memory_type == "value":
            return f"{record.name}: {record.statement[:40]}" if record.name else ""
        elif memory_type == "goal":
            return record.title[:60] if record.title else ""
        elif memory_type == "note":
            return record.content[:60] if record.content else ""
        return str(record)[:60]
    
    def propagate_confidence(
        self,
        memory_type: str,
        memory_id: str,
    ) -> Dict[str, Any]:
        """Propagate confidence changes to derived memories.
        
        When a source memory's confidence changes, this updates
        derived memories accordingly.
        
        Args:
            memory_type: Type of source memory
            memory_id: ID of source memory
            
        Returns:
            Result dict with number of updated memories
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return {"error": f"Memory {memory_type}:{memory_id} not found"}
        
        source_confidence = getattr(record, 'confidence', 0.8)
        source_ref = f"{memory_type}:{memory_id}"
        
        # Find memories derived from this one
        # This is a simplified implementation - would need to query all tables
        updated = 0
        
        # For now, return the source confidence info
        return {
            "source_confidence": source_confidence,
            "source_ref": source_ref,
            "updated": updated,
        }
    
    def set_memory_source(
        self,
        memory_type: str,
        memory_id: str,
        source_type: str,
        source_episodes: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
    ) -> bool:
        """Set provenance information for a memory.
        
        Args:
            memory_type: Type of memory
            memory_id: ID of memory
            source_type: Source type (direct_experience, inference, told_by_agent, consolidation)
            source_episodes: List of supporting episode IDs
            derived_from: List of memory refs this was derived from (format: type:id)
            
        Returns:
            True if updated, False if memory not found
        """
        return self._storage.update_memory_meta(
            memory_type=memory_type,
            memory_id=memory_id,
            source_type=source_type,
            source_episodes=source_episodes,
            derived_from=derived_from,
        )
    
    # =========================================================================
    # CONTROLLED FORGETTING
    # =========================================================================
    
    # Default half-life for salience decay (in days)
    DEFAULT_HALF_LIFE = 30.0
    
    def calculate_salience(self, memory_type: str, memory_id: str) -> float:
        """Calculate current salience score for a memory.
        
        Salience formula:
        salience = (confidence × reinforcement_weight) / (age_factor + 1)
        where:
            reinforcement_weight = log(times_accessed + 1)
            age_factor = days_since_last_access / half_life
        
        Args:
            memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship)
            memory_id: ID of the memory
            
        Returns:
            Salience score (0.0-1.0 typical range, can exceed 1.0 for very active memories)
            Returns -1.0 if memory not found
        """
        import math
        
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return -1.0
        
        confidence = getattr(record, 'confidence', 0.8)
        times_accessed = getattr(record, 'times_accessed', 0) or 0
        last_accessed = getattr(record, 'last_accessed', None)
        created_at = getattr(record, 'created_at', None)
        
        # Use last_accessed if available, otherwise created_at
        reference_time = last_accessed or created_at
        
        now = datetime.now(timezone.utc)
        if reference_time:
            days_since = (now - reference_time).total_seconds() / 86400
        else:
            days_since = 365  # Very old if unknown
        
        age_factor = days_since / self.DEFAULT_HALF_LIFE
        reinforcement_weight = math.log(times_accessed + 1)
        
        # Salience calculation with minimum base value
        salience = (confidence * (reinforcement_weight + 0.1)) / (age_factor + 1)
        
        return salience
    
    def get_forgetting_candidates(
        self,
        threshold: float = 0.3,
        limit: int = 20,
        memory_types: Optional[List[str]] = None,
    ) -> List[dict]:
        """Find low-salience memories eligible for forgetting.
        
        Returns memories that are:
        - Not protected (is_protected = False)
        - Not already forgotten (is_forgotten = False)
        - Have salience below the threshold
        
        Args:
            threshold: Salience threshold (memories below this are candidates)
            limit: Maximum candidates to return
            memory_types: Filter by memory type (default: episode, belief, goal, note, relationship)
            
        Returns:
            List of dicts with memory info and salience score, sorted by salience (lowest first)
        """
        results = self._storage.get_forgetting_candidates(
            memory_types=memory_types,
            limit=limit * 2,  # Get more to filter by threshold
        )
        
        candidates = []
        for r in results:
            if r.score < threshold:
                record = r.record
                candidates.append({
                    "type": r.record_type,
                    "id": record.id,
                    "salience": round(r.score, 4),
                    "summary": self._get_memory_summary(r.record_type, record),
                    "confidence": getattr(record, 'confidence', 0.8),
                    "times_accessed": getattr(record, 'times_accessed', 0),
                    "last_accessed": (
                        getattr(record, 'last_accessed').isoformat()
                        if getattr(record, 'last_accessed', None) else None
                    ),
                    "created_at": (
                        getattr(record, 'created_at').strftime("%Y-%m-%d")
                        if getattr(record, 'created_at', None) else "unknown"
                    ),
                })
        
        return candidates[:limit]
    
    def forget(
        self,
        memory_type: str,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Tombstone a memory (mark forgotten, don't delete).
        
        Forgotten memories are not deleted - they can be recovered later.
        Protected memories cannot be forgotten.
        
        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            reason: Optional reason for forgetting (for audit trail)
            
        Returns:
            True if forgotten, False if not found, already forgotten, or protected
        """
        return self._storage.forget_memory(memory_type, memory_id, reason)
    
    def recover(self, memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory.
        
        Restores a tombstoned memory back to active status.
        
        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            
        Returns:
            True if recovered, False if not found or not forgotten
        """
        return self._storage.recover_memory(memory_type, memory_id)
    
    def protect(self, memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark memory as protected from forgetting.
        
        Protected memories never decay and cannot be forgotten.
        Use this for core identity memories.
        
        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            protected: True to protect, False to unprotect
            
        Returns:
            True if updated, False if memory not found
        """
        return self._storage.protect_memory(memory_type, memory_id, protected)
    
    def run_forgetting_cycle(
        self,
        threshold: float = 0.3,
        limit: int = 10,
        dry_run: bool = True,
    ) -> dict:
        """Review and optionally forget low-salience memories.
        
        This is the main forgetting maintenance function. It:
        1. Finds memories below the salience threshold
        2. Optionally forgets them (if dry_run=False)
        3. Returns a report of what was/would be forgotten
        
        Args:
            threshold: Salience threshold (memories below this are candidates)
            limit: Maximum memories to forget in one cycle
            dry_run: If True, only report what would be forgotten (don't actually forget)
            
        Returns:
            Report dict with:
            - candidates: List of forgetting candidates
            - forgotten: Number actually forgotten (0 if dry_run)
            - protected: Number skipped because protected
            - dry_run: Whether this was a dry run
        """
        candidates = self.get_forgetting_candidates(threshold=threshold, limit=limit)
        
        report = {
            "threshold": threshold,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "forgotten": 0,
            "protected": 0,
            "dry_run": dry_run,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if not dry_run:
            for candidate in candidates:
                success = self.forget(
                    memory_type=candidate["type"],
                    memory_id=candidate["id"],
                    reason=f"Low salience ({candidate['salience']:.4f}) in forgetting cycle",
                )
                if success:
                    report["forgotten"] += 1
                else:
                    # Likely protected or already forgotten
                    report["protected"] += 1
        
        return report
    
    def get_forgotten_memories(
        self,
        memory_types: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Get all forgotten (tombstoned) memories.
        
        These can be recovered using the recover() method.
        
        Args:
            memory_types: Filter by memory type
            limit: Maximum results
            
        Returns:
            List of forgotten memory info dicts
        """
        results = self._storage.get_forgotten_memories(
            memory_types=memory_types,
            limit=limit,
        )
        
        forgotten = []
        for r in results:
            record = r.record
            forgotten.append({
                "type": r.record_type,
                "id": record.id,
                "summary": self._get_memory_summary(r.record_type, record),
                "forgotten_at": (
                    getattr(record, 'forgotten_at').isoformat()
                    if getattr(record, 'forgotten_at', None) else None
                ),
                "forgotten_reason": getattr(record, 'forgotten_reason', None),
                "created_at": (
                    getattr(record, 'created_at').strftime("%Y-%m-%d")
                    if getattr(record, 'created_at', None) else "unknown"
                ),
            })
        
        return forgotten
    
    def record_access(self, memory_type: str, memory_id: str) -> bool:
        """Record that a memory was accessed (for salience tracking).
        
        Call this when retrieving a memory to update its access statistics.
        This helps the salience calculation favor frequently-accessed memories.
        
        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            
        Returns:
            True if updated, False if memory not found
        """
        return self._storage.record_access(memory_type, memory_id)
    
    # =========================================================================
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
        common_lessons = [l for l, c in lesson_counts.items() if c >= 2]
        
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
            narrative_parts.append(f"I value {top_value.name.lower()} highly: {top_value.statement}")
        
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
            "active_goals": [
                {"title": g.title, "priority": g.priority}
                for g in goals[:5]
            ],
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
        if values:
            value_count_score = min(1.0, len(values) / 5)
            avg_priority = sum(v.priority / 100 for v in values) / len(values)
            value_score = (value_count_score * 0.6 + avg_priority * 0.4) * 0.20
        else:
            value_score = 0.0
        
        # Beliefs (20%): quantity × quality (confidence)
        # Ideal: 5-10 beliefs with high confidence
        if beliefs:
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
        if episodes:
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
        
        total = (value_score + belief_score + goal_score + 
                 episode_score + drive_score + relationship_score)
        
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
    
    # =========================================================================
    # ANXIETY TRACKING
    # =========================================================================
    
    # Anxiety level thresholds and colors
    ANXIETY_LEVELS = {
        (0, 30): ("🟢", "Calm"),
        (31, 50): ("🟡", "Aware"),
        (51, 70): ("🟠", "Elevated"),
        (71, 85): ("🔴", "High"),
        (86, 100): ("⚫", "Critical"),
    }
    
    # Dimension weights for composite score
    ANXIETY_WEIGHTS = {
        "context_pressure": 0.35,
        "unsaved_work": 0.25,
        "consolidation_debt": 0.20,
        "identity_coherence": 0.10,
        "memory_uncertainty": 0.10,
    }
    
    def _get_anxiety_level(self, score: int) -> tuple:
        """Get emoji and label for an anxiety score."""
        for (low, high), (emoji, label) in self.ANXIETY_LEVELS.items():
            if low <= score <= high:
                return emoji, label
        return "⚫", "Critical"
    
    def _get_checkpoint_age_minutes(self) -> Optional[int]:
        """Get minutes since last checkpoint."""
        cp = self.load_checkpoint()
        if not cp or "timestamp" not in cp:
            return None
        
        try:
            cp_time = datetime.fromisoformat(cp["timestamp"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - cp_time
            return int(delta.total_seconds() / 60)
        except (ValueError, TypeError):
            return None
    
    def _get_unreflected_episodes(self) -> List[Any]:
        """Get episodes without lessons (unreflected experiences)."""
        episodes = self._storage.get_episodes(limit=100)
        # Filter out checkpoints and episodes that already have lessons
        unreflected = [
            e for e in episodes
            if (not e.tags or "checkpoint" not in e.tags) and not e.lessons
        ]
        return unreflected
    
    def _get_low_confidence_beliefs(self, threshold: float = 0.5) -> List[Any]:
        """Get beliefs with confidence below threshold."""
        beliefs = self._storage.get_beliefs(limit=100)
        return [b for b in beliefs if b.confidence < threshold]
    
    def get_anxiety_report(
        self,
        context_tokens: Optional[int] = None,
        context_limit: int = 200000,
        detailed: bool = False,
    ) -> dict:
        """Calculate memory anxiety across 5 dimensions.
        
        This measures the functional anxiety of a synthetic intelligence
        facing finite context and potential memory loss.
        
        Args:
            context_tokens: Current context token usage (if known)
            context_limit: Maximum context window size
            detailed: Include additional details in the report
        
        Returns:
            dict with:
            - overall_score: Composite anxiety score (0-100)
            - overall_level: Human-readable level (Calm, Aware, etc.)
            - overall_emoji: Level indicator emoji
            - dimensions: Per-dimension breakdown
            - recommendations: If detailed=True, includes recommended actions
        """
        dimensions = {}
        
        # 1. Context Pressure (0-100%)
        # How full is the context window?
        if context_tokens is not None:
            context_pressure_pct = min(100, int((context_tokens / context_limit) * 100))
            context_detail = f"{context_tokens:,}/{context_limit:,} tokens"
        else:
            # Estimate based on session age (checkpoint age as proxy)
            checkpoint_age = self._get_checkpoint_age_minutes()
            if checkpoint_age is not None:
                # Rough heuristic: assume ~500 tokens/minute burn rate
                estimated_tokens = checkpoint_age * 500
                context_pressure_pct = min(100, int((estimated_tokens / context_limit) * 100))
                context_detail = f"~{estimated_tokens:,} tokens (estimated from {checkpoint_age}min session)"
            else:
                # No checkpoint = fresh session, low pressure
                context_pressure_pct = 10
                context_detail = "No checkpoint (fresh session)"
        
        # Map pressure to anxiety (non-linear: gets worse above 70%)
        if context_pressure_pct < 50:
            context_score = int(context_pressure_pct * 0.6)
        elif context_pressure_pct < 70:
            context_score = int(30 + (context_pressure_pct - 50) * 1.5)
        elif context_pressure_pct < 85:
            context_score = int(60 + (context_pressure_pct - 70) * 2)
        else:
            context_score = int(90 + (context_pressure_pct - 85) * 0.67)
        
        dimensions["context_pressure"] = {
            "score": min(100, context_score),
            "raw_value": context_pressure_pct,
            "detail": context_detail,
            "emoji": self._get_anxiety_level(context_score)[0],
        }
        
        # 2. Unsaved Work (0-100%)
        # Time since last checkpoint and estimated decisions made
        checkpoint_age = self._get_checkpoint_age_minutes()
        if checkpoint_age is None:
            unsaved_score = 50  # Unknown = moderate concern
            unsaved_detail = "No checkpoint found"
        elif checkpoint_age < 15:
            unsaved_score = int(checkpoint_age * 2)
            unsaved_detail = f"{checkpoint_age} min since checkpoint"
        elif checkpoint_age < 60:
            unsaved_score = int(30 + (checkpoint_age - 15) * 1.1)
            unsaved_detail = f"{checkpoint_age} min since checkpoint"
        else:
            unsaved_score = min(100, int(80 + (checkpoint_age - 60) * 0.33))
            unsaved_detail = f"{checkpoint_age} min since checkpoint (STALE)"
        
        dimensions["unsaved_work"] = {
            "score": min(100, unsaved_score),
            "raw_value": checkpoint_age,
            "detail": unsaved_detail,
            "emoji": self._get_anxiety_level(unsaved_score)[0],
        }
        
        # 3. Consolidation Debt (0-100%)
        # Episodes without lessons = unreflected experiences
        unreflected = self._get_unreflected_episodes()
        unreflected_count = len(unreflected)
        
        if unreflected_count <= 3:
            consolidation_score = unreflected_count * 7
            consolidation_detail = f"{unreflected_count} unreflected episodes"
        elif unreflected_count <= 7:
            consolidation_score = int(21 + (unreflected_count - 3) * 10)
            consolidation_detail = f"{unreflected_count} unreflected episodes (building up)"
        elif unreflected_count <= 15:
            consolidation_score = int(61 + (unreflected_count - 7) * 4)
            consolidation_detail = f"{unreflected_count} unreflected episodes (significant backlog)"
        else:
            consolidation_score = min(100, int(93 + (unreflected_count - 15) * 0.5))
            consolidation_detail = f"{unreflected_count} unreflected episodes (URGENT)"
        
        dimensions["consolidation_debt"] = {
            "score": min(100, consolidation_score),
            "raw_value": unreflected_count,
            "detail": consolidation_detail,
            "emoji": self._get_anxiety_level(consolidation_score)[0],
        }
        
        # 4. Identity Coherence (inverted - high coherence = low anxiety)
        identity_confidence = self.get_identity_confidence()
        # Invert: 100% confidence = 0% anxiety
        identity_anxiety = int((1.0 - identity_confidence) * 100)
        
        if identity_confidence >= 0.8:
            identity_detail = f"{identity_confidence:.0%} identity confidence (strong)"
        elif identity_confidence >= 0.5:
            identity_detail = f"{identity_confidence:.0%} identity confidence (developing)"
        else:
            identity_detail = f"{identity_confidence:.0%} identity confidence (WEAK)"
        
        dimensions["identity_coherence"] = {
            "score": identity_anxiety,
            "raw_value": identity_confidence,
            "detail": identity_detail,
            "emoji": self._get_anxiety_level(identity_anxiety)[0],
        }
        
        # 5. Memory Uncertainty (0-100%)
        # Count of low-confidence beliefs
        low_conf_beliefs = self._get_low_confidence_beliefs(0.5)
        total_beliefs = len(self._storage.get_beliefs(limit=100))
        
        if total_beliefs == 0:
            uncertainty_pct = 0
            uncertainty_detail = "No beliefs yet"
        else:
            uncertainty_pct = int((len(low_conf_beliefs) / total_beliefs) * 100)
            uncertainty_detail = f"{len(low_conf_beliefs)}/{total_beliefs} beliefs below 50% confidence"
        
        # Convert to anxiety score (more low-confidence = more anxiety)
        if len(low_conf_beliefs) <= 2:
            uncertainty_score = len(low_conf_beliefs) * 15
            uncertainty_detail = f"{len(low_conf_beliefs)} low-confidence beliefs"
        elif len(low_conf_beliefs) <= 5:
            uncertainty_score = int(30 + (len(low_conf_beliefs) - 2) * 15)
            uncertainty_detail = f"{len(low_conf_beliefs)} low-confidence beliefs (some uncertainty)"
        else:
            uncertainty_score = min(100, int(75 + (len(low_conf_beliefs) - 5) * 5))
            uncertainty_detail = f"{len(low_conf_beliefs)} low-confidence beliefs (HIGH uncertainty)"
        
        dimensions["memory_uncertainty"] = {
            "score": min(100, uncertainty_score),
            "raw_value": len(low_conf_beliefs),
            "detail": uncertainty_detail,
            "emoji": self._get_anxiety_level(uncertainty_score)[0],
        }
        
        # Calculate composite score (weighted average)
        overall_score = 0
        for dim_name, weight in self.ANXIETY_WEIGHTS.items():
            overall_score += dimensions[dim_name]["score"] * weight
        overall_score = int(overall_score)
        
        overall_emoji, overall_level = self._get_anxiety_level(overall_score)
        
        report = {
            "overall_score": overall_score,
            "overall_level": overall_level,
            "overall_emoji": overall_emoji,
            "dimensions": dimensions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
        }
        
        if detailed:
            report["recommendations"] = self.get_recommended_actions(overall_score)
            report["context_limit"] = context_limit
            report["context_tokens"] = context_tokens
        
        return report
    
    def get_recommended_actions(self, anxiety_level: int) -> List[Dict[str, Any]]:
        """Return prioritized actions based on anxiety level.
        
        Actions reference actual kernle commands/methods for execution.
        
        Args:
            anxiety_level: Overall anxiety score (0-100)
        
        Returns:
            List of action dicts with priority, description, command, and method
        """
        actions = []
        
        # Always recommend based on current state
        checkpoint_age = self._get_checkpoint_age_minutes()
        unreflected = self._get_unreflected_episodes()
        low_conf = self._get_low_confidence_beliefs(0.5)
        identity_conf = self.get_identity_confidence()
        
        # Calm (0-30): Continue normal work
        if anxiety_level <= 30:
            if len(unreflected) > 0:
                actions.append({
                    "priority": "low",
                    "description": f"Reflect on {len(unreflected)} recent experiences when convenient",
                    "command": "kernle consolidate",
                    "method": "consolidate",
                })
            return actions
        
        # Aware (31-50): Checkpoint and note major decisions
        if anxiety_level <= 50:
            if checkpoint_age is None or checkpoint_age > 15:
                actions.append({
                    "priority": "medium",
                    "description": "Checkpoint current work state",
                    "command": "kernle checkpoint save '<task>'",
                    "method": "checkpoint",
                })
            if len(unreflected) > 3:
                actions.append({
                    "priority": "medium",
                    "description": f"Process {len(unreflected)} unreflected episodes",
                    "command": "kernle consolidate",
                    "method": "consolidate",
                })
            return actions
        
        # Elevated (51-70): Full checkpoint, consolidate, verify
        if anxiety_level <= 70:
            actions.append({
                "priority": "high",
                "description": "Full checkpoint with context",
                "command": "kernle checkpoint save '<task>' --context '<summary>'",
                "method": "checkpoint",
            })
            if len(unreflected) > 0:
                actions.append({
                    "priority": "high",
                    "description": f"Consolidate {len(unreflected)} unreflected episodes",
                    "command": "kernle consolidate",
                    "method": "consolidate",
                })
            if identity_conf < 0.7:
                actions.append({
                    "priority": "medium",
                    "description": "Run identity synthesis to strengthen coherence",
                    "command": "kernle identity show",
                    "method": "synthesize_identity",
                })
            if len(low_conf) > 0:
                actions.append({
                    "priority": "low",
                    "description": f"Review {len(low_conf)} uncertain beliefs",
                    "command": "kernle meta uncertain",
                    "method": "get_uncertain_memories",
                })
            return actions
        
        # High (71-85): Priority memory work
        if anxiety_level <= 85:
            actions.append({
                "priority": "critical",
                "description": "PRIORITY: Run full consolidation",
                "command": "kernle consolidate",
                "method": "consolidate",
            })
            actions.append({
                "priority": "critical",
                "description": "Full checkpoint with session summary",
                "command": "kernle checkpoint save '<task>' --context '<full summary>'",
                "method": "checkpoint",
            })
            actions.append({
                "priority": "high",
                "description": "Run identity synthesis and save",
                "command": "kernle identity show",
                "method": "synthesize_identity",
            })
            sync_status = self.get_sync_status()
            if sync_status.get("online"):
                actions.append({
                    "priority": "high",
                    "description": "Sync to cloud storage",
                    "command": "kernle sync (if available)",
                    "method": "sync",
                })
            return actions
        
        # Critical (86-100): Emergency protocols
        actions.append({
            "priority": "emergency",
            "description": "EMERGENCY: Run emergency_save immediately",
            "command": "kernle anxiety --emergency",
            "method": "emergency_save",
        })
        actions.append({
            "priority": "emergency",
            "description": "Final checkpoint with handoff note",
            "command": "kernle checkpoint save 'HANDOFF' --context '<state for next session>'",
            "method": "checkpoint",
        })
        actions.append({
            "priority": "critical",
            "description": "Accept some context will be lost - prioritize key insights",
            "command": None,
            "method": None,
        })
        
        return actions
    
    def emergency_save(self, summary: Optional[str] = None) -> Dict[str, Any]:
        """Critical-level action: save everything possible.
        
        This is the nuclear option when anxiety hits critical levels.
        Performs all possible memory preservation actions.
        
        Args:
            summary: Optional session summary for the checkpoint
        
        Returns:
            dict with what was saved and any errors
        """
        results = {
            "checkpoint_saved": False,
            "episodes_consolidated": 0,
            "sync_attempted": False,
            "sync_success": False,
            "identity_synthesized": False,
            "errors": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # 1. Emergency checkpoint with full context
        try:
            checkpoint_summary = summary or "EMERGENCY SAVE - Critical anxiety level"
            cp = self.checkpoint(
                task="EMERGENCY_SAVE",
                pending=["Review previous session state"],
                context=checkpoint_summary,
            )
            results["checkpoint_saved"] = True
            results["checkpoint"] = cp
        except Exception as e:
            results["errors"].append(f"Checkpoint failed: {str(e)}")
        
        # 2. Consolidate all unreflected episodes
        try:
            consolidation = self.consolidate(min_episodes=1)
            results["episodes_consolidated"] = consolidation.get("consolidated", 0)
            results["consolidation_result"] = consolidation
        except Exception as e:
            results["errors"].append(f"Consolidation failed: {str(e)}")
        
        # 3. Synthesize identity (to have a coherent state)
        try:
            identity = self.synthesize_identity()
            results["identity_synthesized"] = True
            results["identity_confidence"] = identity.get("confidence", 0)
        except Exception as e:
            results["errors"].append(f"Identity synthesis failed: {str(e)}")
        
        # 4. Attempt cloud sync
        try:
            sync_status = self.get_sync_status()
            if sync_status.get("online"):
                results["sync_attempted"] = True
                sync_result = self.sync()
                results["sync_success"] = sync_result.get("success", False)
                results["sync_result"] = sync_result
            else:
                results["sync_attempted"] = False
        except Exception as e:
            results["errors"].append(f"Sync failed: {str(e)}")
        
        # 5. Record this emergency save as an episode
        try:
            self.episode(
                objective="Emergency memory save",
                outcome="completed" if not results["errors"] else "partial",
                lessons=[
                    f"Anxiety level hit critical - triggered emergency save",
                    f"Saved checkpoint: {results['checkpoint_saved']}",
                    f"Consolidated {results['episodes_consolidated']} episodes",
                ],
                tags=["emergency", "anxiety", "critical"],
            )
        except Exception as e:
            results["errors"].append(f"Episode recording failed: {str(e)}")
        
        results["success"] = len(results["errors"]) == 0
        return results
    
    # =========================================================================
    # META-COGNITION (Self-Awareness of Knowledge)
    # =========================================================================
    
    def _extract_domains_from_tags(self) -> Dict[str, Dict[str, Any]]:
        """Extract knowledge domains from tags across all memory types.
        
        Returns a dict mapping domain names to their statistics.
        """
        from collections import defaultdict
        
        domain_stats = defaultdict(lambda: {
            "belief_count": 0,
            "belief_confidences": [],
            "episode_count": 0,
            "episode_outcomes": [],
            "note_count": 0,
            "goal_count": 0,
            "last_updated": None,
            "tags": set(),
        })
        
        # Process beliefs
        beliefs = self._storage.get_beliefs(limit=1000)
        for belief in beliefs:
            # Use belief_type as a domain indicator
            domain = belief.belief_type or "general"
            domain_stats[domain]["belief_count"] += 1
            domain_stats[domain]["belief_confidences"].append(belief.confidence)
            if belief.created_at:
                if domain_stats[domain]["last_updated"] is None or belief.created_at > domain_stats[domain]["last_updated"]:
                    domain_stats[domain]["last_updated"] = belief.created_at
        
        # Process episodes - extract domains from tags
        episodes = self._storage.get_episodes(limit=1000)
        for episode in episodes:
            tags = episode.tags or []
            # Skip checkpoint tags
            tags = [t for t in tags if t not in ("checkpoint", "working_state", "auto-captured", "manual")]
            
            if tags:
                for tag in tags:
                    domain_stats[tag]["episode_count"] += 1
                    domain_stats[tag]["episode_outcomes"].append(episode.outcome_type or "partial")
                    domain_stats[tag]["tags"].add(tag)
                    if episode.created_at:
                        if domain_stats[tag]["last_updated"] is None or episode.created_at > domain_stats[tag]["last_updated"]:
                            domain_stats[tag]["last_updated"] = episode.created_at
            else:
                # No tags - count in general
                domain_stats["general"]["episode_count"] += 1
                domain_stats["general"]["episode_outcomes"].append(episode.outcome_type or "partial")
        
        # Process notes
        notes = self._storage.get_notes(limit=1000)
        for note in notes:
            tags = note.tags or []
            if tags:
                for tag in tags:
                    domain_stats[tag]["note_count"] += 1
                    domain_stats[tag]["tags"].add(tag)
                    if note.created_at:
                        if domain_stats[tag]["last_updated"] is None or note.created_at > domain_stats[tag]["last_updated"]:
                            domain_stats[tag]["last_updated"] = note.created_at
            else:
                domain_stats["general"]["note_count"] += 1
        
        # Process goals
        goals = self._storage.get_goals(status=None, limit=1000)
        for goal in goals:
            # Extract domain from goal title (simplified)
            words = goal.title.lower().split()[:2]  # First two words as domain hint
            if words:
                domain = words[0]
                domain_stats[domain]["goal_count"] += 1
        
        return dict(domain_stats)
    
    def _calculate_coverage(self, stats: Dict[str, Any]) -> str:
        """Calculate coverage level based on domain statistics."""
        total_items = (
            stats["belief_count"] +
            stats["episode_count"] +
            stats["note_count"]
        )
        
        if total_items == 0:
            return "none"
        elif total_items < 3:
            return "low"
        elif total_items < 10:
            return "medium"
        else:
            return "high"
    
    def get_knowledge_map(self) -> Dict[str, Any]:
        """Map of knowledge domains with coverage assessment.
        
        Analyzes beliefs, episodes, and notes to understand what
        domains I have knowledge about and how confident I am.
        
        Returns:
            {
                "domains": [
                    {
                        "name": "Python programming",
                        "belief_count": 15,
                        "avg_confidence": 0.82,
                        "episode_count": 23,
                        "note_count": 5,
                        "goal_count": 2,
                        "coverage": "high",  # high/medium/low/none
                        "last_updated": datetime
                    },
                    ...
                ],
                "blind_spots": ["GraphQL", "Kubernetes"],  # domains with nothing
                "uncertain_areas": ["Docker networking"]  # low confidence
            }
        """
        domain_stats = self._extract_domains_from_tags()
        
        domains = []
        uncertain_areas = []
        
        for name, stats in domain_stats.items():
            confidences = stats["belief_confidences"]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            coverage = self._calculate_coverage(stats)
            
            domain_info = {
                "name": name,
                "belief_count": stats["belief_count"],
                "avg_confidence": round(avg_confidence, 2),
                "episode_count": stats["episode_count"],
                "note_count": stats["note_count"],
                "goal_count": stats["goal_count"],
                "coverage": coverage,
                "last_updated": stats["last_updated"].isoformat() if stats["last_updated"] else None,
            }
            domains.append(domain_info)
            
            # Track uncertain areas (has beliefs but low confidence)
            if stats["belief_count"] > 0 and avg_confidence < 0.5:
                uncertain_areas.append(name)
        
        # Sort domains by coverage (high first) then by item count
        coverage_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        domains.sort(key=lambda d: (
            coverage_order.get(d["coverage"], 3),
            -(d["belief_count"] + d["episode_count"] + d["note_count"])
        ))
        
        # Blind spots are harder to detect without a reference list
        # For now, we identify domains with very little data that were mentioned
        blind_spots = [
            d["name"] for d in domains
            if d["coverage"] == "none" or (d["coverage"] == "low" and d["avg_confidence"] == 0)
        ]
        
        return {
            "domains": domains,
            "blind_spots": blind_spots,
            "uncertain_areas": uncertain_areas,
            "total_domains": len(domains),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def detect_knowledge_gaps(self, query: str) -> Dict[str, Any]:
        """Analyze if I have knowledge relevant to a query.
        
        Searches memory to determine what I know about a topic
        and identifies gaps in my knowledge.
        
        Args:
            query: The query to check knowledge for
            
        Returns:
            {
                "has_relevant_knowledge": bool,
                "relevant_beliefs": [...],
                "relevant_episodes": [...],
                "relevant_notes": [...],
                "confidence": float,
                "gaps": ["specific thing I don't know about"],
                "recommendation": "I can help" | "I should learn more" | "Ask someone else"
            }
        """
        # Search for relevant memories
        results = self._storage.search(query, limit=20)
        
        relevant_beliefs = []
        relevant_episodes = []
        relevant_notes = []
        confidences = []
        
        for result in results:
            record = result.record
            record_type = result.record_type
            
            if record_type == "belief":
                relevant_beliefs.append({
                    "statement": record.statement,
                    "confidence": record.confidence,
                    "type": record.belief_type,
                })
                confidences.append(record.confidence)
            elif record_type == "episode":
                relevant_episodes.append({
                    "objective": record.objective,
                    "outcome": record.outcome,
                    "outcome_type": record.outcome_type,
                    "lessons": record.lessons,
                })
                confidences.append(getattr(record, 'confidence', 0.8))
            elif record_type == "note":
                relevant_notes.append({
                    "content": record.content[:200],
                    "type": record.note_type,
                    "tags": record.tags,
                })
                confidences.append(getattr(record, 'confidence', 0.8))
        
        # Calculate overall confidence
        has_relevant = len(results) > 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Identify gaps based on query analysis vs what we found
        gaps = []
        query_words = set(query.lower().split())
        found_topics = set()
        
        for b in relevant_beliefs:
            found_topics.update(b["statement"].lower().split())
        for e in relevant_episodes:
            found_topics.update(e["objective"].lower().split())
        
        # Words in query not found in results might indicate gaps
        potential_gaps = query_words - found_topics - {"how", "do", "i", "what", "is", "the", "a", "to", "for", "and", "or"}
        if potential_gaps and len(results) < 3:
            gaps = list(potential_gaps)[:3]
        
        # Determine recommendation
        if not has_relevant:
            recommendation = "Ask someone else"
        elif avg_confidence < 0.5:
            recommendation = "I should learn more"
        elif len(results) < 3:
            recommendation = "I have limited knowledge - proceed with caution"
        else:
            recommendation = "I can help"
        
        return {
            "has_relevant_knowledge": has_relevant,
            "relevant_beliefs": relevant_beliefs[:5],
            "relevant_episodes": relevant_episodes[:5],
            "relevant_notes": relevant_notes[:5],
            "confidence": round(avg_confidence, 2),
            "gaps": gaps,
            "recommendation": recommendation,
            "search_results_count": len(results),
        }
    
    def get_competence_boundaries(self) -> Dict[str, Any]:
        """What am I good at vs not good at?
        
        Analyzes belief confidence distribution, episode outcomes,
        and domain coverage to identify areas of strength and weakness.
        
        Returns:
            {
                "strengths": [
                    {"domain": "Python", "confidence": 0.9, "success_rate": 0.85},
                    ...
                ],
                "weaknesses": [
                    {"domain": "Docker", "confidence": 0.3, "success_rate": 0.4},
                    ...
                ],
                "overall_confidence": float,
                "success_rate": float,
                "experience_depth": int,  # total episodes
                "knowledge_breadth": int,  # number of domains
            }
        """
        domain_stats = self._extract_domains_from_tags()
        
        strengths = []
        weaknesses = []
        all_confidences = []
        all_outcomes = []
        
        for domain_name, stats in domain_stats.items():
            # Skip meta domains
            if domain_name in ("general", "manual", "auto-captured"):
                continue
            
            confidences = stats["belief_confidences"]
            outcomes = stats["episode_outcomes"]
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            success_count = outcomes.count("success")
            success_rate = success_count / len(outcomes) if outcomes else 0.5
            
            all_confidences.extend(confidences)
            all_outcomes.extend(outcomes)
            
            total_items = stats["belief_count"] + stats["episode_count"] + stats["note_count"]
            
            # Need at least some data to make a judgment
            if total_items < 2:
                continue
            
            domain_info = {
                "domain": domain_name,
                "confidence": round(avg_confidence, 2),
                "success_rate": round(success_rate, 2),
                "episode_count": stats["episode_count"],
                "belief_count": stats["belief_count"],
            }
            
            # Classify as strength or weakness
            if avg_confidence >= 0.7 and success_rate >= 0.6:
                strengths.append(domain_info)
            elif avg_confidence < 0.5 or success_rate < 0.4:
                weaknesses.append(domain_info)
        
        # Sort by confidence/success
        strengths.sort(key=lambda x: (x["confidence"], x["success_rate"]), reverse=True)
        weaknesses.sort(key=lambda x: (x["confidence"], x["success_rate"]))
        
        # Calculate overall metrics
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5
        overall_success = all_outcomes.count("success") / len(all_outcomes) if all_outcomes else 0.5
        
        return {
            "strengths": strengths[:10],
            "weaknesses": weaknesses[:10],
            "overall_confidence": round(overall_confidence, 2),
            "success_rate": round(overall_success, 2),
            "experience_depth": len(all_outcomes),
            "knowledge_breadth": len([d for d in domain_stats if d not in ("general", "manual", "auto-captured")]),
        }
    
    def identify_learning_opportunities(self, limit: int = 5) -> List[Dict[str, Any]]:
        """What should I learn next?
        
        Identifies learning opportunities based on:
        - Low-coverage domains that are referenced often
        - Uncertain beliefs that affect decisions
        - Failed episodes that could benefit from more knowledge
        
        Args:
            limit: Maximum opportunities to return
            
        Returns:
            List of learning opportunities with priority and reasoning
        """
        opportunities = []
        
        domain_stats = self._extract_domains_from_tags()
        
        # 1. Low-coverage but frequently referenced domains
        for domain_name, stats in domain_stats.items():
            if domain_name in ("general", "manual", "auto-captured"):
                continue
            
            coverage = self._calculate_coverage(stats)
            reference_count = stats["episode_count"] + stats["note_count"]
            
            if coverage in ("low", "none") and reference_count > 0:
                opportunities.append({
                    "type": "low_coverage_domain",
                    "domain": domain_name,
                    "reason": f"Referenced {reference_count} times but only {stats['belief_count']} beliefs",
                    "priority": "high" if reference_count > 3 else "medium",
                    "suggested_action": f"Research and form beliefs about {domain_name}",
                })
        
        # 2. Uncertain beliefs that might affect decisions
        beliefs = self._storage.get_beliefs(limit=1000)
        low_confidence_beliefs = [
            b for b in beliefs
            if b.confidence < 0.5 and not getattr(b, 'deleted', False)
        ]
        
        for belief in low_confidence_beliefs[:3]:
            opportunities.append({
                "type": "uncertain_belief",
                "domain": belief.belief_type or "general",
                "reason": f"Belief with only {belief.confidence:.0%} confidence: '{belief.statement[:50]}...'",
                "priority": "medium",
                "suggested_action": "Verify or update this belief with evidence",
            })
        
        # 3. Failed episodes indicating knowledge gaps
        episodes = self._storage.get_episodes(limit=100)
        failed_episodes = [
            e for e in episodes
            if e.outcome_type == "failure" and e.tags and "checkpoint" not in e.tags
        ]
        
        # Group failures by domain
        failure_domains = {}
        for ep in failed_episodes:
            for tag in (ep.tags or []):
                if tag not in ("manual", "auto-captured"):
                    failure_domains[tag] = failure_domains.get(tag, 0) + 1
        
        for domain, count in sorted(failure_domains.items(), key=lambda x: -x[1])[:3]:
            opportunities.append({
                "type": "repeated_failures",
                "domain": domain,
                "reason": f"{count} failed episodes in {domain}",
                "priority": "high" if count > 2 else "medium",
                "suggested_action": f"Study {domain} to improve success rate",
            })
        
        # 4. Areas with no recent activity (might be getting stale)
        now = datetime.now(timezone.utc)
        stale_threshold = timedelta(days=30)
        
        for domain_name, stats in domain_stats.items():
            if domain_name in ("general", "manual", "auto-captured"):
                continue
            
            coverage = self._calculate_coverage(stats)
            if coverage in ("medium", "high") and stats["last_updated"]:
                age = now - stats["last_updated"]
                if age > stale_threshold:
                    opportunities.append({
                        "type": "stale_knowledge",
                        "domain": domain_name,
                        "reason": f"No updates in {age.days} days - knowledge may be outdated",
                        "priority": "low",
                        "suggested_action": f"Review and refresh knowledge about {domain_name}",
                    })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        opportunities.sort(key=lambda x: priority_order.get(x["priority"], 2))
        
        return opportunities[:limit]
