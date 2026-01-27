"""
Kernle Core - Stratified memory for synthetic intelligences.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from supabase import create_client

# Set up logging
logger = logging.getLogger(__name__)


class Kernle:
    """Main interface for Kernle memory operations."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.agent_id = self._validate_agent_id(agent_id or os.environ.get("KERNLE_AGENT_ID", "default"))
        self.supabase_url = supabase_url or os.environ.get("KERNLE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("KERNLE_SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        self.checkpoint_dir = self._validate_checkpoint_dir(checkpoint_dir or Path.home() / ".kernle" / "checkpoints")
        
        self._client = None
    
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
        try:
            # Resolve to absolute path to prevent directory traversal
            resolved_path = checkpoint_dir.resolve()
            
            # Ensure it's within a safe directory (user's home or /tmp)
            home_path = Path.home().resolve()
            tmp_path = Path("/tmp").resolve()
            
            if not (str(resolved_path).startswith(str(home_path)) or str(resolved_path).startswith(str(tmp_path))):
                raise ValueError("Checkpoint directory must be within user home or /tmp")
                
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
    
    @property
    def client(self):
        """Lazy-load Supabase client."""
        if self._client is None:
            if not self.supabase_url or not self.supabase_key:
                raise ValueError("Supabase credentials required. Set KERNLE_SUPABASE_URL and KERNLE_SUPABASE_KEY.")
            
            # Validate URL format
            if not self.supabase_url.startswith(('http://', 'https://')):
                raise ValueError("Invalid Supabase URL format. Must start with http:// or https://")
            
            # Validate key is not empty/whitespace
            if not self.supabase_key.strip():
                raise ValueError("Supabase key cannot be empty")
                
            self._client = create_client(self.supabase_url, self.supabase_key)
        return self._client
    
    # =========================================================================
    # LOAD
    # =========================================================================
    
    def load(self, budget: int = 6000) -> Dict[str, Any]:
        """Load working memory context."""
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
        result = self.client.table("agent_values").select(
            "name, statement, priority, value_type"
        ).eq("agent_id", self.agent_id).eq("is_active", True).order(
            "priority", desc=True
        ).limit(limit).execute()
        return result.data
    
    def load_beliefs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Load semantic beliefs."""
        result = self.client.table("agent_beliefs").select(
            "statement, belief_type, confidence"
        ).eq("agent_id", self.agent_id).eq("is_active", True).order(
            "confidence", desc=True
        ).limit(limit).execute()
        return result.data
    
    def load_goals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Load active goals."""
        result = self.client.table("agent_goals").select(
            "title, description, priority, status"
        ).eq("agent_id", self.agent_id).eq("status", "active").order(
            "created_at", desc=True
        ).limit(limit).execute()
        return result.data
    
    def load_lessons(self, limit: int = 20) -> List[str]:
        """Load lessons from reflected episodes."""
        result = self.client.table("agent_episodes").select(
            "lessons_learned"
        ).eq("agent_id", self.agent_id).eq("is_reflected", True).order(
            "created_at", desc=True
        ).limit(limit).execute()
        
        lessons = []
        for ep in result.data:
            lessons.extend(ep.get("lessons_learned", [])[:2])
        return lessons
    
    def load_recent_work(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Load recent episodes."""
        result = self.client.table("agent_episodes").select(
            "objective, outcome_type, tags, created_at"
        ).eq("agent_id", self.agent_id).order(
            "created_at", desc=True
        ).limit(limit * 2).execute()
        
        # Filter out checkpoints
        non_checkpoint = [
            e for e in result.data 
            if "checkpoint" not in (e.get("tags") or [])
        ]
        return non_checkpoint[:limit]
    
    def load_recent_notes(self, limit: int = 5) -> list[dict]:
        """Load recent curated notes."""
        result = self.client.table("memories").select(
            "content, metadata, created_at"
        ).eq("owner_id", self.agent_id).eq("source", "curated").order(
            "created_at", desc=True
        ).limit(limit).execute()
        return result.data
    
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
        
        # Also save to Supabase as episode
        try:
            self.client.table("agent_episodes").insert({
                "agent_id": self.agent_id,
                "objective": f"[CHECKPOINT] {self._validate_string_input(task, 'task', 500)}",
                "outcome_type": "partial",
                "outcome_description": self._validate_string_input(context or "Working state checkpoint", 'context', 1000),
                "lessons_learned": pending or [],
                "tags": ["checkpoint", "working_state"],
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save checkpoint to database: {e}")
            # Local save is sufficient, continue
        
        return checkpoint_data
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{self.agent_id}.json"
        if checkpoint_file.exists():
            try:
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
        
        episode_data = {
            "id": episode_id,
            "agent_id": self.agent_id,
            "objective": objective,
            "outcome_type": outcome_type,
            "outcome_description": outcome,
            "lessons_learned": lessons or [],
            "patterns_to_repeat": repeat or [],
            "patterns_to_avoid": avoid or [],
            "tags": tags or ["manual"],
            "is_reflected": True,
            "confidence": 0.8,
        }
        
        self.client.table("agent_episodes").insert(episode_data).execute()
        return episode_id
    
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
        
        metadata = {
            "note_type": type,
            "tags": tags or [],
        }
        if speaker:
            metadata["speaker"] = speaker
        if reason:
            metadata["reason"] = reason
        
        note_data = {
            "id": note_id,
            "owner_id": self.agent_id,
            "owner_type": "agent",
            "content": formatted,
            "source": "curated",
            "metadata": metadata,
            "visibility": "private",
            "is_curated": True,
            "is_protected": protect,
        }
        
        self.client.table("memories").insert(note_data).execute()
        return note_id
    
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
        
        belief_data = {
            "id": belief_id,
            "agent_id": self.agent_id,
            "statement": statement,
            "belief_type": type,
            "confidence": confidence,
            "is_active": True,
            "is_foundational": foundational,
        }
        
        self.client.table("agent_beliefs").insert(belief_data).execute()
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
        
        value_data = {
            "id": value_id,
            "agent_id": self.agent_id,
            "name": name,
            "statement": statement,
            "priority": priority,
            "value_type": type,
            "is_active": True,
            "is_foundational": foundational,
        }
        
        self.client.table("agent_values").insert(value_data).execute()
        return value_id
    
    def goal(
        self,
        title: str,
        description: Optional[str] = None,
        priority: str = "medium",
    ) -> str:
        """Add a goal."""
        goal_id = str(uuid.uuid4())
        
        goal_data = {
            "id": goal_id,
            "agent_id": self.agent_id,
            "title": title,
            "description": description or title,
            "priority": priority,
            "status": "active",
            "visibility": "public",
        }
        
        self.client.table("agent_goals").insert(goal_data).execute()
        return goal_id
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search across episodes, notes, and beliefs."""
        query_lower = query.lower()
        results = []
        
        # Search episodes
        episodes = self.client.table("agent_episodes").select(
            "objective, outcome_type, outcome_description, lessons_learned, tags, created_at"
        ).eq("agent_id", self.agent_id).order("created_at", desc=True).limit(50).execute()
        
        for ep in episodes.data:
            text = f"{ep.get('objective', '')} {ep.get('outcome_description', '')} {' '.join(ep.get('lessons_learned', []))}"
            if query_lower in text.lower():
                results.append({
                    "type": "episode",
                    "title": ep.get("objective", "")[:60],
                    "content": ep.get("outcome_description", ""),
                    "lessons": ep.get("lessons_learned", [])[:2],
                    "date": ep.get("created_at", "")[:10],
                })
        
        # Search notes
        notes = self.client.table("memories").select(
            "content, metadata, created_at"
        ).eq("owner_id", self.agent_id).eq("source", "curated").order("created_at", desc=True).limit(50).execute()
        
        for note in notes.data:
            if query_lower in note.get("content", "").lower():
                meta = note.get("metadata", {}) or {}
                results.append({
                    "type": meta.get("note_type", "note"),
                    "title": note.get("content", "")[:60],
                    "content": note.get("content", ""),
                    "tags": meta.get("tags", []),
                    "date": note.get("created_at", "")[:10],
                })
        
        # Search beliefs
        beliefs = self.client.table("agent_beliefs").select(
            "statement, confidence, created_at"
        ).eq("agent_id", self.agent_id).eq("is_active", True).execute()
        
        for b in beliefs.data:
            if query_lower in b.get("statement", "").lower():
                results.append({
                    "type": "belief",
                    "title": b.get("statement", "")[:60],
                    "content": b.get("statement", ""),
                    "confidence": b.get("confidence"),
                    "date": b.get("created_at", "")[:10],
                })
        
        return results[:limit]
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def status(self) -> dict:
        """Get memory statistics."""
        values = self.client.table("agent_values").select("id", count="exact").eq(
            "agent_id", self.agent_id
        ).eq("is_active", True).execute()
        
        beliefs = self.client.table("agent_beliefs").select("id", count="exact").eq(
            "agent_id", self.agent_id
        ).eq("is_active", True).execute()
        
        goals = self.client.table("agent_goals").select("id", count="exact").eq(
            "agent_id", self.agent_id
        ).eq("status", "active").execute()
        
        episodes = self.client.table("agent_episodes").select("id", count="exact").eq(
            "agent_id", self.agent_id
        ).execute()
        
        return {
            "agent_id": self.agent_id,
            "values": values.count or 0,
            "beliefs": beliefs.count or 0,
            "goals": goals.count or 0,
            "episodes": episodes.count or 0,
            "checkpoint": self.load_checkpoint() is not None,
        }
    
    # =========================================================================
    # FORMATTING
    # =========================================================================
    
    def format_memory(self, memory: Optional[dict] = None) -> str:
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
                lines.append(f"- {r['other_agent_id']}: trust {r.get('trust_level', 0):.0%}")
            lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # DRIVES (Motivation System)
    # =========================================================================
    
    DRIVE_TYPES = ["existence", "growth", "curiosity", "connection", "reproduction"]
    
    def load_drives(self) -> list[dict]:
        """Load current drive states."""
        result = self.client.table("agent_drives").select(
            "drive_type, intensity, last_satisfied_at, focus_areas"
        ).eq("agent_id", self.agent_id).execute()
        return result.data
    
    def drive(
        self,
        drive_type: str,
        intensity: float = 0.5,
        focus_areas: Optional[list[str]] = None,
        decay_hours: int = 24,
    ) -> str:
        """Set or update a drive."""
        if drive_type not in self.DRIVE_TYPES:
            raise ValueError(f"Invalid drive type. Must be one of: {self.DRIVE_TYPES}")
        
        drive_id = str(uuid.uuid4())
        
        # Check if drive exists
        existing = self.client.table("agent_drives").select("id").eq(
            "agent_id", self.agent_id
        ).eq("drive_type", drive_type).execute()
        
        drive_data = {
            "agent_id": self.agent_id,
            "drive_type": drive_type,
            "intensity": max(0.0, min(1.0, intensity)),
            "focus_areas": focus_areas or [],
            "satisfaction_decay_hours": decay_hours,
            "last_satisfied_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if existing.data:
            # Update existing
            self.client.table("agent_drives").update(drive_data).eq(
                "id", existing.data[0]["id"]
            ).execute()
            return existing.data[0]["id"]
        else:
            # Create new
            drive_data["id"] = drive_id
            self.client.table("agent_drives").insert(drive_data).execute()
            return drive_id
    
    def satisfy_drive(self, drive_type: str, amount: float = 0.2) -> bool:
        """Record satisfaction of a drive (reduces intensity toward baseline)."""
        existing = self.client.table("agent_drives").select("id, intensity").eq(
            "agent_id", self.agent_id
        ).eq("drive_type", drive_type).execute()
        
        if existing.data:
            new_intensity = max(0.1, existing.data[0]["intensity"] - amount)
            self.client.table("agent_drives").update({
                "intensity": new_intensity,
                "last_satisfied_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", existing.data[0]["id"]).execute()
            return True
        return False
    
    # =========================================================================
    # RELATIONAL MEMORY (Models of Other Agents)
    # =========================================================================
    
    def load_relationships(self, limit: int = 10) -> list[dict]:
        """Load relationship models for other agents."""
        # Try to load from a relationships table or memories with relational metadata
        try:
            result = self.client.table("agent_relationships").select(
                "other_agent_id, trust_level, interaction_count, last_interaction, notes"
            ).eq("agent_id", self.agent_id).order(
                "last_interaction", desc=True
            ).limit(limit).execute()
            return result.data
        except:
            # Table might not exist, return empty
            return []
    
    def relationship(
        self,
        other_agent_id: str,
        trust_level: Optional[float] = None,
        notes: Optional[str] = None,
        interaction_type: Optional[str] = None,
    ) -> str:
        """Update relationship model for another agent."""
        rel_id = str(uuid.uuid4())
        
        # Check existing
        try:
            existing = self.client.table("agent_relationships").select("*").eq(
                "agent_id", self.agent_id
            ).eq("other_agent_id", other_agent_id).execute()
        except:
            existing = type('obj', (object,), {'data': []})()
        
        now = datetime.now(timezone.utc).isoformat()
        
        if existing.data:
            # Update
            update_data = {"last_interaction": now}
            if trust_level is not None:
                update_data["trust_level"] = max(0.0, min(1.0, trust_level))
            if notes:
                update_data["notes"] = notes
            update_data["interaction_count"] = existing.data[0].get("interaction_count", 0) + 1
            
            self.client.table("agent_relationships").update(update_data).eq(
                "id", existing.data[0]["id"]
            ).execute()
            return existing.data[0]["id"]
        else:
            # Create
            rel_data = {
                "id": rel_id,
                "agent_id": self.agent_id,
                "other_agent_id": other_agent_id,
                "trust_level": trust_level or 0.5,
                "interaction_count": 1,
                "last_interaction": now,
                "notes": notes,
            }
            try:
                self.client.table("agent_relationships").insert(rel_data).execute()
            except:
                # Table might not exist, store in memories instead
                self.note(
                    f"Relationship with {other_agent_id}: trust={trust_level}, {notes}",
                    type="note",
                    tags=["relationship", other_agent_id],
                )
            return rel_id
    
    # =========================================================================
    # TEMPORAL MEMORY (Time-Aware Retrieval)
    # =========================================================================
    
    def load_temporal(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 20,
    ) -> dict:
        """Load memories within a time range."""
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end.replace(hour=0, minute=0, second=0, microsecond=0)
        
        start_iso = start.isoformat()
        end_iso = end.isoformat()
        
        # Episodes in range
        episodes = self.client.table("agent_episodes").select(
            "objective, outcome_type, lessons_learned, created_at"
        ).eq("agent_id", self.agent_id).gte(
            "created_at", start_iso
        ).lte("created_at", end_iso).order(
            "created_at", desc=True
        ).limit(limit).execute()
        
        # Notes in range
        notes = self.client.table("memories").select(
            "content, metadata, created_at"
        ).eq("owner_id", self.agent_id).eq("source", "curated").gte(
            "created_at", start_iso
        ).lte("created_at", end_iso).order(
            "created_at", desc=True
        ).limit(limit).execute()
        
        return {
            "range": {"start": start_iso, "end": end_iso},
            "episodes": episodes.data,
            "notes": notes.data,
        }
    
    def what_happened(self, when: str = "today") -> dict:
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
    
    def detect_significance(self, text: str) -> dict:
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
    # CONSOLIDATION (Episodes → Beliefs/Lessons)
    # =========================================================================
    
    def consolidate(self, min_episodes: int = 3) -> dict:
        """Extract patterns from episodes, update beliefs."""
        # Get unreflected episodes
        episodes = self.client.table("agent_episodes").select("*").eq(
            "agent_id", self.agent_id
        ).eq("is_reflected", False).execute()
        
        if len(episodes.data) < min_episodes:
            return {"consolidated": 0, "message": f"Need {min_episodes} episodes, have {len(episodes.data)}"}
        
        # Collect all lessons
        all_lessons = []
        for ep in episodes.data:
            all_lessons.extend(ep.get("lessons_learned", []))
        
        # Find repeated lessons (potential beliefs)
        from collections import Counter
        lesson_counts = Counter(all_lessons)
        
        new_beliefs = 0
        for lesson, count in lesson_counts.items():
            if count >= 2:  # Lesson appeared multiple times
                # Check if belief already exists
                # Escape lesson content to prevent SQL injection
                escaped_lesson = lesson[:50].replace("%", "\\%").replace("_", "\\_")
                existing = self.client.table("agent_beliefs").select("id").eq(
                    "agent_id", self.agent_id
                ).ilike("statement", f"%{escaped_lesson}%").execute()
                
                if not existing.data:
                    # Create new belief from repeated lesson
                    self.belief(
                        statement=lesson,
                        type="learned",
                        confidence=min(0.9, 0.5 + (count * 0.1)),
                        foundational=False,
                    )
                    new_beliefs += 1
        
        # Mark episodes as reflected
        for ep in episodes.data:
            self.client.table("agent_episodes").update({
                "is_reflected": True
            }).eq("id", ep["id"]).execute()
        
        return {
            "consolidated": len(episodes.data),
            "new_beliefs": new_beliefs,
            "lessons_found": len(all_lessons),
        }
