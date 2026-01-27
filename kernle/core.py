"""
Kernle Core - Stratified memory for synthetic intelligences.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from supabase import create_client


class Kernle:
    """Main interface for Kernle memory operations."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.agent_id = agent_id or os.environ.get("KERNLE_AGENT_ID", "default")
        self.supabase_url = supabase_url or os.environ.get("KERNLE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("KERNLE_SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        self.checkpoint_dir = checkpoint_dir or Path.home() / ".kernle" / "checkpoints"
        
        self._client = None
    
    @property
    def client(self):
        """Lazy-load Supabase client."""
        if self._client is None:
            if not self.supabase_url or not self.supabase_key:
                raise ValueError("Supabase credentials required. Set KERNLE_SUPABASE_URL and KERNLE_SUPABASE_KEY.")
            self._client = create_client(self.supabase_url, self.supabase_key)
        return self._client
    
    # =========================================================================
    # LOAD
    # =========================================================================
    
    def load(self, budget: int = 6000) -> dict:
        """Load working memory context."""
        return {
            "checkpoint": self.load_checkpoint(),
            "values": self.load_values(),
            "beliefs": self.load_beliefs(),
            "goals": self.load_goals(),
            "lessons": self.load_lessons(),
            "recent_work": self.load_recent_work(),
            "recent_notes": self.load_recent_notes(),
        }
    
    def load_values(self, limit: int = 10) -> list[dict]:
        """Load normative values (highest authority)."""
        result = self.client.table("agent_values").select(
            "name, statement, priority, value_type"
        ).eq("agent_id", self.agent_id).eq("is_active", True).order(
            "priority", desc=True
        ).limit(limit).execute()
        return result.data
    
    def load_beliefs(self, limit: int = 20) -> list[dict]:
        """Load semantic beliefs."""
        result = self.client.table("agent_beliefs").select(
            "statement, belief_type, confidence"
        ).eq("agent_id", self.agent_id).eq("is_active", True).order(
            "confidence", desc=True
        ).limit(limit).execute()
        return result.data
    
    def load_goals(self, limit: int = 10) -> list[dict]:
        """Load active goals."""
        result = self.client.table("agent_goals").select(
            "title, description, priority, status"
        ).eq("agent_id", self.agent_id).eq("status", "active").order(
            "created_at", desc=True
        ).limit(limit).execute()
        return result.data
    
    def load_lessons(self, limit: int = 20) -> list[str]:
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
    
    def load_recent_work(self, limit: int = 5) -> list[dict]:
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
        
        # Save locally
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.checkpoint_dir / f"{self.agent_id}.json"
        
        existing = []
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except:
                existing = []
        
        existing.append(checkpoint_data)
        existing = existing[-10:]  # Keep last 10
        
        with open(checkpoint_file, "w") as f:
            json.dump(existing, f, indent=2)
        
        # Also save to Supabase as episode
        try:
            self.client.table("agent_episodes").insert({
                "agent_id": self.agent_id,
                "objective": f"[CHECKPOINT] {task}",
                "outcome_type": "partial",
                "outcome_description": context or "Working state checkpoint",
                "lessons_learned": pending or [],
                "tags": ["checkpoint", "working_state"],
            }).execute()
        except Exception as e:
            pass  # Local save is sufficient
        
        return checkpoint_data
    
    def load_checkpoint(self) -> Optional[dict]:
        """Load most recent checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{self.agent_id}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    checkpoints = json.load(f)
                    if isinstance(checkpoints, list) and checkpoints:
                        return checkpoints[-1]
                    elif isinstance(checkpoints, dict):
                        return checkpoints
            except:
                pass
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
        lessons: Optional[list[str]] = None,
        repeat: Optional[list[str]] = None,
        avoid: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Record an episodic experience."""
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
        tags: Optional[list[str]] = None,
        protect: bool = False,
    ) -> str:
        """Capture a quick note (decision, insight, quote)."""
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
        
        return "\n".join(lines)
