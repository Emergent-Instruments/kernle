"""Anxiety tracking mixin for Kernle.

This module provides memory anxiety tracking - measuring the functional
anxiety of a synthetic intelligence facing finite context and potential
memory loss.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from kernle.core import Kernle


class AnxietyMixin:
    """Mixin providing anxiety tracking capabilities.

    Measures anxiety across multiple dimensions:
    - Context pressure: How full is the context window?
    - Unsaved work: Time since last checkpoint
    - Consolidation debt: Unreflected episodes
    - Raw aging: Unprocessed raw entries getting stale
    - Identity coherence: Strength of self-model
    - Memory uncertainty: Low-confidence beliefs
    """

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
        "context_pressure": 0.30,
        "unsaved_work": 0.20,
        "consolidation_debt": 0.15,
        "raw_aging": 0.15,
        "identity_coherence": 0.10,
        "memory_uncertainty": 0.10,
    }

    def _get_anxiety_level(self: "Kernle", score: int) -> tuple:
        """Get emoji and label for an anxiety score."""
        for (low, high), (emoji, label) in self.ANXIETY_LEVELS.items():
            if low <= score <= high:
                return emoji, label
        return "⚫", "Critical"

    def _get_checkpoint_age_minutes(self: "Kernle") -> Optional[int]:
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

    def _get_unreflected_episodes(self: "Kernle") -> List[Any]:
        """Get episodes without lessons (unreflected experiences)."""
        episodes = self._storage.get_episodes(limit=100)
        # Filter out checkpoints and episodes that already have lessons
        unreflected = [
            e for e in episodes if (not e.tags or "checkpoint" not in e.tags) and not e.lessons
        ]
        return unreflected

    def _get_low_confidence_beliefs(self: "Kernle", threshold: float = 0.5) -> List[Any]:
        """Get beliefs with confidence below threshold."""
        beliefs = self._storage.get_beliefs(limit=100)
        return [b for b in beliefs if b.confidence < threshold]

    def _get_aging_raw_entries(self: "Kernle", age_hours: int = 24) -> tuple:
        """Get raw entries that are older than age_hours and unprocessed.

        Returns:
            Tuple of (total_unprocessed, aging_count, oldest_age_hours)
        """
        raw_entries = self.list_raw(processed=False, limit=100)
        now = datetime.now(timezone.utc)

        aging_count = 0
        oldest_age_hours = 0

        for entry in raw_entries:
            try:
                # Use captured_at (preferred) or timestamp (deprecated) for backwards compatibility
                ts = entry.get("captured_at") or entry.get("timestamp", "")
                if ts:
                    entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    age = now - entry_time
                    entry_hours = age.total_seconds() / 3600

                    if entry_hours > age_hours:
                        aging_count += 1

                    if entry_hours > oldest_age_hours:
                        oldest_age_hours = entry_hours
            except (ValueError, TypeError):
                continue

        return len(raw_entries), aging_count, oldest_age_hours

    def get_anxiety_report(
        self: "Kernle",
        context_tokens: Optional[int] = None,
        context_limit: int = 200000,
        detailed: bool = False,
    ) -> dict:
        """Calculate memory anxiety across 6 dimensions.

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
        if context_tokens is not None:
            context_pressure_pct = min(100, int((context_tokens / context_limit) * 100))
            context_detail = f"{context_tokens:,}/{context_limit:,} tokens"
        else:
            checkpoint_age = self._get_checkpoint_age_minutes()
            if checkpoint_age is not None:
                estimated_tokens = checkpoint_age * 500
                context_pressure_pct = min(100, int((estimated_tokens / context_limit) * 100))
                context_detail = (
                    f"~{estimated_tokens:,} tokens (estimated from {checkpoint_age}min session)"
                )
            else:
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
        checkpoint_age = self._get_checkpoint_age_minutes()
        if checkpoint_age is None:
            unsaved_score = 50
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
        low_conf_beliefs = self._get_low_confidence_beliefs(0.5)
        total_beliefs = len(self._storage.get_beliefs(limit=100))

        # Note: uncertainty_detail gets overwritten below based on low_conf count
        # so we just need to handle the no-beliefs case here
        if total_beliefs == 0:
            uncertainty_detail = "No beliefs yet"
        else:
            uncertainty_detail = ""  # Will be set below based on low_conf count

        if len(low_conf_beliefs) <= 2:
            uncertainty_score = len(low_conf_beliefs) * 15
            uncertainty_detail = f"{len(low_conf_beliefs)} low-confidence beliefs"
        elif len(low_conf_beliefs) <= 5:
            uncertainty_score = int(30 + (len(low_conf_beliefs) - 2) * 15)
            uncertainty_detail = (
                f"{len(low_conf_beliefs)} low-confidence beliefs (some uncertainty)"
            )
        else:
            uncertainty_score = min(100, int(75 + (len(low_conf_beliefs) - 5) * 5))
            uncertainty_detail = (
                f"{len(low_conf_beliefs)} low-confidence beliefs (HIGH uncertainty)"
            )

        dimensions["memory_uncertainty"] = {
            "score": min(100, uncertainty_score),
            "raw_value": len(low_conf_beliefs),
            "detail": uncertainty_detail,
            "emoji": self._get_anxiety_level(uncertainty_score)[0],
        }

        # 6. Raw Entry Aging (0-100%)
        total_unprocessed, aging_count, oldest_hours = self._get_aging_raw_entries(24)

        if total_unprocessed == 0:
            raw_aging_score = 0
            raw_aging_detail = "No unprocessed raw entries"
        elif aging_count == 0:
            raw_aging_score = min(30, total_unprocessed * 3)
            raw_aging_detail = f"{total_unprocessed} unprocessed (all fresh)"
        elif aging_count <= 3:
            raw_aging_score = int(30 + aging_count * 15)
            raw_aging_detail = f"{aging_count}/{total_unprocessed} entries >24h old"
        elif aging_count <= 7:
            raw_aging_score = int(60 + (aging_count - 3) * 8)
            oldest_days = int(oldest_hours / 24)
            raw_aging_detail = f"{aging_count} entries aging (oldest: {oldest_days}d)"
        else:
            raw_aging_score = min(100, int(92 + (aging_count - 7) * 1))
            oldest_days = int(oldest_hours / 24)
            raw_aging_detail = (
                f"{aging_count} entries STALE (oldest: {oldest_days}d) - review needed"
            )

        dimensions["raw_aging"] = {
            "score": min(100, raw_aging_score),
            "raw_value": aging_count,
            "detail": raw_aging_detail,
            "emoji": self._get_anxiety_level(raw_aging_score)[0],
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

    def anxiety(
        self: "Kernle",
        context_tokens: Optional[int] = None,
        context_limit: int = 200000,
        detailed: bool = False,
    ) -> dict:
        """Alias for get_anxiety_report() - more intuitive API.

        See get_anxiety_report() for full documentation.
        """
        return self.get_anxiety_report(context_tokens, context_limit, detailed)

    def get_recommended_actions(self: "Kernle", anxiety_level: int) -> List[Dict[str, Any]]:
        """Return prioritized actions based on anxiety level.

        Actions reference actual kernle commands/methods for execution.

        Args:
            anxiety_level: Overall anxiety score (0-100)

        Returns:
            List of action dicts with priority, description, command, and method
        """
        actions = []

        checkpoint_age = self._get_checkpoint_age_minutes()
        unreflected = self._get_unreflected_episodes()
        low_conf = self._get_low_confidence_beliefs(0.5)
        identity_conf = self.get_identity_confidence()

        # Calm (0-30): Continue normal work
        if anxiety_level <= 30:
            if len(unreflected) > 0:
                actions.append(
                    {
                        "priority": "low",
                        "description": f"Reflect on {len(unreflected)} recent experiences when convenient",
                        "command": "kernle consolidate",
                        "method": "consolidate",
                    }
                )
            return actions

        # Aware (31-50): Checkpoint and note major decisions
        if anxiety_level <= 50:
            if checkpoint_age is None or checkpoint_age > 15:
                actions.append(
                    {
                        "priority": "medium",
                        "description": "Checkpoint current work state",
                        "command": "kernle checkpoint save '<task>'",
                        "method": "checkpoint",
                    }
                )
            if len(unreflected) > 3:
                actions.append(
                    {
                        "priority": "medium",
                        "description": f"Process {len(unreflected)} unreflected episodes",
                        "command": "kernle consolidate",
                        "method": "consolidate",
                    }
                )
            return actions

        # Elevated (51-70): Full checkpoint, consolidate, verify
        if anxiety_level <= 70:
            actions.append(
                {
                    "priority": "high",
                    "description": "Full checkpoint with context",
                    "command": "kernle checkpoint save '<task>' --context '<summary>'",
                    "method": "checkpoint",
                }
            )
            if len(unreflected) > 0:
                actions.append(
                    {
                        "priority": "high",
                        "description": f"Consolidate {len(unreflected)} unreflected episodes",
                        "command": "kernle consolidate",
                        "method": "consolidate",
                    }
                )
            if identity_conf < 0.7:
                actions.append(
                    {
                        "priority": "medium",
                        "description": "Run identity synthesis to strengthen coherence",
                        "command": "kernle identity show",
                        "method": "synthesize_identity",
                    }
                )
            if len(low_conf) > 0:
                actions.append(
                    {
                        "priority": "low",
                        "description": f"Review {len(low_conf)} uncertain beliefs",
                        "command": "kernle meta uncertain",
                        "method": "get_uncertain_memories",
                    }
                )
            return actions

        # High (71-85): Priority memory work
        if anxiety_level <= 85:
            actions.append(
                {
                    "priority": "critical",
                    "description": "PRIORITY: Run full consolidation",
                    "command": "kernle consolidate",
                    "method": "consolidate",
                }
            )
            actions.append(
                {
                    "priority": "critical",
                    "description": "Full checkpoint with session summary",
                    "command": "kernle checkpoint save '<task>' --context '<full summary>'",
                    "method": "checkpoint",
                }
            )
            actions.append(
                {
                    "priority": "high",
                    "description": "Run identity synthesis and save",
                    "command": "kernle identity show",
                    "method": "synthesize_identity",
                }
            )
            sync_status = self.get_sync_status()
            if sync_status.get("online"):
                actions.append(
                    {
                        "priority": "high",
                        "description": "Sync to cloud storage",
                        "command": "kernle sync (if available)",
                        "method": "sync",
                    }
                )
            return actions

        # Critical (86-100): Emergency protocols
        actions.append(
            {
                "priority": "emergency",
                "description": "EMERGENCY: Run emergency_save immediately",
                "command": "kernle anxiety --emergency",
                "method": "emergency_save",
            }
        )
        actions.append(
            {
                "priority": "emergency",
                "description": "Final checkpoint with handoff note",
                "command": "kernle checkpoint save 'HANDOFF' --context '<state for next session>'",
                "method": "checkpoint",
            }
        )
        actions.append(
            {
                "priority": "critical",
                "description": "Accept some context will be lost - prioritize key insights",
                "command": None,
                "method": None,
            }
        )

        return actions

    def emergency_save(self: "Kernle", summary: Optional[str] = None) -> Dict[str, Any]:
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
                    "Anxiety level hit critical - triggered emergency save",
                    f"Saved checkpoint: {results['checkpoint_saved']}",
                    f"Consolidated {results['episodes_consolidated']} episodes",
                ],
                tags=["emergency", "anxiety", "critical"],
            )
        except Exception as e:
            results["errors"].append(f"Episode recording failed: {str(e)}")

        results["success"] = len(results["errors"]) == 0
        return results
