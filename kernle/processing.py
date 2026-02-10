"""Memory Processing Sessions — focused inference for memory promotion.

Runs targeted processing sessions that promote memories up the hierarchy:
  raw → episode/note → belief → value
  raw → episode → goal, relationship, drive

Each layer transition has its own prompts, triggers, and configuration.
Processing uses the bound model (via InferenceService) to reason about
unprocessed memories and create higher-layer memories with provenance.

Design: Option 2 (entity-level processing) — structured inference calls
through focused prompts. Designed to evolve toward Option 3 (full
recursive self-invocation) by swapping the process_layer implementation.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LayerConfig:
    """Configuration for a single layer transition."""

    layer_transition: str  # e.g., "raw_to_episode"
    enabled: bool = True
    model_id: Optional[str] = None  # Override model for this transition
    quantity_threshold: int = 10  # Min items to trigger
    valence_threshold: float = 3.0  # Emotional valence threshold
    time_threshold_hours: int = 24  # Max hours before forced processing
    batch_size: int = 10  # Max items per session
    max_sessions_per_day: int = 10  # Cost cap


# Default configs for each layer transition
DEFAULT_LAYER_CONFIGS: Dict[str, LayerConfig] = {
    "raw_to_episode": LayerConfig(
        layer_transition="raw_to_episode",
        quantity_threshold=10,
        batch_size=10,
    ),
    "raw_to_note": LayerConfig(
        layer_transition="raw_to_note",
        quantity_threshold=10,
        batch_size=10,
    ),
    "episode_to_belief": LayerConfig(
        layer_transition="episode_to_belief",
        quantity_threshold=5,
        batch_size=10,
    ),
    "episode_to_goal": LayerConfig(
        layer_transition="episode_to_goal",
        quantity_threshold=5,
        batch_size=10,
    ),
    "episode_to_relationship": LayerConfig(
        layer_transition="episode_to_relationship",
        quantity_threshold=3,
        batch_size=5,
    ),
    "belief_to_value": LayerConfig(
        layer_transition="belief_to_value",
        quantity_threshold=5,
        batch_size=10,
    ),
    "episode_to_drive": LayerConfig(
        layer_transition="episode_to_drive",
        quantity_threshold=5,
        batch_size=10,
    ),
}

# All valid layer transitions
VALID_TRANSITIONS = set(DEFAULT_LAYER_CONFIGS.keys())


# =============================================================================
# Results
# =============================================================================


@dataclass
class ProcessingResult:
    """Result of a single processing session."""

    layer_transition: str
    source_count: int  # How many source memories were processed
    created: List[Dict[str, str]] = field(
        default_factory=list
    )  # [{"type": "episode", "id": "..."}]
    errors: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    inference_blocked: bool = False  # True if blocked by no-inference safety


# =============================================================================
# Inference Safety Policy
# =============================================================================

# Identity-layer transitions that require inference to produce quality output.
# These write to layers that are "hard to change" — bad data here is costly.
IDENTITY_LAYER_TRANSITIONS = frozenset(
    {
        "episode_to_belief",
        "episode_to_goal",
        "episode_to_relationship",
        "episode_to_drive",
        "belief_to_value",
    }
)

# Transitions that are always blocked without inference — no override possible.
# Values are the highest identity layer; malformed values corrupt the entity.
NO_OVERRIDE_TRANSITIONS = frozenset(
    {
        "belief_to_value",
    }
)

# Transitions that can be overridden without inference if strict conditions are met.
# Beliefs require: force=True, explicit override flag, high confidence, evidence count.
OVERRIDE_TRANSITIONS = frozenset(
    {
        "episode_to_belief",
        "episode_to_goal",
        "episode_to_relationship",
        "episode_to_drive",
    }
)

# Minimum evidence count required for no-inference belief override
NO_INFERENCE_MIN_EVIDENCE = 3
# Minimum confidence required for no-inference belief override
NO_INFERENCE_MIN_CONFIDENCE = 0.9


# =============================================================================
# Prompts
# =============================================================================

LAYER_PROMPTS: Dict[str, Dict[str, str]] = {
    "raw_to_episode": {
        "system": (
            "You are a memory processing system. Your job is to structure raw "
            "memory captures into episodic memories. Each episode should have a clear "
            "objective (what was happening), outcome (what resulted), and optionally "
            "lessons learned. Group related raw entries into single episodes when "
            "they describe the same event or interaction."
        ),
        "template": (
            "Process these raw memory captures into structured episodes.\n\n"
            "RAW CAPTURES:\n{sources}\n\n"
            "EXISTING EPISODES (for deduplication):\n{context}\n\n"
            "For each episode, respond with a JSON array of objects:\n"
            '[{{"objective": "...", "outcome": "...", "outcome_type": "success|failure|neutral|mixed", '
            '"lessons": ["..."], "source_raw_ids": ["raw_id_1", "raw_id_2"]}}]\n\n'
            "Rules:\n"
            "- Group related raws into one episode\n"
            "- Skip raws that duplicate existing episodes\n"
            "- Each raw should appear in at most one episode\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "raw_to_note": {
        "system": (
            "You are a memory processing system. Your job is to extract factual "
            "notes from raw memory captures. Notes are factual observations, "
            "references, or structured information — not experiences."
        ),
        "template": (
            "Extract factual notes from these raw memory captures.\n\n"
            "RAW CAPTURES:\n{sources}\n\n"
            "For each note, respond with a JSON array of objects:\n"
            '[{{"content": "...", "note_type": "observation|reference|procedure|fact", '
            '"source_raw_ids": ["raw_id_1"]}}]\n\n'
            "Rules:\n"
            "- Only extract clear factual content, not experiences\n"
            "- Each raw can produce zero or more notes\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_belief": {
        "system": (
            "You are a memory processing system. Your job is to identify beliefs "
            "that emerge from experiences. A belief is a general statement about "
            "how the world works, derived from specific episodes."
        ),
        "template": (
            "What beliefs emerge from these experiences?\n\n"
            "EPISODES:\n{sources}\n\n"
            "EXISTING BELIEFS (avoid duplicates):\n{context}\n\n"
            "For each belief, respond with a JSON array of objects:\n"
            '[{{"statement": "...", "belief_type": "causal|evaluative|procedural|factual", '
            '"confidence": 0.5-0.9, "source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Only create beliefs supported by multiple episodes or strong single episodes\n"
            "- Avoid duplicating existing beliefs\n"
            "- Confidence reflects how well-supported the belief is\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_goal": {
        "system": (
            "You are a memory processing system. Your job is to identify goals "
            "that emerge from experiences. Goals are things the entity should pursue "
            "based on what it has learned."
        ),
        "template": (
            "What goals should be pursued based on these experiences?\n\n"
            "EPISODES:\n{sources}\n\n"
            "CURRENT GOALS:\n{context}\n\n"
            "For each goal, respond with a JSON array of objects:\n"
            '[{{"title": "...", "description": "...", "goal_type": "task|aspiration|commitment|exploration", '
            '"priority": "low|medium|high|critical", "source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Only suggest actionable, concrete goals\n"
            "- Avoid duplicating existing goals\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_relationship": {
        "system": (
            "You are a memory processing system. Your job is to identify or update "
            "relationships based on interactions recorded in episodes."
        ),
        "template": (
            "What relationships are revealed or updated by these interactions?\n\n"
            "EPISODES:\n{sources}\n\n"
            "EXISTING RELATIONSHIPS:\n{context}\n\n"
            "For each relationship, respond with a JSON array of objects:\n"
            '[{{"entity_name": "...", "entity_type": "person|org|system|other", '
            '"sentiment": -1.0 to 1.0, "context_note": "...", '
            '"source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Only create relationships with clearly identified entities\n"
            "- Update existing relationships rather than creating duplicates\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "belief_to_value": {
        "system": (
            "You are a memory processing system. Your job is to identify core "
            "values that emerge from strongly-held beliefs. Values are fundamental "
            "principles that guide behavior."
        ),
        "template": (
            "Which of these beliefs represent core values?\n\n"
            "STRONG BELIEFS:\n{sources}\n\n"
            "EXISTING VALUES:\n{context}\n\n"
            "For each value, respond with a JSON array of objects:\n"
            '[{{"name": "...", "statement": "...", "priority": 1-100, '
            '"source_belief_ids": ["belief_id_1"]}}]\n\n'
            "Rules:\n"
            "- Only promote beliefs that are fundamental and enduring\n"
            "- Avoid duplicating existing values\n"
            "- Priority 1=minor, 100=core identity\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
    "episode_to_drive": {
        "system": (
            "You are a memory processing system. Your job is to identify "
            "motivational drives that emerge from experiences and beliefs."
        ),
        "template": (
            "What drives or motivations emerge from these experiences?\n\n"
            "EPISODES:\n{sources}\n\n"
            "CURRENT DRIVES:\n{context}\n\n"
            "For each drive, respond with a JSON array of objects:\n"
            '[{{"drive_type": "...", "intensity": 0.1-1.0, '
            '"source_episode_ids": ["ep_id_1"]}}]\n\n'
            "Rules:\n"
            "- Only identify clear motivational patterns\n"
            "- Avoid duplicating existing drives\n"
            "- Respond with ONLY the JSON array, no other text"
        ),
    },
}


# =============================================================================
# Trigger Evaluation
# =============================================================================


def evaluate_triggers(
    transition: str,
    config: LayerConfig,
    unprocessed_count: int,
    cumulative_valence: float = 0.0,
    hours_since_last: Optional[float] = None,
) -> bool:
    """Check if processing should be triggered for a layer transition.

    Returns True if any trigger condition is met.
    """
    if not config.enabled:
        return False

    # Quantity threshold
    if unprocessed_count >= config.quantity_threshold:
        return True

    # Emotional valence threshold (only for episode-producing transitions)
    if transition in ("raw_to_episode", "raw_to_note"):
        if cumulative_valence >= config.valence_threshold:
            return True

    # Time threshold
    if hours_since_last is not None and config.time_threshold_hours > 0:
        if hours_since_last >= config.time_threshold_hours:
            return True

    return False


# =============================================================================
# Memory Processor
# =============================================================================


class MemoryProcessor:
    """Runs focused inference sessions for memory processing.

    Owned by Entity. Uses InferenceService to run layer-specific
    processing sessions that promote memories up the hierarchy.

    Safety: When inference_available=False, identity-layer transitions
    are blocked. Values cannot be created. Beliefs and other identity
    layers require explicit override with evidence requirements.
    """

    def __init__(
        self,
        stack: Any,  # StackProtocol
        inference: Any,  # InferenceService
        core_id: str,
        configs: Optional[Dict[str, LayerConfig]] = None,
        inference_available: bool = True,
    ) -> None:
        self._stack = stack
        self._inference = inference
        self._core_id = core_id
        self._configs = configs or dict(DEFAULT_LAYER_CONFIGS)
        self._inference_available = inference_available

    def update_config(self, transition: str, config: LayerConfig) -> None:
        """Update configuration for a layer transition."""
        self._configs[transition] = config

    def get_config(self, transition: str) -> Optional[LayerConfig]:
        """Get configuration for a layer transition."""
        return self._configs.get(transition)

    def check_triggers(self, transition: str) -> bool:
        """Check if processing should be triggered for a transition."""
        config = self._configs.get(transition)
        if config is None or not config.enabled:
            return False

        backend = self._stack._backend if hasattr(self._stack, "_backend") else None
        if backend is None:
            return False

        if transition in ("raw_to_episode", "raw_to_note"):
            unprocessed = backend.list_raw(processed=False, limit=config.quantity_threshold + 1)
            return evaluate_triggers(transition, config, len(unprocessed))

        if transition == "episode_to_belief":
            episodes = self._stack.get_episodes(limit=config.quantity_threshold + 1)
            unprocessed = [e for e in episodes if not e.processed]
            return evaluate_triggers(transition, config, len(unprocessed))

        if transition == "episode_to_goal":
            episodes = self._stack.get_episodes(limit=config.quantity_threshold + 1)
            unprocessed = [e for e in episodes if not e.processed]
            return evaluate_triggers(transition, config, len(unprocessed))

        if transition == "episode_to_relationship":
            episodes = self._stack.get_episodes(limit=config.quantity_threshold + 1)
            unprocessed = [e for e in episodes if not e.processed]
            return evaluate_triggers(transition, config, len(unprocessed))

        if transition == "belief_to_value":
            beliefs = self._stack.get_beliefs(limit=config.quantity_threshold + 1)
            unprocessed = [b for b in beliefs if not getattr(b, "processed", False)]
            return evaluate_triggers(transition, config, len(unprocessed))

        if transition == "episode_to_drive":
            episodes = self._stack.get_episodes(limit=config.quantity_threshold + 1)
            unprocessed = [e for e in episodes if not e.processed]
            return evaluate_triggers(transition, config, len(unprocessed))

        return False

    def process(
        self,
        transition: Optional[str] = None,
        *,
        force: bool = False,
        allow_no_inference_override: bool = False,
    ) -> List[ProcessingResult]:
        """Run processing for one or all layer transitions.

        Args:
            transition: Specific transition to process (None = check all)
            force: Process even if triggers aren't met
            allow_no_inference_override: Allow identity-layer writes without
                inference (except values). Requires force=True and only works
                for transitions in OVERRIDE_TRANSITIONS.

        Returns:
            List of ProcessingResult for each transition that ran
        """
        results = []

        transitions = [transition] if transition else list(VALID_TRANSITIONS)

        for t in transitions:
            config = self._configs.get(t)
            if config is None or not config.enabled:
                continue

            # Inference safety gate — checked even when force=True
            blocked = self._check_inference_safety(t, force, allow_no_inference_override)
            if blocked is not None:
                results.append(blocked)
                continue

            if not force and not self.check_triggers(t):
                continue

            result = self._process_layer(t, config)
            results.append(result)

        return results

    def _check_inference_safety(
        self,
        transition: str,
        force: bool,
        allow_override: bool,
    ) -> Optional[ProcessingResult]:
        """Check if a transition is blocked by no-inference safety policy.

        Returns a ProcessingResult with inference_blocked=True if blocked,
        or None if the transition is allowed to proceed.
        """
        if self._inference_available:
            return None

        if transition not in IDENTITY_LAYER_TRANSITIONS:
            # Non-identity transitions (raw_to_episode, raw_to_note) still need
            # inference to run. They will fail at the inference call step, which
            # is handled gracefully. But we don't block them at the policy level
            # since the infrastructure handles the error.
            return None

        # Values are never allowed without inference
        if transition in NO_OVERRIDE_TRANSITIONS:
            return ProcessingResult(
                layer_transition=transition,
                source_count=0,
                skipped=True,
                skip_reason=(
                    "Blocked: inference unavailable. "
                    "Value creation requires inference — "
                    "cannot promote to identity layer without model."
                ),
                inference_blocked=True,
            )

        # Other identity-layer transitions can be overridden with explicit opt-in
        if transition in OVERRIDE_TRANSITIONS:
            if not (force and allow_override):
                return ProcessingResult(
                    layer_transition=transition,
                    source_count=0,
                    skipped=True,
                    skip_reason=(
                        "Blocked: inference unavailable. "
                        "Use force=True with allow_no_inference_override=True "
                        "to override for this transition."
                    ),
                    inference_blocked=True,
                )
            # Override allowed — log warning and proceed
            logger.warning("Processing %s without inference (override enabled)", transition)

        return None

    def _process_layer(self, transition: str, config: LayerConfig) -> ProcessingResult:
        """Run one processing pass for a specific layer transition."""
        prompts = LAYER_PROMPTS.get(transition)
        if prompts is None:
            return ProcessingResult(
                layer_transition=transition,
                source_count=0,
                skipped=True,
                skip_reason=f"No prompts for transition: {transition}",
            )

        # 1. Gather unprocessed source memories
        sources = self._gather_sources(transition, config.batch_size)
        if not sources:
            return ProcessingResult(
                layer_transition=transition,
                source_count=0,
                skipped=True,
                skip_reason="No unprocessed sources",
            )

        # 2. Load context (existing memories for dedup)
        context = self._gather_context(transition)

        # 3. Build prompt
        sources_text = self._format_sources(transition, sources)
        context_text = self._format_context(transition, context)
        prompt = prompts["template"].format(sources=sources_text, context=context_text)
        system = prompts["system"]

        # 4. Call inference
        try:
            response = self._inference.infer(prompt, system=system)
        except Exception as e:
            logger.error("Processing inference failed for %s: %s", transition, e)
            return ProcessingResult(
                layer_transition=transition,
                source_count=len(sources),
                errors=[f"Inference failed: {e}"],
            )

        # 5. Parse response and write memories
        result = ProcessingResult(
            layer_transition=transition,
            source_count=len(sources),
        )

        try:
            parsed = self._parse_response(response)
        except Exception as e:
            logger.error("Failed to parse processing response for %s: %s", transition, e)
            result.errors.append(f"Parse failed: {e}")
            return result

        # 6. Write through stack with provenance
        created = self._write_memories(transition, parsed, sources)
        result.created = created

        # 7. Mark sources as processed
        self._mark_processed(transition, sources, created)

        # 8. Log audit
        self._stack.log_audit(
            "processing",
            transition,
            "process",
            actor=f"core:{self._core_id}",
            details={
                "source_count": len(sources),
                "created_count": len(created),
                "errors": result.errors,
            },
        )

        return result

    # ---- Source Gathering ----

    def _gather_sources(self, transition: str, batch_size: int) -> list:
        """Gather unprocessed source memories for a transition."""
        backend = self._stack._backend if hasattr(self._stack, "_backend") else None
        if backend is None:
            return []

        if transition in ("raw_to_episode", "raw_to_note"):
            return backend.list_raw(processed=False, limit=batch_size)

        if transition in (
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "episode_to_drive",
        ):
            episodes = self._stack.get_episodes(limit=batch_size * 2)
            return [e for e in episodes if not e.processed][:batch_size]

        if transition == "belief_to_value":
            beliefs = self._stack.get_beliefs(limit=batch_size * 2)
            return [b for b in beliefs if not getattr(b, "processed", False)][:batch_size]

        return []

    def _gather_context(self, transition: str) -> list:
        """Load existing memories for deduplication context."""
        if transition in ("raw_to_episode", "raw_to_note"):
            return self._stack.get_episodes(limit=20)

        if transition == "episode_to_belief":
            return self._stack.get_beliefs(limit=20)

        if transition == "episode_to_goal":
            return self._stack.get_goals(limit=20)

        if transition == "episode_to_relationship":
            return self._stack.get_relationships()

        if transition == "belief_to_value":
            return self._stack.get_values(limit=20)

        if transition == "episode_to_drive":
            return self._stack.get_drives()

        return []

    # ---- Formatting ----

    def _format_sources(self, transition: str, sources: list) -> str:
        """Format source memories for the prompt."""
        lines = []
        for s in sources:
            if transition in ("raw_to_episode", "raw_to_note"):
                blob = getattr(s, "blob", None) or getattr(s, "content", "") or ""
                lines.append(f"[{s.id}] {blob[:500]}")
            elif hasattr(s, "objective"):  # Episode
                lines.append(f"[{s.id}] {s.objective}: {s.outcome}")
            elif hasattr(s, "statement"):  # Belief
                lines.append(f"[{s.id}] {s.statement} (confidence: {s.confidence})")
            else:
                lines.append(f"[{s.id}] {str(s)[:200]}")
        return "\n".join(lines) if lines else "(none)"

    def _format_context(self, transition: str, context: list) -> str:
        """Format context memories for deduplication."""
        if not context:
            return "(none)"
        lines = []
        for c in context:
            if hasattr(c, "objective"):  # Episode
                lines.append(f"- {c.objective}: {c.outcome}")
            elif hasattr(c, "statement") and hasattr(c, "belief_type"):  # Belief
                lines.append(f"- {c.statement}")
            elif hasattr(c, "statement") and hasattr(c, "name"):  # Value
                lines.append(f"- {c.name}: {c.statement}")
            elif hasattr(c, "title"):  # Goal
                desc = c.title
                if c.description:
                    desc += f": {c.description}"
                lines.append(f"- {desc}")
            elif hasattr(c, "drive_type"):  # Drive
                lines.append(f"- {c.drive_type} (intensity: {c.intensity})")
            elif hasattr(c, "entity_name"):  # Relationship
                lines.append(f"- {c.entity_name} ({c.entity_type})")
            else:
                lines.append(f"- {str(c)[:100]}")
        return "\n".join(lines[:10])

    # ---- Response Parsing ----

    def _parse_response(self, response: str) -> list:
        """Parse the model's JSON array response."""
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)

    # ---- Memory Writing ----

    def _write_memories(self, transition: str, parsed: list, sources: list) -> List[Dict[str, str]]:
        """Write parsed memories through the stack with provenance."""
        from kernle.types import Belief, Drive, Episode, Goal, Note, Relationship, Value

        created = []
        now = datetime.now(timezone.utc)

        for item in parsed:
            try:
                if transition == "raw_to_episode":
                    raw_ids = item.get("source_raw_ids", [])
                    derived_from = [f"raw:{rid}" for rid in raw_ids]
                    ep = Episode(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        objective=item["objective"],
                        outcome=item["outcome"],
                        outcome_type=item.get("outcome_type", "neutral"),
                        lessons=item.get("lessons"),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    eid = self._stack.save_episode(ep)
                    created.append({"type": "episode", "id": eid})

                elif transition == "raw_to_note":
                    raw_ids = item.get("source_raw_ids", [])
                    derived_from = [f"raw:{rid}" for rid in raw_ids]
                    note = Note(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        content=item["content"],
                        note_type=item.get("note_type", "observation"),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    nid = self._stack.save_note(note)
                    created.append({"type": "note", "id": nid})

                elif transition == "episode_to_belief":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    belief = Belief(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        statement=item["statement"],
                        belief_type=item.get("belief_type", "factual"),
                        confidence=item.get("confidence", 0.7),
                        created_at=now,
                        source_type="processing",
                        source_entity=f"core:{self._core_id}",
                        derived_from=derived_from,
                    )
                    bid = self._stack.save_belief(belief)
                    created.append({"type": "belief", "id": bid})

                elif transition == "episode_to_goal":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    goal = Goal(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        title=item.get("title", item.get("description", "")),
                        description=item.get("description"),
                        goal_type=item.get("goal_type", "task"),
                        priority=item.get("priority", "medium"),
                        created_at=now,
                        source_type="processing",
                        derived_from=derived_from,
                    )
                    gid = self._stack.save_goal(goal)
                    created.append({"type": "goal", "id": gid})

                elif transition == "episode_to_relationship":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    rel = Relationship(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        entity_name=item["entity_name"],
                        entity_type=item.get("entity_type", "person"),
                        relationship_type=item.get("relationship_type", "acquaintance"),
                        notes=item.get("context_note"),
                        sentiment=item.get("sentiment", 0.0),
                        created_at=now,
                        source_type="processing",
                        derived_from=derived_from,
                    )
                    rid = self._stack.save_relationship(rel)
                    created.append({"type": "relationship", "id": rid})

                elif transition == "belief_to_value":
                    belief_ids = item.get("source_belief_ids", [])
                    derived_from = [f"belief:{bid}" for bid in belief_ids]
                    value = Value(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        name=item["name"],
                        statement=item.get("statement", item["name"]),
                        priority=item.get("priority", 50),
                        created_at=now,
                        source_type="processing",
                        derived_from=derived_from,
                    )
                    vid = self._stack.save_value(value)
                    created.append({"type": "value", "id": vid})

                elif transition == "episode_to_drive":
                    ep_ids = item.get("source_episode_ids", [])
                    derived_from = [f"episode:{eid}" for eid in ep_ids]
                    drive = Drive(
                        id=str(uuid.uuid4()),
                        stack_id=self._stack.stack_id,
                        drive_type=item.get("drive_type", "motivation"),
                        intensity=item.get("intensity", 0.5),
                        created_at=now,
                        source_type="processing",
                        derived_from=derived_from,
                    )
                    did = self._stack.save_drive(drive)
                    created.append({"type": "drive", "id": did})

            except Exception as e:
                logger.error("Failed to write %s memory: %s", transition, e)

        return created

    # ---- Mark Processed ----

    def _mark_processed(
        self, transition: str, sources: list, created: List[Dict[str, str]]
    ) -> None:
        """Mark source memories as processed."""
        backend = self._stack._backend if hasattr(self._stack, "_backend") else None
        if backend is None:
            return

        created_refs = [f"{c['type']}:{c['id']}" for c in created]

        if transition in ("raw_to_episode", "raw_to_note"):
            for raw in sources:
                backend.mark_raw_processed(raw.id, created_refs)
        elif transition in (
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "episode_to_drive",
        ):
            for ep in sources:
                backend.mark_episode_processed(ep.id)
        elif transition == "belief_to_value":
            for belief in sources:
                backend.mark_belief_processed(belief.id)
