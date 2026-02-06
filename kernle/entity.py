"""
Entity — the Core coordinator/bus for kernle.

Entity implements CoreProtocol. It manages stack composition,
plugin lifecycle, model binding, and routes memory operations
to the active stack with provenance enforcement.

The entity is not the agent. The entity is the composition —
no single component IS the entity.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kernle.discovery import discover_plugins
from kernle.protocols import (
    Binding,
    InferenceService,
    ModelProtocol,
    NoActiveStackError,
    PluginHealth,
    PluginInfo,
    PluginProtocol,
    SearchResult,
    StackInfo,
    StackProtocol,
    SyncResult,
)
from kernle.types import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    RawEntry,
    Relationship,
    TrustAssessment,
    Value,
)
from kernle.utils import get_kernle_home

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return str(uuid.uuid4())


class _PluginContextImpl:
    """Concrete PluginContext that mediates access from a plugin to the core.

    All memory writes are attributed with source=f"plugin:{plugin_name}".
    Read operations return empty results if no active stack.
    """

    def __init__(self, entity: Entity, plugin_name: str) -> None:
        self._entity = entity
        self._plugin_name = plugin_name

    @property
    def core_id(self) -> str:
        return self._entity.core_id

    @property
    def active_stack_id(self) -> Optional[str]:
        stack = self._entity.active_stack
        return stack.stack_id if stack else None

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    def episode(
        self,
        objective: str,
        outcome: str,
        *,
        lessons: Optional[list[str]] = None,
        repeat: Optional[list[str]] = None,
        avoid: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.episode(
            objective,
            outcome,
            lessons=lessons,
            tags=tags,
            source=f"plugin:{self._plugin_name}",
            context=context,
        )

    def belief(
        self,
        statement: str,
        *,
        belief_type: str = "fact",
        confidence: float = 0.8,
        context: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.belief(
            statement,
            type=belief_type,
            confidence=confidence,
            source=f"plugin:{self._plugin_name}",
            context=context,
        )

    def value(
        self,
        name: str,
        statement: str,
        *,
        priority: int = 50,
        context: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.value(
            name,
            statement,
            priority=priority,
            context=context,
        )

    def goal(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        goal_type: str = "task",
        priority: str = "medium",
        context: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.goal(
            title,
            description=description,
            goal_type=goal_type,
            priority=priority,
            context=context,
        )

    def note(
        self,
        content: str,
        *,
        note_type: str = "note",
        tags: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.note(
            content,
            type=note_type,
            tags=tags,
            source=f"plugin:{self._plugin_name}",
            context=context,
        )

    def relationship(
        self,
        other_entity_id: str,
        *,
        trust_level: Optional[float] = None,
        interaction_type: Optional[str] = None,
        notes: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.relationship(
            other_entity_id,
            trust_level=trust_level,
            notes=notes,
            interaction_type=interaction_type,
            entity_type=entity_type,
        )

    def raw(
        self,
        content: str,
        *,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.raw(
            content,
            tags=tags,
            source=f"plugin:{self._plugin_name}",
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> list[SearchResult]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return stack.search(query, limit=limit, record_types=record_types, context=context)

    def get_relationships(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_trust: Optional[float] = None,
    ) -> list[Relationship]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return stack.get_relationships(
            entity_id=entity_id, entity_type=entity_type, min_trust=min_trust
        )

    def get_goals(
        self,
        *,
        status: Optional[str] = None,
        context: Optional[str] = None,
    ) -> list[Goal]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return stack.get_goals(status=status, context=context)

    def trust_set(
        self,
        entity: str,
        domain: str,
        score: float,
        *,
        evidence: Optional[str] = None,
    ) -> Optional[str]:
        stack = self._entity.active_stack
        if stack is None:
            return None
        return self._entity.trust_set(entity, domain, score, evidence=evidence)

    def trust_get(
        self,
        entity: str,
        *,
        domain: Optional[str] = None,
    ) -> list[TrustAssessment]:
        stack = self._entity.active_stack
        if stack is None:
            return []
        return self._entity.trust_get(entity, domain=domain)

    def get_data_dir(self) -> Path:
        data_dir = self._entity._data_dir / "plugins" / self._plugin_name / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def get_config(self, key: str, default: Any = None) -> Any:
        config = self._entity._plugin_configs.get(self._plugin_name, {})
        return config.get(key, default)

    def get_secret(self, key: str) -> Optional[str]:
        secrets = self._entity._plugin_secrets.get(self._plugin_name, {})
        return secrets.get(key)


class Entity:
    """The Core coordinator/bus — implements CoreProtocol.

    Entity manages stack composition, plugin lifecycle, model binding,
    and routes memory operations to the active stack with provenance
    enforcement.
    """

    def __init__(
        self,
        core_id: str,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._core_id = core_id
        self._data_dir = data_dir or get_kernle_home()
        self._model: Optional[ModelProtocol] = None
        self._stacks: dict[str, StackProtocol] = {}
        self._active_stack_alias: Optional[str] = None
        self._plugins: dict[str, PluginProtocol] = {}
        self._plugin_contexts: dict[str, _PluginContextImpl] = {}
        self._plugin_configs: dict[str, dict[str, Any]] = {}
        self._plugin_secrets: dict[str, dict[str, str]] = {}

    # ---- Core Properties ----

    @property
    def core_id(self) -> str:
        return self._core_id

    @property
    def model(self) -> Optional[ModelProtocol]:
        return self._model

    def set_model(self, model: ModelProtocol) -> None:
        self._model = model
        inference = self._get_inference_service()
        for stack in self._stacks.values():
            stack.on_model_changed(inference)

    @property
    def active_stack(self) -> Optional[StackProtocol]:
        if self._active_stack_alias:
            return self._stacks.get(self._active_stack_alias)
        return None

    @property
    def stacks(self) -> dict[str, StackInfo]:
        result: dict[str, StackInfo] = {}
        for alias, stack in self._stacks.items():
            result[alias] = StackInfo(
                stack_id=stack.stack_id,
                alias=alias,
                schema_version=stack.schema_version,
                stats=stack.get_stats(),
                is_active=alias == self._active_stack_alias,
            )
        return result

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        result: dict[str, PluginInfo] = {}
        for name, plugin in self._plugins.items():
            result[name] = PluginInfo(
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                capabilities=plugin.capabilities(),
                is_loaded=True,
            )
        return result

    # ---- Stack Management ----

    def attach_stack(
        self,
        stack: StackProtocol,
        *,
        alias: Optional[str] = None,
        set_active: bool = True,
    ) -> str:
        resolved_alias = alias or stack.stack_id
        self._stacks[resolved_alias] = stack
        stack.on_attach(self._core_id, self._get_inference_service())
        if set_active:
            self._active_stack_alias = resolved_alias
        return resolved_alias

    def detach_stack(self, alias: str) -> None:
        stack = self._stacks.pop(alias, None)
        if stack:
            stack.on_detach(self._core_id)
        if self._active_stack_alias == alias:
            self._active_stack_alias = None

    def set_active_stack(self, alias: str) -> None:
        if alias not in self._stacks:
            raise ValueError(f"No stack with alias '{alias}'")
        self._active_stack_alias = alias

    # ---- Plugin Management ----

    def load_plugin(self, plugin: PluginProtocol) -> None:
        context = _PluginContextImpl(self, plugin.name)
        plugin.activate(context)
        self._plugins[plugin.name] = plugin
        self._plugin_contexts[plugin.name] = context

    def unload_plugin(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.deactivate()
        self._plugin_contexts.pop(name, None)

    def discover_plugins(self) -> list[PluginInfo]:
        discovered = discover_plugins()
        loaded_names = set(self._plugins.keys())
        result: list[PluginInfo] = []
        for comp in discovered:
            result.append(
                PluginInfo(
                    name=comp.name,
                    version=comp.dist_version or "unknown",
                    description="",
                    is_loaded=comp.name in loaded_names,
                )
            )
        return result

    # ---- Routed Memory Operations (Provenance Enforcement) ----

    def _require_active_stack(self) -> StackProtocol:
        stack = self.active_stack
        if stack is None:
            raise NoActiveStackError("No active stack attached")
        return stack

    def episode(
        self,
        objective: str,
        outcome: str,
        *,
        lessons: Optional[list[str]] = None,
        repeat: Optional[list[str]] = None,
        avoid: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
    ) -> str:
        stack = self._require_active_stack()
        ep = Episode(
            id=_generate_id(),
            stack_id=stack.stack_id,
            objective=objective,
            outcome=outcome,
            lessons=lessons,
            tags=tags,
            created_at=datetime.now(timezone.utc),
            source_type="direct_experience",
            derived_from=derived_from,
            context=context,
            context_tags=context_tags,
        )
        if source:
            ep.source_entity = source
        else:
            ep.source_entity = f"core:{self._core_id}"
        return stack.save_episode(ep)

    def belief(
        self,
        statement: str,
        *,
        type: str = "fact",
        confidence: float = 0.8,
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        derived_from: Optional[list[str]] = None,
    ) -> str:
        stack = self._require_active_stack()
        b = Belief(
            id=_generate_id(),
            stack_id=stack.stack_id,
            statement=statement,
            belief_type=type,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
            is_protected=foundational,
            source_type="direct_experience",
            derived_from=derived_from,
            context=context,
            context_tags=context_tags,
        )
        if source:
            b.source_entity = source
        else:
            b.source_entity = f"core:{self._core_id}"
        return stack.save_belief(b)

    def value(
        self,
        name: str,
        statement: str,
        *,
        priority: int = 50,
        type: str = "core_value",
        foundational: bool = False,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
    ) -> str:
        stack = self._require_active_stack()
        v = Value(
            id=_generate_id(),
            stack_id=stack.stack_id,
            name=name,
            statement=statement,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            is_protected=foundational,
            source_type="direct_experience",
            context=context,
            context_tags=context_tags,
        )
        return stack.save_value(v)

    def goal(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        goal_type: str = "task",
        priority: str = "medium",
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
    ) -> str:
        stack = self._require_active_stack()
        g = Goal(
            id=_generate_id(),
            stack_id=stack.stack_id,
            title=title,
            description=description,
            goal_type=goal_type,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            source_type="direct_experience",
            context=context,
            context_tags=context_tags,
        )
        return stack.save_goal(g)

    def note(
        self,
        content: str,
        *,
        type: str = "note",
        speaker: Optional[str] = None,
        reason: Optional[str] = None,
        tags: Optional[list[str]] = None,
        protect: bool = False,
        derived_from: Optional[list[str]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
    ) -> str:
        stack = self._require_active_stack()
        n = Note(
            id=_generate_id(),
            stack_id=stack.stack_id,
            content=content,
            note_type=type,
            speaker=speaker,
            reason=reason,
            tags=tags,
            created_at=datetime.now(timezone.utc),
            is_protected=protect,
            source_type="direct_experience",
            derived_from=derived_from,
            context=context,
            context_tags=context_tags,
        )
        if source:
            n.source_entity = source
        else:
            n.source_entity = f"core:{self._core_id}"
        return stack.save_note(n)

    def drive(
        self,
        drive_type: str,
        *,
        intensity: float = 0.5,
        focus_areas: Optional[list[str]] = None,
        decay_hours: int = 24,
        context: Optional[str] = None,
        context_tags: Optional[list[str]] = None,
    ) -> str:
        stack = self._require_active_stack()
        d = Drive(
            id=_generate_id(),
            stack_id=stack.stack_id,
            drive_type=drive_type,
            intensity=intensity,
            focus_areas=focus_areas,
            created_at=datetime.now(timezone.utc),
            source_type="direct_experience",
            context=context,
            context_tags=context_tags,
        )
        return stack.save_drive(d)

    def relationship(
        self,
        other_stack_id: str,
        *,
        trust_level: Optional[float] = None,
        notes: Optional[str] = None,
        interaction_type: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> str:
        stack = self._require_active_stack()
        r = Relationship(
            id=_generate_id(),
            stack_id=stack.stack_id,
            entity_name=other_stack_id,
            entity_type=entity_type or "unknown",
            relationship_type=interaction_type or "known",
            notes=notes,
            sentiment=trust_level if trust_level is not None else 0.0,
            created_at=datetime.now(timezone.utc),
            source_type="direct_experience",
        )
        return stack.save_relationship(r)

    def raw(
        self,
        content: str,
        *,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> str:
        stack = self._require_active_stack()
        r = RawEntry(
            id=_generate_id(),
            stack_id=stack.stack_id,
            blob=content,
            captured_at=datetime.now(timezone.utc),
            source=source or f"core:{self._core_id}",
            tags=tags,
        )
        return stack.save_raw(r)

    # ---- Routed Search & Load ----

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> list[SearchResult]:
        stack = self._require_active_stack()
        return stack.search(query, limit=limit, record_types=record_types, context=context)

    def load(
        self,
        *,
        token_budget: int = 8000,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        stack = self._require_active_stack()
        result = stack.load(token_budget=token_budget, context=context)
        for plugin in self._plugins.values():
            try:
                plugin.on_load(result)
            except Exception:
                logger.exception("Plugin %s failed on_load", plugin.name)
        return result

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "core_id": self._core_id,
            "model": self._model.model_id if self._model else None,
            "stacks": {},
            "plugins": {},
        }
        for alias, stack in self._stacks.items():
            result["stacks"][alias] = {
                "stack_id": stack.stack_id,
                "active": alias == self._active_stack_alias,
                "stats": stack.get_stats(),
            }
        for name, plugin in self._plugins.items():
            try:
                health = plugin.health_check()
            except Exception:
                health = PluginHealth(healthy=False, message="health_check failed")
            result["plugins"][name] = {
                "version": plugin.version,
                "health": {"healthy": health.healthy, "message": health.message},
            }
        for plugin in self._plugins.values():
            try:
                plugin.on_status(result)
            except Exception:
                logger.exception("Plugin %s failed on_status", plugin.name)
        return result

    # ---- Routed Trust ----

    def trust_set(
        self,
        entity: str,
        domain: str,
        score: float,
        *,
        evidence: Optional[str] = None,
    ) -> str:
        stack = self._require_active_stack()
        assessment = TrustAssessment(
            id=_generate_id(),
            stack_id=stack.stack_id,
            entity=entity,
            dimensions={domain: {"score": score}},
            evidence_episode_ids=[evidence] if evidence else None,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        return stack.save_trust_assessment(assessment)

    def trust_get(
        self,
        entity: str,
        *,
        domain: Optional[str] = None,
    ) -> list[TrustAssessment]:
        stack = self._require_active_stack()
        return stack.get_trust_assessments(entity_id=entity, domain=domain)

    def trust_list(
        self,
        *,
        domain: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[TrustAssessment]:
        stack = self._require_active_stack()
        assessments = stack.get_trust_assessments(domain=domain)
        if min_score is not None:
            filtered = []
            for a in assessments:
                for dim_data in a.dimensions.values():
                    if isinstance(dim_data, dict) and dim_data.get("score", 0) >= min_score:
                        filtered.append(a)
                        break
            return filtered
        return assessments

    # ---- Routed Sync ----

    def sync(self) -> SyncResult:
        stack = self._require_active_stack()
        return stack.sync()

    def checkpoint(self, message: str = "") -> str:
        self._require_active_stack()
        checkpoint_dir = self._data_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cp_id = f"{self._core_id}_{ts}"
        cp_path = checkpoint_dir / f"{cp_id}.json"
        binding = self.get_binding()
        data = {
            "checkpoint_id": cp_id,
            "message": message,
            "binding": {
                "core_id": binding.core_id,
                "model_config": binding.model_config,
                "stacks": binding.stacks,
                "active_stack_alias": binding.active_stack_alias,
                "plugins": binding.plugins,
            },
            "created_at": _now_iso(),
        }
        cp_path.write_text(json.dumps(data, indent=2))
        return cp_id

    # ---- Binding Management ----

    def get_binding(self) -> Binding:
        stack_map: dict[str, str] = {}
        for alias, stack in self._stacks.items():
            stack_map[alias] = stack.stack_id
        return Binding(
            core_id=self._core_id,
            model_config={"model_id": self._model.model_id} if self._model else {},
            stacks=stack_map,
            active_stack_alias=self._active_stack_alias,
            plugins=list(self._plugins.keys()),
            created_at=datetime.now(timezone.utc),
        )

    def save_binding(self, path: Optional[Path] = None) -> Path:
        binding = self.get_binding()
        if path is None:
            bindings_dir = self._data_dir / "bindings"
            bindings_dir.mkdir(parents=True, exist_ok=True)
            path = bindings_dir / f"{self._core_id}.json"
        data = {
            "core_id": binding.core_id,
            "model_config": binding.model_config,
            "stacks": binding.stacks,
            "active_stack_alias": binding.active_stack_alias,
            "plugins": binding.plugins,
            "created_at": binding.created_at.isoformat() if binding.created_at else None,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def from_binding(cls, binding: Binding | Path) -> Entity:
        if isinstance(binding, Path):
            data = json.loads(binding.read_text())
            binding = Binding(
                core_id=data["core_id"],
                model_config=data.get("model_config", {}),
                stacks=data.get("stacks", {}),
                active_stack_alias=data.get("active_stack_alias"),
                plugins=data.get("plugins", []),
            )
        return cls(core_id=binding.core_id)

    # ---- Internal Helpers ----

    def _get_inference_service(self) -> Optional[InferenceService]:
        # Full InferenceService implementation deferred to v0.5.0.
        # For now, return None. Stacks degrade gracefully without it.
        return None
