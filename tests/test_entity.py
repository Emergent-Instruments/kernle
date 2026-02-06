"""Tests for kernle.entity.Entity — CoreProtocol implementation."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kernle.entity import Entity
from kernle.protocols import (
    Binding,
    NoActiveStackError,
    PluginHealth,
    PluginInfo,
    StackInfo,
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

# ---- Fixtures ----


def _make_mock_stack(stack_id="test-stack", schema_version=22):
    """Create a mock StackProtocol."""
    stack = MagicMock()
    type(stack).stack_id = PropertyMock(return_value=stack_id)
    type(stack).schema_version = PropertyMock(return_value=schema_version)
    stack.get_stats.return_value = {"episodes": 5, "beliefs": 3}
    stack.on_attach.return_value = None
    stack.on_detach.return_value = None
    stack.on_model_changed.return_value = None
    # Write ops return IDs
    stack.save_episode.side_effect = lambda ep: ep.id
    stack.save_belief.side_effect = lambda b: b.id
    stack.save_value.side_effect = lambda v: v.id
    stack.save_goal.side_effect = lambda g: g.id
    stack.save_note.side_effect = lambda n: n.id
    stack.save_drive.side_effect = lambda d: d.id
    stack.save_relationship.side_effect = lambda r: r.id
    stack.save_raw.side_effect = lambda r: r.id
    # Read ops
    stack.search.return_value = []
    stack.load.return_value = {"identity": {}, "beliefs": []}
    stack.sync.return_value = SyncResult()
    stack.get_trust_assessments.return_value = []
    stack.save_trust_assessment.side_effect = lambda a: a.id
    stack.get_relationships.return_value = []
    stack.get_goals.return_value = []
    return stack


def _make_mock_plugin(name="test-plugin", version="1.0.0"):
    """Create a mock PluginProtocol."""
    plugin = MagicMock()
    type(plugin).name = PropertyMock(return_value=name)
    type(plugin).version = PropertyMock(return_value=version)
    type(plugin).protocol_version = PropertyMock(return_value=1)
    type(plugin).description = PropertyMock(return_value=f"Test plugin: {name}")
    plugin.capabilities.return_value = ["testing"]
    plugin.activate.return_value = None
    plugin.deactivate.return_value = None
    plugin.health_check.return_value = PluginHealth(healthy=True, message="ok")
    plugin.on_load.return_value = None
    plugin.on_status.return_value = None
    plugin.register_cli.return_value = None
    plugin.register_tools.return_value = []
    return plugin


def _make_mock_model(model_id="test-model"):
    """Create a mock ModelProtocol."""
    model = MagicMock()
    type(model).model_id = PropertyMock(return_value=model_id)
    return model


@pytest.fixture
def entity(tmp_path):
    return Entity(core_id="test-core", data_dir=tmp_path)


@pytest.fixture
def stack():
    return _make_mock_stack()


@pytest.fixture
def plugin():
    return _make_mock_plugin()


# ---- Basic Properties ----


class TestEntityProperties:
    def test_core_id(self, entity):
        assert entity.core_id == "test-core"

    def test_model_initially_none(self, entity):
        assert entity.model is None

    def test_active_stack_initially_none(self, entity):
        assert entity.active_stack is None

    def test_stacks_initially_empty(self, entity):
        assert entity.stacks == {}

    def test_plugins_initially_empty(self, entity):
        assert entity.plugins == {}


# ---- Stack Management ----


class TestStackManagement:
    def test_attach_stack_default_alias(self, entity, stack):
        alias = entity.attach_stack(stack)
        assert alias == "test-stack"
        stack.on_attach.assert_called_once_with("test-core", None)

    def test_attach_stack_custom_alias(self, entity, stack):
        alias = entity.attach_stack(stack, alias="my-alias")
        assert alias == "my-alias"

    def test_attach_stack_sets_active(self, entity, stack):
        entity.attach_stack(stack)
        assert entity.active_stack is stack

    def test_attach_stack_no_set_active(self, entity, stack):
        entity.attach_stack(stack, set_active=False)
        assert entity.active_stack is None

    def test_stacks_property_returns_stack_info(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        stacks = entity.stacks
        assert "primary" in stacks
        info = stacks["primary"]
        assert isinstance(info, StackInfo)
        assert info.stack_id == "test-stack"
        assert info.is_active is True
        assert info.schema_version == 22
        assert info.stats == {"episodes": 5, "beliefs": 3}

    def test_detach_stack(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        entity.detach_stack("primary")
        assert entity.active_stack is None
        assert "primary" not in entity.stacks
        stack.on_detach.assert_called_once_with("test-core")

    def test_detach_nonexistent_stack_is_noop(self, entity):
        entity.detach_stack("nonexistent")  # Should not raise

    def test_detach_clears_active_if_matching(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        second = _make_mock_stack(stack_id="second-stack")
        entity.attach_stack(second, alias="secondary", set_active=False)
        entity.detach_stack("primary")
        assert entity.active_stack is None
        assert "secondary" in entity.stacks

    def test_set_active_stack(self, entity):
        s1 = _make_mock_stack(stack_id="s1")
        s2 = _make_mock_stack(stack_id="s2")
        entity.attach_stack(s1, alias="first")
        entity.attach_stack(s2, alias="second")
        assert entity.active_stack is s2  # Last attached is active
        entity.set_active_stack("first")
        assert entity.active_stack is s1

    def test_set_active_stack_invalid_alias_raises(self, entity):
        with pytest.raises(ValueError, match="No stack with alias"):
            entity.set_active_stack("nonexistent")

    def test_multiple_stacks(self, entity):
        s1 = _make_mock_stack(stack_id="s1")
        s2 = _make_mock_stack(stack_id="s2")
        entity.attach_stack(s1, alias="first", set_active=False)
        entity.attach_stack(s2, alias="second", set_active=True)
        stacks = entity.stacks
        assert len(stacks) == 2
        assert stacks["first"].is_active is False
        assert stacks["second"].is_active is True


# ---- Model Management ----


class TestModelManagement:
    def test_set_model(self, entity):
        model = _make_mock_model()
        entity.set_model(model)
        assert entity.model is model
        assert entity.model.model_id == "test-model"

    def test_set_model_notifies_stacks(self, entity, stack):
        entity.attach_stack(stack)
        model = _make_mock_model()
        entity.set_model(model)
        stack.on_model_changed.assert_called_once()

    def test_set_model_replaces_previous(self, entity):
        m1 = _make_mock_model(model_id="first")
        m2 = _make_mock_model(model_id="second")
        entity.set_model(m1)
        entity.set_model(m2)
        assert entity.model.model_id == "second"


# ---- Routed Memory Operations ----


class TestRoutedOperations:
    def test_episode_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        mem_id = entity.episode("test objective", "test outcome")
        assert mem_id is not None
        stack.save_episode.assert_called_once()
        ep = stack.save_episode.call_args[0][0]
        assert isinstance(ep, Episode)
        assert ep.objective == "test objective"
        assert ep.outcome == "test outcome"
        assert ep.stack_id == "test-stack"

    def test_episode_populates_provenance(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out", source="user:alice", context="project:foo")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_entity == "user:alice"
        assert ep.context == "project:foo"
        assert ep.created_at is not None

    def test_episode_default_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.episode("obj", "out")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_entity == "core:test-core"

    def test_belief_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        mem_id = entity.belief("the sky is blue", confidence=0.9)
        assert mem_id is not None
        b = stack.save_belief.call_args[0][0]
        assert isinstance(b, Belief)
        assert b.statement == "the sky is blue"
        assert b.confidence == 0.9

    def test_belief_default_source(self, entity, stack):
        entity.attach_stack(stack)
        entity.belief("test")
        b = stack.save_belief.call_args[0][0]
        assert b.source_entity == "core:test-core"

    def test_value_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.value("honesty", "I value truthfulness", priority=90)
        v = stack.save_value.call_args[0][0]
        assert isinstance(v, Value)
        assert v.name == "honesty"
        assert v.priority == 90

    def test_goal_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.goal("learn rust", description="Start with the book", goal_type="aspiration")
        g = stack.save_goal.call_args[0][0]
        assert isinstance(g, Goal)
        assert g.title == "learn rust"
        assert g.goal_type == "aspiration"

    def test_note_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.note("important thought", tags=["meta"])
        n = stack.save_note.call_args[0][0]
        assert isinstance(n, Note)
        assert n.content == "important thought"
        assert n.tags == ["meta"]

    def test_drive_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.drive("curiosity", intensity=0.8, focus_areas=["AI"])
        d = stack.save_drive.call_args[0][0]
        assert isinstance(d, Drive)
        assert d.drive_type == "curiosity"
        assert d.intensity == 0.8

    def test_relationship_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.relationship("other-entity", trust_level=0.7, entity_type="agent")
        r = stack.save_relationship.call_args[0][0]
        assert isinstance(r, Relationship)
        assert r.entity_name == "other-entity"
        assert r.entity_type == "agent"
        assert r.sentiment == 0.7

    def test_raw_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.raw("some unstructured content", tags=["dump"])
        r = stack.save_raw.call_args[0][0]
        assert isinstance(r, RawEntry)
        assert r.blob == "some unstructured content"
        assert r.source == "core:test-core"


# ---- NoActiveStackError ----


class TestNoActiveStack:
    def test_episode_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.episode("obj", "out")

    def test_belief_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.belief("test")

    def test_value_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.value("v", "s")

    def test_goal_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.goal("g")

    def test_note_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.note("n")

    def test_drive_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.drive("d")

    def test_relationship_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.relationship("other")

    def test_raw_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.raw("r")

    def test_search_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.search("query")

    def test_load_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.load()

    def test_trust_set_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.trust_set("entity", "domain", 0.5)

    def test_trust_get_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.trust_get("entity")

    def test_sync_raises_without_stack(self, entity):
        with pytest.raises(NoActiveStackError):
            entity.sync()


# ---- Search & Load ----


class TestSearchAndLoad:
    def test_search_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.search("test query", limit=5)
        stack.search.assert_called_once_with("test query", limit=5, record_types=None, context=None)

    def test_load_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        entity.load(token_budget=4000, context="project:x")
        stack.load.assert_called_once_with(token_budget=4000, context="project:x")

    def test_load_calls_plugin_on_load(self, entity, stack, plugin):
        entity.attach_stack(stack)
        entity.load_plugin(plugin)
        entity.load()
        plugin.on_load.assert_called_once()

    def test_load_plugin_error_does_not_crash(self, entity, stack):
        entity.attach_stack(stack)
        bad_plugin = _make_mock_plugin(name="bad-plugin")
        bad_plugin.on_load.side_effect = RuntimeError("boom")
        entity.load_plugin(bad_plugin)
        result = entity.load()  # Should not raise
        assert result is not None


# ---- Trust Operations ----


class TestTrustOperations:
    def test_trust_set(self, entity, stack):
        entity.attach_stack(stack)
        mem_id = entity.trust_set("alice", "general", 0.9, evidence="ep-123")
        assert mem_id is not None
        assessment = stack.save_trust_assessment.call_args[0][0]
        assert isinstance(assessment, TrustAssessment)
        assert assessment.entity == "alice"
        assert assessment.dimensions == {"general": {"score": 0.9}}

    def test_trust_get(self, entity, stack):
        entity.attach_stack(stack)
        entity.trust_get("alice", domain="general")
        stack.get_trust_assessments.assert_called_once_with(entity_id="alice", domain="general")

    def test_trust_list(self, entity, stack):
        entity.attach_stack(stack)
        entity.trust_list(domain="general")
        stack.get_trust_assessments.assert_called_once_with(domain="general")

    def test_trust_list_filters_by_min_score(self, entity, stack):
        entity.attach_stack(stack)
        high = TrustAssessment(
            id="a1",
            stack_id="test-stack",
            entity="alice",
            dimensions={"general": {"score": 0.9}},
        )
        low = TrustAssessment(
            id="a2",
            stack_id="test-stack",
            entity="bob",
            dimensions={"general": {"score": 0.2}},
        )
        stack.get_trust_assessments.return_value = [high, low]
        result = entity.trust_list(min_score=0.5)
        assert len(result) == 1
        assert result[0].entity == "alice"


# ---- Plugin Management ----


class TestPluginManagement:
    def test_load_plugin(self, entity, plugin):
        entity.load_plugin(plugin)
        plugin.activate.assert_called_once()
        assert "test-plugin" in entity.plugins
        info = entity.plugins["test-plugin"]
        assert isinstance(info, PluginInfo)
        assert info.is_loaded is True

    def test_unload_plugin(self, entity, plugin):
        entity.load_plugin(plugin)
        entity.unload_plugin("test-plugin")
        plugin.deactivate.assert_called_once()
        assert "test-plugin" not in entity.plugins

    def test_unload_nonexistent_plugin_is_noop(self, entity):
        entity.unload_plugin("nonexistent")  # Should not raise

    def test_plugin_context_provides_core_id(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.core_id == "test-core"

    def test_plugin_context_returns_none_stack_id_without_stack(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.active_stack_id is None

    def test_plugin_context_returns_stack_id_with_stack(self, entity, stack, plugin):
        entity.attach_stack(stack)
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.active_stack_id == "test-stack"

    def test_plugin_context_write_returns_none_without_stack(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.episode("obj", "out") is None
        assert ctx.belief("stmt") is None
        assert ctx.note("content") is None
        assert ctx.raw("blob") is None

    def test_plugin_context_write_routes_through_entity(self, entity, stack, plugin):
        entity.attach_stack(stack)
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        ctx.episode("plugin objective", "plugin outcome")
        ep = stack.save_episode.call_args[0][0]
        assert ep.source_entity == "plugin:test-plugin"

    def test_plugin_context_search_returns_empty_without_stack(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.search("query") == []

    def test_plugin_context_get_data_dir(self, entity, plugin):
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        data_dir = ctx.get_data_dir()
        assert data_dir.exists()
        assert "test-plugin" in str(data_dir)

    def test_plugin_context_config_and_secrets(self, entity, plugin):
        entity._plugin_configs["test-plugin"] = {"api_url": "https://example.com"}
        entity._plugin_secrets["test-plugin"] = {"api_key": "secret123"}
        entity.load_plugin(plugin)
        ctx = entity._plugin_contexts["test-plugin"]
        assert ctx.get_config("api_url") == "https://example.com"
        assert ctx.get_config("missing", "default") == "default"
        assert ctx.get_secret("api_key") == "secret123"
        assert ctx.get_secret("missing") is None

    @patch("kernle.entity.discover_plugins")
    def test_discover_plugins(self, mock_discover, entity, plugin):
        from kernle.discovery import DiscoveredComponent

        mock_discover.return_value = [
            DiscoveredComponent(
                name="found-plugin",
                group="kernle.plugins",
                module="found_plugin",
                attr="FoundPlugin",
                dist_version="2.0.0",
            )
        ]
        entity.load_plugin(plugin)
        discovered = entity.discover_plugins()
        assert len(discovered) == 1
        assert discovered[0].name == "found-plugin"
        assert discovered[0].is_loaded is False


# ---- Status Assembly ----


class TestStatus:
    def test_status_basic(self, entity):
        result = entity.status()
        assert result["core_id"] == "test-core"
        assert result["model"] is None
        assert result["stacks"] == {}
        assert result["plugins"] == {}

    def test_status_with_model(self, entity):
        entity.set_model(_make_mock_model(model_id="claude"))
        result = entity.status()
        assert result["model"] == "claude"

    def test_status_with_stack(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        result = entity.status()
        assert "primary" in result["stacks"]
        assert result["stacks"]["primary"]["stack_id"] == "test-stack"
        assert result["stacks"]["primary"]["active"] is True

    def test_status_with_plugin(self, entity, plugin):
        entity.load_plugin(plugin)
        result = entity.status()
        assert "test-plugin" in result["plugins"]
        assert result["plugins"]["test-plugin"]["health"]["healthy"] is True

    def test_status_plugin_health_failure_isolated(self, entity):
        bad_plugin = _make_mock_plugin(name="bad")
        bad_plugin.health_check.side_effect = RuntimeError("crash")
        entity.load_plugin(bad_plugin)
        result = entity.status()
        assert result["plugins"]["bad"]["health"]["healthy"] is False

    def test_status_calls_plugin_on_status(self, entity, plugin):
        entity.load_plugin(plugin)
        entity.status()
        plugin.on_status.assert_called_once()


# ---- Binding ----


class TestBinding:
    def test_get_binding(self, entity, stack):
        entity.attach_stack(stack, alias="primary")
        entity.load_plugin(_make_mock_plugin())
        binding = entity.get_binding()
        assert isinstance(binding, Binding)
        assert binding.core_id == "test-core"
        assert binding.stacks == {"primary": "test-stack"}
        assert binding.active_stack_alias == "primary"
        assert "test-plugin" in binding.plugins

    def test_save_binding(self, entity, tmp_path, stack):
        entity.attach_stack(stack)
        path = entity.save_binding()
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert data["core_id"] == "test-core"

    def test_save_binding_custom_path(self, entity, tmp_path):
        path = tmp_path / "custom.json"
        result = entity.save_binding(path=path)
        assert result == path
        assert path.exists()

    def test_from_binding_object(self):
        binding = Binding(
            core_id="restored-core",
            model_config={},
            stacks={},
        )
        restored = Entity.from_binding(binding)
        assert restored.core_id == "restored-core"

    def test_from_binding_path(self, entity, tmp_path, stack):
        entity.attach_stack(stack)
        path = entity.save_binding()
        restored = Entity.from_binding(path)
        assert restored.core_id == "test-core"


# ---- Checkpoint ----


class TestCheckpoint:
    def test_checkpoint(self, entity, stack):
        entity.attach_stack(stack)
        cp_id = entity.checkpoint("test checkpoint")
        assert "test-core" in cp_id
        cp_dir = entity._data_dir / "checkpoints"
        assert cp_dir.exists()
        files = list(cp_dir.glob("*.json"))
        assert len(files) == 1


# ---- Sync ----


class TestSync:
    def test_sync_routes_to_stack(self, entity, stack):
        entity.attach_stack(stack)
        result = entity.sync()
        stack.sync.assert_called_once()
        assert isinstance(result, SyncResult)
