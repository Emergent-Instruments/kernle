"""Integration tests for the Binding save/restore system.

Tests the full lifecycle of creating an Entity + SQLiteStack composition,
saving the binding, and restoring it. Uses real SQLiteStack instances
(not mocks) to verify the binding system works end-to-end.
"""

import json
from unittest.mock import MagicMock, PropertyMock

import pytest

from kernle.entity import Entity
from kernle.protocols import Binding, PluginHealth
from kernle.stack import SQLiteStack

CORE_ID = "binding-test-core"
STACK_ID = "binding-test-stack"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "kernle_data"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "binding_test.db"


@pytest.fixture
def entity(data_dir):
    return Entity(core_id=CORE_ID, data_dir=data_dir)


@pytest.fixture
def stack(db_path):
    return SQLiteStack(stack_id=STACK_ID, db_path=db_path, enforce_provenance=False)


def _make_mock_plugin(name="test-plugin"):
    plugin = MagicMock()
    type(plugin).name = PropertyMock(return_value=name)
    type(plugin).version = PropertyMock(return_value="1.0.0")
    type(plugin).protocol_version = PropertyMock(return_value=1)
    type(plugin).description = PropertyMock(return_value=f"Plugin: {name}")
    plugin.capabilities.return_value = ["testing"]
    plugin.activate.return_value = None
    plugin.deactivate.return_value = None
    plugin.health_check.return_value = PluginHealth(healthy=True)
    plugin.on_load.return_value = None
    plugin.on_status.return_value = None
    plugin.register_cli.return_value = None
    plugin.register_tools.return_value = []
    return plugin


def _make_mock_model(model_id="test-model"):
    model = MagicMock()
    type(model).model_id = PropertyMock(return_value=model_id)
    return model


# ============================================================================
# 1. Full Roundtrip: Create, Save, Restore
# ============================================================================


class TestBindingRoundtrip:
    def test_basic_roundtrip(self, entity, stack, tmp_path):
        """Create Entity + Stack, save binding, restore, verify."""
        entity.attach_stack(stack, alias="primary")
        entity.episode("Test roundtrip", "It works")

        # Save
        binding_path = tmp_path / "binding.json"
        path = entity.save_binding(path=binding_path)
        assert path.exists()

        # Restore
        restored = Entity.from_binding(path)
        assert restored.core_id == CORE_ID

    def test_roundtrip_preserves_core_id(self, entity, stack, tmp_path):
        entity.attach_stack(stack)
        path = entity.save_binding(path=tmp_path / "b.json")

        restored = Entity.from_binding(path)
        assert restored.core_id == entity.core_id

    def test_roundtrip_with_model(self, entity, stack, tmp_path):
        model = _make_mock_model("claude-test")
        entity.set_model(model)
        entity.attach_stack(stack, alias="main")

        path = entity.save_binding(path=tmp_path / "b.json")
        data = json.loads(path.read_text())

        assert data["model_config"]["model_id"] == "claude-test"

    def test_roundtrip_with_multiple_stacks(self, entity, tmp_path):
        s1 = SQLiteStack(stack_id="s1", db_path=tmp_path / "s1.db", enforce_provenance=False)
        s2 = SQLiteStack(stack_id="s2", db_path=tmp_path / "s2.db", enforce_provenance=False)

        entity.attach_stack(s1, alias="alpha")
        entity.attach_stack(s2, alias="beta", set_active=False)

        path = entity.save_binding(path=tmp_path / "multi.json")
        data = json.loads(path.read_text())

        assert data["stacks"]["alpha"] == "s1"
        assert data["stacks"]["beta"] == "s2"
        assert data["active_stack_alias"] == "alpha"

    def test_roundtrip_with_plugins(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="main")
        entity.load_plugin(_make_mock_plugin("plugin-a"))
        entity.load_plugin(_make_mock_plugin("plugin-b"))

        path = entity.save_binding(path=tmp_path / "plugins.json")
        data = json.loads(path.read_text())

        assert "plugin-a" in data["plugins"]
        assert "plugin-b" in data["plugins"]


# ============================================================================
# 2. Binding File Format
# ============================================================================


class TestBindingFileFormat:
    def test_binding_is_valid_json(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="main")
        path = entity.save_binding(path=tmp_path / "format.json")

        content = path.read_text()
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_binding_is_human_readable(self, entity, stack, tmp_path):
        """Binding file should be indented / pretty-printed."""
        entity.attach_stack(stack, alias="main")
        path = entity.save_binding(path=tmp_path / "pretty.json")

        content = path.read_text()
        # Pretty-printed JSON has newlines and indentation
        assert "\n" in content
        assert "  " in content

    def test_binding_has_required_fields(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="main")
        path = entity.save_binding(path=tmp_path / "fields.json")

        data = json.loads(path.read_text())
        required = ["core_id", "model_config", "stacks", "active_stack_alias", "plugins"]
        for field in required:
            assert field in data, f"Missing required field: {field}"

    def test_binding_has_created_at(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="main")
        path = entity.save_binding(path=tmp_path / "ts.json")

        data = json.loads(path.read_text())
        assert "created_at" in data
        assert data["created_at"] is not None

    def test_binding_stacks_map_alias_to_id(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="memory")
        path = entity.save_binding(path=tmp_path / "stacks.json")

        data = json.loads(path.read_text())
        assert data["stacks"]["memory"] == STACK_ID

    def test_empty_entity_binding(self, entity, tmp_path):
        """Entity with no stacks, no model, no plugins."""
        path = entity.save_binding(path=tmp_path / "empty.json")

        data = json.loads(path.read_text())
        assert data["core_id"] == CORE_ID
        assert data["stacks"] == {}
        assert data["plugins"] == []
        assert data["model_config"] == {}
        assert data["active_stack_alias"] is None


# ============================================================================
# 3. Memories Survive Binding Roundtrip
# ============================================================================


class TestMemoriesSurviveBinding:
    """Save memories through Entity, save binding, verify stack still has them."""

    def test_memories_persist_after_binding_save(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="main")

        # Write various memory types
        ep_id = entity.episode("Learn Rust", "Built CLI tool")
        b_id = entity.belief("Rust is safe")
        v_id = entity.value("Safety", "Memory safety matters")
        g_id = entity.goal("Ship v2")
        n_id = entity.note("Check performance")

        # Save binding
        entity.save_binding(path=tmp_path / "persist.json")

        # Verify memories still in stack
        episodes = stack.get_episodes()
        assert any(e.id == ep_id for e in episodes)

        beliefs = stack.get_beliefs()
        assert any(b.id == b_id for b in beliefs)

        values = stack.get_values()
        assert any(v.id == v_id for v in values)

        goals = stack.get_goals()
        assert any(g.id == g_id for g in goals)

        notes = stack.get_notes()
        assert any(n.id == n_id for n in notes)

    def test_stack_accessible_after_restore(self, entity, stack, db_path, tmp_path):
        """After binding restore, stack data is still on disk and accessible."""
        entity.attach_stack(stack, alias="main")
        ep_id = entity.episode("Persist test", "Memory survives")

        path = entity.save_binding(path=tmp_path / "access.json")

        # Restore creates a new Entity (no stacks yet, but same core_id)
        restored = Entity.from_binding(path)
        assert restored.core_id == CORE_ID

        # Re-open the same database to verify data persisted
        reopened_stack = SQLiteStack(stack_id=STACK_ID, db_path=db_path, enforce_provenance=False)
        episodes = reopened_stack.get_episodes()
        assert any(e.id == ep_id for e in episodes)


# ============================================================================
# 4. Edge Cases and Error Handling
# ============================================================================


class TestBindingEdgeCases:
    def test_save_binding_default_path(self, entity, stack, data_dir):
        entity.attach_stack(stack, alias="main")
        path = entity.save_binding()

        assert path.exists()
        assert "bindings" in str(path)

    def test_binding_overwrite(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="main")
        path = tmp_path / "overwrite.json"

        entity.save_binding(path=path)
        data1 = json.loads(path.read_text())

        # Load a plugin, save again
        entity.load_plugin(_make_mock_plugin("new-plugin"))
        entity.save_binding(path=path)
        data2 = json.loads(path.read_text())

        assert data1["plugins"] == []
        assert "new-plugin" in data2["plugins"]

    def test_from_binding_invalid_path(self, tmp_path):
        bad_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            Entity.from_binding(bad_path)

    def test_from_binding_corrupted_json(self, tmp_path):
        bad_path = tmp_path / "corrupt.json"
        bad_path.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            Entity.from_binding(bad_path)

    def test_from_binding_minimal_json(self, tmp_path):
        """Binding with only core_id should still restore."""
        minimal = tmp_path / "minimal.json"
        minimal.write_text(json.dumps({"core_id": "minimal-core"}))

        restored = Entity.from_binding(minimal)
        assert restored.core_id == "minimal-core"

    def test_get_binding_detached_entity(self, entity):
        """Entity with no stacks can still produce a binding."""
        binding = entity.get_binding()
        assert binding.core_id == CORE_ID
        assert binding.stacks == {}
        assert binding.active_stack_alias is None

    def test_binding_after_detach(self, entity, stack, tmp_path):
        entity.attach_stack(stack, alias="temp")
        entity.detach_stack("temp")

        binding = entity.get_binding()
        assert binding.stacks == {}
        assert binding.active_stack_alias is None


# ============================================================================
# 5. Binding Object (Dataclass)
# ============================================================================


class TestBindingDataclass:
    def test_binding_fields(self):
        b = Binding(
            core_id="test",
            model_config={"model_id": "m1"},
            stacks={"main": "s1"},
            active_stack_alias="main",
            plugins=["p1"],
        )
        assert b.core_id == "test"
        assert b.model_config == {"model_id": "m1"}
        assert b.stacks == {"main": "s1"}
        assert b.active_stack_alias == "main"
        assert b.plugins == ["p1"]

    def test_binding_defaults(self):
        b = Binding(core_id="test", model_config={}, stacks={})
        assert b.active_stack_alias is None
        assert b.plugins == []
        assert b.created_at is None
        assert b.metadata == {}
