"""Tests for plugin CLI and MCP tool registration wiring (#264).

Verifies that Entity.load_plugin() registers tools from plugins,
Entity.unload_plugin() cleans them up, and the MCP server can
expose plugin tools.
"""

from unittest.mock import MagicMock, PropertyMock

import pytest

from kernle.entity import Entity
from kernle.protocols import PluginHealth, ToolDefinition

# ---- Helpers ----


def _make_mock_plugin(name="test-plugin", version="1.0.0", tools=None):
    """Create a mock PluginProtocol with configurable tool registration."""
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
    plugin.register_tools.return_value = tools if tools is not None else []
    return plugin


def _make_tool(name="do-thing", description="Does a thing"):
    """Create a ToolDefinition."""
    return ToolDefinition(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        handler=lambda args: f"handled {name}",
    )


@pytest.fixture
def entity(tmp_path):
    return Entity(core_id="test-core", data_dir=tmp_path)


# ---- Tool Registration ----


class TestLoadPluginRegistersTools:
    def test_load_plugin_registers_tools(self, entity):
        tools = [_make_tool("search"), _make_tool("fetch")]
        plugin = _make_mock_plugin(tools=tools)

        entity.load_plugin(plugin)

        assert "test-plugin" in entity._plugin_tools
        assert len(entity._plugin_tools["test-plugin"]) == 2
        assert entity._plugin_tools["test-plugin"][0].name == "search"
        assert entity._plugin_tools["test-plugin"][1].name == "fetch"

    def test_load_plugin_no_tools_returns_empty(self, entity):
        plugin = _make_mock_plugin(tools=[])

        entity.load_plugin(plugin)

        assert "test-plugin" not in entity._plugin_tools

    def test_load_plugin_tool_registration_failure_isolated(self, entity):
        """Tool registration failure must not prevent plugin activation."""
        plugin = _make_mock_plugin()
        plugin.register_tools.side_effect = RuntimeError("tool registration boom")

        entity.load_plugin(plugin)

        # Plugin was still activated and registered
        plugin.activate.assert_called_once()
        assert "test-plugin" in entity.plugins
        health = entity.plugin_health("test-plugin")
        assert health is not None
        assert health.healthy is False
        assert "tool registration failed" in health.message
        # But no tools were registered
        assert "test-plugin" not in entity._plugin_tools

    def test_load_plugin_tool_registration_fail_fast_raises(self, entity):
        """Fail-fast plugin registration should rollback plugin load and raise."""
        plugin = _make_mock_plugin()
        plugin.register_tools.side_effect = RuntimeError("tool registration boom")

        with pytest.raises(RuntimeError, match="tool registration failed"):
            entity.load_plugin(plugin, fail_fast=True)

        assert "test-plugin" not in entity.plugins
        assert entity.plugin_health("test-plugin") is None
        assert "test-plugin" not in entity._plugin_tools
        plugin.deactivate.assert_called_once()


# ---- Tool Cleanup on Unload ----


class TestUnloadPluginRemovesTools:
    def test_unload_plugin_removes_tools(self, entity):
        tools = [_make_tool()]
        plugin = _make_mock_plugin(tools=tools)

        entity.load_plugin(plugin)
        assert "test-plugin" in entity._plugin_tools

        entity.unload_plugin("test-plugin")
        assert "test-plugin" not in entity._plugin_tools

    def test_unload_nonexistent_plugin_is_noop(self, entity):
        """Unloading a plugin that was never loaded should not error."""
        entity.unload_plugin("nonexistent")  # Should not raise


# ---- get_all_plugin_tools ----


class TestGetAllPluginTools:
    def test_get_all_plugin_tools_aggregates(self, entity):
        plugin_a = _make_mock_plugin(name="alpha", tools=[_make_tool("a1"), _make_tool("a2")])
        plugin_b = _make_mock_plugin(name="beta", tools=[_make_tool("b1")])

        entity.load_plugin(plugin_a)
        entity.load_plugin(plugin_b)

        all_tools = entity.get_all_plugin_tools()
        assert len(all_tools) == 3
        names = {t.name for t in all_tools}
        assert names == {"a1", "a2", "b1"}

    def test_get_all_plugin_tools_empty_when_no_plugins(self, entity):
        assert entity.get_all_plugin_tools() == []

    def test_get_all_plugin_tools_empty_when_plugins_have_no_tools(self, entity):
        plugin = _make_mock_plugin(tools=[])
        entity.load_plugin(plugin)
        assert entity.get_all_plugin_tools() == []


# ---- CLI Registration via subparsers ----


class TestLoadPluginCLIRegistration:
    def test_load_plugin_with_subparsers_calls_register_cli(self, entity):
        plugin = _make_mock_plugin()
        subparsers = MagicMock()

        entity.load_plugin(plugin, subparsers=subparsers)

        plugin.register_cli.assert_called_once_with(subparsers)

    def test_load_plugin_without_subparsers_skips_register_cli(self, entity):
        plugin = _make_mock_plugin()

        entity.load_plugin(plugin)

        plugin.register_cli.assert_not_called()

    def test_load_plugin_cli_registration_failure_isolated(self, entity):
        """CLI registration failure must not prevent plugin activation or tool registration."""
        tools = [_make_tool()]
        plugin = _make_mock_plugin(tools=tools)
        plugin.register_cli.side_effect = RuntimeError("cli boom")
        subparsers = MagicMock()

        entity.load_plugin(plugin, subparsers=subparsers)

        # Plugin was still activated
        plugin.activate.assert_called_once()
        assert "test-plugin" in entity.plugins
        # Tools were still registered
        assert "test-plugin" in entity._plugin_tools
        health = entity.plugin_health("test-plugin")
        assert health is not None
        assert health.healthy is False
        assert "CLI registration failed" in health.message

    def test_load_plugin_cli_registration_fail_fast_raises(self, entity):
        """Fail-fast CLI registration should rollback plugin load and raise."""
        tools = [_make_tool()]
        plugin = _make_mock_plugin(tools=tools)
        plugin.register_cli.side_effect = RuntimeError("cli boom")
        subparsers = MagicMock()

        with pytest.raises(RuntimeError, match="CLI registration failed"):
            entity.load_plugin(plugin, subparsers=subparsers, fail_fast=True)

        assert "test-plugin" not in entity.plugins
        assert entity.plugin_health("test-plugin") is None
        assert "test-plugin" not in entity._plugin_tools
        plugin.deactivate.assert_called_once()


# ---- Full Lifecycle ----


class TestPluginToolLifecycle:
    def test_plugin_tool_lifecycle(self, entity):
        """Load -> tools available -> unload -> tools gone."""
        tools = [_make_tool("my-tool")]
        plugin = _make_mock_plugin(tools=tools)

        # Before load: no tools
        assert entity.get_all_plugin_tools() == []

        # Load: tools available
        entity.load_plugin(plugin)
        all_tools = entity.get_all_plugin_tools()
        assert len(all_tools) == 1
        assert all_tools[0].name == "my-tool"

        # Unload: tools gone
        entity.unload_plugin("test-plugin")
        assert entity.get_all_plugin_tools() == []


# ---- MCP Server Plugin Tool Integration ----


class TestMCPPluginTools:
    def test_register_plugin_tools_adds_to_registry(self):
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
        )

        # Clean state
        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td = ToolDefinition(
                name="greet",
                description="Says hello",
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                handler=lambda args: f"Hello {args.get('name', 'world')}",
            )
            register_plugin_tools("myplugin", [td])

            assert "myplugin.greet" in _plugin_tools
            assert _plugin_tools["myplugin.greet"].name == "myplugin.greet"
            assert "[myplugin]" in _plugin_tools["myplugin.greet"].description
            assert "myplugin.greet" in _plugin_handlers
            assert "myplugin.greet" in _plugin_schemas
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()

    def test_unregister_plugin_tools_removes_from_registry(self):
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
            unregister_plugin_tools,
        )

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td = ToolDefinition(
                name="greet",
                description="Says hello",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: "hi",
            )
            register_plugin_tools("myplugin", [td])
            assert "myplugin.greet" in _plugin_tools

            unregister_plugin_tools("myplugin")
            assert "myplugin.greet" not in _plugin_tools
            assert "myplugin.greet" not in _plugin_handlers
            assert "myplugin.greet" not in _plugin_schemas
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()

    def test_unregister_only_removes_target_plugin(self):
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
            unregister_plugin_tools,
        )

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td_a = ToolDefinition(
                name="a",
                description="A",
                input_schema={"type": "object"},
                handler=lambda args: "a",
            )
            td_b = ToolDefinition(
                name="b",
                description="B",
                input_schema={"type": "object"},
                handler=lambda args: "b",
            )
            register_plugin_tools("alpha", [td_a])
            register_plugin_tools("beta", [td_b])

            unregister_plugin_tools("alpha")
            assert "alpha.a" not in _plugin_tools
            assert "beta.b" in _plugin_tools
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()

    def test_validate_tool_input_passes_plugin_tools(self):
        """Plugin tools must pass through validate_tool_input without raising."""
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
            validate_tool_input,
        )

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td = ToolDefinition(
                name="greet",
                description="Says hello",
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                handler=lambda args: f"Hello {args.get('name', 'world')}",
            )
            register_plugin_tools("myplugin", [td])

            # This should NOT raise ValueError("Unknown tool")
            result = validate_tool_input("myplugin.greet", {"name": "test"})
            assert result == {"name": "test"}
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()

    def test_validate_tool_input_rejects_unknown_tools(self):
        """Unknown tools should still raise ValueError."""
        from kernle.mcp.server import validate_tool_input

        with pytest.raises(ValueError, match="Unknown tool"):
            validate_tool_input("totally_unknown_tool", {})

    def test_validate_tool_input_rejects_non_object_arguments(self):
        from kernle.mcp.server import validate_tool_input

        with pytest.raises(ValueError, match="arguments must be an object"):
            validate_tool_input("memory_load", "not-a-dict")  # type: ignore[arg-type]

    def test_validate_tool_input_enforces_plugin_schema(self):
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
            validate_tool_input,
        )

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td = ToolDefinition(
                name="strict",
                description="Strict schema",
                input_schema={
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
                handler=lambda args: args,
            )
            register_plugin_tools("myplugin", [td])

            with pytest.raises(ValueError, match="Schema validation failed|invalid type"):
                validate_tool_input("myplugin.strict", {"count": "2"})
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()

    def test_validate_tool_input_enforces_plugin_payload_size_limit(self):
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
            validate_tool_input,
        )

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td = ToolDefinition(
                name="big",
                description="Large payload tool",
                input_schema={"type": "object", "properties": {"blob": {"type": "string"}}},
                handler=lambda args: args,
            )
            register_plugin_tools("myplugin", [td])

            too_large = {"blob": "x" * 70000}
            with pytest.raises(ValueError, match="payload too large"):
                validate_tool_input("myplugin.big", too_large)
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()

    def test_tool_without_handler_not_in_handlers(self):
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
        )

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td = ToolDefinition(
                name="info",
                description="Info tool",
                input_schema={"type": "object"},
                handler=None,
            )
            register_plugin_tools("myplugin", [td])

            assert "myplugin.info" in _plugin_tools
            assert "myplugin.info" not in _plugin_handlers
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()

    def test_validate_tool_input_rejects_plugin_without_schema(self):
        from kernle.mcp.server import (
            _plugin_handlers,
            _plugin_schema_validators,
            _plugin_schemas,
            _plugin_tools,
            register_plugin_tools,
            validate_tool_input,
        )

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        try:
            td = ToolDefinition(
                name="schemaless",
                description="Intentionally missing schema",
                input_schema={},
                handler=lambda args: args,
            )
            with pytest.raises(ValueError, match="must provide an input schema"):
                register_plugin_tools("myplugin", [td])

            # simulate metadata drift after successful registration
            td = ToolDefinition(
                name="schemaless",
                description="Schema can be removed later",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: args,
            )
            register_plugin_tools("myplugin", [td])
            _plugin_schemas.pop("myplugin.schemaless", None)

            with pytest.raises(ValueError, match="metadata missing"):
                validate_tool_input("myplugin.schemaless", {"foo": "bar"})
        finally:
            _plugin_tools.clear()
            _plugin_handlers.clear()
            _plugin_schemas.clear()
            _plugin_schema_validators.clear()
