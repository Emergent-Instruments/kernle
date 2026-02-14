"""Coverage tests for kernle/mcp/server.py.

Targets uncovered lines: set_stack_id, handle_tool_error branches,
plugin tool call_tool path, main(), run_server(), and __main__ block.
"""

import json
import runpy
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from kernle.mcp.server import (
    _plugin_handlers,
    _plugin_schema_validators,
    _plugin_schemas,
    _plugin_tools,
    call_tool,
    get_kernle,
    handle_tool_error,
    main,
    run_server,
    set_stack_id,
    validate_tool_input,
)


class TestSetStackId:
    """Tests for set_stack_id() — covers lines 51-54."""

    def test_set_stack_id_updates_global(self):
        """set_stack_id updates the global _mcp_stack_id."""
        import kernle.mcp.server as srv

        old = srv._mcp_stack_id
        try:
            set_stack_id("new-stack-id")
            assert srv._mcp_stack_id == "new-stack-id"
        finally:
            srv._mcp_stack_id = old

    def test_set_stack_id_clears_cached_instance(self):
        """set_stack_id clears cached Kernle instance if one exists."""
        # Attach a fake cached instance
        get_kernle._instance = "fake"  # type: ignore[attr-defined]
        try:
            set_stack_id("another-id")
            assert not hasattr(get_kernle, "_instance")
        finally:
            # Clean up
            if hasattr(get_kernle, "_instance"):
                delattr(get_kernle, "_instance")

    def test_set_stack_id_no_cached_instance(self):
        """set_stack_id works fine when no cached instance exists."""
        # Ensure no cached instance
        if hasattr(get_kernle, "_instance"):
            delattr(get_kernle, "_instance")
        # Should not raise
        set_stack_id("yet-another-id")
        assert not hasattr(get_kernle, "_instance")


class TestHandleToolErrorBranches:
    """Tests for handle_tool_error() — covers lines 121-130."""

    def test_permission_error_returns_access_denied(self):
        """PermissionError returns 'Access denied' message."""
        result = handle_tool_error(PermissionError("forbidden"), "test_tool", {"key": "val"})
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Access denied"

    def test_file_not_found_error_returns_resource_not_found(self):
        """FileNotFoundError returns 'Resource not found' message."""
        result = handle_tool_error(FileNotFoundError("missing"), "test_tool", {"key": "val"})
        assert len(result) == 1
        assert result[0].text == "Resource not found"

    def test_connection_error_returns_service_unavailable(self):
        """ConnectionError returns 'Service temporarily unavailable'."""
        result = handle_tool_error(ConnectionError("db down"), "test_tool", {"key": "val"})
        assert len(result) == 1
        assert result[0].text == "Service temporarily unavailable"


class TestCallToolPluginHandler:
    """Tests for plugin tool dispatching in call_tool() — covers lines 172-183."""

    @pytest.mark.asyncio
    async def test_plugin_handler_returns_string(self, monkeypatch):
        """Plugin handler returning a string is passed through directly."""
        # Register a fake plugin handler with a schema
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()
        _plugin_handlers["fake_plugin.do_thing"] = lambda args: f"result: {args['x']}"
        _plugin_tools["fake_plugin.do_thing"] = MagicMock()
        _plugin_schemas["fake_plugin.do_thing"] = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }

        # Mock get_kernle to avoid real Kernle creation
        mock_kernle = MagicMock()
        monkeypatch.setattr("kernle.mcp.server.get_kernle", lambda: mock_kernle)

        try:
            result = await call_tool("fake_plugin.do_thing", {"x": 42})
            assert len(result) == 1
            assert result[0].text == "result: 42"
        finally:
            _plugin_handlers.pop("fake_plugin.do_thing", None)
            _plugin_tools.pop("fake_plugin.do_thing", None)
            _plugin_schemas.pop("fake_plugin.do_thing", None)
            _plugin_schema_validators.pop("fake_plugin.do_thing", None)

    @pytest.mark.asyncio
    async def test_plugin_handler_returns_dict(self, monkeypatch):
        """Plugin handler returning a dict is JSON-serialized."""
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()
        _plugin_handlers["fake_plugin.info"] = lambda args: {"status": "ok", "count": 3}
        _plugin_tools["fake_plugin.info"] = MagicMock()
        _plugin_schemas["fake_plugin.info"] = {"type": "object", "properties": {}}

        mock_kernle = MagicMock()
        monkeypatch.setattr("kernle.mcp.server.get_kernle", lambda: mock_kernle)

        try:
            result = await call_tool("fake_plugin.info", {})
            assert len(result) == 1
            parsed = json.loads(result[0].text)
            assert parsed["status"] == "ok"
            assert parsed["count"] == 3
        finally:
            _plugin_handlers.pop("fake_plugin.info", None)
            _plugin_tools.pop("fake_plugin.info", None)
            _plugin_schemas.pop("fake_plugin.info", None)
            _plugin_schema_validators.pop("fake_plugin.info", None)

    @pytest.mark.asyncio
    async def test_unknown_tool_after_validation_returns_not_available(self, monkeypatch):
        """Tool name not in HANDLERS or _plugin_handlers returns 'not available'."""
        # We need to bypass validate_tool_input — register the name there
        # but NOT in HANDLERS or _plugin_handlers
        mock_kernle = MagicMock()
        monkeypatch.setattr("kernle.mcp.server.get_kernle", lambda: mock_kernle)

        # Monkeypatch validate_tool_input to return args as-is
        monkeypatch.setattr("kernle.mcp.server.validate_tool_input", lambda name, args: dict(args))

        result = await call_tool("nonexistent_tool_xyz", {})
        assert len(result) == 1
        assert "not available" in result[0].text

    @pytest.mark.asyncio
    async def test_plugin_schema_validation_failure_returns_invalid_input(self, monkeypatch):
        """Schema/type failures for plugin tool args should be rejected at MCP boundary."""
        import kernle.mcp.server as srv
        from kernle.protocols import ToolDefinition

        td = ToolDefinition(
            name="sum",
            description="Adds values",
            input_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            handler=lambda args: str(args["count"]),
        )

        srv.register_plugin_tools("validator", [td])
        monkeypatch.setattr("kernle.mcp.server.get_kernle", lambda: MagicMock())

        try:
            result = await call_tool("validator.sum", {"count": "2"})
            assert len(result) == 1
            assert "Invalid input" in result[0].text
        finally:
            srv.unregister_plugin_tools("validator")


class TestValidateToolInput:
    """Negative tests for validate_tool_input()."""

    def test_tool_name_must_be_non_empty_string(self):
        """Empty tool names are rejected."""
        with pytest.raises(ValueError, match="tool name must not be empty"):
            validate_tool_input("", {})

    def test_tool_name_must_be_string(self):
        """Non-string tool names are rejected."""
        with pytest.raises(ValueError, match="tool name must be a string"):
            validate_tool_input(123, {})

    def test_arguments_must_be_object(self):
        """Non-dict arguments are rejected early."""
        with pytest.raises(ValueError, match="arguments must be an object"):
            validate_tool_input("some_tool", [])


class TestMain:
    """Tests for main() entry point — covers lines 207-213."""

    def test_main_calls_set_stack_id_and_run_server(self, monkeypatch):
        """main() resolves stack ID, sets it, and runs the server."""
        captured_stack_id = {}
        captured_run = {"called": False}

        def fake_set_stack_id(sid):
            captured_stack_id["id"] = sid

        def fake_run(coro):
            coro.close()  # Properly clean up the coroutine
            captured_run["called"] = True

        monkeypatch.setattr("kernle.mcp.server.set_stack_id", fake_set_stack_id)
        monkeypatch.setattr("asyncio.run", fake_run)
        monkeypatch.setattr("kernle.utils.resolve_stack_id", lambda x: x or "resolved-default")

        main(stack_id="my-agent")
        assert captured_stack_id["id"] == "my-agent"
        assert captured_run["called"]

    def test_main_default_stack_id_resolves(self, monkeypatch):
        """main() with default stack_id triggers resolve_stack_id(None)."""
        resolved_args = {}

        def fake_resolve(explicit):
            resolved_args["explicit"] = explicit
            return "auto-12345678"

        monkeypatch.setattr("kernle.utils.resolve_stack_id", fake_resolve)
        monkeypatch.setattr("kernle.mcp.server.set_stack_id", lambda sid: None)
        monkeypatch.setattr("asyncio.run", lambda coro: coro.close())

        main()  # default stack_id="default"
        assert resolved_args["explicit"] is None


class TestRunServer:
    """Tests for run_server() — covers lines 191-192."""

    @pytest.mark.asyncio
    async def test_run_server_opens_stdio_and_runs_mcp(self):
        """run_server() opens stdio_server context and calls mcp.run()."""
        import kernle.mcp.server as srv

        fake_read = MagicMock(name="read_stream")
        fake_write = MagicMock(name="write_stream")

        @asynccontextmanager
        async def fake_stdio_server():
            yield (fake_read, fake_write)

        fake_mcp_run = AsyncMock(return_value=None)
        fake_init_opts = {"server_name": "kernle"}

        original_run = srv.mcp.run
        original_init = srv.mcp.create_initialization_options
        try:
            srv.mcp.run = fake_mcp_run
            srv.mcp.create_initialization_options = MagicMock(return_value=fake_init_opts)

            with patch("kernle.mcp.server.stdio_server", fake_stdio_server):
                await run_server()

            fake_mcp_run.assert_called_once_with(fake_read, fake_write, fake_init_opts)
        finally:
            srv.mcp.run = original_run
            srv.mcp.create_initialization_options = original_init


class TestHandleToolErrorAdditionalBranches:
    """Tests for ValueError and generic Exception branches in handle_tool_error()."""

    def test_value_error_returns_invalid_input(self):
        """ValueError returns 'Invalid input' message."""
        result = handle_tool_error(ValueError("bad value"), "test_tool", {"key": "val"})
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Invalid input" in result[0].text
        assert "bad value" in result[0].text

    def test_generic_exception_returns_internal_error(self):
        """Unknown exception types return 'Internal server error'."""
        result = handle_tool_error(RuntimeError("boom"), "test_tool", {"key": "val"})
        assert len(result) == 1
        assert result[0].text == "Internal server error"

    def test_generic_exception_with_non_dict_arguments(self):
        """Generic exception with non-dict arguments doesn't crash."""
        result = handle_tool_error(RuntimeError("crash"), "test_tool", "not-a-dict")
        assert len(result) == 1
        assert result[0].text == "Internal server error"


class TestRegisterPluginTools:
    """Tests for register_plugin_tools() validation branches."""

    def _make_td(self, name="tool1", schema=None, handler=None):
        from kernle.protocols import ToolDefinition

        return ToolDefinition(
            name=name,
            description="Test tool",
            input_schema=schema or {"type": "object", "properties": {}},
            handler=handler,
        )

    def setup_method(self):
        """Clear plugin registries before each test."""
        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

    def test_rejects_non_dict_schema(self):
        """Schema that is not a dict raises ValueError."""
        td = self._make_td(schema="not-a-dict")
        with pytest.raises(ValueError, match="must provide an input schema dictionary"):
            from kernle.mcp.server import register_plugin_tools

            register_plugin_tools("test", [td])

    def test_rejects_missing_type_field(self):
        """Schema missing 'type' field raises ValueError."""
        td = self._make_td(schema={"properties": {}})
        with pytest.raises(ValueError, match="must provide an input schema"):
            from kernle.mcp.server import register_plugin_tools

            register_plugin_tools("test", [td])

    def test_rejects_non_object_type(self):
        """Schema with type != 'object' raises ValueError."""
        td = self._make_td(schema={"type": "array", "items": {"type": "string"}})
        with pytest.raises(ValueError, match="must use an object input schema"):
            from kernle.mcp.server import register_plugin_tools

            register_plugin_tools("test", [td])

    def test_rejects_non_dict_properties(self, monkeypatch):
        """Schema with 'properties' that's not a dict raises ValueError.

        Must disable jsonschema since it catches this before manual checks.
        """
        import kernle.mcp.server as srv

        monkeypatch.setattr(srv, "_HAS_JSONSCHEMA", False)
        td = self._make_td(schema={"type": "object", "properties": "not-dict"})
        with pytest.raises(ValueError, match="'properties' must be an object"):
            srv.register_plugin_tools("test", [td])

    def test_rejects_non_list_required(self, monkeypatch):
        """Schema with 'required' that's not a list raises ValueError.

        Must disable jsonschema since it catches this before manual checks.
        """
        import kernle.mcp.server as srv

        monkeypatch.setattr(srv, "_HAS_JSONSCHEMA", False)
        td = self._make_td(schema={"type": "object", "properties": {}, "required": "not-list"})
        with pytest.raises(ValueError, match="'required' must be an array"):
            srv.register_plugin_tools("test", [td])

    def test_rejects_invalid_jsonschema(self):
        """Invalid JSON schema (caught by jsonschema.check_schema) raises ValueError."""
        td = self._make_td(schema={"type": "object", "properties": {"x": {"type": "nonsense"}}})
        with pytest.raises(ValueError, match="invalid schema"):
            from kernle.mcp.server import register_plugin_tools

            register_plugin_tools("test", [td])

    def test_successful_registration(self):
        """Valid tool definition registers correctly."""
        from kernle.mcp.server import register_plugin_tools

        def handler(args):
            return "ok"

        td = self._make_td(
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=handler,
        )
        register_plugin_tools("myplugin", [td])

        assert "myplugin.tool1" in _plugin_tools
        assert "myplugin.tool1" in _plugin_schemas
        assert "myplugin.tool1" in _plugin_handlers
        assert _plugin_handlers["myplugin.tool1"] is handler

    def test_registration_without_handler(self):
        """Tool without handler registers but has no handler entry."""
        from kernle.mcp.server import register_plugin_tools

        td = self._make_td(handler=None)
        register_plugin_tools("nohandler", [td])

        assert "nohandler.tool1" in _plugin_tools
        assert "nohandler.tool1" not in _plugin_handlers

    def test_unregister_clears_all_registries(self):
        """unregister_plugin_tools removes all entries for a plugin."""
        from kernle.mcp.server import register_plugin_tools, unregister_plugin_tools

        td = self._make_td(handler=lambda args: "ok")
        register_plugin_tools("cleanup", [td])
        assert "cleanup.tool1" in _plugin_tools

        unregister_plugin_tools("cleanup")
        assert "cleanup.tool1" not in _plugin_tools
        assert "cleanup.tool1" not in _plugin_handlers
        assert "cleanup.tool1" not in _plugin_schemas
        assert "cleanup.tool1" not in _plugin_schema_validators


class TestValidatePluginToolInput:
    """Tests for _validate_plugin_tool_input() edge cases."""

    def setup_method(self):
        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

    def test_oversized_payload_rejected(self):
        """Payload exceeding 64 KB is rejected."""
        from kernle.mcp.server import _validate_plugin_tool_input, register_plugin_tools
        from kernle.protocols import ToolDefinition

        td = ToolDefinition(
            name="big",
            description="big tool",
            input_schema={"type": "object", "properties": {"data": {"type": "string"}}},
            handler=lambda args: "ok",
        )
        register_plugin_tools("size", [td])

        big_data = {"data": "x" * (64 * 1024 + 1)}
        with pytest.raises(ValueError, match="payload too large"):
            _validate_plugin_tool_input("size.big", big_data)

    def test_missing_schema_rejected(self):
        """Tool name not in schema registry is rejected."""
        from kernle.mcp.server import _validate_plugin_tool_input

        with pytest.raises(ValueError, match="plugin tool metadata missing"):
            _validate_plugin_tool_input("unknown.tool", {"key": "val"})

    def test_schema_validation_error_reports_path(self):
        """Schema validation error includes the failing path."""
        from kernle.mcp.server import _validate_plugin_tool_input, register_plugin_tools
        from kernle.protocols import ToolDefinition

        td = ToolDefinition(
            name="typed",
            description="typed tool",
            input_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            handler=lambda args: "ok",
        )
        register_plugin_tools("schema", [td])

        with pytest.raises(ValueError, match="Schema validation failed"):
            _validate_plugin_tool_input("schema.typed", {"count": "not-int"})

    def test_valid_input_returns_dict(self):
        """Valid input passes validation and returns a dict."""
        from kernle.mcp.server import _validate_plugin_tool_input, register_plugin_tools
        from kernle.protocols import ToolDefinition

        td = ToolDefinition(
            name="ok",
            description="ok tool",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            handler=lambda args: "ok",
        )
        register_plugin_tools("valid", [td])

        result = _validate_plugin_tool_input("valid.ok", {"name": "test"})
        assert result == {"name": "test"}


class TestFallbackSchemaValidation:
    """Tests for _validate_fallback_schema() when jsonschema is not available."""

    def test_missing_required_property(self):
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"required": ["name"], "properties": {"name": {"type": "string"}}}
        with pytest.raises(ValueError, match="Missing required property: name"):
            _validate_fallback_schema({}, schema)

    def test_unexpected_properties_with_additional_false(self):
        from kernle.mcp.server import _validate_fallback_schema

        schema = {
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValueError, match="Unexpected properties: extra"):
            _validate_fallback_schema({"name": "ok", "extra": "bad"}, schema)

    def test_type_mismatch_string(self):
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"name": {"type": "string"}}}
        with pytest.raises(ValueError, match="invalid type"):
            _validate_fallback_schema({"name": 42}, schema)

    def test_type_mismatch_integer(self):
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"count": {"type": "integer"}}}
        with pytest.raises(ValueError, match="invalid type"):
            _validate_fallback_schema({"count": "not-int"}, schema)

    def test_type_mismatch_boolean(self):
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"flag": {"type": "boolean"}}}
        with pytest.raises(ValueError, match="invalid type"):
            _validate_fallback_schema({"flag": "yes"}, schema)

    def test_type_union_accepts_valid(self):
        """Union type (list) accepts any matching type."""
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"val": {"type": ["string", "null"]}}}
        # Should not raise — None matches "null"
        _validate_fallback_schema({"val": None}, schema)

    def test_type_union_rejects_invalid(self):
        """Union type (list) rejects non-matching types."""
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"val": {"type": ["string", "null"]}}}
        with pytest.raises(ValueError, match="invalid type"):
            _validate_fallback_schema({"val": 42}, schema)

    def test_enum_validation_accepts_valid(self):
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"color": {"type": "string", "enum": ["red", "blue"]}}}
        # Should not raise
        _validate_fallback_schema({"color": "red"}, schema)

    def test_enum_validation_rejects_invalid(self):
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"color": {"type": "string", "enum": ["red", "blue"]}}}
        with pytest.raises(ValueError, match="must be one of"):
            _validate_fallback_schema({"color": "green"}, schema)

    def test_additional_properties_true_allows_extras(self):
        """When additionalProperties is True (default), extra keys are fine."""
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"name": {"type": "string"}}}
        # Should not raise
        _validate_fallback_schema({"name": "ok", "extra": "fine"}, schema)

    def test_no_properties_in_schema(self):
        """Schema with no properties allows anything."""
        from kernle.mcp.server import _validate_fallback_schema

        # Should not raise
        _validate_fallback_schema({"anything": "goes"}, {})

    def test_property_without_dict_schema_skipped(self):
        """Non-dict property schema is skipped (no validation)."""
        from kernle.mcp.server import _validate_fallback_schema

        schema = {"properties": {"weird": "not-a-dict"}}
        # Should not raise — non-dict property schema is ignored
        _validate_fallback_schema({"weird": 42}, schema)


class TestJsonTypeMatches:
    """Tests for _json_type_matches() helper."""

    def test_object_type(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches({"key": "val"}, "object") is True
        assert _json_type_matches([1, 2], "object") is False

    def test_array_type(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches([1, 2], "array") is True
        assert _json_type_matches({"key": "val"}, "array") is False

    def test_string_type(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches("hello", "string") is True
        assert _json_type_matches(42, "string") is False

    def test_integer_type_excludes_bool(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches(42, "integer") is True
        assert _json_type_matches(True, "integer") is False

    def test_number_type_excludes_bool(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches(3.14, "number") is True
        assert _json_type_matches(42, "number") is True
        assert _json_type_matches(True, "number") is False

    def test_boolean_type(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches(True, "boolean") is True
        assert _json_type_matches(False, "boolean") is True
        assert _json_type_matches(1, "boolean") is False

    def test_null_type(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches(None, "null") is True
        assert _json_type_matches("", "null") is False

    def test_unknown_type_returns_true(self):
        from kernle.mcp.server import _json_type_matches

        assert _json_type_matches("anything", "unknown_type") is True


class TestFallbackSchemaValidationIntegration:
    """Test the full fallback path when jsonschema is mocked away."""

    def test_plugin_validation_falls_back_without_jsonschema(self, monkeypatch):
        """When _HAS_JSONSCHEMA is False, fallback validation is used."""
        import kernle.mcp.server as srv
        from kernle.protocols import ToolDefinition

        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

        td = ToolDefinition(
            name="fb",
            description="fallback tool",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=lambda args: "ok",
        )

        # Must register while _HAS_JSONSCHEMA is True (for validation to pass)
        srv.register_plugin_tools("fallback", [td])

        # Now simulate no jsonschema for the validation path
        monkeypatch.setattr(srv, "_HAS_JSONSCHEMA", False)
        # Remove compiled validator so fallback path triggers
        _plugin_schema_validators.pop("fallback.fb", None)

        try:
            # Valid input should pass fallback validation
            result = srv._validate_plugin_tool_input("fallback.fb", {"name": "test"})
            assert result == {"name": "test"}

            # Missing required property should fail
            with pytest.raises(ValueError, match="Missing required property"):
                srv._validate_plugin_tool_input("fallback.fb", {})
        finally:
            srv.unregister_plugin_tools("fallback")


class TestValidateToolInputPluginPath:
    """Test validate_tool_input routing to plugin validation."""

    def setup_method(self):
        _plugin_tools.clear()
        _plugin_handlers.clear()
        _plugin_schemas.clear()
        _plugin_schema_validators.clear()

    def test_unknown_tool_raises(self):
        """Unknown tool name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid input"):
            validate_tool_input("completely_unknown_tool", {})

    def test_plugin_tool_routes_to_plugin_validation(self):
        """Plugin tool names route through _validate_plugin_tool_input."""
        from kernle.mcp.server import register_plugin_tools
        from kernle.protocols import ToolDefinition

        td = ToolDefinition(
            name="route",
            description="route test",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
            handler=lambda args: "ok",
        )
        register_plugin_tools("routing", [td])

        try:
            result = validate_tool_input("routing.route", {"x": 42})
            assert result == {"x": 42}
        finally:
            from kernle.mcp.server import unregister_plugin_tools

            unregister_plugin_tools("routing")


class TestMainModule:
    """Tests for if __name__ == '__main__' block — covers line 217."""

    def test_dunder_main_invokes_main(self):
        """Running server.py as __main__ calls main() at line 217."""
        import importlib
        import sys

        # Remove the module from sys.modules so runpy can re-execute it
        saved_modules = {}
        for key in list(sys.modules):
            if key == "kernle.mcp.server" or key.startswith("kernle.mcp.server."):
                saved_modules[key] = sys.modules.pop(key)

        try:
            # Patch asyncio.run to prevent the real server from starting
            with patch("asyncio.run", side_effect=lambda coro: coro.close()):
                runpy.run_module("kernle.mcp.server", run_name="__main__", alter_sys=False)
        finally:
            # Restore original modules
            sys.modules.update(saved_modules)
            # Re-import to restore normal state
            importlib.import_module("kernle.mcp.server")
