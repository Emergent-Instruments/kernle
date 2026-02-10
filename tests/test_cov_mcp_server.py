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
    _plugin_tools,
    call_tool,
    get_kernle,
    handle_tool_error,
    main,
    run_server,
    set_stack_id,
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
        # Register a fake plugin handler
        _plugin_handlers["fake_plugin.do_thing"] = lambda args: f"result: {args['x']}"
        # Also register in _plugin_tools so validate_tool_input doesn't reject it
        _plugin_tools["fake_plugin.do_thing"] = MagicMock()

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

    @pytest.mark.asyncio
    async def test_plugin_handler_returns_dict(self, monkeypatch):
        """Plugin handler returning a dict is JSON-serialized."""
        _plugin_handlers["fake_plugin.info"] = lambda args: {"status": "ok", "count": 3}
        _plugin_tools["fake_plugin.info"] = MagicMock()

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
