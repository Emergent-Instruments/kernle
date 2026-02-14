"""Tests for CLI dispatch logic in kernle/cli/__main__.py.

Targets coverage for:
- main() argparse error handling (missing command, bad args)
- _import_devtools error paths (missing package, version error)
- cmd_mcp validation
- Kernle init failure path in main()
- Command exception handling in dispatch
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from kernle.cli.__main__ import _import_devtools, cmd_mcp, main

# =============================================================================
# main() - Missing / Invalid Command
# =============================================================================


class TestMainDispatchErrors:
    """Tests for the main() function dispatch error paths."""

    def test_no_command_exits_with_error(self):
        """Calling main() with no arguments should exit with error code 2.

        argparse has `required=True` on subparsers, so omitting
        the command triggers a SystemExit(2).
        """
        with patch("sys.argv", ["kernle"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # argparse exits with code 2 for usage errors
        assert exc_info.value.code == 2

    def test_unknown_command_exits_with_error(self):
        """Calling main() with an unrecognized command should exit with error."""
        with patch("sys.argv", ["kernle", "totally-fake-command"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 2

    def test_invalid_stack_id_exits(self):
        """Calling main() with a stack_id containing path separators exits 1."""
        with patch("sys.argv", ["kernle", "--stack", "a/b", "status"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # SQLiteStorage raises ValueError for path separators in stack_id
        assert exc_info.value.code == 1

    def test_kernle_init_failure_exits_1(self):
        """When Kernle() constructor raises ValueError, main() exits with code 1."""
        with patch("sys.argv", ["kernle", "--stack", "good-stack", "status"]):
            with patch("kernle.cli.__main__.Kernle", side_effect=ValueError("boom")):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_command_runtime_error_exits_1(self):
        """When a command handler raises an exception, main() exits with code 1."""
        mock_kernle = MagicMock()
        with patch("sys.argv", ["kernle", "--stack", "test", "status"]):
            with patch("kernle.cli.__main__.Kernle", return_value=mock_kernle):
                with patch(
                    "kernle.cli.__main__.resolve_stack_id",
                    return_value="test",
                ):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=RuntimeError("oops"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

        assert exc_info.value.code == 1

    def test_value_error_in_dispatch_exits_1(self):
        """When a command handler raises ValueError, main() exits with code 1."""
        mock_kernle = MagicMock()
        with patch("sys.argv", ["kernle", "status"]):
            with patch("kernle.cli.__main__.Kernle", return_value=mock_kernle):
                with patch(
                    "kernle.cli.__main__.resolve_stack_id",
                    return_value="test",
                ):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=ValueError("bad input"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

        assert exc_info.value.code == 1


# =============================================================================
# _import_devtools
# =============================================================================


class TestImportDevtools:
    """Tests for the _import_devtools helper function."""

    def test_missing_devtools_package_exits(self):
        """When kernle_devtools is not installed, _import_devtools prints message and exits."""
        error = ModuleNotFoundError("No module named 'kernle_devtools'")
        error.name = "kernle_devtools"

        with patch("kernle.cli.__main__.importlib.import_module", side_effect=error):
            with pytest.raises(SystemExit) as exc_info:
                _import_devtools("kernle_devtools.admin_health", "some_function")

        assert exc_info.value.code == 2

    def test_unrelated_module_not_found_propagates(self):
        """When a non-devtools ModuleNotFoundError occurs, it propagates."""
        error = ModuleNotFoundError("No module named 'numpy'")
        error.name = "numpy"

        with patch("kernle.cli.__main__.importlib.import_module", side_effect=error):
            with pytest.raises(ModuleNotFoundError, match="numpy"):
                _import_devtools("kernle_devtools.admin_health", "some_function")

    def test_devtools_version_error_exits(self):
        """When DevtoolsVersionError is raised, _import_devtools exits with code 2."""

        # Create a custom ImportError subclass that mimics DevtoolsVersionError
        class DevtoolsVersionError(ImportError):
            __module__ = "kernle_devtools.compat"

        error = DevtoolsVersionError("version mismatch: need >=1.0")
        error.__class__.__name__ = "DevtoolsVersionError"

        with patch("kernle.cli.__main__.importlib.import_module", side_effect=error):
            with pytest.raises(SystemExit) as exc_info:
                _import_devtools("kernle_devtools.admin_health", "some_function")

        assert exc_info.value.code == 2

    def test_unrelated_import_error_propagates(self):
        """When a generic ImportError occurs (not DevtoolsVersionError), it propagates."""
        error = ImportError("some random import failure")

        with patch("kernle.cli.__main__.importlib.import_module", side_effect=error):
            with pytest.raises(ImportError, match="some random import failure"):
                _import_devtools("kernle_devtools.admin_health", "some_function")

    def test_successful_import_returns_symbol(self):
        """When the module and symbol exist, _import_devtools returns the symbol."""
        mock_module = MagicMock()
        mock_module.my_function = lambda: "hello"

        with patch("kernle.cli.__main__.importlib.import_module", return_value=mock_module):
            result = _import_devtools("some.module.path", "my_function")

        assert result is mock_module.my_function

    def test_module_not_found_with_none_name_propagates(self):
        """When ModuleNotFoundError has name=None, it propagates (not devtools)."""
        error = ModuleNotFoundError("weird error")
        error.name = None

        with patch("kernle.cli.__main__.importlib.import_module", side_effect=error):
            with pytest.raises(ModuleNotFoundError):
                _import_devtools("kernle_devtools.something", "func")


# =============================================================================
# cmd_mcp validation
# =============================================================================


class TestCmdMcpValidation:
    """Tests for cmd_mcp stack_id validation."""

    def test_mcp_no_stack_uses_default(self):
        """When no stack is provided, cmd_mcp uses 'default'."""
        args = argparse.Namespace(stack=None)

        with patch("kernle.mcp.server.main") as mock_mcp:
            cmd_mcp(args)
            mock_mcp.assert_called_once_with(stack_id="default")

    def test_mcp_valid_stack(self):
        """When a valid stack is provided, cmd_mcp passes it through."""
        args = argparse.Namespace(stack="my-agent")

        with patch("kernle.mcp.server.main") as mock_mcp:
            cmd_mcp(args)
            mock_mcp.assert_called_once_with(stack_id="my-agent")

    def test_mcp_empty_stack_raises(self):
        """When stack is an empty string, cmd_mcp exits with code 2."""
        args = argparse.Namespace(stack="")

        with pytest.raises(SystemExit) as exc_info:
            cmd_mcp(args)

        assert exc_info.value.code == 2

    def test_mcp_whitespace_stack_raises(self):
        """When stack is whitespace, cmd_mcp exits with code 2."""
        args = argparse.Namespace(stack="   ")

        with pytest.raises(SystemExit) as exc_info:
            cmd_mcp(args)

        assert exc_info.value.code == 2

    def test_mcp_non_string_stack_raises(self):
        """When stack is not a string, cmd_mcp exits with code 2."""
        args = argparse.Namespace(stack=12345)

        with pytest.raises(SystemExit) as exc_info:
            cmd_mcp(args)

        assert exc_info.value.code == 2
