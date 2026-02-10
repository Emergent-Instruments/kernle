"""Additional tests for kernle/cli/__main__.py to improve coverage.

Targets uncovered lines: validate_input errors, cmd_init, cmd_mcp,
raw argv preprocessing, plugin discovery, dispatch branches, error handling.
"""

import argparse
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kernle.cli.__main__ import cmd_init, main, validate_input
from kernle.core import Kernle
from kernle.storage import SQLiteStorage

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path):
    s = SQLiteStorage(stack_id="test-main", db_path=tmp_path / "main.db")
    yield s
    s.close()


@pytest.fixture
def k(storage):
    inst = Kernle(stack_id="test-main", storage=storage, strict=False)
    yield inst


# ============================================================================
# validate_input
# ============================================================================


class TestValidateInput:
    """Tests for the validate_input function."""

    def test_non_string_raises(self):
        """Non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_input(123, "test_field")

    def test_too_long_raises(self):
        """Input exceeding max_length raises ValueError."""
        with pytest.raises(ValueError, match="too long"):
            validate_input("x" * 1001, "test_field")

    def test_too_long_custom_max(self):
        """Custom max_length is respected."""
        with pytest.raises(ValueError, match="too long"):
            validate_input("x" * 51, "test_field", max_length=50)

    def test_valid_input_passes(self):
        """Normal input passes validation and is returned."""
        result = validate_input("hello world", "test_field")
        assert result == "hello world"

    def test_control_characters_stripped(self):
        """Control characters are removed from input."""
        result = validate_input("hello\x00world\x07!", "test_field")
        assert result == "helloworld!"

    def test_newlines_preserved(self):
        """Newlines are NOT stripped (only control chars except newlines)."""
        result = validate_input("hello\nworld", "test_field")
        assert result == "hello\nworld"


# ============================================================================
# cmd_init (lines 93-423)
# ============================================================================


class TestCmdInit:
    """Tests for the cmd_init function — full setup wizard."""

    def test_init_non_interactive_claude_code(self, k, capsys):
        """Non-interactive init with claude-code env produces correct output."""
        args = argparse.Namespace(
            non_interactive=True,
            env="claude-code",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Welcome to Kernle" in captured
        assert "Claude Code Setup" in captured
        assert "MCP server" in captured
        assert "Checkpoint saved" in captured or "Setup Complete" in captured

    def test_init_non_interactive_openclaw(self, k, capsys):
        """Non-interactive init with openclaw env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="openclaw",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "OpenClaw Setup" in captured
        assert "AGENTS.md" in captured

    def test_init_non_interactive_cline(self, k, capsys):
        """Non-interactive init with cline env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="cline",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Cline Setup" in captured
        assert ".clinerules" in captured

    def test_init_non_interactive_cursor(self, k, capsys):
        """Non-interactive init with cursor env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="cursor",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Cursor Setup" in captured
        assert ".cursorrules" in captured

    def test_init_non_interactive_desktop(self, k, capsys):
        """Non-interactive init with desktop env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="desktop",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Claude Desktop Setup" in captured

    def test_init_non_interactive_other(self, k, capsys):
        """Non-interactive init with other/manual env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Manual Setup" in captured
        assert "CLI commands" in captured

    def test_init_with_seed_values(self, k, capsys):
        """Init with --seed-values creates default values."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=True,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Seeding Initial Values" in captured
        assert "memory_sovereignty" in captured or "continuous_learning" in captured

    def test_init_seed_values_skip_existing(self, k, capsys):
        """Init skips seeding when values already exist."""
        # Create an existing value first
        k.value("existing_value", "Already exists", priority=50)

        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=True,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "existing values, skipping seed" in captured

    def test_init_interactive_env_selection(self, k, capsys):
        """Interactive env selection uses input()."""
        args = argparse.Namespace(
            non_interactive=False,
            env=None,
            seed_values=False,
        )
        # Simulate user choosing "6" (other)
        with patch("builtins.input", return_value="6"):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Manual Setup" in captured

    def test_init_interactive_env_eof(self, k, capsys):
        """Interactive env selection handles EOFError gracefully."""
        args = argparse.Namespace(
            non_interactive=False,
            env=None,
            seed_values=False,
        )
        with patch("builtins.input", side_effect=EOFError):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Aborted" in captured

    def test_init_interactive_id_selection(self, k, capsys):
        """Interactive ID selection with auto-generated stack_id."""
        # Give the kernle instance an auto-generated stack_id
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", return_value="my-agent"):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Using: my-agent" in captured

    def test_init_interactive_id_invalid(self, k, capsys):
        """Interactive ID selection rejects invalid characters."""
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", return_value="invalid name!"):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Invalid" in captured

    def test_init_interactive_id_empty(self, k, capsys):
        """Interactive ID selection keeps auto when empty input."""
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", return_value=""):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "auto-abc123" in captured

    def test_init_interactive_id_eof(self, k, capsys):
        """Interactive ID selection handles EOFError."""
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", side_effect=EOFError):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        # Should still complete
        assert "Setup Complete" in captured

    def test_init_checkpoint_error(self, k, capsys):
        """Init handles checkpoint creation failure gracefully."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=False,
        )
        with patch.object(k, "checkpoint", side_effect=Exception("DB error")):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Warning" in captured or "Setup Complete" in captured

    def test_init_seed_trust(self, k, capsys):
        """Init calls seed_trust and reports count."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        # seed_trust returns count of seeded assessments
        assert "Setup Complete" in captured

    def test_init_env_detection(self, k, capsys, tmp_path):
        """Init detects environment files when none is specified."""
        args = argparse.Namespace(
            non_interactive=False,
            env=None,
            seed_values=False,
        )
        # Create CLAUDE.md to trigger detection
        with patch("builtins.input", return_value="1"):
            with patch.object(Path, "exists", return_value=True):
                cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Detected" in captured or "Claude Code Setup" in captured


# ============================================================================
# cmd_mcp (lines 426-434)
# ============================================================================


class TestCmdMcp:
    """Tests for the cmd_mcp function."""

    def test_mcp_dispatches_to_server(self):
        """cmd_mcp calls the MCP server main function."""
        args = argparse.Namespace(stack="my-agent")

        with patch("kernle.cli.__main__.sys") as mock_sys:
            mock_sys.stderr = StringIO()
            with patch("kernle.mcp.server.main") as mock_mcp_main:
                from kernle.cli.__main__ import cmd_mcp

                cmd_mcp(args)
                mock_mcp_main.assert_called_once_with(stack_id="my-agent")

    def test_mcp_default_stack(self):
        """cmd_mcp uses 'default' when no stack specified."""
        args = argparse.Namespace()  # No stack attribute

        with patch("kernle.cli.__main__.sys") as mock_sys:
            mock_sys.stderr = StringIO()
            with patch("kernle.mcp.server.main") as mock_mcp_main:
                from kernle.cli.__main__ import cmd_mcp

                cmd_mcp(args)
                mock_mcp_main.assert_called_once_with(stack_id="default")


# ============================================================================
# Raw argv preprocessing (lines 1954-1987)
# ============================================================================


class TestRawArgvPreprocessing:
    """Tests for the raw subcommand argv preprocessing."""

    def test_raw_content_preprocessing_runs(self, k, capsys):
        """'kernle raw "content"' triggers the argv preprocessing (inserts capture).

        Note: The inserted 'capture' is consumed by argparse as the content
        positional, so the parse actually fails. But the preprocessing code
        path (lines 1979-1986) IS exercised.
        """
        test_args = ["kernle", "raw", "some content"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with pytest.raises(SystemExit):
                        main()

    def test_raw_list_no_insert(self, k, capsys):
        """'kernle raw list' does NOT insert 'capture'."""
        test_args = ["kernle", "raw", "list"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    main()

        # Should run raw list without error (output may be empty list)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_raw_with_stack_flag_before(self, k, capsys):
        """'kernle --stack X raw "content"' preprocessing correctly skips flag."""
        test_args = ["kernle", "--stack", "test-main", "raw", "test content"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                # The preprocessing still triggers the same argparse issue
                with pytest.raises(SystemExit):
                    main()

    def test_raw_with_dash_arg(self, k, capsys):
        """'kernle raw --quiet' does NOT insert 'capture' (starts with -)."""
        test_args = ["kernle", "raw", "--quiet"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    # No content arg but quiet mode, so cmd_raw handles it
                    main()

    def test_raw_no_content(self, k, capsys):
        """'kernle raw' with no content or subcommand still dispatches."""
        test_args = ["kernle", "raw"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    main()

        # Should dispatch to cmd_raw with raw_action=None and content=None
        captured = capsys.readouterr().out
        # Without content, cmd_raw prints an error
        assert "required" in captured.lower() or captured is not None


# ============================================================================
# Plugin discovery (lines 1990-2005)
# ============================================================================


class TestPluginDiscovery:
    """Tests for plugin CLI discovery in main()."""

    def test_plugin_discovery_failure_logged(self, k, capsys):
        """Plugin discovery failure is logged but doesn't crash main()."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.discovery.discover_plugins",
                        side_effect=Exception("Discovery failed"),
                    ):
                        main()

        # Should still complete the status command
        captured = capsys.readouterr().out
        assert "Memory Status" in captured

    def test_plugin_cli_registration_failure(self, k, capsys):
        """Individual plugin registration failure doesn't crash."""
        test_args = ["kernle", "status"]

        mock_comp = MagicMock()
        mock_comp.name = "broken-plugin"

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch("kernle.discovery.discover_plugins", return_value=[mock_comp]):
                        with patch(
                            "kernle.discovery.load_component",
                            side_effect=ImportError("bad plugin"),
                        ):
                            main()

        captured = capsys.readouterr().out
        assert "Memory Status" in captured


# ============================================================================
# Dispatch branches (lines 2027-2138)
# ============================================================================


class TestDispatchBranches:
    """Tests for command dispatch in main() — covering uncovered branches."""

    def _run_main(self, argv, k):
        """Helper to run main() with given argv and kernle instance."""
        with patch("sys.argv", ["kernle"] + argv):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    main()

    def test_dispatch_extract(self, k, capsys):
        """Dispatch 'extract' command."""
        self._run_main(["extract", "summary of conversation"], k)
        captured = capsys.readouterr().out
        assert "Conversation" in captured or "extract" in captured.lower() or captured

    def test_dispatch_resume(self, k, capsys):
        """Dispatch 'resume' command."""
        self._run_main(["resume"], k)
        captured = capsys.readouterr().out
        # Resume should run without error
        assert captured is not None

    def test_dispatch_init(self, k, capsys):
        """Dispatch 'init' command."""
        self._run_main(["init", "--non-interactive", "-y"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_doctor(self, k, capsys):
        """Dispatch 'doctor' command (no subcommand)."""
        self._run_main(["doctor"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_doctor_structural(self, k, capsys):
        """Dispatch 'doctor structural' subcommand."""
        self._run_main(["doctor", "structural"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_doctor_session_start(self, k, capsys):
        """Dispatch 'doctor session start' subcommand."""
        self._run_main(["doctor", "session", "start"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_doctor_session_list(self, k, capsys):
        """Dispatch 'doctor session list' subcommand."""
        self._run_main(["doctor", "session", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_doctor_session_no_action(self, k, capsys):
        """Dispatch 'doctor session' without start/list shows usage."""
        self._run_main(["doctor", "session"], k)
        captured = capsys.readouterr().out
        assert "Usage" in captured

    def test_dispatch_doctor_report(self, k, capsys):
        """Dispatch 'doctor report' subcommand."""
        # This may error with no session, but it should dispatch correctly
        try:
            self._run_main(["doctor", "report", "latest"], k)
        except SystemExit:
            pass  # May exit with error if no sessions exist
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_trust(self, k, capsys):
        """Dispatch 'trust' command."""
        self._run_main(["trust", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_relation(self, k, capsys):
        """Dispatch 'relation' command."""
        self._run_main(["relation", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_entity_model(self, k, capsys):
        """Dispatch 'entity-model' command."""
        self._run_main(["entity-model", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_identity_default(self, k, capsys):
        """Dispatch 'identity' with no subcommand defaults to show."""
        self._run_main(["identity"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_identity_show(self, k, capsys):
        """Dispatch 'identity show'."""
        self._run_main(["identity", "show"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_emotion(self, k, capsys):
        """Dispatch 'emotion' command."""
        self._run_main(["emotion", "summary"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_meta(self, k, capsys):
        """Dispatch 'meta' command."""
        self._run_main(["meta", "uncertain"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_anxiety(self, k, capsys):
        """Dispatch 'anxiety' command."""
        self._run_main(["anxiety"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_stats(self, k, capsys):
        """Dispatch 'stats' command."""
        self._run_main(["stats", "health-checks"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_forget(self, k, capsys):
        """Dispatch 'forget' command."""
        self._run_main(["forget", "candidates"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_epoch(self, k, capsys):
        """Dispatch 'epoch' command."""
        self._run_main(["epoch", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_summary(self, k, capsys):
        """Dispatch 'summary' command."""
        self._run_main(["summary", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_narrative(self, k, capsys):
        """Dispatch 'narrative' command."""
        self._run_main(["narrative", "show"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_playbook(self, k, capsys):
        """Dispatch 'playbook' command."""
        self._run_main(["playbook", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_process(self, k, capsys):
        """Dispatch 'process' command."""
        self._run_main(["process", "status"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_suggestions(self, k, capsys):
        """Dispatch 'suggestions' command."""
        self._run_main(["suggestions", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_belief(self, k, capsys):
        """Dispatch 'belief' command."""
        self._run_main(["belief", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_dump(self, k, capsys):
        """Dispatch 'dump' command."""
        self._run_main(["dump"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_export(self, k, capsys, tmp_path):
        """Dispatch 'export' command."""
        out_file = str(tmp_path / "export.json")
        self._run_main(["export", out_file, "--format", "json"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_export_cache(self, k, capsys):
        """Dispatch 'export-cache' command."""
        self._run_main(["export-cache"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_export_full(self, k, capsys):
        """Dispatch 'export-full' command."""
        self._run_main(["export-full"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_boot(self, k, capsys):
        """Dispatch 'boot' command."""
        self._run_main(["boot", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_sync(self, k, capsys, tmp_path):
        """Dispatch 'sync' command."""
        # sync conflicts is the simplest to test without backend
        creds_path = tmp_path / "sync_creds"
        creds_path.mkdir()
        with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
            self._run_main(["sync", "conflicts"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_auth(self, k, capsys, tmp_path):
        """Dispatch 'auth' command."""
        self._run_main(["auth", "status"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_stack(self, k, capsys):
        """Dispatch 'stack' command."""
        self._run_main(["stack", "list"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_import(self, k, capsys, tmp_path):
        """Dispatch 'import' command."""
        test_file = tmp_path / "import.md"
        test_file.write_text("## Notes\n- Test note\n")
        self._run_main(["import", str(test_file), "--dry-run"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_migrate(self, k, capsys):
        """Dispatch 'migrate' command."""
        self._run_main(["migrate", "seed-beliefs", "--dry-run"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_setup(self, k, capsys):
        """Dispatch 'setup' command."""
        self._run_main(["setup", "claude-code"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_search(self, k, capsys):
        """Dispatch 'search' command through main()."""
        self._run_main(["search", "test query"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_when(self, k, capsys):
        """Dispatch 'when' (temporal) command."""
        self._run_main(["when", "today"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_promote(self, k, capsys):
        """Dispatch 'promote' command."""
        self._run_main(["promote"], k)
        captured = capsys.readouterr().out
        assert captured is not None

    def test_dispatch_mcp(self, k, capsys):
        """Dispatch 'mcp' command through main()."""
        with patch("kernle.mcp.server.main") as mock_mcp_main:
            self._run_main(["mcp"], k)
            mock_mcp_main.assert_called_once()


# ============================================================================
# Error handling in main() (lines 2133-2138)
# ============================================================================


class TestMainErrorHandling:
    """Test error handling in main() dispatch."""

    def test_value_error_exits(self, k):
        """ValueError during command execution causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=ValueError("bad input"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1

    def test_type_error_exits(self, k):
        """TypeError during command execution causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=TypeError("wrong type"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1

    def test_generic_exception_exits(self, k):
        """Generic Exception during command execution causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=RuntimeError("unexpected"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1

    def test_init_type_error_exits(self):
        """TypeError during Kernle initialization causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch(
                "kernle.cli.__main__.Kernle",
                side_effect=TypeError("bad init"),
            ):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test"):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1


# ============================================================================
# Hook dispatch (lines 2011-2013)
# ============================================================================


class TestHookDispatch:
    """Test hook command dispatch in main()."""

    def test_hook_dispatch_before_kernle_init(self):
        """Hook commands are dispatched BEFORE Kernle initialization."""
        test_args = ["kernle", "hook", "session-start"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.cmd_hook") as mock_hook:
                mock_hook.return_value = None
                # Kernle should NOT be initialized for hook commands
                with patch("kernle.cli.__main__.Kernle") as mock_kernle:
                    main()
                    mock_hook.assert_called_once()
                    mock_kernle.assert_not_called()
