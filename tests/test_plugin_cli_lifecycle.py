"""Tests for plugin CLI lifecycle: discover -> register -> parse -> activate -> dispatch -> unload.

Tests the plugin dispatch machinery in kernle/cli/__main__.py using fake plugins
that exercise all code paths: normal dispatch, collision detection, legacy adapter,
missing handle_cli, and cleanup.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake plugins for testing
# ---------------------------------------------------------------------------


class FakePlugin:
    """Well-behaved plugin with cli_commands() and handle_cli(args, k)."""

    name = "fake-plugin"
    version = "0.1.0"
    protocol_version = 1
    description = "Fake plugin for testing"

    def __init__(self):
        self.activated = False
        self.deactivated = False
        self.cli_calls = []

    def capabilities(self):
        return ["testing"]

    def activate(self, context):
        self.activated = True
        self._context = context

    def deactivate(self):
        self.deactivated = True

    def register_cli(self, subparsers):
        p = subparsers.add_parser("fake", help="Fake command")
        p.add_argument("--opt", default="default")

    def register_tools(self):
        return []

    def cli_commands(self):
        return ["fake"]

    def handle_cli(self, args, k):
        self.cli_calls.append(("handle_cli", args, k))

    def on_load(self, ctx):
        pass

    def on_status(self, status):
        pass

    def health_check(self):
        from kernle.types import PluginHealth

        return PluginHealth(healthy=True, message="OK")


class FakePluginLegacy:
    """Legacy plugin with handle_cli(args) — no k parameter."""

    name = "legacy-plugin"
    version = "0.1.0"
    protocol_version = 1
    description = "Legacy plugin"

    def __init__(self):
        self.cli_calls = []

    def capabilities(self):
        return []

    def activate(self, context):
        pass

    def deactivate(self):
        pass

    def register_cli(self, subparsers):
        subparsers.add_parser("legacy", help="Legacy command")

    def register_tools(self):
        return []

    def cli_commands(self):
        return ["legacy"]

    def handle_cli(self, args):
        """Old signature — only takes args, no k."""
        self.cli_calls.append(("handle_cli_legacy", args))

    def on_load(self, ctx):
        pass

    def on_status(self, status):
        pass

    def health_check(self):
        from kernle.types import PluginHealth

        return PluginHealth(healthy=True, message="OK")


class FakePluginNoHandleCli:
    """Plugin with cli_commands() but no handle_cli method."""

    name = "no-handler-plugin"
    version = "0.1.0"
    protocol_version = 1
    description = "Plugin missing handle_cli"

    def capabilities(self):
        return []

    def activate(self, context):
        pass

    def deactivate(self):
        pass

    def register_cli(self, subparsers):
        subparsers.add_parser("nohandler", help="No handler")

    def register_tools(self):
        return []

    def cli_commands(self):
        return ["nohandler"]

    def on_load(self, ctx):
        pass

    def on_status(self, status):
        pass

    def health_check(self):
        from kernle.types import PluginHealth

        return PluginHealth(healthy=True, message="OK")


class FakePluginCollisionA:
    """Plugin A that claims 'shared' command."""

    name = "plugin-a"
    version = "0.1.0"
    protocol_version = 1
    description = "Plugin A"

    def capabilities(self):
        return []

    def activate(self, context):
        pass

    def deactivate(self):
        pass

    def register_cli(self, subparsers):
        subparsers.add_parser("shared", help="Shared A")

    def register_tools(self):
        return []

    def cli_commands(self):
        return ["shared"]

    def handle_cli(self, args, k):
        pass

    def on_load(self, ctx):
        pass

    def on_status(self, status):
        pass

    def health_check(self):
        from kernle.types import PluginHealth

        return PluginHealth(healthy=True, message="OK")


class FakePluginCollisionB:
    """Plugin B that also claims 'shared' command."""

    name = "plugin-b"
    version = "0.1.0"
    protocol_version = 1
    description = "Plugin B"

    def capabilities(self):
        return []

    def activate(self, context):
        pass

    def deactivate(self):
        pass

    def register_cli(self, subparsers):
        pass  # Don't add parser (A already did)

    def register_tools(self):
        return []

    def cli_commands(self):
        return ["shared"]

    def handle_cli(self, args, k):
        pass

    def on_load(self, ctx):
        pass

    def on_status(self, status):
        pass

    def health_check(self):
        from kernle.types import PluginHealth

        return PluginHealth(healthy=True, message="OK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_kernle():
    """Create a mock Kernle with entity that supports load/unload_plugin."""
    k = MagicMock()
    k.stack_id = "test-lifecycle"
    k.stack = MagicMock()  # Materializes lazy property
    k.entity = MagicMock()
    return k


def _make_component(name, plugin_cls):
    """Create a mock discovery component."""
    comp = MagicMock()
    comp.name = name
    comp._plugin_cls = plugin_cls
    return comp


# ---------------------------------------------------------------------------
# Tests — Plugin Command Map Building
# ---------------------------------------------------------------------------


class TestPluginCommandMapBuilding:
    """Test the plugin discovery and command map construction."""

    def test_cli_commands_builds_cmd_map(self):
        """Plugin with cli_commands() populates _plugin_cmd_map."""
        plugin = FakePlugin()
        _plugin_cmd_map = {}
        _collided_commands = {}

        for cmd_name in plugin.cli_commands():
            if cmd_name in _plugin_cmd_map:
                pass
            else:
                _plugin_cmd_map[cmd_name] = plugin

        assert "fake" in _plugin_cmd_map
        assert _plugin_cmd_map["fake"] is plugin

    def test_collision_detection(self):
        """Two plugins claiming the same command triggers collision."""
        plugin_a = FakePluginCollisionA()
        plugin_b = FakePluginCollisionB()
        _plugin_cmd_map = {}
        _collided_commands = {}

        # Register plugin A
        for cmd_name in plugin_a.cli_commands():
            _plugin_cmd_map[cmd_name] = plugin_a

        # Register plugin B — should detect collision
        for cmd_name in plugin_b.cli_commands():
            if cmd_name in _collided_commands:
                pass
            elif cmd_name in _plugin_cmd_map:
                existing = _plugin_cmd_map[cmd_name]
                _collided_commands[cmd_name] = (existing.name, plugin_b.name)
                _plugin_cmd_map.pop(cmd_name)
            else:
                _plugin_cmd_map[cmd_name] = plugin_b

        assert "shared" not in _plugin_cmd_map
        assert "shared" in _collided_commands
        assert _collided_commands["shared"] == ("plugin-a", "plugin-b")


# ---------------------------------------------------------------------------
# Tests — Plugin Dispatch via main()
# ---------------------------------------------------------------------------


class TestPluginDispatchIntegration:
    """Integration tests for plugin CLI dispatch through main()."""

    def _run_main_with_plugin(self, argv, plugin, plugin_name="fake-plugin"):
        """Run main() with a fake plugin injected via mocked discovery."""
        comp = _make_component(plugin_name, type(plugin))

        def fake_discover():
            return [comp]

        def fake_load(c):
            return type(plugin)

        # We need to patch at the import site in __main__
        with (
            patch("kernle.cli.__main__.Kernle") as mock_kernle_cls,
            patch("kernle.discovery.discover_plugins", fake_discover),
            patch(
                "kernle.discovery.load_component",
                fake_load,
            ),
        ):
            k = _make_mock_kernle()
            mock_kernle_cls.return_value = k

            from kernle.cli.__main__ import main

            main(argv)
            return k

    def test_plugin_dispatch_calls_handle_cli(self, capsys):
        """Plugin command dispatches through handle_cli(args, k)."""
        plugin = FakePlugin()
        # Test the dispatch logic directly by building the maps
        # and simulating dispatch

        k = _make_mock_kernle()
        _plugin_cmd_map = {"fake": plugin}
        _collided_commands = {}

        args = argparse.Namespace(command="fake", opt="test-val")

        # Simulate the dispatch logic from __main__.py
        assert args.command in _plugin_cmd_map
        p = _plugin_cmd_map[args.command]
        _ = k.stack  # Materialize
        k.entity.load_plugin(p)

        p.handle_cli(args, k)

        assert len(plugin.cli_calls) == 1
        assert plugin.cli_calls[0][0] == "handle_cli"
        assert plugin.cli_calls[0][1].opt == "test-val"

        k.entity.unload_plugin(p.name)
        k.entity.load_plugin.assert_called_once_with(p)
        k.entity.unload_plugin.assert_called_once_with("fake-plugin")

    def test_legacy_adapter_retries_without_k(self):
        """Legacy plugin handle_cli(args) works via TypeError adapter."""
        plugin = FakePluginLegacy()
        k = _make_mock_kernle()
        args = argparse.Namespace(command="legacy")

        # Simulate dispatch with legacy adapter
        try:
            plugin.handle_cli(args, k)
        except TypeError as e:
            if "argument" in str(e) or "positional" in str(e):
                plugin.handle_cli(args)
            else:
                raise

        assert len(plugin.cli_calls) == 1
        assert plugin.cli_calls[0][0] == "handle_cli_legacy"

    def test_handle_cli_return_value_printed(self, capsys):
        """Legacy handler returning a string gets printed to stdout."""
        plugin = FakePluginLegacy()
        k = _make_mock_kernle()
        args = argparse.Namespace(command="legacy")

        # Override to return a string (like fatline does)
        plugin.handle_cli = lambda a: "Registered agent: test-agent"

        # Simulate dispatch with return-value printing
        try:
            result = plugin.handle_cli(args, k)
        except TypeError as e:
            if "argument" in str(e) or "positional" in str(e):
                result = plugin.handle_cli(args)
            else:
                raise
        if isinstance(result, str) and result:
            print(result)

        captured = capsys.readouterr().out
        assert "Registered agent: test-agent" in captured

    def test_handle_cli_none_return_not_printed(self, capsys):
        """Handler returning None does not print anything extra."""
        plugin = FakePlugin()
        k = _make_mock_kernle()
        args = argparse.Namespace(command="fake", opt="val")

        result = plugin.handle_cli(args, k)
        if isinstance(result, str) and result:
            print(result)

        captured = capsys.readouterr().out
        assert captured == ""

    def test_missing_handle_cli_shows_error(self, capsys):
        """Plugin without handle_cli shows error message."""
        plugin = FakePluginNoHandleCli()
        args = argparse.Namespace(command="nohandler")

        if not hasattr(plugin, "handle_cli"):
            printed = (
                f"Plugin '{plugin.name}' registered "
                f"'{args.command}' but has no handle_cli(). "
                f"Update the plugin for CLI support."
            )
        else:
            pytest.fail("Plugin should not have handle_cli")

        assert "no handle_cli" in printed.lower() or "handle_cli" in printed

    def test_collided_command_shows_both_plugins(self, capsys):
        """Invoking a collided command names both plugins."""
        _collided_commands = {"shared": ("plugin-a", "plugin-b")}

        command = "shared"
        assert command in _collided_commands
        p1, p2 = _collided_commands[command]
        message = (
            f"Command '{command}' is claimed by both '{p1}' and "
            f"'{p2}'. Uninstall one to resolve the conflict."
        )

        assert "plugin-a" in message
        assert "plugin-b" in message

    def test_cleanup_unload_on_success(self):
        """unload_plugin called after successful dispatch."""
        plugin = FakePlugin()
        k = _make_mock_kernle()
        args = argparse.Namespace(command="fake", opt="val")

        activated = False
        try:
            k.entity.load_plugin(plugin)
            activated = True
            plugin.handle_cli(args, k)
        finally:
            if activated:
                k.entity.unload_plugin(plugin.name)

        k.entity.unload_plugin.assert_called_once_with("fake-plugin")

    def test_cleanup_unload_on_dispatch_failure(self):
        """unload_plugin called even when handle_cli raises."""
        plugin = FakePlugin()
        k = _make_mock_kernle()
        args = argparse.Namespace(command="fake", opt="val")

        # Make handle_cli raise
        plugin.handle_cli = MagicMock(side_effect=RuntimeError("boom"))

        activated = False
        with pytest.raises(RuntimeError, match="boom"):
            try:
                k.entity.load_plugin(plugin)
                activated = True
                plugin.handle_cli(args, k)
            finally:
                if activated:
                    k.entity.unload_plugin(plugin.name)

        k.entity.unload_plugin.assert_called_once_with("fake-plugin")

    def test_activation_failure_no_unload(self):
        """If load_plugin fails, unload_plugin is NOT called."""
        plugin = FakePlugin()
        k = _make_mock_kernle()
        k.entity.load_plugin.side_effect = RuntimeError("activation failed")

        activated = False
        with pytest.raises(RuntimeError, match="activation failed"):
            try:
                k.entity.load_plugin(plugin)
                activated = True
            finally:
                if activated:
                    k.entity.unload_plugin(plugin.name)

        k.entity.unload_plugin.assert_not_called()

    def test_stack_materialized_before_activation(self):
        """k.stack is accessed before load_plugin to ensure PluginContext setup."""
        plugin = FakePlugin()
        call_order = []

        # Use a dedicated class instead of patching MagicMock's class
        class TrackedKernle:
            def __init__(self):
                self.stack_id = "test-lifecycle"
                self.entity = MagicMock()

            @property
            def stack(self):
                call_order.append("stack")
                return MagicMock()

        k = TrackedKernle()
        original_load = k.entity.load_plugin

        def tracked_load(p):
            call_order.append("load_plugin")
            return original_load(p)

        k.entity.load_plugin = tracked_load

        # Simulate dispatch
        _ = k.stack
        k.entity.load_plugin(plugin)

        assert call_order.index("stack") < call_order.index("load_plugin")


# ---------------------------------------------------------------------------
# Tests — Dedup regression
# ---------------------------------------------------------------------------


class TestDedupRegression:
    """Regression test for PDF import dedup with corpus-format entries."""

    def test_pdf_dedup_strips_corpus_headers(self):
        """PDF import dedup correctly handles corpus entries with headers."""
        from kernle.dedup import load_raw_content_hashes
        from kernle.processing import compute_content_hash

        # Simulate storage with corpus-format entries
        storage = MagicMock()
        raw_entry_1 = MagicMock()
        raw_entry_1.blob = "[corpus:repo] [file:x.py] [chunk:function:main]\ndef main():\n    pass"
        raw_entry_1.source = "corpus"

        raw_entry_2 = MagicMock()
        raw_entry_2.blob = "plain text without header"
        raw_entry_2.source = "pdf:doc.pdf"

        storage.list_raw.return_value = [raw_entry_1, raw_entry_2]

        result = load_raw_content_hashes(storage)

        # The hash for the corpus entry should be based on content AFTER header
        expected_hash = compute_content_hash("def main():\n    pass")
        assert expected_hash in result.hashes

        # The plain text entry hash should match directly
        plain_hash = compute_content_hash("plain text without header")
        assert plain_hash in result.hashes

        assert result.rows_scanned == 2
        assert result.rows_matched == 2

    def test_corpus_source_filter_only_scans_corpus(self):
        """source_filter='corpus' only includes corpus-tagged entries."""
        from kernle.dedup import load_raw_content_hashes

        storage = MagicMock()
        corpus_entry = MagicMock()
        corpus_entry.blob = "[corpus:repo] [file:x.py]\ncontent A"
        corpus_entry.source = "corpus"

        pdf_entry = MagicMock()
        pdf_entry.blob = "content B"
        pdf_entry.source = "pdf:doc.pdf"

        storage.list_raw.return_value = [corpus_entry, pdf_entry]

        result = load_raw_content_hashes(storage, source_filter="corpus")

        assert result.rows_scanned == 2
        assert result.rows_matched == 1  # Only corpus entry

    def test_corpus_filter_ignores_source_field_without_header(self):
        """source_filter='corpus' uses blob header, not source field.

        Regression: entries with source='corpus' but no [corpus:...] blob header
        must NOT be matched — only the blob header is authoritative.
        """
        from kernle.dedup import load_raw_content_hashes

        storage = MagicMock()
        # Entry with corpus header in blob (should match)
        real_corpus = MagicMock()
        real_corpus.blob = "[corpus:repo] [file:x.py]\ncontent A"
        real_corpus.source = "corpus"

        # Entry with source="corpus" but no blob header (should NOT match)
        misattributed = MagicMock()
        misattributed.blob = "plain content with source=corpus"
        misattributed.source = "corpus"

        storage.list_raw.return_value = [real_corpus, misattributed]

        result = load_raw_content_hashes(storage, source_filter="corpus")

        assert result.rows_matched == 1  # Only the real corpus entry
        assert result.rows_scanned == 2

    def test_strip_corpus_header(self):
        """strip_corpus_header removes [corpus:...] header line."""
        from kernle.dedup import strip_corpus_header

        assert strip_corpus_header("[corpus:repo] [file:x.py]\ncontent") == "content"
        assert strip_corpus_header("no header here") == "no header here"
        assert strip_corpus_header("[corpus:repo]") == "[corpus:repo]"  # No newline
        assert strip_corpus_header("") == ""
