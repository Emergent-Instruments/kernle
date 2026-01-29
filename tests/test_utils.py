"""Tests for kernle.utils module."""

import os
from unittest.mock import patch

from kernle.utils import generate_default_agent_id, resolve_agent_id


class TestGenerateDefaultAgentId:
    """Tests for generate_default_agent_id function."""

    def test_returns_auto_prefixed_id(self):
        """Generated ID should have 'auto-' prefix."""
        agent_id = generate_default_agent_id()
        assert agent_id.startswith("auto-")

    def test_returns_8_char_hash(self):
        """Generated ID should have 8-character hash suffix."""
        agent_id = generate_default_agent_id()
        # Format: auto-XXXXXXXX (5 + 8 = 13 chars)
        assert len(agent_id) == 13
        # Verify the hash portion is hex
        hash_part = agent_id[5:]
        assert all(c in "0123456789abcdef" for c in hash_part)

    def test_consistent_for_same_environment(self):
        """Same machine + cwd should produce same ID."""
        id1 = generate_default_agent_id()
        id2 = generate_default_agent_id()
        assert id1 == id2

    @patch("kernle.utils._get_git_root")
    @patch("os.getcwd")
    @patch("platform.node")
    def test_different_for_different_machines(self, mock_node, mock_cwd, mock_git):
        """Different machines should produce different IDs."""
        mock_git.return_value = None
        mock_cwd.return_value = "/same/path"

        mock_node.return_value = "machine-a"
        id_a = generate_default_agent_id()

        mock_node.return_value = "machine-b"
        id_b = generate_default_agent_id()

        assert id_a != id_b

    @patch("kernle.utils._get_git_root")
    @patch("os.getcwd")
    @patch("platform.node")
    def test_different_for_different_paths(self, mock_node, mock_cwd, mock_git):
        """Different paths should produce different IDs."""
        mock_git.return_value = None
        mock_node.return_value = "same-machine"

        mock_cwd.return_value = "/path/one"
        id_a = generate_default_agent_id()

        mock_cwd.return_value = "/path/two"
        id_b = generate_default_agent_id()

        assert id_a != id_b

    @patch("kernle.utils._get_git_root")
    @patch("os.getcwd")
    @patch("platform.node")
    def test_prefers_git_root_over_cwd(self, mock_node, mock_cwd, mock_git):
        """When in a git repo, should use git root for path."""
        mock_node.return_value = "test-machine"
        mock_git.return_value = "/git/root"
        mock_cwd.return_value = "/git/root/subdir/deep"

        id_from_subdir = generate_default_agent_id()

        # Simulate being at git root
        mock_cwd.return_value = "/git/root"
        id_from_root = generate_default_agent_id()

        # Both should be the same since git root is used
        assert id_from_subdir == id_from_root

    @patch("platform.node")
    def test_handles_empty_hostname(self, mock_node):
        """Should handle empty hostname gracefully."""
        mock_node.return_value = ""
        agent_id = generate_default_agent_id()
        assert agent_id.startswith("auto-")
        assert len(agent_id) == 13


class TestResolveAgentId:
    """Tests for resolve_agent_id function."""

    def test_explicit_id_takes_priority(self):
        """Explicit ID should override everything."""
        result = resolve_agent_id("my-explicit-id")
        assert result == "my-explicit-id"

    def test_default_string_triggers_fallback(self):
        """'default' should be treated as no explicit ID."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("kernle.utils.generate_default_agent_id", return_value="auto-abcd1234"):
                result = resolve_agent_id("default")
                assert result == "auto-abcd1234"

    def test_env_var_second_priority(self):
        """KERNLE_AGENT_ID env var should be used if no explicit ID."""
        with patch.dict(os.environ, {"KERNLE_AGENT_ID": "env-agent"}):
            result = resolve_agent_id(None)
            assert result == "env-agent"

    def test_auto_generate_as_fallback(self):
        """Should auto-generate if no explicit ID and no env var."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear KERNLE_AGENT_ID if it exists
            if "KERNLE_AGENT_ID" in os.environ:
                del os.environ["KERNLE_AGENT_ID"]
            result = resolve_agent_id(None)
            assert result.startswith("auto-")

    def test_explicit_overrides_env_var(self):
        """Explicit ID should override env var."""
        with patch.dict(os.environ, {"KERNLE_AGENT_ID": "env-agent"}):
            result = resolve_agent_id("explicit-agent")
            assert result == "explicit-agent"

    def test_none_with_env_var(self):
        """None as explicit should fall through to env var."""
        with patch.dict(os.environ, {"KERNLE_AGENT_ID": "from-env"}):
            result = resolve_agent_id(None)
            assert result == "from-env"
