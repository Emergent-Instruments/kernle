"""Tests for CLI agent command module."""

import sqlite3
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from kernle.cli.commands.stack import _delete_stack, _list_stacks, cmd_stack


class TestCmdAgent:
    """Test the cmd_stack dispatcher function."""

    def test_dispatches_to_list(self, capsys):
        """Test cmd_stack dispatches to list handler."""
        k = MagicMock()
        k.stack_id = "test-agent"

        args = Namespace(stack_action="list")

        with patch("kernle.cli.commands.stack._list_stacks") as mock_list:
            cmd_stack(args, k)
            mock_list.assert_called_once_with(args, k)

    def test_dispatches_to_delete(self, capsys):
        """Test cmd_stack dispatches to delete handler."""
        k = MagicMock()
        k.stack_id = "test-agent"

        args = Namespace(stack_action="delete", name="other-agent")

        with patch("kernle.cli.commands.stack._delete_stack") as mock_delete:
            cmd_stack(args, k)
            mock_delete.assert_called_once_with(args, k)


class TestListAgentsNoKernleDir:
    """Test _list_stacks when Kernle directory doesn't exist."""

    def test_no_kernle_dir(self, capsys, tmp_path):
        """Test when ~/.kernle doesn't exist."""
        k = MagicMock()
        k.stack_id = "test-agent"

        args = Namespace()

        # Use a non-existent directory as home
        with patch.object(Path, "home", return_value=tmp_path / "nonexistent"):
            _list_stacks(args, k)

        captured = capsys.readouterr()
        assert "No agents found (Kernle not initialized)" in captured.out


class TestListAgentsWithDatabase:
    """Test _list_stacks with SQLite database."""

    def test_agents_from_database(self, capsys, tmp_path):
        """Test listing agents from SQLite database."""
        k = MagicMock()
        k.stack_id = "agent-1"

        args = Namespace()

        # Create fake .kernle directory with database
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('agent-1', 'ep1')")
        conn.execute("INSERT INTO episodes VALUES ('agent-1', 'ep2')")
        conn.execute("INSERT INTO notes VALUES ('agent-1', 'n1')")
        conn.execute("INSERT INTO beliefs VALUES ('agent-1', 'b1')")
        conn.execute("INSERT INTO episodes VALUES ('agent-2', 'ep3')")
        conn.commit()
        conn.close()

        with patch.object(Path, "home", return_value=tmp_path):
            _list_stacks(args, k)

        captured = capsys.readouterr()
        assert "Local Stacks (2 total)" in captured.out
        assert "agent-1" in captured.out
        assert "current" in captured.out  # agent-1 should be marked as current
        assert "agent-2" in captured.out
        assert "Episodes: 2" in captured.out  # agent-1 has 2 episodes

    def test_agents_from_directories(self, capsys, tmp_path):
        """Test listing agents from directory structure."""
        k = MagicMock()
        k.stack_id = "agent-dir"

        args = Namespace()

        # Create fake .kernle directory with agent subdirectories
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create agent directories
        agent_dir = kernle_dir / "agent-dir"
        agent_dir.mkdir()
        raw_dir = agent_dir / "raw"
        raw_dir.mkdir()
        (raw_dir / "entry1.md").write_text("test")
        (raw_dir / "entry2.md").write_text("test")

        # Create another agent without raw dir
        (kernle_dir / "agent-simple").mkdir()

        # Directories that should be skipped
        (kernle_dir / "logs").mkdir()
        (kernle_dir / "cache").mkdir()
        (kernle_dir / "__pycache__").mkdir()

        with patch.object(Path, "home", return_value=tmp_path):
            _list_stacks(args, k)

        captured = capsys.readouterr()
        assert "Local Stacks" in captured.out
        assert "agent-dir" in captured.out
        assert "agent-simple" in captured.out
        assert "Raw: 2" in captured.out
        assert "logs" not in captured.out
        assert "cache" not in captured.out

    def test_no_agents_found(self, capsys, tmp_path):
        """Test when kernle dir exists but no agents."""
        k = MagicMock()
        k.stack_id = "test-agent"

        args = Namespace()

        # Create empty .kernle directory
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        with patch.object(Path, "home", return_value=tmp_path):
            _list_stacks(args, k)

        captured = capsys.readouterr()
        assert "No agents found" in captured.out

    def test_db_error_handled_gracefully(self, capsys, tmp_path):
        """Test database errors are handled gracefully."""
        k = MagicMock()
        k.stack_id = "agent-1"

        args = Namespace()

        # Create .kernle directory with invalid database
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create a corrupt/invalid database file
        db_path = kernle_dir / "memories.db"
        db_path.write_text("not a valid database")

        # Create a valid agent directory to show something
        agent_dir = kernle_dir / "agent-1"
        agent_dir.mkdir()

        with patch.object(Path, "home", return_value=tmp_path):
            _list_stacks(args, k)

        captured = capsys.readouterr()
        # Should still list the directory-based agent
        assert "agent-1" in captured.out


class TestDeleteAgent:
    """Test _delete_stack function."""

    def _make_kernle_mock(self, stack_id="current-agent"):
        """Create a Kernle mock with working _validate_stack_id."""
        k = MagicMock()
        k.stack_id = stack_id
        k._validate_stack_id = lambda name: name  # pass-through for valid names
        return k

    def test_cannot_delete_current_agent(self, capsys):
        """Test error when trying to delete current agent."""
        k = self._make_kernle_mock("current-agent")

        args = Namespace(name="current-agent", force=False)

        _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Cannot delete current agent" in captured.out
        assert "Switch to a different stack" in captured.out

    def test_path_traversal_rejected(self, capsys):
        """Test that path traversal attempts are rejected."""
        k = MagicMock()
        k.stack_id = "current-agent"
        k._validate_stack_id.side_effect = ValueError("Stack ID must not contain path separators")

        args = Namespace(name="../../../etc/passwd", force=True)

        _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Invalid stack name" in captured.out

    def test_dotdot_traversal_rejected(self, capsys):
        """Test that .. traversal attempts are rejected."""
        k = MagicMock()
        k.stack_id = "current-agent"
        k._validate_stack_id.side_effect = ValueError("Stack ID must not contain path separators")

        args = Namespace(name="../../secret", force=True)

        _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Invalid stack name" in captured.out

    def test_agent_not_found(self, capsys, tmp_path):
        """Test error when agent doesn't exist."""
        k = self._make_kernle_mock()

        args = Namespace(name="nonexistent-agent", force=True)

        # Create empty .kernle directory
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Stack 'nonexistent-agent' not found" in captured.out

    def test_delete_with_force(self, capsys, tmp_path):
        """Test deleting agent with --force flag."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=True)

        # Set up .kernle directory with database and agent directory
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database with agent data
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep1')")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep2')")
        conn.execute("INSERT INTO notes VALUES ('other-agent', 'n1')")
        conn.commit()
        conn.close()

        # Create agent directory
        agent_dir = kernle_dir / "other-agent"
        agent_dir.mkdir()
        (agent_dir / "some_file.txt").write_text("test")

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Stack 'other-agent' deleted" in captured.out
        assert "Deleted directory" in captured.out
        assert not agent_dir.exists()

        # Verify database records were deleted
        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE stack_id = ?", ("other-agent",)
        ).fetchone()[0]
        conn.close()
        assert count == 0

    def test_delete_cancelled_on_wrong_confirmation(self, capsys, tmp_path, monkeypatch):
        """Test deletion is cancelled when wrong name is entered."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=False)

        # Set up .kernle directory with agent
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database with agent data
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep1')")
        conn.commit()
        conn.close()

        # User enters wrong confirmation
        monkeypatch.setattr("builtins.input", lambda _: "wrong-name")

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "About to delete agent 'other-agent'" in captured.out
        assert "Deletion cancelled" in captured.out

    def test_delete_confirmed_with_correct_name(self, capsys, tmp_path, monkeypatch):
        """Test deletion proceeds with correct confirmation."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=False)

        # Set up .kernle directory with agent
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database with agent data
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep1')")
        conn.commit()
        conn.close()

        # Create agent directory
        agent_dir = kernle_dir / "other-agent"
        agent_dir.mkdir()

        # User enters correct confirmation
        monkeypatch.setattr("builtins.input", lambda _: "other-agent")

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Stack 'other-agent' deleted" in captured.out

    def test_delete_shows_counts_in_confirmation(self, capsys, tmp_path, monkeypatch):
        """Test that confirmation message shows record counts."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=False)

        # Set up .kernle directory with agent
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database with various records
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep1')")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep2')")
        conn.execute("INSERT INTO notes VALUES ('other-agent', 'n1')")
        conn.execute("INSERT INTO beliefs VALUES ('other-agent', 'b1')")
        conn.execute("INSERT INTO beliefs VALUES ('other-agent', 'b2')")
        conn.execute("INSERT INTO beliefs VALUES ('other-agent', 'b3')")
        conn.execute("INSERT INTO goals VALUES ('other-agent', 'g1')")
        conn.execute("INSERT INTO agent_values VALUES ('other-agent', 'v1')")
        conn.commit()
        conn.close()

        # Cancel deletion
        monkeypatch.setattr("builtins.input", lambda _: "no")

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Episodes: 2" in captured.out
        assert "Notes: 1" in captured.out
        assert "Beliefs: 3" in captured.out
        assert "Goals: 1" in captured.out
        assert "Values: 1" in captured.out

    def test_delete_db_only_agent(self, capsys, tmp_path):
        """Test deleting agent that only exists in database (no directory)."""
        k = self._make_kernle_mock()

        args = Namespace(name="db-only-agent", force=True)

        # Set up .kernle directory with database only
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database with agent data but no directory
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('db-only-agent', 'ep1')")
        conn.commit()
        conn.close()

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Stack 'db-only-agent' deleted" in captured.out
        # Should not mention directory deletion since there was none
        assert "Deleted directory" not in captured.out

    def test_delete_dir_only_agent(self, capsys, tmp_path):
        """Test deleting agent that only exists as directory (no DB records)."""
        k = self._make_kernle_mock()

        args = Namespace(name="dir-only-agent", force=True)

        # Set up .kernle directory with agent directory but no DB records
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create empty database
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.commit()
        conn.close()

        # Create agent directory
        agent_dir = kernle_dir / "dir-only-agent"
        agent_dir.mkdir()
        (agent_dir / "data.txt").write_text("test")

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Stack 'dir-only-agent' deleted" in captured.out
        assert "Deleted directory" in captured.out
        assert not agent_dir.exists()

    def test_delete_handles_additional_tables(self, capsys, tmp_path):
        """Test deletion cleans up all related tables."""
        k = self._make_kernle_mock()

        args = Namespace(name="full-agent", force=True)

        # Set up .kernle directory with full database schema
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database with all tables the delete function handles
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        tables = [
            "episodes",
            "notes",
            "beliefs",
            "goals",
            "agent_values",
            "checkpoints",
            "drives",
            "relationships",
            "playbooks",
            "raw_entries",
            "sync_queue",
        ]
        for table in tables:
            conn.execute(f"CREATE TABLE {table} (stack_id TEXT, id TEXT)")
            conn.execute(f"INSERT INTO {table} VALUES ('full-agent', 'id1')")
        conn.commit()
        conn.close()

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Stack 'full-agent' deleted" in captured.out

        # Verify all tables were cleaned
        conn = sqlite3.connect(str(db_path))
        for table in tables:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE stack_id = ?", ("full-agent",)
            ).fetchone()[0]
            assert count == 0, f"Table {table} should be empty"
        conn.close()

    def test_delete_handles_db_error_checking_existence(self, capsys, tmp_path):
        """Test deletion handles DB error when checking if agent exists."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=True)

        # Set up .kernle directory with invalid database
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create a corrupt database (can't execute SQL)
        db_path = kernle_dir / "memories.db"
        db_path.write_text("not a valid database")

        # But create the agent directory so agent exists
        agent_dir = kernle_dir / "other-agent"
        agent_dir.mkdir()

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        # Should still delete the directory-based agent
        assert "Stack 'other-agent' deleted" in captured.out or "Error" in captured.out

    def test_delete_handles_db_error_getting_counts(self, capsys, tmp_path, monkeypatch):
        """Test deletion handles DB error when getting counts for confirmation."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=False)

        # Set up .kernle directory
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create valid database with minimal schema
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep1')")
        # Missing other tables - will cause error when getting counts
        conn.commit()
        conn.close()

        # Cancel deletion
        monkeypatch.setattr("builtins.input", lambda _: "no")

        with patch.object(Path, "home", return_value=tmp_path):
            _delete_stack(args, k)

        captured = capsys.readouterr()
        # Should handle the error gracefully and still show confirmation
        assert "About to delete agent" in captured.out

    def test_delete_handles_db_error_during_cleanup(self, capsys, tmp_path):
        """Test deletion handles DB error during cleanup."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=True)

        # Set up .kernle directory
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database with agent data
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep1')")
        conn.commit()
        conn.close()

        # Create agent directory so it exists via directory
        agent_dir = kernle_dir / "other-agent"
        agent_dir.mkdir()

        # Track connect calls and fail on later ones (during deletion cleanup)
        call_count = [0]
        original_connect = sqlite3.connect

        def mock_connect(path):
            call_count[0] += 1
            # Let first 2 calls through (existence check, count fetch)
            # Fail on 3rd call (deletion cleanup)
            if call_count[0] >= 3:
                raise Exception("Database locked")
            return original_connect(path)

        with patch.object(Path, "home", return_value=tmp_path):
            with patch("sqlite3.connect", side_effect=mock_connect):
                _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Error cleaning database" in captured.out

    def test_delete_handles_directory_deletion_error(self, capsys, tmp_path):
        """Test deletion handles error when deleting directory."""
        k = self._make_kernle_mock()

        args = Namespace(name="other-agent", force=True)

        # Set up .kernle directory
        kernle_dir = tmp_path / ".kernle"
        kernle_dir.mkdir()

        # Create database
        db_path = kernle_dir / "memories.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE episodes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE notes (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE beliefs (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE goals (stack_id TEXT, id TEXT)")
        conn.execute("CREATE TABLE agent_values (stack_id TEXT, id TEXT)")
        conn.execute("INSERT INTO episodes VALUES ('other-agent', 'ep1')")
        conn.commit()
        conn.close()

        # Create agent directory
        agent_dir = kernle_dir / "other-agent"
        agent_dir.mkdir()

        with patch.object(Path, "home", return_value=tmp_path):
            with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
                _delete_stack(args, k)

        captured = capsys.readouterr()
        assert "Error deleting directory" in captured.out
