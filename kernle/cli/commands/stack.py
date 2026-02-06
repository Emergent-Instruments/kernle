"""Stack management commands (list, delete)."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from kernle.storage.sqlite import validate_table_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


def cmd_stack(args: "argparse.Namespace", k: "Kernle") -> None:
    """Handle agent subcommands."""
    action = args.stack_action

    if action == "list":
        _list_stacks(args, k)
    elif action == "delete":
        _delete_stack(args, k)


def _list_stacks(args: "argparse.Namespace", k: "Kernle") -> None:
    """List all local agents."""
    kernle_dir = Path.home() / ".kernle"

    if not kernle_dir.exists():
        print("No agents found (Kernle not initialized)")
        return

    # Find agent directories (those with memory.db or raw/ subdirectory)
    agents = []

    # Check for multi-agent SQLite structure
    db_path = kernle_dir / "memories.db"
    if db_path.exists():
        # Query SQLite for distinct stack_ids
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                "SELECT DISTINCT stack_id FROM episodes "
                "UNION SELECT DISTINCT stack_id FROM notes "
                "UNION SELECT DISTINCT stack_id FROM beliefs"
            )
            for row in cursor:
                agents.append(row[0])
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to query agents from database: {e}")

    # Also check for per-agent directories (for raw layer)
    for item in kernle_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Skip non-agent directories
            if item.name in ("logs", "cache", "__pycache__"):
                continue
            if item.name not in agents:
                agents.append(item.name)

    if not agents:
        print("No agents found")
        return

    agents.sort()

    print(f"Local Stacks ({len(agents)} total)")
    print("=" * 50)

    for stack_id in agents:
        agent_dir = kernle_dir / stack_id
        raw_count = 0
        has_dir = agent_dir.exists()

        if has_dir:
            raw_dir = agent_dir / "raw"
            if raw_dir.exists():
                raw_count = sum(1 for f in raw_dir.glob("*.md"))

        # Get episode/note counts from SQLite
        episode_count = note_count = belief_count = 0
        if db_path.exists():
            import sqlite3

            try:
                conn = sqlite3.connect(str(db_path))
                episode_count = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE stack_id = ?", (stack_id,)
                ).fetchone()[0]
                note_count = conn.execute(
                    "SELECT COUNT(*) FROM notes WHERE stack_id = ?", (stack_id,)
                ).fetchone()[0]
                belief_count = conn.execute(
                    "SELECT COUNT(*) FROM beliefs WHERE stack_id = ?", (stack_id,)
                ).fetchone()[0]
                conn.close()
            except Exception as e:
                logger.debug(f"Failed to get counts for agent '{stack_id}': {e}")

        # Mark current agent
        marker = " ← current" if stack_id == k.stack_id else ""
        print(f"\n  {stack_id}{marker}")
        print(
            f"    Episodes: {episode_count}  Notes: {note_count}  Beliefs: {belief_count}  Raw: {raw_count}"
        )


def _delete_stack(args: "argparse.Namespace", k: "Kernle") -> None:
    """Delete an agent and all its data."""
    stack_id = args.name
    force = getattr(args, "force", False)

    if stack_id == k.stack_id:
        print(f"❌ Cannot delete current agent '{stack_id}'")
        print("   Switch to a different stack first with: kernle -s <other> ...")
        return

    kernle_dir = Path.home() / ".kernle"
    db_path = kernle_dir / "memories.db"
    agent_dir = kernle_dir / stack_id

    # Check if agent exists
    has_db_data = False
    has_dir = agent_dir.exists()

    if db_path.exists():
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_path))
            count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE stack_id = ?", (stack_id,)
            ).fetchone()[0]
            has_db_data = count > 0
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to check agent in database: {e}")

    if not has_db_data and not has_dir:
        print(f"❌ Stack '{stack_id}' not found")
        return

    # Get counts for confirmation
    episode_count = note_count = belief_count = goal_count = value_count = 0
    if db_path.exists():
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_path))
            episode_count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE stack_id = ?", (stack_id,)
            ).fetchone()[0]
            note_count = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE stack_id = ?", (stack_id,)
            ).fetchone()[0]
            belief_count = conn.execute(
                "SELECT COUNT(*) FROM beliefs WHERE stack_id = ?", (stack_id,)
            ).fetchone()[0]
            goal_count = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE stack_id = ?", (stack_id,)
            ).fetchone()[0]
            value_count = conn.execute(
                "SELECT COUNT(*) FROM agent_values WHERE stack_id = ?", (stack_id,)
            ).fetchone()[0]
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to get deletion counts: {e}")

    total_records = episode_count + note_count + belief_count + goal_count + value_count

    if not force:
        print(f"⚠️  About to delete agent '{stack_id}':")
        print(f"   Episodes: {episode_count}")
        print(f"   Notes: {note_count}")
        print(f"   Beliefs: {belief_count}")
        print(f"   Goals: {goal_count}")
        print(f"   Values: {value_count}")
        if has_dir:
            print(f"   Directory: {agent_dir}")
        print()
        confirm = input("Type the agent name to confirm deletion: ")
        if confirm != stack_id:
            print("❌ Deletion cancelled")
            return

    # Delete from SQLite
    deleted_tables = []
    if db_path.exists():
        import sqlite3

        try:
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
                try:
                    validate_table_name(table)  # Security: validate before SQL use
                    cursor = conn.execute(f"DELETE FROM {table} WHERE stack_id = ?", (stack_id,))
                    if cursor.rowcount > 0:
                        deleted_tables.append(f"{table}: {cursor.rowcount}")
                except Exception as e:
                    logger.debug(f"Failed to delete from table '{table}': {e}")

            # Also delete from vec_embeddings if exists
            try:
                conn.execute("DELETE FROM vec_embeddings WHERE id LIKE ?", (f"%:{stack_id}:%",))
            except Exception as e:
                logger.debug(f"Failed to delete embeddings: {e}")

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️  Error cleaning database: {e}")

    # Delete agent directory
    if has_dir:
        try:
            shutil.rmtree(agent_dir)
            print(f"✓ Deleted directory: {agent_dir}")
        except Exception as e:
            print(f"⚠️  Error deleting directory: {e}")

    print(f"✓ Stack '{stack_id}' deleted ({total_records} records)")
    if deleted_tables:
        print(f"   Cleaned: {', '.join(deleted_tables)}")
