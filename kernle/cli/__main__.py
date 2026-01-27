"""
Kernle CLI - Command-line interface for stratified memory.

Usage:
    kernle load [--json]
    kernle checkpoint save TASK [--pending P]... [--context CTX]
    kernle checkpoint load [--json]
    kernle checkpoint clear
    kernle episode OBJECTIVE OUTCOME [--lesson L]... [--tag T]...
    kernle note CONTENT [--type TYPE] [--speaker S] [--reason R]
    kernle search QUERY [--limit N]
    kernle status
"""

import argparse
import json
import sys
import re
import logging

from kernle import Kernle

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def validate_input(value: str, field_name: str, max_length: int = 1000) -> str:
    """Validate and sanitize CLI inputs."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    if len(value) > max_length:
        raise ValueError(f"{field_name} too long (max {max_length} characters)")
    
    # Remove null bytes and control characters except newlines
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
    
    return sanitized


def cmd_load(args, k: Kernle):
    """Load and display working memory."""
    memory = k.load()
    if args.json:
        print(json.dumps(memory, indent=2, default=str))
    else:
        print(k.format_memory(memory))


def cmd_checkpoint(args, k: Kernle):
    """Handle checkpoint subcommands."""
    if args.checkpoint_action == "save":
        task = validate_input(args.task, "task", 500)
        pending = [validate_input(p, "pending item", 200) for p in (args.pending or [])]
        context = validate_input(args.context, "context", 1000) if args.context else None
        
        result = k.checkpoint(task, pending, context)
        print(f"✓ Checkpoint saved: {result['current_task']}")
        if result.get("pending"):
            print(f"  Pending: {len(result['pending'])} items")
    
    elif args.checkpoint_action == "load":
        cp = k.load_checkpoint()
        if cp:
            if args.json:
                print(json.dumps(cp, indent=2, default=str))
            else:
                print(f"Task: {cp.get('current_task', 'unknown')}")
                print(f"When: {cp.get('timestamp', 'unknown')}")
                if cp.get("pending"):
                    print("Pending:")
                    for p in cp["pending"]:
                        print(f"  - {p}")
                if cp.get("context"):
                    print(f"Context: {cp['context']}")
        else:
            print("No checkpoint found.")
    
    elif args.checkpoint_action == "clear":
        if k.clear_checkpoint():
            print("✓ Checkpoint cleared")
        else:
            print("No checkpoint to clear")


def cmd_episode(args, k: Kernle):
    """Record an episode."""
    objective = validate_input(args.objective, "objective", 1000)
    outcome = validate_input(args.outcome, "outcome", 1000)
    lessons = [validate_input(l, "lesson", 500) for l in (args.lesson or [])]
    tags = [validate_input(t, "tag", 100) for t in (args.tag or [])]
    
    episode_id = k.episode(
        objective=objective,
        outcome=outcome,
        lessons=lessons,
        tags=tags,
    )
    print(f"✓ Episode saved: {episode_id[:8]}...")
    if args.lesson:
        print(f"  Lessons: {len(args.lesson)}")


def cmd_note(args, k: Kernle):
    """Capture a note."""
    content = validate_input(args.content, "content", 2000)
    speaker = validate_input(args.speaker, "speaker", 200) if args.speaker else None
    reason = validate_input(args.reason, "reason", 1000) if args.reason else None
    tags = [validate_input(t, "tag", 100) for t in (args.tag or [])]
    
    note_id = k.note(
        content=content,
        type=args.type,
        speaker=speaker,
        reason=reason,
        tags=tags,
        protect=args.protect,
    )
    print(f"✓ Note saved: {args.content[:50]}...")
    if args.tag:
        print(f"  Tags: {', '.join(args.tag)}")


def cmd_search(args, k: Kernle):
    """Search memory."""
    query = validate_input(args.query, "query", 500)
    results = k.search(query, args.limit)
    if not results:
        print(f"No results for '{args.query}'")
        return
    
    print(f"Found {len(results)} result(s) for '{args.query}':\n")
    for i, r in enumerate(results, 1):
        # Handle potentially malformed results gracefully
        result_type = r.get('type', 'unknown')
        title = r.get('title', '(no title)')
        print(f"{i}. [{result_type}] {title}")
        if r.get("lessons"):
            for lesson in r["lessons"]:
                print(f"     → {lesson[:50]}...")
        if r.get("tags"):
            print(f"     tags: {', '.join(r['tags'])}")
        if r.get("confidence"):
            print(f"     confidence: {r['confidence']}")
        if r.get("date"):
            print(f"     {r['date']}")
        print()


def cmd_status(args, k: Kernle):
    """Show memory status."""
    status = k.status()
    print(f"Memory Status for {status['agent_id']}")
    print("=" * 40)
    print(f"Values:     {status['values']}")
    print(f"Beliefs:    {status['beliefs']}")
    print(f"Goals:      {status['goals']} active")
    print(f"Episodes:   {status['episodes']}")
    print(f"Checkpoint: {'Yes' if status['checkpoint'] else 'No'}")


def cmd_drive(args, k: Kernle):
    """Set or view drives."""
    if args.drive_action == "list":
        drives = k.load_drives()
        if not drives:
            print("No drives set.")
            return
        print("Drives:")
        for d in drives:
            focus = f" → {', '.join(d.get('focus_areas', []))}" if d.get('focus_areas') else ""
            print(f"  {d['drive_type']}: {d['intensity']:.0%}{focus}")
    
    elif args.drive_action == "set":
        k.drive(args.type, args.intensity, args.focus)
        print(f"✓ Drive '{args.type}' set to {args.intensity:.0%}")
    
    elif args.drive_action == "satisfy":
        if k.satisfy_drive(args.type, args.amount):
            print(f"✓ Satisfied drive '{args.type}'")
        else:
            print(f"Drive '{args.type}' not found")


def cmd_consolidate(args, k: Kernle):
    """Run memory consolidation."""
    result = k.consolidate(args.min_episodes)
    print(f"Consolidation complete:")
    print(f"  Episodes processed: {result['consolidated']}")
    print(f"  New beliefs: {result.get('new_beliefs', 0)}")
    print(f"  Lessons found: {result.get('lessons_found', 0)}")


def cmd_temporal(args, k: Kernle):
    """Query memories by time."""
    result = k.what_happened(args.when)
    
    print(f"What happened {args.when}:")
    print(f"  Time range: {result['range']['start'][:10]} to {result['range']['end'][:10]}")
    print()
    
    if result.get("episodes"):
        print("Episodes:")
        for ep in result["episodes"][:5]:
            print(f"  - {ep['objective'][:60]} [{ep.get('outcome_type', '?')}]")
    
    if result.get("notes"):
        print("Notes:")
        for n in result["notes"][:5]:
            print(f"  - {n['content'][:60]}...")


def cmd_mcp(args):
    """Start MCP server."""
    try:
        from kernle.mcp.server import main as mcp_main
        mcp_main()
    except ImportError as e:
        logger.error(f"MCP dependencies not installed. Run: pip install kernle[mcp]")
        logger.error(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="kernle",
        description="Stratified memory for synthetic intelligences",
    )
    parser.add_argument("--agent", "-a", help="Agent ID", default=None)
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # load
    p_load = subparsers.add_parser("load", help="Load working memory")
    p_load.add_argument("--json", "-j", action="store_true")
    
    # checkpoint
    p_checkpoint = subparsers.add_parser("checkpoint", help="Checkpoint operations")
    cp_sub = p_checkpoint.add_subparsers(dest="checkpoint_action", required=True)
    
    cp_save = cp_sub.add_parser("save", help="Save checkpoint")
    cp_save.add_argument("task", help="Current task")
    cp_save.add_argument("--pending", "-p", action="append", help="Pending item")
    cp_save.add_argument("--context", "-c", help="Additional context")
    
    cp_load = cp_sub.add_parser("load", help="Load checkpoint")
    cp_load.add_argument("--json", "-j", action="store_true")
    
    cp_sub.add_parser("clear", help="Clear checkpoint")
    
    # episode
    p_episode = subparsers.add_parser("episode", help="Record an episode")
    p_episode.add_argument("objective", help="What was the objective?")
    p_episode.add_argument("outcome", help="What was the outcome?")
    p_episode.add_argument("--lesson", "-l", action="append", help="Lesson learned")
    p_episode.add_argument("--tag", "-t", action="append", help="Tag")
    
    # note
    p_note = subparsers.add_parser("note", help="Capture a note")
    p_note.add_argument("content", help="Note content")
    p_note.add_argument("--type", choices=["note", "decision", "insight", "quote"], default="note")
    p_note.add_argument("--speaker", "-s", help="Speaker (for quotes)")
    p_note.add_argument("--reason", "-r", help="Reason (for decisions)")
    p_note.add_argument("--tag", action="append", help="Tag")
    p_note.add_argument("--protect", "-p", action="store_true", help="Protect from forgetting")
    
    # search
    p_search = subparsers.add_parser("search", help="Search memory")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    
    # status
    subparsers.add_parser("status", help="Show memory status")
    
    # drive
    p_drive = subparsers.add_parser("drive", help="Manage drives")
    drive_sub = p_drive.add_subparsers(dest="drive_action", required=True)
    
    drive_sub.add_parser("list", help="List drives")
    
    drive_set = drive_sub.add_parser("set", help="Set a drive")
    drive_set.add_argument("type", choices=["existence", "growth", "curiosity", "connection", "reproduction"])
    drive_set.add_argument("intensity", type=float, help="Intensity 0.0-1.0")
    drive_set.add_argument("--focus", "-f", action="append", help="Focus area")
    
    drive_satisfy = drive_sub.add_parser("satisfy", help="Satisfy a drive")
    drive_satisfy.add_argument("type", help="Drive type")
    drive_satisfy.add_argument("--amount", "-a", type=float, default=0.2)
    
    # consolidate
    p_consolidate = subparsers.add_parser("consolidate", help="Run memory consolidation")
    p_consolidate.add_argument("--min-episodes", "-m", type=int, default=3)
    
    # temporal
    p_temporal = subparsers.add_parser("when", help="Query by time")
    p_temporal.add_argument("when", nargs="?", default="today", 
                           choices=["today", "yesterday", "this week", "last hour"])
    
    # mcp
    subparsers.add_parser("mcp", help="Start MCP server (stdio transport)")
    
    args = parser.parse_args()
    
    # Initialize Kernle with error handling
    try:
        agent_id = validate_input(args.agent, "agent_id", 100) if args.agent else None
        k = Kernle(agent_id=agent_id)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to initialize Kernle: {e}")
        sys.exit(1)
    
    # Dispatch with error handling
    try:
        if args.command == "load":
            cmd_load(args, k)
        elif args.command == "checkpoint":
            cmd_checkpoint(args, k)
        elif args.command == "episode":
            cmd_episode(args, k)
        elif args.command == "note":
            cmd_note(args, k)
        elif args.command == "search":
            cmd_search(args, k)
        elif args.command == "status":
            cmd_status(args, k)
        elif args.command == "drive":
            cmd_drive(args, k)
        elif args.command == "consolidate":
            cmd_consolidate(args, k)
        elif args.command == "when":
            cmd_temporal(args, k)
        elif args.command == "mcp":
            cmd_mcp(args)
    except (ValueError, TypeError) as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
