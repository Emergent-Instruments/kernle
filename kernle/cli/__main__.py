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

from kernle import Kernle


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
        result = k.checkpoint(args.task, args.pending, args.context)
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
    episode_id = k.episode(
        objective=args.objective,
        outcome=args.outcome,
        lessons=args.lesson,
        tags=args.tag,
    )
    print(f"✓ Episode saved: {episode_id[:8]}...")
    if args.lesson:
        print(f"  Lessons: {len(args.lesson)}")


def cmd_note(args, k: Kernle):
    """Capture a note."""
    note_id = k.note(
        content=args.content,
        type=args.type,
        speaker=args.speaker,
        reason=args.reason,
        tags=args.tag,
        protect=args.protect,
    )
    print(f"✓ Note saved: {args.content[:50]}...")
    if args.tag:
        print(f"  Tags: {', '.join(args.tag)}")


def cmd_search(args, k: Kernle):
    """Search memory."""
    results = k.search(args.query, args.limit)
    if not results:
        print(f"No results for '{args.query}'")
        return
    
    print(f"Found {len(results)} result(s) for '{args.query}':\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['type']}] {r['title']}")
        if r.get("lessons"):
            for lesson in r["lessons"]:
                print(f"     → {lesson[:50]}...")
        if r.get("tags"):
            print(f"     tags: {', '.join(r['tags'])}")
        if r.get("confidence"):
            print(f"     confidence: {r['confidence']}")
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
    
    args = parser.parse_args()
    
    # Initialize Kernle
    k = Kernle(agent_id=args.agent)
    
    # Dispatch
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


if __name__ == "__main__":
    main()
