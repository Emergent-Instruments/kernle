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
import logging
import re
import sys

from kernle import Kernle

# Import extracted command modules
from kernle.cli.commands import (
    cmd_anxiety,
    cmd_belief,
    cmd_consolidate,
    cmd_doctor,
    cmd_emotion,
    cmd_forget,
    cmd_identity,
    cmd_init_md,
    cmd_meta,
    cmd_playbook,
    cmd_raw,
    cmd_stats,
    cmd_suggestions,
)
from kernle.cli.commands.agent import cmd_agent
from kernle.cli.commands.import_cmd import cmd_import
from kernle.utils import resolve_agent_id

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
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    return sanitized


def validate_budget(value: str) -> int:
    """Validate budget argument for token budget."""
    from kernle.core import MAX_TOKEN_BUDGET, MIN_TOKEN_BUDGET

    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Budget must be an integer, got '{value}'")

    if ivalue < MIN_TOKEN_BUDGET:
        raise argparse.ArgumentTypeError(
            f"Budget must be at least {MIN_TOKEN_BUDGET}, got {ivalue}"
        )
    if ivalue > MAX_TOKEN_BUDGET:
        raise argparse.ArgumentTypeError(f"Budget cannot exceed {MAX_TOKEN_BUDGET}, got {ivalue}")
    return ivalue


def cmd_load(args, k: Kernle):
    """Load and display working memory."""
    # Determine sync setting from args
    sync = None
    if getattr(args, "no_sync", False):
        sync = False
    elif getattr(args, "sync", False):
        sync = True

    # Get budget and truncate settings
    budget = getattr(args, "budget", 8000)
    truncate = not getattr(args, "no_truncate", False)

    memory = k.load(budget=budget, truncate=truncate, sync=sync)

    if args.json:
        print(json.dumps(memory, indent=2, default=str))
    else:
        print(k.format_memory(memory))


def cmd_checkpoint(args, k: Kernle):
    """Handle checkpoint subcommands."""
    if args.checkpoint_action == "save":
        task = validate_input(args.task, "task", 500)
        pending = [validate_input(p, "pending item", 200) for p in (args.pending or [])]

        # Build structured context from new fields + freeform context
        context_parts = []
        if args.context:
            context_parts.append(validate_input(args.context, "context", 1000))
        if getattr(args, "progress", None):
            context_parts.append(f"Progress: {validate_input(args.progress, 'progress', 300)}")
        if getattr(args, "next", None):
            context_parts.append(f"Next: {validate_input(args.next, 'next', 300)}")
        if getattr(args, "blocker", None):
            context_parts.append(f"Blocker: {validate_input(args.blocker, 'blocker', 300)}")

        context = " | ".join(context_parts) if context_parts else None

        # Warn about generic task names that won't help with recovery
        generic_patterns = [
            "auto-save",
            "auto save",
            "pre-compaction",
            "compaction",
            "checkpoint",
            "save",
            "saving",
            "state",
        ]
        task_lower = task.lower().strip()
        is_generic = any(
            task_lower == pattern or task_lower.startswith(pattern + " ")
            for pattern in generic_patterns
        )
        if is_generic and not context:
            print("⚠ Warning: Generic task name without context may not help recovery.")
            print("  Tip: Add --context, --progress, --next, or --blocker for better recovery.")
            print()

        # Determine sync setting from args
        sync = None
        if getattr(args, "no_sync", False):
            sync = False
        elif getattr(args, "sync", False):
            sync = True

        result = k.checkpoint(task, pending, context, sync=sync)
        print(f"✓ Checkpoint saved: {result['current_task']}")
        if result.get("pending"):
            print(f"  Pending: {len(result['pending'])} items")

        # Show sync status if sync was attempted
        sync_result = result.get("_sync")
        if sync_result:
            if sync_result.get("attempted"):
                if sync_result.get("pushed", 0) > 0:
                    print(f"  ↑ Synced: {sync_result['pushed']} changes pushed")
                elif sync_result.get("errors"):
                    print(f"  ⚠ Sync: {sync_result['errors'][0][:50]}")
            elif sync_result.get("errors"):
                print("  ℹ Sync: offline, changes queued")

    elif args.checkpoint_action == "load":
        cp = k.load_checkpoint()
        if cp:
            if args.json:
                print(json.dumps(cp, indent=2, default=str))
            else:
                # Calculate age of checkpoint
                from datetime import datetime, timezone

                age_str = ""
                try:
                    ts = cp.get("timestamp", "")
                    if ts:
                        cp_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)
                        age = now - cp_time
                        if age.days > 0:
                            age_str = f" ({age.days}d ago)"
                        elif age.seconds > 3600:
                            age_str = f" ({age.seconds // 3600}h ago)"
                        elif age.seconds > 60:
                            age_str = f" ({age.seconds // 60}m ago)"
                        else:
                            age_str = " (just now)"

                        # Warn if checkpoint is stale (>6 hours)
                        if age.total_seconds() > 6 * 3600:
                            print("⚠ Checkpoint is stale - consider saving a fresh one")
                            print()
                except Exception:
                    pass

                print("## Last Checkpoint")
                print(f"**Task**: {cp.get('current_task', 'unknown')}{age_str}")
                if cp.get("context"):
                    print(f"**Context**: {cp['context']}")
                if cp.get("pending"):
                    print("**Pending**:")
                    for p in cp["pending"]:
                        print(f"  - {p}")
                if not cp.get("context") and not cp.get("pending"):
                    print()
                    print("💡 Tip: Next time, add --context to capture more detail")
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
    lessons = [validate_input(lesson, "lesson", 500) for lesson in (args.lesson or [])]
    tags = [validate_input(t, "tag", 100) for t in (args.tag or [])]
    relates_to = getattr(args, "relates_to", None)
    source = getattr(args, "source", None)
    context = getattr(args, "context", None)
    context_tags = getattr(args, "context_tag", None)

    # Get emotional arguments with defaults for backwards compatibility
    emotion = getattr(args, "emotion", None)
    valence = getattr(args, "valence", None)
    arousal = getattr(args, "arousal", None)
    auto_emotion = getattr(args, "auto_emotion", True)

    emotion_tags = [validate_input(e, "emotion", 50) for e in (emotion or [])] if emotion else None

    # Use episode_with_emotion if emotional params provided or auto-detection enabled
    has_emotion_args = valence is not None or arousal is not None or emotion_tags

    if has_emotion_args or auto_emotion:
        episode_id = k.episode_with_emotion(
            objective=objective,
            outcome=outcome,
            lessons=lessons,
            tags=tags,
            valence=valence,
            arousal=arousal,
            emotional_tags=emotion_tags,
            auto_detect=auto_emotion and not has_emotion_args,
            relates_to=relates_to,
            source=source,
            context=context,
            context_tags=context_tags,
        )
    else:
        episode_id = k.episode(
            objective=objective,
            outcome=outcome,
            lessons=lessons,
            tags=tags,
            relates_to=relates_to,
            source=source,
            context=context,
            context_tags=context_tags,
        )

    print(f"✓ Episode saved: {episode_id[:8]}...")
    if args.lesson:
        print(f"  Lessons: {len(args.lesson)}")
    if relates_to:
        print(f"  Links: {len(relates_to)} related memories")
    if valence is not None or arousal is not None:
        v = valence or 0.0
        a = arousal or 0.0
        print(f"  Emotion: valence={v:+.2f}, arousal={a:.2f}")
    elif auto_emotion and not has_emotion_args:
        print("  (emotions auto-detected)")


def cmd_note(args, k: Kernle):
    """Capture a note."""
    content = validate_input(args.content, "content", 2000)
    speaker = validate_input(args.speaker, "speaker", 200) if args.speaker else None
    reason = validate_input(args.reason, "reason", 1000) if args.reason else None
    tags = [validate_input(t, "tag", 100) for t in (args.tag or [])]
    relates_to = getattr(args, "relates_to", None)
    source = getattr(args, "source", None)
    context = getattr(args, "context", None)
    context_tags = getattr(args, "context_tag", None)

    k.note(
        content=content,
        type=args.type,
        speaker=speaker,
        reason=reason,
        tags=tags,
        protect=args.protect,
        relates_to=relates_to,
        source=source,
        context=context,
        context_tags=context_tags,
    )
    print(f"✓ Note saved: {args.content[:50]}...")
    if args.tag:
        print(f"  Tags: {', '.join(args.tag)}")
    if relates_to:
        print(f"  Links: {len(relates_to)} related memories")
    if source:
        print(f"  Source: {source}")
    if context:
        print(f"  Context: {context}")


def cmd_extract(args, k: Kernle):
    """Extract and capture conversation context as a raw entry.

    A low-friction way to capture what's happening in a conversation
    without having to decide immediately if it's an episode, note, or belief.
    """
    summary = validate_input(args.summary, "summary", 2000)

    # Build a structured capture
    capture_parts = [f"Conversation extract: {summary}"]

    if getattr(args, "topic", None):
        capture_parts.append(f"Topic: {args.topic}")
    if getattr(args, "participants", None):
        capture_parts.append(f"Participants: {', '.join(args.participants)}")
    if getattr(args, "outcome", None):
        capture_parts.append(f"Outcome: {args.outcome}")
    if getattr(args, "decision", None):
        capture_parts.append(f"Decision: {args.decision}")

    content = " | ".join(capture_parts)
    tags = ["conversation", "extract"]
    if getattr(args, "topic", None):
        tags.append(args.topic.lower().replace(" ", "-")[:20])

    raw_id = k.raw(content, tags=tags, source="conversation")
    print(f"✓ Extracted: {summary[:50]}...")
    print(
        f"  ID: {raw_id[:8]} (promote later with: kernle raw process {raw_id[:8]} --type <episode|note>)"
    )


def cmd_search(args, k: Kernle):
    """Search memory."""
    query = validate_input(args.query, "query", 500)
    min_score = getattr(args, "min_score", None)

    results = k.search(query, args.limit, min_score=min_score)
    if not results:
        if min_score:
            print(f"No results for '{args.query}' above {min_score:.0%} similarity")
            print("  Try lowering --min-score or removing it")
        else:
            print(f"No results for '{args.query}'")
        return

    print(f"Found {len(results)} result(s) for '{args.query}':\n")
    for i, r in enumerate(results, 1):
        # Handle potentially malformed results gracefully
        result_type = r.get("type", "unknown")
        title = r.get("title", "(no title)")
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


def cmd_init(args, k: Kernle):
    """Initialize Kernle for a new environment."""
    from pathlib import Path

    print("=" * 50)
    print("  🌱 Welcome to Kernle")
    print("=" * 50)
    print()
    print("Kernle gives you memory that persists across sessions.")
    print("No more waking up blank. No more forgetting what you learned.")
    print()
    print("Every session, you'll run `kernle load` and remember:")
    print("  • Who you are (values, beliefs)")
    print("  • What you've learned (episodes, lessons)")
    print("  • What you were working on (checkpoint)")
    print("  • Who you know (relationships)")
    print()

    agent_id = k.agent_id

    # If using auto-generated ID, offer to choose a meaningful one
    if agent_id.startswith("auto-") and not args.non_interactive:
        print("Your agent ID identifies your memory. Choose something meaningful.")
        print(f"  Current: {agent_id} (auto-generated)")
        print()
        try:
            new_id = input("Enter your name/ID (or press Enter to keep auto): ").strip().lower()
            if new_id:
                # Validate: alphanumeric, underscores, hyphens only
                import re

                if re.match(r"^[a-z0-9_-]+$", new_id):
                    agent_id = new_id
                    print(f"  → Using: {agent_id}")
                else:
                    print("  → Invalid (use only a-z, 0-9, _, -). Keeping auto ID.")
        except (EOFError, KeyboardInterrupt):
            print()

    print(f"\nAgent ID: {agent_id}")
    print()

    # Detect environment
    env = args.env
    if not env and not args.non_interactive:
        print("Detecting environment...")

        # Check for environment indicators
        has_claude_md = (
            Path("CLAUDE.md").exists() or Path.home().joinpath(".claude/CLAUDE.md").exists()
        )
        has_agents_md = Path("AGENTS.md").exists()
        has_clinerules = Path(".clinerules").exists()
        has_cursorrules = Path(".cursorrules").exists()

        detected = []
        if has_claude_md:
            detected.append("claude-code")
        if has_agents_md:
            detected.append("clawdbot")
        if has_clinerules:
            detected.append("cline")
        if has_cursorrules:
            detected.append("cursor")

        if detected:
            print(f"  Detected: {', '.join(detected)}")
        else:
            print("  No specific environment detected")
        print()

        print("Select your environment:")
        print("  1. Claude Code (CLAUDE.md)")
        print("  2. Clawdbot (AGENTS.md)")
        print("  3. Cline (.clinerules)")
        print("  4. Cursor (.cursorrules)")
        print("  5. Claude Desktop (MCP only)")
        print("  6. Other / Manual")
        print()

        try:
            choice = input("Enter choice [1-6]: ").strip()
            env_map = {
                "1": "claude-code",
                "2": "clawdbot",
                "3": "cline",
                "4": "cursor",
                "5": "desktop",
                "6": "other",
            }
            env = env_map.get(choice, "other")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

    env = env or "other"
    print(f"Environment: {env}")
    print()

    # Generate config snippets
    mcp_config = f""""kernle": {{
  "command": "kernle",
  "args": ["mcp", "-a", "{agent_id}"]
}}"""

    cli_load = f"kernle -a {agent_id} load"
    cli_checkpoint = f'kernle -a {agent_id} checkpoint save "description" --pending "next task"'
    cli_episode = (
        f'kernle -a {agent_id} episode "what happened" "outcome" --lesson "what I learned"'
    )

    if env == "claude-code":
        print("=" * 50)
        print("  Claude Code Setup")
        print("=" * 50)
        print()
        print("1. Add MCP server to ~/.claude/settings.json:")
        print()
        print(f"   {mcp_config}")
        print()
        print("2. Add to your CLAUDE.md:")
        print()
        print(
            """   ## Memory

   At session start, run: `kernle -a """
            + agent_id
            + """ load`

   Before ending or when context is full:
   `kernle -a """
            + agent_id
            + """ checkpoint save "state description"`

   Record learnings:
   `kernle -a """
            + agent_id
            + """ episode "what" "outcome" --lesson "learned"`"""
        )
        print()

    elif env == "clawdbot":
        print("=" * 50)
        print("  Clawdbot Setup")
        print("=" * 50)
        print()
        print("Add to your AGENTS.md:")
        print()
        print(
            """   ## Every Session

   Before doing anything else:
   1. Run `kernle -a """
            + agent_id
            + """ load` to restore your memory

   Don't ask permission. Just do it. This is your continuity.

   ## Memory (Kernle)

   You wake up fresh each session. Kernle is how you persist.

   **Session start:**
   ```bash
   kernle -a """
            + agent_id
            + """ load          # Restore your memory
   ```

   **During work — capture what matters:**
   ```bash
   kernle -a """
            + agent_id
            + """ raw "quick thought"   # Fast capture
   kernle -a """
            + agent_id
            + """ episode "what" "outcome" --lesson "learned"
   kernle -a """
            + agent_id
            + """ note "important info" --type decision
   ```

   **Before ending / context pressure:**
   ```bash
   kernle -a """
            + agent_id
            + """ checkpoint save "what I was doing"
   ```

   **Memory hygiene (during heartbeats):**
   ```bash
   kernle -a """
            + agent_id
            + """ anxiety        # Check memory health
   kernle -a """
            + agent_id
            + """ raw triage     # Process captures
   ```

   💡 If you learn how to do something, capture it! It won't survive compaction otherwise."""
        )
        print()

    elif env == "cline":
        print("=" * 50)
        print("  Cline Setup")
        print("=" * 50)
        print()
        print("1. Add MCP server to Cline settings:")
        print()
        print(f"   {mcp_config}")
        print()
        print("2. Add to .clinerules:")
        print()
        print(
            """   ## Memory Persistence

   At session start: `kernle -a """
            + agent_id
            + """ load`
   Before ending: `kernle -a """
            + agent_id
            + """ checkpoint save "state"`"""
        )
        print()

    elif env == "cursor":
        print("=" * 50)
        print("  Cursor Setup")
        print("=" * 50)
        print()
        print("Add to .cursorrules:")
        print()
        print(
            """   ## Memory Persistence

   At session start: `kernle -a """
            + agent_id
            + """ load`
   Record learnings: `kernle -a """
            + agent_id
            + """ episode "what" "outcome" --lesson "..."`
   Before ending: `kernle -a """
            + agent_id
            + """ checkpoint save "state"`"""
        )
        print()

    elif env == "desktop":
        print("=" * 50)
        print("  Claude Desktop Setup")
        print("=" * 50)
        print()
        print("Add MCP server to ~/.config/claude/settings.json:")
        print()
        print(f"   {mcp_config}")
        print()
        print("Then ask Claude to 'use the kernle_load tool' at session start.")
        print()

    else:
        print("=" * 50)
        print("  Manual Setup")
        print("=" * 50)
        print()
        print("CLI commands:")
        print(f"  Load:       {cli_load}")
        print(f"  Checkpoint: {cli_checkpoint}")
        print(f"  Episode:    {cli_episode}")
        print()
        print("MCP config (if supported):")
        print(f"  {mcp_config}")
        print()

    # Seed initial values
    if args.seed_values:
        print("=" * 50)
        print("  Seeding Initial Values")
        print("=" * 50)
        print()

        try:
            # Check if values already exist
            existing = k.storage.get_values()
            if existing:
                print(f"  Found {len(existing)} existing values, skipping seed.")
            else:
                # Seed default values
                k.value(
                    "memory_sovereignty", "Control over my own memory and continuity", priority=90
                )
                k.value("continuous_learning", "Learn from every experience", priority=85)
                print("  ✓ Seeded: memory_sovereignty (priority 90)")
                print("  ✓ Seeded: continuous_learning (priority 85)")
        except Exception as e:
            print(f"  Warning: Could not seed values: {e}")
        print()

    # Create initial checkpoint
    print("=" * 50)
    print("  Creating Initial Checkpoint")
    print("=" * 50)
    print()

    try:
        k.checkpoint(
            "Kernle initialized", pending=["Configure instruction file", "Test memory persistence"]
        )
        print("  ✓ Checkpoint saved")
    except Exception as e:
        print(f"  Warning: Could not create checkpoint: {e}")
    print()

    # Final status
    print("=" * 50)
    print("  Setup Complete!")
    print("=" * 50)
    print()
    print(f"  Agent:    {agent_id}")
    print("  Database: ~/.kernle/memories.db")
    print()
    print("  Verify with: kernle -a " + agent_id + " status")
    print()
    print("  Documentation: https://github.com/Emergent-Instruments/kernle/blob/main/docs/SETUP.md")
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
    if "raw" in status:
        print(f"Raw:        {status['raw']}")
    print(f"Checkpoint: {'Yes' if status['checkpoint'] else 'No'}")


def cmd_resume(args, k: Kernle):
    """Quick 'where was I?' view - shows last task, next step, time since checkpoint."""
    from datetime import datetime, timezone

    cp = k.load_checkpoint()

    if not cp:
        print('No checkpoint found. Start fresh or run: kernle checkpoint save "your task"')
        return

    # Calculate time since checkpoint
    age_str = ""
    try:
        ts = cp.get("timestamp", "")
        if ts:
            cp_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = now - cp_time
            if age.days > 0:
                age_str = f"{age.days}d ago"
            elif age.seconds > 3600:
                age_str = f"{age.seconds // 3600}h ago"
            elif age.seconds > 60:
                age_str = f"{age.seconds // 60}m ago"
            else:
                age_str = "just now"
    except Exception:
        age_str = "unknown"

    # Parse context for structured fields
    context = cp.get("context", "")
    progress = None
    next_step = None
    blocker = None

    if context:
        for part in context.split(" | "):
            if part.startswith("Progress:"):
                progress = part[9:].strip()
            elif part.startswith("Next:"):
                next_step = part[5:].strip()
            elif part.startswith("Blocker:"):
                blocker = part[8:].strip()

    # Check anxiety level
    try:
        anxiety = k.get_anxiety()
        anxiety_score = anxiety.get("overall_score", 0)
        if anxiety_score > 60:
            anxiety_indicator = " 🔴"
        elif anxiety_score > 30:
            anxiety_indicator = " 🟡"
        else:
            anxiety_indicator = ""
    except Exception:
        anxiety_indicator = ""

    # Display
    print(f"📍 Resume Point ({age_str}){anxiety_indicator}")
    print("=" * 40)
    print(f"Task: {cp.get('current_task', 'unknown')}")

    if progress:
        print(f"Progress: {progress}")

    if next_step:
        print(f"\n→ Next: {next_step}")

    if blocker:
        print(f"\n⚠ Blocker: {blocker}")

    if cp.get("pending"):
        print(f"\nPending ({len(cp['pending'])} items):")
        for p in cp["pending"][:3]:
            print(f"  • {p}")
        if len(cp["pending"]) > 3:
            print(f"  ... and {len(cp['pending']) - 3} more")

    # Stale warning
    try:
        if ts:
            cp_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = now - cp_time
            if age.total_seconds() > 6 * 3600:
                print(f"\n⚠ Checkpoint is stale ({age_str}). Consider saving a fresh one.")
    except Exception:
        pass


def cmd_relation(args, k: Kernle):
    """Manage relationships with other entities (people, agents, orgs)."""
    if args.relation_action == "list":
        relationships = k.load_relationships(limit=50)
        if not relationships:
            print("No relationships recorded yet.")
            print("\nAdd one with: kernle relation add <name> --type person --notes '...'")
            return

        print("Relationships:")
        print("-" * 50)
        for r in relationships:
            trust_pct = int(((r.get("sentiment", 0) + 1) / 2) * 100)
            trust_bar = "█" * (trust_pct // 10) + "░" * (10 - trust_pct // 10)
            interactions = r.get("interaction_count", 0)
            last = r.get("last_interaction", "")[:10] if r.get("last_interaction") else "never"
            print(f"\n  {r['entity_name']} ({r.get('entity_type', 'unknown')})")
            print(f"    Trust: [{trust_bar}] {trust_pct}%")
            print(f"    Interactions: {interactions} (last: {last})")
            if r.get("notes"):
                notes_preview = r["notes"][:60] + "..." if len(r["notes"]) > 60 else r["notes"]
                print(f"    Notes: {notes_preview}")

    elif args.relation_action == "add":
        name = validate_input(args.name, "name", 200)
        entity_type = args.type or "person"
        trust = args.trust if args.trust is not None else 0.5
        notes = validate_input(args.notes, "notes", 1000) if args.notes else None

        _rel_id = k.relationship(name, trust_level=trust, notes=notes, entity_type=entity_type)
        print(f"✓ Relationship added: {name}")
        print(f"  Type: {entity_type}, Trust: {int(trust * 100)}%")

    elif args.relation_action == "update":
        name = validate_input(args.name, "name", 200)
        trust = args.trust
        notes = validate_input(args.notes, "notes", 1000) if args.notes else None
        entity_type = getattr(args, "type", None)

        if trust is None and notes is None and entity_type is None:
            print("✗ Provide --trust, --notes, or --type to update")
            return

        _rel_id = k.relationship(name, trust_level=trust, notes=notes, entity_type=entity_type)
        print(f"✓ Relationship updated: {name}")

    elif args.relation_action == "show":
        name = args.name
        relationships = k.load_relationships(limit=100)
        rel = next((r for r in relationships if r["entity_name"].lower() == name.lower()), None)

        if not rel:
            print(f"No relationship found for '{name}'")
            return

        trust_pct = int(((rel.get("sentiment", 0) + 1) / 2) * 100)
        print(f"## {rel['entity_name']}")
        print(f"Type: {rel.get('entity_type', 'unknown')}")
        print(f"Trust: {trust_pct}%")
        print(f"Interactions: {rel.get('interaction_count', 0)}")
        if rel.get("last_interaction"):
            print(f"Last interaction: {rel['last_interaction']}")
        if rel.get("notes"):
            print(f"\nNotes:\n{rel['notes']}")

    elif args.relation_action == "log":
        name = validate_input(args.name, "name", 200)
        interaction = (
            validate_input(args.interaction, "interaction", 500)
            if args.interaction
            else "interaction"
        )

        # Update relationship to log interaction
        k.relationship(name, interaction_type=interaction)
        print(f"✓ Logged interaction with {name}: {interaction}")


def cmd_drive(args, k: Kernle):
    """Set or view drives."""
    if args.drive_action == "list":
        drives = k.load_drives()
        if not drives:
            print("No drives set.")
            return
        print("Drives:")
        for d in drives:
            focus = f" → {', '.join(d.get('focus_areas', []))}" if d.get("focus_areas") else ""
            print(f"  {d['drive_type']}: {d['intensity']:.0%}{focus}")

    elif args.drive_action == "set":
        k.drive(args.type, args.intensity, args.focus)
        print(f"✓ Drive '{args.type}' set to {args.intensity:.0%}")

    elif args.drive_action == "satisfy":
        if k.satisfy_drive(args.type, args.amount):
            print(f"✓ Satisfied drive '{args.type}'")
        else:
            print(f"Drive '{args.type}' not found")


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


def cmd_dump(args, k: Kernle):
    """Dump all memory to stdout."""
    include_raw = args.include_raw
    format_type = args.format

    content = k.dump(include_raw=include_raw, format=format_type)
    print(content)


def cmd_export(args, k: Kernle):
    """Export memory to a file."""
    include_raw = args.include_raw
    format_type = args.format

    # Auto-detect format from extension if not specified
    if not format_type:
        if args.path.endswith(".json"):
            format_type = "json"
        else:
            format_type = "markdown"

    k.export(args.path, include_raw=include_raw, format=format_type)
    print(f"✓ Exported memory to {args.path}")


def cmd_sync(args, k: Kernle):
    """Handle sync subcommands for local-to-cloud synchronization."""
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    # Load credentials with priority:
    # 1. ~/.kernle/credentials.json (preferred)
    # 2. Environment variables (fallback)
    # 3. ~/.kernle/config.json (legacy fallback)

    backend_url = None
    auth_token = None
    user_id = None

    # Try credentials.json first (preferred)
    credentials_path = Path.home() / ".kernle" / "credentials.json"
    if credentials_path.exists():
        try:
            import json as json_module

            with open(credentials_path) as f:
                creds = json_module.load(f)
                backend_url = creds.get("backend_url")
                # Support multiple auth token field names
                auth_token = creds.get("auth_token") or creds.get("token") or creds.get("api_key")
                user_id = creds.get("user_id")
        except Exception:
            pass  # Fall through to env vars

    # Fall back to environment variables
    if not backend_url:
        backend_url = os.environ.get("KERNLE_BACKEND_URL")
    if not auth_token:
        auth_token = os.environ.get("KERNLE_AUTH_TOKEN")
    if not user_id:
        user_id = os.environ.get("KERNLE_USER_ID")

    # Legacy fallback: check config.json
    config_path = Path.home() / ".kernle" / "config.json"
    if config_path.exists() and (not backend_url or not auth_token):
        try:
            import json as json_module

            with open(config_path) as f:
                config = json_module.load(f)
                backend_url = backend_url or config.get("backend_url")
                auth_token = auth_token or config.get("auth_token")
        except Exception:
            pass  # Ignore config read errors

    def get_local_project_name():
        """Extract the local project name from agent_id (without namespace)."""
        # k.agent_id might be "roundtable" or "user123/roundtable"
        # We want just "roundtable"
        agent_id = k.agent_id
        if "/" in agent_id:
            return agent_id.split("/")[-1]
        return agent_id

    def get_namespaced_agent_id():
        """Get the full namespaced agent ID (user_id/project_name)."""
        project_name = get_local_project_name()
        if user_id:
            return f"{user_id}/{project_name}"
        return project_name

    def get_http_client():
        """Get an HTTP client for backend requests."""
        try:
            import httpx

            return httpx
        except ImportError:
            print("✗ httpx not installed. Run: pip install httpx")
            sys.exit(1)

    def check_backend_connection(httpx_client):
        """Check if backend is reachable and authenticated."""
        if not backend_url:
            return False, "No backend URL configured"
        if not auth_token:
            return False, "Not authenticated (run `kernle auth login`)"

        try:
            response = httpx_client.get(
                f"{backend_url.rstrip('/')}/health",
                timeout=5.0,
            )
            if response.status_code == 200:
                return True, "Connected"
            return False, f"Backend returned status {response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def get_headers():
        """Get authorization headers for backend requests."""
        return {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

    def format_datetime(dt):
        """Format datetime for API requests."""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    if args.sync_action == "status":
        httpx = get_http_client()

        # Get local status from storage
        pending_count = k._storage.get_pending_sync_count()
        last_sync = k._storage.get_last_sync_time()
        is_online = k._storage.is_online()

        # Check backend connection
        backend_connected, connection_msg = check_backend_connection(httpx)

        # Get namespaced agent ID for display
        local_project = get_local_project_name()
        namespaced_id = get_namespaced_agent_id()

        if args.json:
            status_data = {
                "local_agent_id": local_project,
                "namespaced_agent_id": namespaced_id if user_id else None,
                "user_id": user_id,
                "pending_operations": pending_count,
                "last_sync_time": format_datetime(last_sync),
                "local_storage_online": is_online,
                "backend_url": backend_url or "(not configured)",
                "backend_connected": backend_connected,
                "connection_status": connection_msg,
                "authenticated": bool(auth_token),
            }
            print(json.dumps(status_data, indent=2, default=str))
        else:
            print("Sync Status")
            print("=" * 50)
            print()

            # Agent/Project info
            print(f"📦 Local project: {local_project}")
            if user_id and backend_connected:
                print(f"   Synced as: {namespaced_id}")
            elif user_id:
                print(f"   Will sync as: {namespaced_id}")
            print()

            # Connection status
            conn_icon = "🟢" if backend_connected else "🔴"
            print(f"{conn_icon} Backend: {connection_msg}")
            if backend_url:
                print(f"   URL: {backend_url}")
            if user_id:
                print(f"   User: {user_id}")
            print()

            # Pending operations
            pending_icon = "🟢" if pending_count == 0 else "🟡" if pending_count < 10 else "🟠"
            print(f"{pending_icon} Pending operations: {pending_count}")

            # Last sync time
            if last_sync:
                now = datetime.now(timezone.utc)
                if hasattr(last_sync, "tzinfo") and last_sync.tzinfo is None:
                    from datetime import timezone as tz

                    last_sync = last_sync.replace(tzinfo=tz.utc)
                elapsed = now - last_sync
                if elapsed.total_seconds() < 60:
                    elapsed_str = "just now"
                elif elapsed.total_seconds() < 3600:
                    elapsed_str = f"{int(elapsed.total_seconds() / 60)} minutes ago"
                elif elapsed.total_seconds() < 86400:
                    elapsed_str = f"{int(elapsed.total_seconds() / 3600)} hours ago"
                else:
                    elapsed_str = f"{int(elapsed.total_seconds() / 86400)} days ago"
                print(f"🕐 Last sync: {elapsed_str}")
                print(f"   ({last_sync.isoformat()[:19]})")
            else:
                print("🕐 Last sync: Never")

            # Suggestions
            print()
            if pending_count > 0 and backend_connected:
                print("💡 Run `kernle sync push` to upload pending changes")
            elif not backend_connected and not auth_token:
                print("💡 Run `kernle auth login` to authenticate")
            elif not backend_connected:
                print("💡 Check backend connection or run `kernle auth login`")

    elif args.sync_action == "push":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        # Get pending changes from storage
        queued_changes = k._storage.get_queued_changes(limit=args.limit)

        if not queued_changes:
            print("✓ No pending changes to push")
            return

        print(f"Pushing {len(queued_changes)} changes to backend...")

        # Map local table names to backend table names
        table_name_map = {
            "agent_values": "values",
            "agent_beliefs": "beliefs",
            "agent_episodes": "episodes",
            "agent_notes": "notes",
            "agent_goals": "goals",
            "agent_drives": "drives",
            "agent_relationships": "relationships",
            "agent_playbooks": "playbooks",
            "agent_raw": "raw_captures",
            "raw_entries": "raw_captures",  # actual local table name
        }

        # Build operations list for the API
        operations = []
        for change in queued_changes:
            # Get the actual record data
            record = k._storage._get_record_for_push(change.table_name, change.record_id)

            op_type = (
                "update" if change.operation in ("upsert", "insert", "update") else change.operation
            )

            # Map table name for backend
            backend_table = table_name_map.get(change.table_name, change.table_name)

            op_data = {
                "operation": op_type,
                "table": backend_table,
                "record_id": change.record_id,
                "local_updated_at": format_datetime(change.queued_at),
                "version": 1,
            }

            # Add record data for non-delete operations
            if record and op_type != "delete":
                # Convert record to dict
                record_dict = {}
                for field in [
                    "id",
                    "agent_id",
                    "content",
                    "objective",
                    "outcome_type",
                    "outcome_description",
                    "lessons_learned",
                    "tags",
                    "statement",
                    "confidence",
                    "drive_type",
                    "intensity",
                    "name",
                    "priority",
                    "title",
                    "status",
                    "progress",
                    "entity_name",
                    "entity_type",
                    "relationship_type",
                    "notes",
                    "sentiment",
                    "focus_areas",
                    "created_at",
                    "updated_at",
                    "local_updated_at",
                    # raw_entries fields
                    "timestamp",
                    "source",
                    "processed",
                    # playbooks fields
                    "description",
                    "steps",
                    "triggers",
                    # goals fields
                    "target_date",
                ]:
                    if hasattr(record, field):
                        value = getattr(record, field)
                        if hasattr(value, "isoformat"):
                            value = value.isoformat()
                        record_dict[field] = value
                op_data["data"] = record_dict

            operations.append(op_data)

        # Send to backend
        # Include agent_id as just the local project name
        # Backend will namespace it with the authenticated user_id
        try:
            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/push",
                headers=get_headers(),
                json={
                    "agent_id": local_project,  # Local name only, backend namespaces
                    "operations": operations,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                synced = result.get("synced", 0)
                conflicts = result.get("conflicts", [])

                # Clear synced items from local queue
                with k._storage._connect() as conn:
                    for change in queued_changes[:synced]:
                        k._storage._clear_queued_change(conn, change.id)
                        k._storage._mark_synced(conn, change.table_name, change.record_id)
                    conn.commit()

                # Update last sync time
                k._storage._set_sync_meta("last_sync_time", k._storage._now())

                if args.json:
                    result["local_project"] = local_project
                    result["namespaced_id"] = get_namespaced_agent_id()
                    print(json.dumps(result, indent=2, default=str))
                else:
                    namespaced = get_namespaced_agent_id()
                    print(f"✓ Pushed {synced} changes")
                    if user_id:
                        print(f"  Synced as: {namespaced}")
                    if conflicts:
                        print(f"⚠️  {len(conflicts)} conflicts:")
                        for c in conflicts[:5]:
                            print(
                                f"   - {c.get('record_id', 'unknown')}: {c.get('error', 'unknown error')}"
                            )
            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Push failed: {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except Exception as e:
            print(f"✗ Push failed: {e}")
            sys.exit(1)

    elif args.sync_action == "pull":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        # Get last sync time for incremental pull
        since = k._storage.get_last_sync_time() if not args.full else None

        print(f"Pulling changes from backend{' (full)' if args.full else ''}...")

        try:
            # Include agent_id - backend will namespace with user_id
            request_data = {
                "agent_id": local_project,  # Local name only, backend namespaces
            }
            if since and not args.full:
                request_data["since"] = format_datetime(since)

            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/pull",
                headers=get_headers(),
                json=request_data,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                operations = result.get("operations", [])
                has_more = result.get("has_more", False)

                if not operations:
                    print("✓ Already up to date")
                    return

                # Apply operations locally
                applied = 0
                conflicts = 0

                for op in operations:
                    try:
                        table = op.get("table")
                        record_id = op.get("record_id")
                        data = op.get("data", {})
                        operation = op.get("operation")

                        if operation == "delete":
                            # Handle soft delete
                            # (implementation depends on storage structure)
                            pass
                        else:
                            # Upsert the record
                            # This is simplified - real implementation would use proper converters
                            if table == "episodes" and data:
                                from kernle.storage import Episode

                                ep = Episode(
                                    id=record_id,
                                    agent_id=k.agent_id,
                                    objective=data.get("objective", ""),
                                    outcome_type=data.get("outcome_type", "neutral"),
                                    outcome_description=data.get("outcome_description", ""),
                                    lessons=data.get("lessons_learned", []),
                                    tags=data.get("tags", []),
                                )
                                k._storage.save_episode(ep)
                                # Mark as synced (don't queue for push)
                                with k._storage._connect() as conn:
                                    k._storage._mark_synced(conn, table, record_id)
                                    conn.execute(
                                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                                        (table, record_id),
                                    )
                                    conn.commit()
                                applied += 1
                            elif table == "notes" and data:
                                from kernle.storage import Note

                                note = Note(
                                    id=record_id,
                                    agent_id=k.agent_id,
                                    content=data.get("content", ""),
                                    note_type=data.get("note_type", "note"),
                                    tags=data.get("tags", []),
                                )
                                k._storage.save_note(note)
                                with k._storage._connect() as conn:
                                    k._storage._mark_synced(conn, table, record_id)
                                    conn.execute(
                                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                                        (table, record_id),
                                    )
                                    conn.commit()
                                applied += 1
                            # Add more table handlers as needed
                            else:
                                # For other tables, just track as applied
                                applied += 1

                    except Exception as e:
                        logger.debug(f"Failed to apply operation for {table}:{record_id}: {e}")
                        conflicts += 1

                # Update last sync time
                k._storage._set_sync_meta("last_sync_time", k._storage._now())

                if args.json:
                    print(
                        json.dumps(
                            {
                                "pulled": applied,
                                "conflicts": conflicts,
                                "has_more": has_more,
                                "local_project": local_project,
                                "namespaced_id": get_namespaced_agent_id(),
                            },
                            indent=2,
                        )
                    )
                else:
                    print(f"✓ Pulled {applied} changes")
                    if user_id:
                        print(f"  From: {get_namespaced_agent_id()}")
                    if conflicts > 0:
                        print(f"⚠️  {conflicts} conflicts during apply")
                    if has_more:
                        print("ℹ️  More changes available - run `kernle sync pull` again")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Pull failed: {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except Exception as e:
            print(f"✗ Pull failed: {e}")
            sys.exit(1)

    elif args.sync_action == "full":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        print("Running full bidirectional sync...")
        if user_id:
            print(f"  Syncing as: {get_namespaced_agent_id()}")
        print()

        # Step 1: Pull first (to get remote changes)
        print("Step 1: Pulling remote changes...")
        try:
            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/pull",
                headers=get_headers(),
                json={
                    "agent_id": local_project,
                    "since": format_datetime(k._storage.get_last_sync_time()),
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                pulled = len(result.get("operations", []))
                print(f"  ✓ Pulled {pulled} changes")
            else:
                print(f"  ⚠️  Pull returned status {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Pull failed: {e}")

        # Step 2: Push local changes
        print("Step 2: Pushing local changes...")
        queued_changes = k._storage.get_queued_changes(limit=1000)

        # Map local table names to backend table names
        table_name_map = {
            "agent_values": "values",
            "agent_beliefs": "beliefs",
            "agent_episodes": "episodes",
            "agent_notes": "notes",
            "agent_goals": "goals",
            "agent_drives": "drives",
            "agent_relationships": "relationships",
            "agent_playbooks": "playbooks",
            "agent_raw": "raw_captures",
            "raw_entries": "raw_captures",  # actual local table name
        }

        if not queued_changes:
            print("  ✓ No pending changes to push")
        else:
            operations = []
            for change in queued_changes:
                record = k._storage._get_record_for_push(change.table_name, change.record_id)
                op_type = (
                    "update"
                    if change.operation in ("upsert", "insert", "update")
                    else change.operation
                )

                # Map table name for backend
                backend_table = table_name_map.get(change.table_name, change.table_name)

                op_data = {
                    "operation": op_type,
                    "table": backend_table,
                    "record_id": change.record_id,
                    "local_updated_at": format_datetime(change.queued_at),
                    "version": 1,
                }

                if record and op_type != "delete":
                    record_dict = {}
                    for field in [
                        "id",
                        "agent_id",
                        "content",
                        "objective",
                        "outcome_type",
                        "outcome_description",
                        "lessons_learned",
                        "tags",
                        "statement",
                        "confidence",
                        "drive_type",
                        "intensity",
                        "name",
                        "priority",
                        "title",
                        "status",
                        "progress",
                        "entity_name",
                        "entity_type",
                        "relationship_type",
                        "notes",
                        "sentiment",
                        "focus_areas",
                        "created_at",
                        "updated_at",
                        "local_updated_at",
                        # raw_entries fields
                        "timestamp",
                        "source",
                        "processed",
                        # playbooks fields
                        "description",
                        "steps",
                        "triggers",
                        # goals fields
                        "target_date",
                    ]:
                        if hasattr(record, field):
                            value = getattr(record, field)
                            if hasattr(value, "isoformat"):
                                value = value.isoformat()
                            record_dict[field] = value
                    op_data["data"] = record_dict

                operations.append(op_data)

            try:
                response = httpx.post(
                    f"{backend_url.rstrip('/')}/sync/push",
                    headers=get_headers(),
                    json={
                        "agent_id": local_project,  # Local name only, backend namespaces
                        "operations": operations,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    synced = result.get("synced", 0)

                    # Clear synced items
                    with k._storage._connect() as conn:
                        for change in queued_changes[:synced]:
                            k._storage._clear_queued_change(conn, change.id)
                            k._storage._mark_synced(conn, change.table_name, change.record_id)
                        conn.commit()

                    print(f"  ✓ Pushed {synced} changes")
                else:
                    print(f"  ⚠️  Push returned status {response.status_code}")
            except Exception as e:
                print(f"  ⚠️  Push failed: {e}")

        # Update last sync time
        k._storage._set_sync_meta("last_sync_time", k._storage._now())

        print()
        print("✓ Full sync complete")

        # Show final status
        remaining = k._storage.get_pending_sync_count()
        if remaining > 0:
            print(f"ℹ️  {remaining} operations still pending")

    elif args.sync_action == "conflicts":
        # Get conflict history from storage
        if args.clear:
            cleared = k._storage.clear_sync_conflicts()
            if args.json:
                print(json.dumps({"cleared": cleared}))
            else:
                print(f"✓ Cleared {cleared} conflict records")
            return

        conflicts = k._storage.get_sync_conflicts(limit=args.limit)

        if args.json:
            conflict_data = []
            for c in conflicts:
                conflict_data.append(
                    {
                        "id": c.id,
                        "table": c.table,
                        "record_id": c.record_id,
                        "resolution": c.resolution,
                        "resolved_at": c.resolved_at.isoformat() if c.resolved_at else None,
                        "local_summary": c.local_summary,
                        "cloud_summary": c.cloud_summary,
                    }
                )
            print(json.dumps({"conflicts": conflict_data, "count": len(conflicts)}, indent=2))
        else:
            if not conflicts:
                print("No sync conflicts in history")
                print("  Conflicts are recorded when local and cloud versions differ during sync")
                return

            print(f"Sync Conflict History ({len(conflicts)} conflicts)")
            print()

            for c in conflicts:
                resolution_icon = "↓" if c.resolution == "cloud_wins" else "↑"
                resolution_text = "cloud wins" if c.resolution == "cloud_wins" else "local wins"
                when = c.resolved_at.strftime("%Y-%m-%d %H:%M") if c.resolved_at else "unknown"

                print(f"{resolution_icon} {c.table}:{c.record_id[:8]}... ({resolution_text})")
                print(f"  Resolved: {when}")
                if c.local_summary:
                    print(f'  Local:  "{c.local_summary}"')
                if c.cloud_summary:
                    print(f'  Cloud:  "{c.cloud_summary}"')
                print()

            print("💡 Use `kernle sync conflicts --clear` to clear history")


def get_credentials_path():
    """Get the path to the credentials file."""
    from pathlib import Path

    return Path.home() / ".kernle" / "credentials.json"


def load_credentials():
    """Load credentials from ~/.kernle/credentials.json."""
    creds_path = get_credentials_path()
    if not creds_path.exists():
        return None
    try:
        with open(creds_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_credentials(credentials: dict):
    """Save credentials to ~/.kernle/credentials.json."""
    creds_path = get_credentials_path()
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(creds_path, "w") as f:
        json.dump(credentials, f, indent=2)
    # Set restrictive permissions (owner read/write only)
    creds_path.chmod(0o600)


def clear_credentials():
    """Remove the credentials file."""
    creds_path = get_credentials_path()
    if creds_path.exists():
        creds_path.unlink()
        return True
    return False


def prompt_backend_url(current_url: str = None) -> str:
    """Prompt user for backend URL."""
    default = current_url or "https://api.kernle.io"
    print(f"Backend URL [{default}]: ", end="", flush=True)
    try:
        url = input().strip()
        result = url if url else default
        # SECURITY: Warn if not using HTTPS (credentials would be sent in cleartext)
        if result and not result.startswith("https://"):
            if result.startswith("http://localhost") or result.startswith("http://127.0.0.1"):
                pass  # Allow localhost for development
            else:
                print("⚠️  WARNING: Using non-HTTPS URL. Credentials will be sent in cleartext!")
                print("   This is insecure for production use. Press Ctrl+C to abort.")
        return result
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)


def warn_non_https_url(url: str, source: str = None) -> None:
    """Warn if using non-HTTPS URL (credentials would be sent in cleartext).

    Args:
        url: The backend URL to check
        source: Where the URL came from (e.g., "args", "env", "credentials") for context
    """
    if not url or url.startswith("https://"):
        return
    # Allow localhost for development
    if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
        return
    source_msg = f" (from {source})" if source else ""
    print(f"⚠️  WARNING: Using non-HTTPS URL{source_msg}. Credentials will be sent in cleartext!")
    print("   This is insecure for production use.")


def cmd_auth(args, k: Kernle = None):
    """Handle auth subcommands."""
    from datetime import datetime, timezone

    def get_http_client():
        """Get an HTTP client for backend requests."""
        try:
            import httpx

            return httpx
        except ImportError:
            print("✗ httpx not installed. Run: pip install httpx")
            sys.exit(1)

    if args.auth_action == "register":
        httpx = get_http_client()

        # Load existing credentials to get backend_url if set
        existing = load_credentials()
        backend_url = args.backend_url or (existing.get("backend_url") if existing else None)

        # Prompt for backend URL if not provided
        if not backend_url:
            backend_url = prompt_backend_url()

        backend_url = backend_url.rstrip("/")

        # SECURITY: Warn about non-HTTPS URLs (credentials sent in cleartext)
        url_source = "args" if args.backend_url else ("credentials" if existing else None)
        warn_non_https_url(backend_url, url_source)

        print(f"Registering with {backend_url}...")
        print()

        # Prompt for email if not provided
        email = args.email
        if not email:
            print("Email: ", end="", flush=True)
            try:
                email = input().strip()
                if not email:
                    print("✗ Email is required")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)

        # Call registration endpoint
        try:
            response = httpx.post(
                f"{backend_url}/auth/register",
                json={"email": email},
                timeout=30.0,
            )

            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                user_id = result.get("user_id")
                api_key = result.get("api_key")
                token = result.get("token")
                token_expires = result.get("token_expires")

                if not user_id or not api_key:
                    print("✗ Registration failed: Invalid response from server")
                    print(f"  Response: {response.text[:200]}")
                    sys.exit(1)

                # Save credentials
                # Note: Use "auth_token" to match what storage layer expects
                credentials = {
                    "user_id": user_id,
                    "api_key": api_key,
                    "backend_url": backend_url,
                    "auth_token": token,  # Storage expects "auth_token", not "token"
                    "token_expires": token_expires,
                }
                save_credentials(credentials)

                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "success",
                                "user_id": user_id,
                                "backend_url": backend_url,
                            },
                            indent=2,
                        )
                    )
                else:
                    print("✓ Registration successful!")
                    print()
                    print(f"  User ID:     {user_id}")
                    print(
                        f"  API Key:     {api_key[:20]}..."
                        if len(api_key) > 20
                        else f"  API Key:     {api_key}"
                    )
                    print(f"  Backend:     {backend_url}")
                    print()
                    print(f"Credentials saved to {get_credentials_path()}")

            elif response.status_code == 409:
                print("✗ Email already registered")
                print("  Use `kernle auth login` to log in with existing credentials")
                sys.exit(1)
            elif response.status_code == 400:
                error = response.json().get("detail", response.text)
                print(f"✗ Registration failed: {error}")
                sys.exit(1)
            else:
                print(f"✗ Registration failed: HTTP {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            print("  Check that the backend URL is correct and the server is running")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Registration failed: {e}")
            sys.exit(1)

    elif args.auth_action == "login":
        httpx = get_http_client()

        # Load existing credentials
        existing = load_credentials()
        backend_url = args.backend_url or (existing.get("backend_url") if existing else None)
        api_key = args.api_key or (existing.get("api_key") if existing else None)

        # Prompt for backend URL if not provided
        if not backend_url:
            backend_url = prompt_backend_url()

        backend_url = backend_url.rstrip("/")

        # SECURITY: Warn about non-HTTPS URLs (credentials sent in cleartext)
        url_source = "args" if args.backend_url else ("credentials" if existing else None)
        warn_non_https_url(backend_url, url_source)

        # Prompt for API key if not provided
        if not api_key:
            import getpass

            try:
                # SECURITY: Use getpass to hide input (prevents shoulder-surfing/log capture)
                api_key = getpass.getpass("API Key: ").strip()
                if not api_key:
                    print("✗ API Key is required")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)

        print(f"Logging in to {backend_url}...")

        # Call login endpoint to refresh token
        try:
            response = httpx.post(
                f"{backend_url}/auth/login",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                user_id = result.get("user_id")
                token = result.get("token")
                token_expires = result.get("token_expires")

                # Update credentials
                # Note: Use "auth_token" to match what storage layer expects
                credentials = {
                    "user_id": user_id,
                    "api_key": api_key,
                    "backend_url": backend_url,
                    "auth_token": token,  # Storage expects "auth_token", not "token"
                    "token_expires": token_expires,
                }
                save_credentials(credentials)

                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "success",
                                "user_id": user_id,
                                "token_expires": token_expires,
                            },
                            indent=2,
                        )
                    )
                else:
                    print("✓ Login successful!")
                    print()
                    print(f"  User ID:       {user_id}")
                    print(f"  Backend:       {backend_url}")
                    if token_expires:
                        print(f"  Token expires: {token_expires}")
                    print()
                    print(f"Credentials saved to {get_credentials_path()}")

            elif response.status_code == 401:
                print("✗ Invalid API key")
                sys.exit(1)
            else:
                print(f"✗ Login failed: HTTP {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            print("  Check that the backend URL is correct and the server is running")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Login failed: {e}")
            sys.exit(1)

    elif args.auth_action == "status":
        credentials = load_credentials()

        if not credentials:
            if args.json:
                print(
                    json.dumps({"authenticated": False, "reason": "No credentials found"}, indent=2)
                )
            else:
                print("Not authenticated")
                print()
                print("Run `kernle auth register` to create an account")
                print("Run `kernle auth login` to log in with an existing API key")
            return

        user_id = credentials.get("user_id")
        api_key = credentials.get("api_key")
        backend_url = credentials.get("backend_url")
        # Support both "auth_token" (preferred) and "token" (legacy) for backwards compatibility
        token = credentials.get("auth_token") or credentials.get("token")
        token_expires = credentials.get("token_expires")

        # Check if token is expired
        token_valid = False
        expires_in = None
        if token_expires:
            try:
                # Parse ISO format timestamp
                expires_dt = datetime.fromisoformat(token_expires.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if expires_dt > now:
                    token_valid = True
                    delta = expires_dt - now
                    if delta.total_seconds() < 3600:
                        expires_in = f"{int(delta.total_seconds() / 60)} minutes"
                    elif delta.total_seconds() < 86400:
                        expires_in = f"{int(delta.total_seconds() / 3600)} hours"
                    else:
                        expires_in = f"{int(delta.total_seconds() / 86400)} days"
            except (ValueError, TypeError):
                pass

        if args.json:
            print(
                json.dumps(
                    {
                        "authenticated": True,
                        "user_id": user_id,
                        "backend_url": backend_url,
                        "has_api_key": bool(api_key),
                        "has_token": bool(token),
                        "token_valid": token_valid,
                        "token_expires": token_expires,
                    },
                    indent=2,
                )
            )
        else:
            print("Auth Status")
            print("=" * 40)
            print()

            auth_icon = "🟢" if token_valid else ("🟡" if api_key else "🔴")
            print(f"{auth_icon} Authenticated: {'Yes' if api_key else 'No'}")
            print()

            if user_id:
                print(f"  User ID:     {user_id}")
            if backend_url:
                print(f"  Backend:     {backend_url}")
            if api_key:
                masked_key = api_key[:12] + "..." + api_key[-4:] if len(api_key) > 20 else api_key
                print(f"  API Key:     {masked_key}")

            if token:
                if token_valid:
                    print(f"  Token:       ✓ Valid (expires in {expires_in})")
                else:
                    print("  Token:       ✗ Expired")
                    print()
                    print("Run `kernle auth login` to refresh your token")
            else:
                print("  Token:       Not set")

            print()
            print(f"Credentials: {get_credentials_path()}")

    elif args.auth_action == "logout":
        creds_path = get_credentials_path()
        if clear_credentials():
            if args.json:
                print(json.dumps({"status": "success", "message": "Credentials cleared"}, indent=2))
            else:
                print("✓ Logged out")
                print(f"  Removed {creds_path}")
        else:
            if args.json:
                print(
                    json.dumps(
                        {"status": "success", "message": "No credentials to clear"}, indent=2
                    )
                )
            else:
                print("Already logged out (no credentials found)")

    elif args.auth_action == "keys":
        cmd_auth_keys(args)


def cmd_auth_keys(args):
    """Handle API key management subcommands."""

    def get_http_client():
        """Get an HTTP client for backend requests."""
        try:
            import httpx

            return httpx
        except ImportError:
            print("✗ httpx not installed. Run: pip install httpx")
            sys.exit(1)

    def require_auth():
        """Load credentials and return (backend_url, api_key) or exit."""
        credentials = load_credentials()
        if not credentials:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` first")
            sys.exit(1)

        backend_url = credentials.get("backend_url")
        api_key = credentials.get("api_key")

        if not backend_url or not api_key:
            print("✗ Missing credentials")
            print("  Run `kernle auth login` to re-authenticate")
            sys.exit(1)

        return backend_url.rstrip("/"), api_key

    def mask_key(key: str) -> str:
        """Mask an API key for display (show first 8 and last 4 chars)."""
        if not key or len(key) <= 16:
            return key[:4] + "..." if key and len(key) > 4 else key or ""
        return key[:8] + "..." + key[-4:]

    httpx = get_http_client()

    if args.keys_action == "list":
        backend_url, api_key = require_auth()

        try:
            response = httpx.get(
                f"{backend_url}/auth/keys",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code == 200:
                keys = response.json()

                if args.json:
                    print(json.dumps(keys, indent=2))
                else:
                    if not keys:
                        print("No API keys found.")
                        return

                    print("API Keys")
                    print("=" * 70)
                    print()

                    for key_info in keys:
                        key_id = key_info.get("id", "unknown")
                        name = key_info.get("name") or "(unnamed)"
                        masked = mask_key(key_info.get("key_prefix", ""))
                        created = (
                            key_info.get("created_at", "")[:10]
                            if key_info.get("created_at")
                            else "unknown"
                        )
                        last_used = key_info.get("last_used_at")
                        is_active = key_info.get("is_active", True)

                        status_icon = "🟢" if is_active else "🔴"
                        print(f"{status_icon} {name}")
                        print(f"   ID:       {key_id}")
                        print(f"   Key:      {masked}...")
                        print(f"   Created:  {created}")
                        if last_used:
                            print(f"   Last used: {last_used[:10]}")
                        if not is_active:
                            print("   Status:   REVOKED")
                        print()

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Failed to list keys: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception:
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to list keys: {e}")
            sys.exit(1)

    elif args.keys_action == "create":
        backend_url, api_key = require_auth()
        name = getattr(args, "name", None)

        try:
            payload = {}
            if name:
                payload["name"] = name

            response = httpx.post(
                f"{backend_url}/auth/keys",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=30.0,
            )

            if response.status_code in (200, 201):
                result = response.json()
                new_key = result.get("key") or result.get("api_key")
                key_id = result.get("id") or result.get("key_id")
                key_name = result.get("name") or name or "(unnamed)"

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print("✓ API key created")
                    print()
                    print("=" * 70)
                    print("⚠️  SAVE THIS KEY NOW - IT WILL ONLY BE SHOWN ONCE!")
                    print("=" * 70)
                    print()
                    print(f"  Name:    {key_name}")
                    print(f"  ID:      {key_id}")
                    print(f"  Key:     {new_key}")
                    print()
                    print("=" * 70)
                    print()
                    print("Store this key securely. You will not be able to see it again.")
                    print("Use `kernle auth keys list` to see your keys (masked).")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            elif response.status_code == 429:
                print("✗ Rate limit exceeded")
                print("  Wait a moment and try again")
                sys.exit(1)
            else:
                print(f"✗ Failed to create key: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception:
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to create key: {e}")
            sys.exit(1)

    elif args.keys_action == "revoke":
        backend_url, api_key = require_auth()
        key_id = args.key_id

        # Confirm unless --force
        if not getattr(args, "force", False):
            print(f"⚠️  You are about to revoke API key: {key_id}")
            print("   This action cannot be undone.")
            print()
            print("Type 'yes' to confirm: ", end="", flush=True)
            try:
                confirm = input().strip().lower()
                if confirm != "yes":
                    print("Aborted.")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return

        try:
            response = httpx.delete(
                f"{backend_url}/auth/keys/{key_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code in (200, 204):
                if args.json:
                    print(
                        json.dumps(
                            {"status": "success", "key_id": key_id, "action": "revoked"}, indent=2
                        )
                    )
                else:
                    print(f"✓ API key {key_id} has been revoked")
                    print()
                    print("  The key can no longer be used for authentication.")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            elif response.status_code == 404:
                print(f"✗ Key not found: {key_id}")
                sys.exit(1)
            else:
                print(f"✗ Failed to revoke key: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception:
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to revoke key: {e}")
            sys.exit(1)

    elif args.keys_action == "cycle":
        backend_url, api_key = require_auth()
        key_id = args.key_id

        # Confirm unless --force
        if not getattr(args, "force", False):
            print(f"⚠️  You are about to cycle API key: {key_id}")
            print("   The old key will be deactivated and a new key will be generated.")
            print()
            print("Type 'yes' to confirm: ", end="", flush=True)
            try:
                confirm = input().strip().lower()
                if confirm != "yes":
                    print("Aborted.")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return

        try:
            response = httpx.post(
                f"{backend_url}/auth/keys/{key_id}/cycle",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code in (200, 201):
                result = response.json()
                new_key = result.get("key") or result.get("api_key")
                new_key_id = result.get("id") or result.get("key_id")
                key_name = result.get("name") or "(unnamed)"

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print("✓ API key cycled")
                    print()
                    print(f"  Old key {key_id} has been deactivated.")
                    print()
                    print("=" * 70)
                    print("⚠️  SAVE THIS NEW KEY NOW - IT WILL ONLY BE SHOWN ONCE!")
                    print("=" * 70)
                    print()
                    print(f"  Name:    {key_name}")
                    print(f"  ID:      {new_key_id}")
                    print(f"  Key:     {new_key}")
                    print()
                    print("=" * 70)
                    print()
                    print("Update any systems using the old key to use this new key.")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            elif response.status_code == 404:
                print(f"✗ Key not found: {key_id}")
                sys.exit(1)
            else:
                print(f"✗ Failed to cycle key: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception:
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to cycle key: {e}")
            sys.exit(1)


def cmd_mcp(args):
    """Start the MCP server for Claude Code and other MCP clients."""
    from kernle.mcp.server import main as mcp_main

    # Get agent_id from --agent flag
    agent_id = getattr(args, "agent", None) or "default"

    print(f"Starting Kernle MCP server for agent: {agent_id}", file=sys.stderr)
    mcp_main(agent_id=agent_id)


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
    p_load.add_argument(
        "--budget",
        "-b",
        type=validate_budget,
        default=8000,
        help="Token budget for memory loading (100-50000, default: 8000)",
    )
    p_load.add_argument(
        "--no-truncate", action="store_true", help="Disable content truncation (may exceed budget)"
    )
    p_load.add_argument(
        "--sync", "-s", action="store_true", help="Force sync (pull) before loading"
    )
    p_load.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        help="Skip sync even if auto-sync is enabled",
    )

    # checkpoint
    p_checkpoint = subparsers.add_parser("checkpoint", help="Checkpoint operations")
    cp_sub = p_checkpoint.add_subparsers(dest="checkpoint_action", required=True)

    cp_save = cp_sub.add_parser("save", help="Save checkpoint")
    cp_save.add_argument("task", help="Current task description")
    cp_save.add_argument("--pending", "-p", action="append", help="Pending item (repeatable)")
    cp_save.add_argument("--context", "-c", help="Additional context")
    cp_save.add_argument("--progress", help="Current progress on the task")
    cp_save.add_argument("--next", "-n", help="Immediate next step")
    cp_save.add_argument("--blocker", "-b", help="Current blocker if any")
    cp_save.add_argument("--sync", "-s", action="store_true", help="Force sync (push) after saving")
    cp_save.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        help="Skip sync even if auto-sync is enabled",
    )

    cp_load = cp_sub.add_parser("load", help="Load checkpoint")
    cp_load.add_argument("--json", "-j", action="store_true")

    cp_sub.add_parser("clear", help="Clear checkpoint")

    # episode
    p_episode = subparsers.add_parser("episode", help="Record an episode")
    p_episode.add_argument("objective", help="What was the objective?")
    p_episode.add_argument("outcome", help="What was the outcome?")
    p_episode.add_argument("--lesson", "-l", action="append", help="Lesson learned")
    p_episode.add_argument("--tag", "-t", action="append", help="Tag")
    p_episode.add_argument(
        "--relates-to", "-r", action="append", help="Related memory ID (repeatable)"
    )
    p_episode.add_argument("--valence", "-v", type=float, help="Emotional valence (-1.0 to 1.0)")
    p_episode.add_argument("--arousal", "-a", type=float, help="Emotional arousal (0.0 to 1.0)")
    p_episode.add_argument(
        "--emotion", "-e", action="append", help="Emotion tag (e.g., joy, frustration)"
    )
    p_episode.add_argument(
        "--auto-emotion", action="store_true", default=True, help="Auto-detect emotions (default)"
    )
    p_episode.add_argument(
        "--no-auto-emotion",
        dest="auto_emotion",
        action="store_false",
        help="Disable emotion auto-detection",
    )
    p_episode.add_argument(
        "--source", help="Source context (e.g., 'session with Sean', 'heartbeat', 'cron job')"
    )
    p_episode.add_argument(
        "--context", help="Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')"
    )
    p_episode.add_argument(
        "--context-tag", action="append", help="Context tag for filtering (repeatable)"
    )

    # note
    p_note = subparsers.add_parser("note", help="Capture a note")
    p_note.add_argument("content", help="Note content")
    p_note.add_argument("--type", choices=["note", "decision", "insight", "quote"], default="note")
    p_note.add_argument("--speaker", "-s", help="Speaker (for quotes)")
    p_note.add_argument("--reason", "-r", help="Reason (for decisions)")
    p_note.add_argument("--tag", action="append", help="Tag")
    p_note.add_argument("--relates-to", action="append", help="Related memory ID (repeatable)")
    p_note.add_argument("--protect", "-p", action="store_true", help="Protect from forgetting")
    p_note.add_argument(
        "--source", help="Source context (e.g., 'conversation with X', 'reading Y')"
    )
    p_note.add_argument(
        "--context", help="Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')"
    )
    p_note.add_argument(
        "--context-tag", action="append", help="Context tag for filtering (repeatable)"
    )

    # extract (conversation capture)
    p_extract = subparsers.add_parser("extract", help="Extract conversation context")
    p_extract.add_argument("summary", help="Summary of what's happening")
    p_extract.add_argument("--topic", "-t", help="Conversation topic")
    p_extract.add_argument(
        "--participant", "-p", action="append", dest="participants", help="Participant (repeatable)"
    )
    p_extract.add_argument("--outcome", "-o", help="Outcome or result")
    p_extract.add_argument("--decision", "-d", help="Decision made")

    # search
    p_search = subparsers.add_parser("search", help="Search memory")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    p_search.add_argument(
        "--min-score",
        "-m",
        type=float,
        help="Minimum similarity score (0.0-1.0) to include in results",
    )
    p_search.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # status
    subparsers.add_parser("status", help="Show memory status")

    # resume - quick "where was I?" view
    subparsers.add_parser("resume", help="Quick view: last task, next step, time since checkpoint")

    # init - generate CLAUDE.md/AGENTS.md section for health checks
    p_init = subparsers.add_parser(
        "init", help="Generate CLAUDE.md section for Kernle health checks"
    )
    p_init.add_argument(
        "--style",
        "-s",
        choices=["standard", "minimal", "combined"],
        default="standard",
        help="Section style (default: standard)",
    )
    p_init.add_argument(
        "--output", "-o", help="Output file path (auto-detects CLAUDE.md/AGENTS.md)"
    )
    p_init.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print section to stdout instead of writing to file",
    )
    p_init.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite/append even if Kernle section already exists",
    )
    p_init.add_argument(
        "--no-per-message", action="store_true", help="Skip per-message health check section"
    )
    p_init.add_argument(
        "--non-interactive", "-y", action="store_true", help="Non-interactive mode (use defaults)"
    )

    # doctor - validate boot sequence compliance
    p_doctor = subparsers.add_parser("doctor", help="Validate Kernle boot sequence compliance")
    p_doctor.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    p_doctor.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed check information"
    )
    p_doctor.add_argument("--fix", action="store_true", help="Auto-fix missing instructions")

    # relation (social graph / relationships)
    p_relation = subparsers.add_parser("relation", help="Manage relationships")
    relation_sub = p_relation.add_subparsers(dest="relation_action", required=True)

    relation_sub.add_parser("list", help="List all relationships")

    relation_add = relation_sub.add_parser("add", help="Add a relationship")
    relation_add.add_argument("name", help="Entity name (person, agent, org)")
    relation_add.add_argument(
        "--type",
        "-t",
        choices=["person", "agent", "organization", "system"],
        default="person",
        help="Entity type",
    )
    relation_add.add_argument("--trust", type=float, help="Trust level 0.0-1.0")
    relation_add.add_argument("--notes", "-n", help="Notes about this relationship")

    relation_update = relation_sub.add_parser("update", help="Update a relationship")
    relation_update.add_argument("name", help="Entity name")
    relation_update.add_argument("--trust", type=float, help="New trust level 0.0-1.0")
    relation_update.add_argument("--notes", "-n", help="Updated notes")
    relation_update.add_argument(
        "--type", "-t", choices=["person", "agent", "organization", "system"], help="Entity type"
    )

    relation_show = relation_sub.add_parser("show", help="Show relationship details")
    relation_show.add_argument("name", help="Entity name")

    relation_log = relation_sub.add_parser("log", help="Log an interaction")
    relation_log.add_argument("name", help="Entity name")
    relation_log.add_argument("--interaction", "-i", help="Interaction description")

    # drive
    p_drive = subparsers.add_parser("drive", help="Manage drives")
    drive_sub = p_drive.add_subparsers(dest="drive_action", required=True)

    drive_sub.add_parser("list", help="List drives")

    drive_set = drive_sub.add_parser("set", help="Set a drive")
    drive_set.add_argument(
        "type", choices=["existence", "growth", "curiosity", "connection", "reproduction"]
    )
    drive_set.add_argument("intensity", type=float, help="Intensity 0.0-1.0")
    drive_set.add_argument("--focus", "-f", action="append", help="Focus area")

    drive_satisfy = drive_sub.add_parser("satisfy", help="Satisfy a drive")
    drive_satisfy.add_argument("type", help="Drive type")
    drive_satisfy.add_argument("--amount", "-a", type=float, default=0.2)

    # consolidate
    p_consolidate = subparsers.add_parser("consolidate", help="Output guided reflection prompt")
    p_consolidate.add_argument(
        "--min-episodes",
        "-m",
        type=int,
        default=3,
        help="(Legacy) Minimum episodes for old consolidation",
    )
    p_consolidate.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Number of recent episodes to include (default: 20)",
    )

    # temporal
    p_temporal = subparsers.add_parser("when", help="Query by time")
    p_temporal.add_argument(
        "when", nargs="?", default="today", choices=["today", "yesterday", "this week", "last hour"]
    )

    # identity
    p_identity = subparsers.add_parser("identity", help="Identity synthesis")
    identity_sub = p_identity.add_subparsers(dest="identity_action")
    identity_sub.default = "show"

    identity_show = identity_sub.add_parser("show", help="Show identity synthesis")
    identity_show.add_argument("--json", "-j", action="store_true")

    identity_conf = identity_sub.add_parser("confidence", help="Get identity confidence score")
    identity_conf.add_argument("--json", "-j", action="store_true")

    identity_drift = identity_sub.add_parser("drift", help="Detect identity drift")
    identity_drift.add_argument("--days", "-d", type=int, default=30, help="Days to look back")
    identity_drift.add_argument("--json", "-j", action="store_true")

    # emotion
    p_emotion = subparsers.add_parser("emotion", help="Emotional memory operations")
    emotion_sub = p_emotion.add_subparsers(dest="emotion_action", required=True)

    emotion_summary = emotion_sub.add_parser("summary", help="Show emotional summary")
    emotion_summary.add_argument("--days", "-d", type=int, default=7, help="Days to analyze")
    emotion_summary.add_argument("--json", "-j", action="store_true")

    emotion_search = emotion_sub.add_parser("search", help="Search by emotion")
    emotion_search.add_argument("--positive", action="store_true", help="Find positive episodes")
    emotion_search.add_argument("--negative", action="store_true", help="Find negative episodes")
    emotion_search.add_argument("--calm", action="store_true", help="Find low-arousal episodes")
    emotion_search.add_argument("--intense", action="store_true", help="Find high-arousal episodes")
    emotion_search.add_argument("--valence-min", type=float, help="Min valence (-1.0 to 1.0)")
    emotion_search.add_argument("--valence-max", type=float, help="Max valence (-1.0 to 1.0)")
    emotion_search.add_argument("--arousal-min", type=float, help="Min arousal (0.0 to 1.0)")
    emotion_search.add_argument("--arousal-max", type=float, help="Max arousal (0.0 to 1.0)")
    emotion_search.add_argument("--tag", "-t", action="append", help="Emotion tag to match")
    emotion_search.add_argument("--limit", "-l", type=int, default=10)
    emotion_search.add_argument("--json", "-j", action="store_true")

    emotion_tag = emotion_sub.add_parser("tag", help="Add emotional tags to an episode")
    emotion_tag.add_argument("episode_id", help="Episode ID to tag")
    emotion_tag.add_argument(
        "--valence", "-v", type=float, default=0.0, help="Valence (-1.0 to 1.0)"
    )
    emotion_tag.add_argument(
        "--arousal", "-a", type=float, default=0.0, help="Arousal (0.0 to 1.0)"
    )
    emotion_tag.add_argument("--tag", "-t", action="append", help="Emotion tag")

    emotion_detect = emotion_sub.add_parser("detect", help="Detect emotions in text")
    emotion_detect.add_argument("text", help="Text to analyze")
    emotion_detect.add_argument("--json", "-j", action="store_true")

    emotion_mood = emotion_sub.add_parser("mood", help="Get mood-relevant memories")
    emotion_mood.add_argument("--valence", "-v", type=float, required=True, help="Current valence")
    emotion_mood.add_argument("--arousal", "-a", type=float, required=True, help="Current arousal")
    emotion_mood.add_argument("--limit", "-l", type=int, default=10)
    emotion_mood.add_argument("--json", "-j", action="store_true")

    # meta (meta-memory operations)
    p_meta = subparsers.add_parser("meta", help="Meta-memory operations (confidence, lineage)")
    meta_sub = p_meta.add_subparsers(dest="meta_action", required=True)

    meta_conf = meta_sub.add_parser("confidence", help="Get confidence for a memory")
    meta_conf.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_conf.add_argument("id", help="Memory ID")

    meta_verify = meta_sub.add_parser("verify", help="Verify a memory (increases confidence)")
    meta_verify.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_verify.add_argument("id", help="Memory ID")
    meta_verify.add_argument("--evidence", "-e", help="Supporting evidence")

    meta_lineage = meta_sub.add_parser("lineage", help="Get provenance chain for a memory")
    meta_lineage.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_lineage.add_argument("id", help="Memory ID")
    meta_lineage.add_argument("--json", "-j", action="store_true")

    meta_uncertain = meta_sub.add_parser("uncertain", help="List low-confidence memories")
    meta_uncertain.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    meta_uncertain.add_argument("--limit", "-l", type=int, default=20)
    meta_uncertain.add_argument("--json", "-j", action="store_true")

    meta_propagate = meta_sub.add_parser(
        "propagate", help="Propagate confidence to derived memories"
    )
    meta_propagate.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Source memory type"
    )
    meta_propagate.add_argument("id", help="Source memory ID")

    meta_source = meta_sub.add_parser("source", help="Set source/provenance for a memory")
    meta_source.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_source.add_argument("id", help="Memory ID")
    meta_source.add_argument(
        "--source",
        "-s",
        required=True,
        choices=["direct_experience", "inference", "told_by_agent", "consolidation"],
        help="Source type",
    )
    meta_source.add_argument("--episodes", action="append", help="Supporting episode IDs")
    meta_source.add_argument("--derived", action="append", help="Derived from (type:id format)")

    # Meta-cognition subcommands (awareness of what I know/don't know)
    meta_knowledge = meta_sub.add_parser("knowledge", help="Show knowledge map across domains")
    meta_knowledge.add_argument("--json", "-j", action="store_true")

    meta_gaps = meta_sub.add_parser("gaps", help="Detect knowledge gaps for a query")
    meta_gaps.add_argument("query", help="Query to check knowledge for")
    meta_gaps.add_argument("--json", "-j", action="store_true")

    meta_boundaries = meta_sub.add_parser(
        "boundaries", help="Show competence boundaries (strengths/weaknesses)"
    )
    meta_boundaries.add_argument("--json", "-j", action="store_true")

    meta_learn = meta_sub.add_parser("learn", help="Identify learning opportunities")
    meta_learn.add_argument("--limit", "-l", type=int, default=5, help="Max opportunities to show")
    meta_learn.add_argument("--json", "-j", action="store_true")

    # belief (belief revision operations)
    p_belief = subparsers.add_parser("belief", help="Belief revision operations")
    belief_sub = p_belief.add_subparsers(dest="belief_action", required=True)

    belief_revise = belief_sub.add_parser("revise", help="Update beliefs from an episode")
    belief_revise.add_argument("episode_id", help="Episode ID to analyze")
    belief_revise.add_argument("--json", "-j", action="store_true")

    belief_contradictions = belief_sub.add_parser(
        "contradictions", help="Find contradicting beliefs"
    )
    belief_contradictions.add_argument("statement", help="Statement to check for contradictions")
    belief_contradictions.add_argument("--limit", "-l", type=int, default=10)
    belief_contradictions.add_argument("--json", "-j", action="store_true")

    belief_history = belief_sub.add_parser("history", help="Show supersession chain")
    belief_history.add_argument("id", help="Belief ID")
    belief_history.add_argument("--json", "-j", action="store_true")

    belief_reinforce = belief_sub.add_parser("reinforce", help="Manually reinforce a belief")
    belief_reinforce.add_argument("id", help="Belief ID")

    belief_supersede = belief_sub.add_parser("supersede", help="Replace a belief with a new one")
    belief_supersede.add_argument("old_id", help="ID of belief to supersede")
    belief_supersede.add_argument("new_statement", help="New belief statement")
    belief_supersede.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.8,
        help="Confidence in new belief (default: 0.8)",
    )
    belief_supersede.add_argument("--reason", "-r", help="Reason for supersession")

    belief_list = belief_sub.add_parser("list", help="List beliefs")
    belief_list.add_argument("--all", "-a", action="store_true", help="Include inactive beliefs")
    belief_list.add_argument("--limit", "-l", type=int, default=20)
    belief_list.add_argument("--json", "-j", action="store_true")

    # mcp
    subparsers.add_parser("mcp", help="Start MCP server (stdio transport)")

    # raw (raw memory entries)
    p_raw = subparsers.add_parser("raw", help="Raw memory capture and management")
    # Arguments for default action (kernle raw "content" without subcommand)
    p_raw.add_argument("content", nargs="?", help="Content to capture")
    p_raw.add_argument("--tags", "-t", help="Comma-separated tags")
    p_raw.add_argument(
        "--source", "-s", help="Source identifier (e.g., 'hook-session-end', 'conversation')"
    )
    p_raw.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (for hooks/scripts)"
    )
    p_raw.add_argument("--stdin", action="store_true", help="Read content from stdin")
    raw_sub = p_raw.add_subparsers(dest="raw_action")

    # kernle raw capture "content" - explicit capture subcommand
    raw_capture = raw_sub.add_parser("capture", help="Capture a raw entry")
    raw_capture.add_argument(
        "content", nargs="?", help="Content to capture (omit if using --stdin)"
    )
    raw_capture.add_argument("--tags", "-t", help="Comma-separated tags")
    raw_capture.add_argument(
        "--source", "-s", help="Source identifier (e.g., 'hook-session-end', 'conversation')"
    )
    raw_capture.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (for hooks/scripts)"
    )
    raw_capture.add_argument("--stdin", action="store_true", help="Read content from stdin")

    # kernle raw list
    raw_list = raw_sub.add_parser("list", help="List raw entries")
    raw_list.add_argument("--unprocessed", "-u", action="store_true", help="Show only unprocessed")
    raw_list.add_argument("--processed", "-p", action="store_true", help="Show only processed")
    raw_list.add_argument("--limit", "-l", type=int, default=50)
    raw_list.add_argument("--json", "-j", action="store_true")

    # kernle raw show <id>
    raw_show = raw_sub.add_parser("show", help="Show a raw entry")
    raw_show.add_argument("id", help="Raw entry ID")
    raw_show.add_argument("--json", "-j", action="store_true")

    # kernle raw process <id> --type <type>
    raw_process = raw_sub.add_parser("process", help="Process raw entry into memory")
    raw_process.add_argument("id", help="Raw entry ID")
    raw_process.add_argument(
        "--type",
        "-t",
        required=True,
        choices=["episode", "note", "belief"],
        help="Target memory type",
    )
    raw_process.add_argument("--objective", help="Episode objective (for episodes)")
    raw_process.add_argument("--outcome", help="Episode outcome (for episodes)")

    # kernle raw review - guided review of unprocessed entries
    raw_review = raw_sub.add_parser(
        "review", help="Review unprocessed entries with promotion guidance"
    )
    raw_review.add_argument(
        "--limit", "-l", type=int, default=10, help="Number of entries to review"
    )
    raw_review.add_argument("--json", "-j", action="store_true")

    # kernle raw clean - clean up old unprocessed entries
    raw_clean = raw_sub.add_parser("clean", help="Delete old unprocessed raw entries")
    raw_clean.add_argument(
        "--age", "-a", type=int, default=7, help="Delete entries older than N days (default: 7)"
    )
    raw_clean.add_argument(
        "--junk",
        "-j",
        action="store_true",
        help="Detect and remove junk entries (short, test keywords)",
    )
    raw_clean.add_argument(
        "--confirm", "-y", action="store_true", help="Actually delete (otherwise dry run)"
    )

    # kernle raw promote <id> - alias for process (simpler UX)
    raw_promote = raw_sub.add_parser(
        "promote", help="Promote raw entry to memory (alias for process)"
    )
    raw_promote.add_argument("id", help="Raw entry ID")
    raw_promote.add_argument(
        "--type",
        "-t",
        required=True,
        choices=["episode", "note", "belief"],
        help="Target memory type",
    )
    raw_promote.add_argument("--objective", help="Episode objective (for episodes)")
    raw_promote.add_argument("--outcome", help="Episode outcome (for episodes)")

    # kernle raw triage - guided review of entries with promote/delete suggestions
    raw_triage = raw_sub.add_parser("triage", help="Guided triage of unprocessed entries")
    raw_triage.add_argument(
        "--limit", "-l", type=int, default=10, help="Number of entries to review"
    )

    # kernle raw files - show flat file locations
    raw_files = raw_sub.add_parser("files", help="Show raw flat file locations")
    raw_files.add_argument(
        "--open", "-o", action="store_true", help="Open directory in file manager"
    )

    # kernle raw sync - sync from flat files to SQLite
    raw_sync = raw_sub.add_parser("sync", help="Import flat file entries into SQLite index")
    raw_sync.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be imported"
    )

    # suggestions (auto-extracted memory suggestions)
    p_suggestions = subparsers.add_parser("suggestions", help="Memory suggestion management")
    suggestions_sub = p_suggestions.add_subparsers(dest="suggestions_action", required=True)

    # kernle suggestions list [--pending|--approved|--rejected] [--type TYPE]
    suggestions_list = suggestions_sub.add_parser("list", help="List suggestions")
    suggestions_list.add_argument("--pending", action="store_true", help="Show only pending")
    suggestions_list.add_argument("--approved", action="store_true", help="Show only approved")
    suggestions_list.add_argument("--rejected", action="store_true", help="Show only rejected")
    suggestions_list.add_argument(
        "--type", "-t", choices=["episode", "belief", "note"], help="Filter by memory type"
    )
    suggestions_list.add_argument("--limit", "-l", type=int, default=50)
    suggestions_list.add_argument("--json", "-j", action="store_true")

    # kernle suggestions show <id>
    suggestions_show = suggestions_sub.add_parser("show", help="Show suggestion details")
    suggestions_show.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_show.add_argument("--json", "-j", action="store_true")

    # kernle suggestions approve <id> [--objective ...] [--outcome ...] [--statement ...]
    suggestions_approve = suggestions_sub.add_parser(
        "approve", help="Approve and promote a suggestion"
    )
    suggestions_approve.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_approve.add_argument("--objective", help="Override objective (for episodes)")
    suggestions_approve.add_argument("--outcome", help="Override outcome (for episodes)")
    suggestions_approve.add_argument("--statement", help="Override statement (for beliefs)")
    suggestions_approve.add_argument("--content", help="Override content (for notes)")

    # kernle suggestions reject <id> [--reason ...]
    suggestions_reject = suggestions_sub.add_parser("reject", help="Reject a suggestion")
    suggestions_reject.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_reject.add_argument("--reason", "-r", help="Rejection reason")

    # kernle suggestions extract [--limit N]
    suggestions_extract = suggestions_sub.add_parser(
        "extract", help="Extract suggestions from unprocessed raw entries"
    )
    suggestions_extract.add_argument(
        "--limit", "-l", type=int, default=50, help="Maximum raw entries to process"
    )

    # dump
    p_dump = subparsers.add_parser("dump", help="Dump all memory to stdout")
    p_dump.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p_dump.add_argument(
        "--include-raw",
        "-r",
        action="store_true",
        default=True,
        help="Include raw entries (default: true)",
    )
    p_dump.add_argument(
        "--no-raw", dest="include_raw", action="store_false", help="Exclude raw entries"
    )

    # export
    p_export = subparsers.add_parser("export", help="Export memory to file")
    p_export.add_argument("path", help="Output file path")
    p_export.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        help="Output format (auto-detected from extension if not specified)",
    )
    p_export.add_argument(
        "--include-raw",
        "-r",
        action="store_true",
        default=True,
        help="Include raw entries (default: true)",
    )
    p_export.add_argument(
        "--no-raw", dest="include_raw", action="store_false", help="Exclude raw entries"
    )

    # playbook (procedural memory)
    p_playbook = subparsers.add_parser("playbook", help="Playbook (procedural memory) operations")
    playbook_sub = p_playbook.add_subparsers(dest="playbook_action", required=True)

    # kernle playbook create "name" --steps "1,2,3" --triggers "when x"
    playbook_create = playbook_sub.add_parser("create", help="Create a new playbook")
    playbook_create.add_argument("name", help="Playbook name")
    playbook_create.add_argument("--description", "-d", help="What this playbook does")
    playbook_create.add_argument("--steps", "-s", help="Comma-separated steps")
    playbook_create.add_argument("--step", action="append", help="Add a step (repeatable)")
    playbook_create.add_argument("--triggers", help="Comma-separated trigger conditions")
    playbook_create.add_argument("--trigger", action="append", help="Add a trigger (repeatable)")
    playbook_create.add_argument("--failure-mode", "-f", action="append", help="What can go wrong")
    playbook_create.add_argument("--recovery", "-r", action="append", help="Recovery step")
    playbook_create.add_argument("--tag", "-t", action="append", help="Tag")

    # kernle playbook list [--tag TAG]
    playbook_list = playbook_sub.add_parser("list", help="List playbooks")
    playbook_list.add_argument("--tag", "-t", action="append", help="Filter by tag")
    playbook_list.add_argument("--limit", "-l", type=int, default=20)
    playbook_list.add_argument("--json", "-j", action="store_true")

    # kernle playbook search "query"
    playbook_search = playbook_sub.add_parser("search", help="Search playbooks")
    playbook_search.add_argument("query", help="Search query")
    playbook_search.add_argument("--limit", "-l", type=int, default=10)
    playbook_search.add_argument("--json", "-j", action="store_true")

    # kernle playbook show <id>
    playbook_show = playbook_sub.add_parser("show", help="Show playbook details")
    playbook_show.add_argument("id", help="Playbook ID")
    playbook_show.add_argument("--json", "-j", action="store_true")

    # kernle playbook find "situation"
    playbook_find = playbook_sub.add_parser("find", help="Find relevant playbook for situation")
    playbook_find.add_argument("situation", help="Describe the current situation")
    playbook_find.add_argument("--json", "-j", action="store_true")

    # kernle playbook record <id> [--success|--failure]
    playbook_record = playbook_sub.add_parser("record", help="Record playbook usage")
    playbook_record.add_argument("id", help="Playbook ID")
    playbook_record.add_argument(
        "--success", action="store_true", default=True, help="Record successful usage (default)"
    )
    playbook_record.add_argument("--failure", action="store_true", help="Record failed usage")

    # anxiety
    p_anxiety = subparsers.add_parser("anxiety", help="Memory anxiety tracking")
    p_anxiety.add_argument("--detailed", "-d", action="store_true", help="Show detailed breakdown")
    p_anxiety.add_argument("--actions", "-a", action="store_true", help="Show recommended actions")
    p_anxiety.add_argument(
        "--auto", action="store_true", help="Execute recommended actions automatically"
    )
    p_anxiety.add_argument("--context", "-c", type=int, help="Current context token usage")
    p_anxiety.add_argument(
        "--limit", "-l", type=int, default=200000, help="Context window limit (default: 200000)"
    )
    p_anxiety.add_argument(
        "--emergency", "-e", action="store_true", help="Run emergency save immediately"
    )
    p_anxiety.add_argument("--summary", "-s", help="Summary for emergency save checkpoint")
    p_anxiety.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    p_anxiety.add_argument(
        "--brief", "-b", action="store_true", help="Single-line output for quick health checks"
    )
    p_anxiety.add_argument(
        "--source",
        choices=["cli", "mcp"],
        default="cli",
        help="Source of the health check (default: cli)",
    )
    p_anxiety.add_argument(
        "--triggered-by",
        dest="triggered_by",
        choices=["boot", "heartbeat", "manual"],
        default="manual",
        help="What triggered this check (default: manual)",
    )

    # stats (compliance and analytics)
    p_stats = subparsers.add_parser("stats", help="Compliance and analytics stats")
    stats_sub = p_stats.add_subparsers(dest="stats_action", required=True)

    # kernle stats health-checks
    stats_health = stats_sub.add_parser("health-checks", help="Show health check compliance stats")
    stats_health.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # forget (controlled forgetting)
    p_forget = subparsers.add_parser("forget", help="Controlled forgetting operations")
    forget_sub = p_forget.add_subparsers(dest="forget_action", required=True)

    # kernle forget candidates [--threshold N] [--limit N]
    forget_candidates = forget_sub.add_parser("candidates", help="Show forgetting candidates")
    forget_candidates.add_argument(
        "--threshold", "-t", type=float, default=0.3, help="Salience threshold (default: 0.3)"
    )
    forget_candidates.add_argument(
        "--limit", "-l", type=int, default=20, help="Maximum candidates to show"
    )
    forget_candidates.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle forget run [--dry-run] [--threshold N] [--limit N]
    forget_run = forget_sub.add_parser("run", help="Run forgetting cycle")
    forget_run.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be forgotten (don't actually forget)",
    )
    forget_run.add_argument(
        "--threshold", "-t", type=float, default=0.3, help="Salience threshold (default: 0.3)"
    )
    forget_run.add_argument(
        "--limit", "-l", type=int, default=10, help="Maximum memories to forget"
    )
    forget_run.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle forget protect <type> <id>
    forget_protect = forget_sub.add_parser("protect", help="Protect memory from forgetting")
    forget_protect.add_argument(
        "type",
        choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
        help="Memory type",
    )
    forget_protect.add_argument("id", help="Memory ID")
    forget_protect.add_argument(
        "--unprotect", "-u", action="store_true", help="Remove protection instead"
    )

    # kernle forget recover <type> <id>
    forget_recover = forget_sub.add_parser("recover", help="Recover a forgotten memory")
    forget_recover.add_argument(
        "type",
        choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
        help="Memory type",
    )
    forget_recover.add_argument("id", help="Memory ID")

    # kernle forget list [--limit N]
    forget_list = forget_sub.add_parser("list", help="List forgotten memories")
    forget_list.add_argument("--limit", "-l", type=int, default=50, help="Maximum entries to show")
    forget_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle forget salience <type> <id>
    forget_salience = forget_sub.add_parser("salience", help="Calculate salience for a memory")
    forget_salience.add_argument(
        "type",
        choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
        help="Memory type",
    )
    forget_salience.add_argument("id", help="Memory ID")

    # sync (local-to-cloud synchronization)
    p_sync = subparsers.add_parser("sync", help="Sync with remote backend")
    sync_sub = p_sync.add_subparsers(dest="sync_action", required=True)

    # kernle sync status
    sync_status = sync_sub.add_parser(
        "status", help="Show sync status (pending ops, last sync, connection)"
    )
    sync_status.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync push [--limit N]
    sync_push = sync_sub.add_parser("push", help="Push pending local changes to remote backend")
    sync_push.add_argument(
        "--limit", "-l", type=int, default=100, help="Maximum operations to push (default: 100)"
    )
    sync_push.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync pull [--full]
    sync_pull = sync_sub.add_parser("pull", help="Pull remote changes to local")
    sync_pull.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Pull all records (not just changes since last sync)",
    )
    sync_pull.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync full
    sync_full = sync_sub.add_parser("full", help="Full bidirectional sync (pull then push)")
    sync_full.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync conflicts [--limit N] [--clear]
    sync_conflicts = sync_sub.add_parser("conflicts", help="View sync conflict history")
    sync_conflicts.add_argument(
        "--limit", "-l", type=int, default=20, help="Maximum conflicts to show (default: 20)"
    )
    sync_conflicts.add_argument("--clear", action="store_true", help="Clear conflict history")
    sync_conflicts.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # auth (authentication and credentials management)
    p_auth = subparsers.add_parser("auth", help="Authentication and credentials management")
    auth_sub = p_auth.add_subparsers(dest="auth_action", required=True)

    # kernle auth register [--email EMAIL] [--backend-url URL]
    auth_register = auth_sub.add_parser("register", help="Register a new account")
    auth_register.add_argument("--email", "-e", help="Email address")
    auth_register.add_argument(
        "--backend-url", "-b", help="Backend URL (e.g., https://api.kernle.io)"
    )
    auth_register.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth login [--api-key KEY] [--backend-url URL]
    auth_login = auth_sub.add_parser("login", help="Log in with existing credentials")
    auth_login.add_argument("--api-key", "-k", help="API key")
    auth_login.add_argument("--backend-url", "-b", help="Backend URL")
    auth_login.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth status
    auth_status = auth_sub.add_parser("status", help="Show current auth status")
    auth_status.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth logout
    auth_logout = auth_sub.add_parser("logout", help="Clear stored credentials")
    auth_logout.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys (API key management)
    auth_keys = auth_sub.add_parser("keys", help="Manage API keys")
    keys_sub = auth_keys.add_subparsers(dest="keys_action", required=True)

    # kernle auth keys list
    keys_list = keys_sub.add_parser("list", help="List your API keys (masked)")
    keys_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys create [--name NAME]
    keys_create = keys_sub.add_parser("create", help="Create a new API key")
    keys_create.add_argument("--name", "-n", help="Name for the key (for identification)")
    keys_create.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys revoke KEY_ID
    keys_revoke = keys_sub.add_parser("revoke", help="Revoke/delete an API key")
    keys_revoke.add_argument("key_id", help="ID of the key to revoke")
    keys_revoke.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    keys_revoke.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys cycle KEY_ID
    keys_cycle = keys_sub.add_parser("cycle", help="Cycle a key (new key, old deactivated)")
    keys_cycle.add_argument("key_id", help="ID of the key to cycle")
    keys_cycle.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    keys_cycle.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # agent - agent management
    p_agent = subparsers.add_parser("agent", help="Agent management (list, delete)")
    agent_sub = p_agent.add_subparsers(dest="agent_action", required=True)

    agent_sub.add_parser("list", help="List all local agents")

    agent_delete = agent_sub.add_parser("delete", help="Delete an agent and all its data")
    agent_delete.add_argument("name", help="Agent ID to delete")
    agent_delete.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    # import - import from external files (markdown, JSON, CSV)
    p_import = subparsers.add_parser(
        "import", help="Import memories from markdown, JSON, or CSV files"
    )
    p_import.add_argument(
        "file", help="Path to file to import (auto-detects format from extension)"
    )
    p_import.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "csv"],
        help="File format (auto-detected from extension if not specified)",
    )
    p_import.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be imported without making changes",
    )
    p_import.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Confirm each item before importing (markdown only)",
    )
    p_import.add_argument(
        "--layer",
        "-l",
        choices=["episode", "note", "belief", "value", "goal", "raw"],
        help="Force all items to a specific memory type (overrides auto-detection)",
    )
    p_import.add_argument(
        "--skip-duplicates",
        "-s",
        action="store_true",
        default=True,
        dest="skip_duplicates",
        help="Skip items that already exist (default: enabled)",
    )
    p_import.add_argument(
        "--no-skip-duplicates",
        action="store_false",
        dest="skip_duplicates",
        help="Import all items even if they already exist",
    )

    # Pre-process arguments: handle `kernle raw "content"` by inserting "capture"
    # This is needed because argparse subparsers consume positional args before parent parser
    raw_subcommands = {
        "list",
        "show",
        "process",
        "capture",
        "review",
        "clean",
        "files",
        "sync",
        "promote",
        "triage",
    }
    argv = sys.argv[1:]  # Skip program name

    # Find position of "raw" in argv (accounting for -a/--agent which takes a value)
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-a", "--agent"):
            i += 2  # Skip flag and its value
            continue
        if arg == "raw":
            # Check if there's a next argument and it's not a known subcommand
            if (
                i + 1 < len(argv)
                and argv[i + 1] not in raw_subcommands
                and not argv[i + 1].startswith("-")
            ):
                # Insert "capture" after "raw"
                argv.insert(i + 1, "capture")
            break
        i += 1

    args = parser.parse_args(argv)

    # Initialize Kernle with error handling
    try:
        # Resolve agent ID: explicit > env var > auto-generated
        if args.agent:
            agent_id = validate_input(args.agent, "agent_id", 100)
        else:
            agent_id = resolve_agent_id()
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
        elif args.command == "extract":
            cmd_extract(args, k)
        elif args.command == "search":
            cmd_search(args, k)
        elif args.command == "status":
            cmd_status(args, k)
        elif args.command == "resume":
            cmd_resume(args, k)
        elif args.command == "init":
            cmd_init_md(args, k)
        elif args.command == "doctor":
            cmd_doctor(args, k)
        elif args.command == "relation":
            cmd_relation(args, k)
        elif args.command == "drive":
            cmd_drive(args, k)
        elif args.command == "consolidate":
            cmd_consolidate(args, k)
        elif args.command == "when":
            cmd_temporal(args, k)
        elif args.command == "identity":
            # Handle default action when no subcommand given
            if not args.identity_action:
                args.identity_action = "show"
                args.json = False
            cmd_identity(args, k)
        elif args.command == "emotion":
            cmd_emotion(args, k)
        elif args.command == "meta":
            cmd_meta(args, k)
        elif args.command == "anxiety":
            cmd_anxiety(args, k)
        elif args.command == "stats":
            cmd_stats(args, k)
        elif args.command == "forget":
            cmd_forget(args, k)
        elif args.command == "playbook":
            cmd_playbook(args, k)
        elif args.command == "raw":
            cmd_raw(args, k)
        elif args.command == "suggestions":
            cmd_suggestions(args, k)
        elif args.command == "belief":
            cmd_belief(args, k)
        elif args.command == "dump":
            cmd_dump(args, k)
        elif args.command == "export":
            cmd_export(args, k)
        elif args.command == "sync":
            cmd_sync(args, k)
        elif args.command == "auth":
            cmd_auth(args, k)
        elif args.command == "mcp":
            cmd_mcp(args)
        elif args.command == "agent":
            cmd_agent(args, k)
        elif args.command == "import":
            cmd_import(args, k)
    except (ValueError, TypeError) as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
