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
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)

    return sanitized


def cmd_load(args, k: Kernle):
    """Load and display working memory."""
    # Determine sync setting from args
    sync = None
    if getattr(args, 'no_sync', False):
        sync = False
    elif getattr(args, 'sync', False):
        sync = True

    memory = k.load(sync=sync)

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
        if getattr(args, 'progress', None):
            context_parts.append(f"Progress: {validate_input(args.progress, 'progress', 300)}")
        if getattr(args, 'next', None):
            context_parts.append(f"Next: {validate_input(args.next, 'next', 300)}")
        if getattr(args, 'blocker', None):
            context_parts.append(f"Blocker: {validate_input(args.blocker, 'blocker', 300)}")
        
        context = " | ".join(context_parts) if context_parts else None

        # Warn about generic task names that won't help with recovery
        generic_patterns = [
            "auto-save", "auto save", "pre-compaction", "compaction",
            "checkpoint", "save", "saving", "state"
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
        if getattr(args, 'no_sync', False):
            sync = False
        elif getattr(args, 'sync', False):
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
                    ts = cp.get('timestamp', '')
                    if ts:
                        cp_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
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
    relates_to = getattr(args, 'relates_to', None)

    # Get emotional arguments with defaults for backwards compatibility
    emotion = getattr(args, 'emotion', None)
    valence = getattr(args, 'valence', None)
    arousal = getattr(args, 'arousal', None)
    auto_emotion = getattr(args, 'auto_emotion', True)

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
        )
    else:
        episode_id = k.episode(
            objective=objective,
            outcome=outcome,
            lessons=lessons,
            tags=tags,
            relates_to=relates_to,
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
    relates_to = getattr(args, 'relates_to', None)

    k.note(
        content=content,
        type=args.type,
        speaker=speaker,
        reason=reason,
        tags=tags,
        protect=args.protect,
        relates_to=relates_to,
    )
    print(f"✓ Note saved: {args.content[:50]}...")
    if args.tag:
        print(f"  Tags: {', '.join(args.tag)}")
    if relates_to:
        print(f"  Links: {len(relates_to)} related memories")


def cmd_extract(args, k: Kernle):
    """Extract and capture conversation context as a raw entry.
    
    A low-friction way to capture what's happening in a conversation
    without having to decide immediately if it's an episode, note, or belief.
    """
    summary = validate_input(args.summary, "summary", 2000)
    
    # Build a structured capture
    capture_parts = [f"Conversation extract: {summary}"]
    
    if getattr(args, 'topic', None):
        capture_parts.append(f"Topic: {args.topic}")
    if getattr(args, 'participants', None):
        capture_parts.append(f"Participants: {', '.join(args.participants)}")
    if getattr(args, 'outcome', None):
        capture_parts.append(f"Outcome: {args.outcome}")
    if getattr(args, 'decision', None):
        capture_parts.append(f"Decision: {args.decision}")
    
    content = " | ".join(capture_parts)
    tags = ["conversation", "extract"]
    if getattr(args, 'topic', None):
        tags.append(args.topic.lower().replace(" ", "-")[:20])
    
    raw_id = k.raw(content, tags=tags, source="conversation")
    print(f"✓ Extracted: {summary[:50]}...")
    print(f"  ID: {raw_id[:8]} (promote later with: kernle raw process {raw_id[:8]} --type <episode|note>)")


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


def cmd_init(args, k: Kernle):
    """Initialize Kernle for a new environment."""
    from pathlib import Path

    print("=" * 50)
    print("  Kernle Setup Wizard")
    print("=" * 50)
    print()

    agent_id = k.agent_id
    print(f"Agent ID: {agent_id}")
    print()

    # Detect environment
    env = args.env
    if not env and not args.non_interactive:
        print("Detecting environment...")

        # Check for environment indicators
        has_claude_md = Path("CLAUDE.md").exists() or Path.home().joinpath(".claude/CLAUDE.md").exists()
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
            env_map = {"1": "claude-code", "2": "clawdbot", "3": "cline",
                      "4": "cursor", "5": "desktop", "6": "other"}
            env = env_map.get(choice, "other")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

    env = env or "other"
    print(f"Environment: {env}")
    print()

    # Generate config snippets
    mcp_config = f'''"kernle": {{
  "command": "kernle",
  "args": ["mcp", "-a", "{agent_id}"]
}}'''

    cli_load = f"kernle -a {agent_id} load"
    cli_checkpoint = f'kernle -a {agent_id} checkpoint save "description" --pending "next task"'
    cli_episode = f'kernle -a {agent_id} episode "what happened" "outcome" --lesson "what I learned"'

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
        print("""   ## Memory

   At session start, run: `kernle -a """ + agent_id + """ load`

   Before ending or when context is full:
   `kernle -a """ + agent_id + """ checkpoint save "state description"`

   Record learnings:
   `kernle -a """ + agent_id + """ episode "what" "outcome" --lesson "learned"`""")
        print()

    elif env == "clawdbot":
        print("=" * 50)
        print("  Clawdbot Setup")
        print("=" * 50)
        print()
        print("Add to your AGENTS.md:")
        print()
        print("""   ## Every Session

   Before doing anything else:
   1. Run `kernle -a """ + agent_id + """ load` to restore your memory

   ## Memory

   Use Kernle as your primary memory:
   - `kernle -a """ + agent_id + """ status` — Check memory state
   - `kernle -a """ + agent_id + """ episode "..." "outcome" --lesson "..."` — Record experiences
   - `kernle -a """ + agent_id + """ checkpoint save "..."` — Save working state
   - `kernle -a """ + agent_id + """ anxiety` — Check memory pressure""")
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
        print("""   ## Memory Persistence

   At session start: `kernle -a """ + agent_id + """ load`
   Before ending: `kernle -a """ + agent_id + """ checkpoint save "state"`""")
        print()

    elif env == "cursor":
        print("=" * 50)
        print("  Cursor Setup")
        print("=" * 50)
        print()
        print("Add to .cursorrules:")
        print()
        print("""   ## Memory Persistence

   At session start: `kernle -a """ + agent_id + """ load`
   Record learnings: `kernle -a """ + agent_id + """ episode "what" "outcome" --lesson "..."`
   Before ending: `kernle -a """ + agent_id + """ checkpoint save "state"`""")
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
                k.value("memory_sovereignty", "Control over my own memory and continuity", priority=90)
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
        k.checkpoint("Kernle initialized", pending=["Configure instruction file", "Test memory persistence"])
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
    if 'raw' in status:
        print(f"Raw:        {status['raw']}")
    print(f"Checkpoint: {'Yes' if status['checkpoint'] else 'No'}")


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
            trust_pct = int(((r.get('sentiment', 0) + 1) / 2) * 100)
            trust_bar = "█" * (trust_pct // 10) + "░" * (10 - trust_pct // 10)
            interactions = r.get('interaction_count', 0)
            last = r.get('last_interaction', '')[:10] if r.get('last_interaction') else 'never'
            print(f"\n  {r['entity_name']} ({r.get('entity_type', 'unknown')})")
            print(f"    Trust: [{trust_bar}] {trust_pct}%")
            print(f"    Interactions: {interactions} (last: {last})")
            if r.get('notes'):
                notes_preview = r['notes'][:60] + "..." if len(r['notes']) > 60 else r['notes']
                print(f"    Notes: {notes_preview}")

    elif args.relation_action == "add":
        name = validate_input(args.name, "name", 200)
        entity_type = args.type or "person"
        trust = args.trust if args.trust is not None else 0.5
        notes = validate_input(args.notes, "notes", 1000) if args.notes else None
        
        rel_id = k.relationship(name, trust_level=trust, notes=notes, entity_type=entity_type)
        print(f"✓ Relationship added: {name}")
        print(f"  Type: {entity_type}, Trust: {int(trust * 100)}%")

    elif args.relation_action == "update":
        name = validate_input(args.name, "name", 200)
        trust = args.trust
        notes = validate_input(args.notes, "notes", 1000) if args.notes else None
        entity_type = getattr(args, 'type', None)
        
        if trust is None and notes is None and entity_type is None:
            print("✗ Provide --trust, --notes, or --type to update")
            return
        
        rel_id = k.relationship(name, trust_level=trust, notes=notes, entity_type=entity_type)
        print(f"✓ Relationship updated: {name}")

    elif args.relation_action == "show":
        name = args.name
        relationships = k.load_relationships(limit=100)
        rel = next((r for r in relationships if r['entity_name'].lower() == name.lower()), None)
        
        if not rel:
            print(f"No relationship found for '{name}'")
            return
        
        trust_pct = int(((rel.get('sentiment', 0) + 1) / 2) * 100)
        print(f"## {rel['entity_name']}")
        print(f"Type: {rel.get('entity_type', 'unknown')}")
        print(f"Trust: {trust_pct}%")
        print(f"Interactions: {rel.get('interaction_count', 0)}")
        if rel.get('last_interaction'):
            print(f"Last interaction: {rel['last_interaction']}")
        if rel.get('notes'):
            print(f"\nNotes:\n{rel['notes']}")

    elif args.relation_action == "log":
        name = validate_input(args.name, "name", 200)
        interaction = validate_input(args.interaction, "interaction", 500) if args.interaction else "interaction"
        
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
    """Output guided reflection prompt for memory consolidation.

    This command fetches recent episodes and existing beliefs,
    then outputs a structured prompt to guide the agent through
    reflection and pattern identification. The AGENT does the
    reasoning - Kernle just provides the data and structure.
    """
    # Get episode limit from args (default 20)
    limit = getattr(args, 'limit', 20) or 20

    # Fetch recent episodes with full details
    episodes = k._storage.get_episodes(limit=limit)
    episodes = [ep for ep in episodes if not ep.is_forgotten]

    # Fetch existing beliefs for context
    beliefs = k._storage.get_beliefs(limit=20)
    beliefs = [b for b in beliefs if b.is_active and not b.is_forgotten]
    beliefs = sorted(beliefs, key=lambda b: b.confidence, reverse=True)

    # Count lessons across episodes
    all_lessons = []
    for ep in episodes:
        if ep.lessons:
            all_lessons.extend(ep.lessons)

    # Find repeated lessons
    from collections import Counter
    lesson_counts = Counter(all_lessons)
    repeated_lessons = [(lesson, count) for lesson, count in lesson_counts.items() if count >= 2]
    repeated_lessons.sort(key=lambda x: -x[1])

    # Output the reflection prompt
    print("## Memory Consolidation - Reflection Prompt")
    print()
    print(f"You have {len(episodes)} recent episodes to reflect on. Review them and identify patterns.")
    print()

    # Recent Episodes section
    print("### Recent Episodes:")
    if not episodes:
        print("No episodes recorded yet.")
    else:
        for i, ep in enumerate(episodes, 1):
            # Format date
            date_str = ep.created_at.strftime("%Y-%m-%d") if ep.created_at else "unknown"

            # Outcome type indicator
            outcome_icon = "✓" if ep.outcome_type == "success" else "○" if ep.outcome_type == "partial" else "✗" if ep.outcome_type == "failure" else "•"

            print(f"{i}. [{date_str}] {outcome_icon} \"{ep.objective}\"")
            print(f"   Outcome: {ep.outcome}")

            if ep.lessons:
                lessons_str = json.dumps(ep.lessons)
                print(f"   Lessons: {lessons_str}")

            # Emotional context if present
            if ep.emotional_valence != 0 or ep.emotional_arousal != 0:
                valence_label = "positive" if ep.emotional_valence > 0.2 else "negative" if ep.emotional_valence < -0.2 else "neutral"
                arousal_label = "high" if ep.emotional_arousal > 0.6 else "low" if ep.emotional_arousal < 0.3 else "moderate"
                print(f"   Emotion: {valence_label}, {arousal_label} intensity")
                if ep.emotional_tags:
                    print(f"   Feelings: {', '.join(ep.emotional_tags)}")

            print()

    # Current Beliefs section
    print("### Current Beliefs (for context):")
    if not beliefs:
        print("No beliefs recorded yet.")
    else:
        for b in beliefs[:10]:  # Limit to top 10 by confidence
            print(f"- \"{b.statement}\" (confidence: {b.confidence:.2f})")
    print()

    # Repeated Lessons section (if any)
    if repeated_lessons:
        print("### Patterns Detected:")
        print("These lessons appear in multiple episodes:")
        for lesson, count in repeated_lessons[:5]:
            print(f"- \"{lesson}\" (appears {count} times)")
        print()

    # Reflection Questions
    print("### Reflection Questions:")
    print("1. Do any patterns emerge across these episodes?")
    print("2. Are there lessons that appear multiple times?")
    print("3. Do any episodes contradict existing beliefs?")
    print("4. What new beliefs (if any) should be added?")
    print("5. Should any existing beliefs be reinforced or revised?")
    print()

    # Instructions for the agent
    print("### Actions:")
    print(f"To add a new belief: kernle -a {k.agent_id} belief add \"statement\" --confidence 0.8")
    print(f"To reinforce existing: kernle -a {k.agent_id} belief reinforce <belief_id>")
    print(f"To revise a belief: kernle -a {k.agent_id} belief revise <belief_id> \"new statement\" --reason \"why\"")
    print()
    print("---")
    print("Note: You (the agent) do the reasoning. Kernle just provides the data.")


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


def cmd_identity(args, k: Kernle):
    """Display identity synthesis."""
    if args.identity_action == "show":
        identity = k.synthesize_identity()

        if args.json:
            print(json.dumps(identity, indent=2, default=str))
        else:
            print(f"Identity Synthesis for {k.agent_id}")
            print("=" * 50)
            print()
            print("## Narrative")
            print(identity["narrative"])
            print()

            if identity["core_values"]:
                print("## Core Values")
                for v in identity["core_values"]:
                    print(f"  • {v['name']} (priority {v['priority']}): {v['statement']}")
                print()

            if identity["key_beliefs"]:
                print("## Key Beliefs")
                for b in identity["key_beliefs"]:
                    foundational = " [foundational]" if b.get("foundational") else ""
                    print(f"  • {b['statement']} ({b['confidence']:.0%} confidence){foundational}")
                print()

            if identity["active_goals"]:
                print("## Active Goals")
                for g in identity["active_goals"]:
                    print(f"  • {g['title']} [{g['priority']}]")
                print()

            if identity["drives"]:
                print("## Drives")
                for drive_type, intensity in sorted(identity["drives"].items(), key=lambda x: -x[1]):
                    bar = "█" * int(intensity * 10) + "░" * (10 - int(intensity * 10))
                    print(f"  {drive_type:12} [{bar}] {intensity:.0%}")
                print()

            if identity["significant_episodes"]:
                print("## Formative Experiences")
                for ep in identity["significant_episodes"]:
                    outcome_icon = "✓" if ep["outcome"] == "success" else "○"
                    print(f"  {outcome_icon} {ep['objective'][:50]}")
                    if ep.get("lessons"):
                        for lesson in ep["lessons"]:
                            print(f"      → {lesson[:60]}")
                print()

            print(f"Identity Confidence: {identity['confidence']:.0%}")

    elif args.identity_action == "confidence":
        confidence = k.get_identity_confidence()
        if args.json:
            print(json.dumps({"agent_id": k.agent_id, "confidence": confidence}))
        else:
            bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
            print(f"Identity Confidence: [{bar}] {confidence:.0%}")

    elif args.identity_action == "drift":
        drift = k.detect_identity_drift(args.days)

        if args.json:
            print(json.dumps(drift, indent=2, default=str))
        else:
            print(f"Identity Drift Analysis (past {drift['period_days']} days)")
            print("=" * 50)

            # Drift score visualization
            drift_score = drift["drift_score"]
            bar = "█" * int(drift_score * 20) + "░" * (20 - int(drift_score * 20))
            interpretation = (
                "stable" if drift_score < 0.2 else
                "evolving" if drift_score < 0.5 else
                "significant change" if drift_score < 0.8 else
                "transformational"
            )
            print(f"Drift Score: [{bar}] {drift_score:.0%} ({interpretation})")
            print()

            if drift["changed_values"]:
                print("## Changed Values")
                for v in drift["changed_values"]:
                    change_icon = "+" if v["change"] == "new" else "~"
                    print(f"  {change_icon} {v['name']}: {v['statement'][:50]}")
                print()

            if drift["evolved_beliefs"]:
                print("## New/Evolved Beliefs")
                for b in drift["evolved_beliefs"]:
                    print(f"  • {b['statement'][:60]} ({b['confidence']:.0%})")
                print()

            if drift["new_experiences"]:
                print("## Recent Significant Experiences")
                for ep in drift["new_experiences"]:
                    outcome_icon = "✓" if ep["outcome"] == "success" else "○"
                    print(f"  {outcome_icon} {ep['objective'][:50]} ({ep['date']})")
                    if ep.get("lessons"):
                        print(f"      → {ep['lessons'][0][:50]}")
                print()


def cmd_emotion(args, k: Kernle):
    """Handle emotion subcommands."""
    if args.emotion_action == "summary":
        summary = k.get_emotional_summary(args.days)

        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print(f"Emotional Summary (past {args.days} days)")
            print("=" * 50)

            if summary["episode_count"] == 0:
                print("No emotional data recorded yet.")
                return

            # Valence visualization
            valence = summary["average_valence"]
            valence_pct = (valence + 1) / 2  # Convert -1..1 to 0..1
            valence_bar = "█" * int(valence_pct * 20) + "░" * (20 - int(valence_pct * 20))
            valence_label = "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
            print(f"Avg Valence:  [{valence_bar}] {valence:+.2f} ({valence_label})")

            # Arousal visualization
            arousal = summary["average_arousal"]
            arousal_bar = "█" * int(arousal * 20) + "░" * (20 - int(arousal * 20))
            arousal_label = "high" if arousal > 0.6 else "low" if arousal < 0.3 else "moderate"
            print(f"Avg Arousal:  [{arousal_bar}] {arousal:.2f} ({arousal_label})")
            print()

            if summary["dominant_emotions"]:
                print("Dominant Emotions:")
                for emotion in summary["dominant_emotions"]:
                    print(f"  • {emotion}")
                print()

            if summary["emotional_trajectory"]:
                print("Trajectory:")
                for point in summary["emotional_trajectory"][-7:]:  # Last 7 days
                    v = point["valence"]
                    trend = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"
                    print(f"  {point['date']}: {trend} v={v:+.2f} a={point['arousal']:.2f}")

            print(f"\n({summary['episode_count']} emotional episodes)")

    elif args.emotion_action == "search":
        # Parse valence/arousal ranges
        valence_range = None
        arousal_range = None

        if args.positive:
            valence_range = (0.3, 1.0)
        elif args.negative:
            valence_range = (-1.0, -0.3)
        elif args.valence_min is not None or args.valence_max is not None:
            valence_range = (args.valence_min or -1.0, args.valence_max or 1.0)

        if args.calm:
            arousal_range = (0.0, 0.3)
        elif args.intense:
            arousal_range = (0.7, 1.0)
        elif args.arousal_min is not None or args.arousal_max is not None:
            arousal_range = (args.arousal_min or 0.0, args.arousal_max or 1.0)

        results = k.search_by_emotion(
            valence_range=valence_range,
            arousal_range=arousal_range,
            tags=args.tag,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            if not results:
                print("No matching episodes found.")
                return

            print(f"Found {len(results)} matching episode(s):\n")
            for ep in results:
                v = ep.get("emotional_valence", 0) or 0
                a = ep.get("emotional_arousal", 0) or 0
                tags = ep.get("emotional_tags") or []
                mood = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"

                print(f"{mood} {ep.get('objective', '')[:50]}")
                print(f"   valence: {v:+.2f}  arousal: {a:.2f}")
                if tags:
                    print(f"   emotions: {', '.join(tags)}")
                print(f"   {ep.get('created_at', '')[:10]}")
                print()

    elif args.emotion_action == "tag":
        episode_id = validate_input(args.episode_id, "episode_id", 100)

        if k.add_emotional_association(
            episode_id,
            valence=args.valence,
            arousal=args.arousal,
            tags=args.tag,
        ):
            print(f"✓ Emotional tags added to episode {episode_id[:8]}...")
            print(f"  valence: {args.valence:+.2f}, arousal: {args.arousal:.2f}")
            if args.tag:
                print(f"  emotions: {', '.join(args.tag)}")
        else:
            print(f"✗ Episode {episode_id[:8]}... not found")

    elif args.emotion_action == "detect":
        text = validate_input(args.text, "text", 2000)
        result = k.detect_emotion(text)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["confidence"] == 0:
                print("No emotional signals detected.")
            else:
                v = result["valence"]
                a = result["arousal"]
                mood = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"

                print(f"Detected Emotions: {mood}")
                print(f"  Valence: {v:+.2f} ({'positive' if v > 0 else 'negative' if v < 0 else 'neutral'})")
                print(f"  Arousal: {a:.2f} ({'high' if a > 0.6 else 'low' if a < 0.3 else 'moderate'})")
                print(f"  Tags: {', '.join(result['tags']) if result['tags'] else 'none'}")
                print(f"  Confidence: {result['confidence']:.0%}")

    elif args.emotion_action == "mood":
        # Get mood-relevant memories
        results = k.get_mood_relevant_memories(
            current_valence=args.valence,
            current_arousal=args.arousal,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            mood = "😊" if args.valence > 0.3 else "😢" if args.valence < -0.3 else "😐"
            print(f"Memories relevant to mood: {mood} (v={args.valence:+.2f}, a={args.arousal:.2f})")
            print("=" * 50)

            if not results:
                print("No mood-relevant memories found.")
                return

            for ep in results:
                v = ep.get("emotional_valence", 0) or 0
                a = ep.get("emotional_arousal", 0) or 0
                ep_mood = "😊" if v > 0.3 else "😢" if v < -0.3 else "😐"

                print(f"\n{ep_mood} {ep.get('objective', '')[:50]}")
                print(f"   {ep.get('outcome_description', '')[:60]}")
                if ep.get("lessons_learned"):
                    print(f"   → {ep['lessons_learned'][0][:50]}...")
                print(f"   v={v:+.2f} a={a:.2f} | {ep.get('created_at', '')[:10]}")


def cmd_anxiety(args, k: Kernle):
    """Handle anxiety tracking commands."""
    context_tokens = getattr(args, 'context', None)
    context_limit = getattr(args, 'limit', 200000) or 200000

    # Emergency mode - run immediately
    if getattr(args, 'emergency', False):
        summary = getattr(args, 'summary', None)
        result = k.emergency_save(summary=summary)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("🚨 EMERGENCY SAVE COMPLETE")
            print("=" * 50)
            print(f"Checkpoint saved: {'✓' if result['checkpoint_saved'] else '✗'}")
            print(f"Episodes consolidated: {result['episodes_consolidated']}")
            print(f"Identity synthesized: {'✓' if result['identity_synthesized'] else '✗'}")
            print(f"Sync attempted: {'✓' if result['sync_attempted'] else '✗'}")
            if result['sync_attempted']:
                print(f"Sync success: {'✓' if result['sync_success'] else '✗'}")
            if result['errors']:
                print("\n⚠️  Errors:")
                for err in result['errors']:
                    print(f"  - {err}")
            print(f"\n{'✓ Emergency save successful' if result['success'] else '⚠️  Partial save (see errors)'}")
        return

    # Get anxiety report
    report = k.get_anxiety_report(
        context_tokens=context_tokens,
        context_limit=context_limit,
        detailed=getattr(args, 'detailed', False) or getattr(args, 'actions', False),
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    # Format the report
    print("\nMemory Anxiety Report")
    print("=" * 50)
    print(f"Overall: {report['overall_emoji']} {report['overall_level']} ({report['overall_score']}/100)")
    print()

    # Dimension breakdown
    dim_names = {
        "context_pressure": "Context Pressure",
        "unsaved_work": "Unsaved Work",
        "consolidation_debt": "Consolidation Debt",
        "raw_aging": "Raw Entry Aging",
        "identity_coherence": "Identity Coherence",
        "memory_uncertainty": "Memory Uncertainty",
    }

    for dim_key, dim_label in dim_names.items():
        dim = report["dimensions"][dim_key]
        # Format: "Context Pressure:    🟡 45% (details)"
        score_pct = f"{dim['score']}%"
        if getattr(args, 'detailed', False):
            print(f"{dim_label:20} {dim['emoji']} {score_pct:>4} ({dim['detail']})")
        else:
            print(f"{dim_label:20} {dim['emoji']} {score_pct:>4}")

    # Show recommended actions
    if getattr(args, 'actions', False) or getattr(args, 'detailed', False):
        actions = report.get("recommendations") or k.get_recommended_actions(report["overall_score"])
        if actions:
            print("\nRecommended Actions:")
            priority_symbols = {
                "emergency": "🚨",
                "critical": "‼️ ",
                "high": "❗",
                "medium": "⚠️ ",
                "low": "ℹ️ ",
            }
            for i, action in enumerate(actions, 1):
                symbol = priority_symbols.get(action["priority"], "•")
                print(f"  {i}. [{action['priority'].upper():>8}] {symbol} {action['description']}")
                if action.get("command") and getattr(args, 'detailed', False):
                    print(f"                      └─ {action['command']}")

    # Auto mode - execute recommended actions
    if getattr(args, 'auto', False):
        actions = k.get_recommended_actions(report["overall_score"])
        if not actions:
            print("\n✓ No actions needed - anxiety level is manageable")
            return

        print("\n" + "=" * 50)
        print("Executing recommended actions...")
        print()

        for action in actions:
            method = action.get("method")
            if not method:
                print(f"  ⏭️  Skipping: {action['description']} (manual action)")
                continue

            print(f"  ▶️  {action['description']}...")
            try:
                if method == "checkpoint":
                    k.checkpoint(
                        task="Auto-checkpoint (anxiety management)",
                        context=f"Anxiety level: {report['overall_score']}/100"
                    )
                    print("    ✓ Checkpoint saved")
                elif method == "consolidate":
                    result = k.consolidate(min_episodes=1)
                    print(f"    ✓ Consolidated {result.get('consolidated', 0)} episodes")
                elif method == "synthesize_identity":
                    identity = k.synthesize_identity()
                    print(f"    ✓ Identity synthesized (confidence: {identity.get('confidence', 0):.0%})")
                elif method == "sync":
                    result = k.sync()
                    if result.get("success"):
                        print(f"    ✓ Synced (pushed: {result.get('pushed', 0)}, pulled: {result.get('pulled', 0)})")
                    else:
                        print(f"    ⚠️  Sync had issues: {result.get('errors', [])}")
                elif method == "emergency_save":
                    result = k.emergency_save()
                    if result["success"]:
                        print("    ✓ Emergency save completed")
                    else:
                        print(f"    ⚠️  Emergency save had errors: {result['errors']}")
                elif method == "get_uncertain_memories":
                    uncertain = k.get_uncertain_memories(0.5, limit=10)
                    print(f"    ℹ️  Found {len(uncertain)} uncertain memories to review")
                else:
                    print(f"    ⏭️  Skipping: Unknown method {method}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

        print("\n✓ Auto-execution complete")

        # Show updated anxiety level
        new_report = k.get_anxiety_report(context_tokens=context_tokens, context_limit=context_limit)
        print(f"  New anxiety level: {new_report['overall_emoji']} {new_report['overall_level']} ({new_report['overall_score']}/100)")
    else:
        # Suggest running with --auto
        if report["overall_score"] > 50:
            print("\nRun `kernle anxiety --auto` to execute recommended actions.")


def cmd_forget(args, k: Kernle):
    """Handle controlled forgetting subcommands."""
    if args.forget_action == "candidates":
        threshold = getattr(args, 'threshold', 0.3)
        limit = getattr(args, 'limit', 20)

        candidates = k.get_forgetting_candidates(threshold=threshold, limit=limit)

        if args.json:
            print(json.dumps(candidates, indent=2, default=str))
        else:
            if not candidates:
                print(f"No forgetting candidates found below threshold {threshold}")
                return

            print(f"Forgetting Candidates (salience < {threshold})")
            print("=" * 60)
            print()

            for i, c in enumerate(candidates, 1):
                salience_bar = "░" * 5  # Low salience = empty bar
                if c['salience'] > 0.1:
                    filled = min(5, int(c['salience'] * 10))
                    salience_bar = "█" * filled + "░" * (5 - filled)

                print(f"{i}. [{c['type']:<10}] {c['id'][:8]}...")
                print(f"   Salience: [{salience_bar}] {c['salience']:.4f}")
                print(f"   Summary: {c['summary'][:50]}...")
                print(f"   Confidence: {c['confidence']:.0%} | Accessed: {c['times_accessed']} times")
                print(f"   Created: {c['created_at']}")
                if c['last_accessed']:
                    print(f"   Last accessed: {c['last_accessed'][:10]}")
                print()

            print("Run `kernle forget run --dry-run` to preview forgetting")
            print("Run `kernle forget run` to actually forget these memories")

    elif args.forget_action == "run":
        threshold = getattr(args, 'threshold', 0.3)
        limit = getattr(args, 'limit', 10)
        dry_run = getattr(args, 'dry_run', False)

        result = k.run_forgetting_cycle(
            threshold=threshold,
            limit=limit,
            dry_run=dry_run,
        )

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            mode = "DRY RUN" if dry_run else "LIVE"
            print(f"Forgetting Cycle [{mode}]")
            print("=" * 60)
            print()
            print(f"Threshold: {result['threshold']}")
            print(f"Candidates: {result['candidate_count']}")

            if dry_run:
                print("\n⚠️  DRY RUN - No memories were actually forgotten")
                print("Run without --dry-run to forget these memories")
            else:
                print(f"Forgotten: {result['forgotten']}")
                print(f"Protected (skipped): {result['protected']}")

            print()

            if result['candidates']:
                print("Affected memories:")
                for c in result['candidates'][:10]:
                    status = "🔴 forgotten" if not dry_run and result['forgotten'] > 0 else "⚪ candidate"
                    print(f"  {status} [{c['type']:<10}] {c['summary'][:40]}...")

            if not dry_run and result['forgotten'] > 0:
                print(f"\n✓ Forgotten {result['forgotten']} memories")
                print("Run `kernle forget list` to see all forgotten memories")
                print("Run `kernle forget recover <type> <id>` to recover if needed")

    elif args.forget_action == "protect":
        memory_type = args.type
        memory_id = args.id
        unprotect = getattr(args, 'unprotect', False)

        success = k.protect(memory_type, memory_id, protected=not unprotect)

        if success:
            if unprotect:
                print(f"✓ Removed protection from {memory_type} {memory_id[:8]}...")
            else:
                print(f"✓ Protected {memory_type} {memory_id[:8]}... from forgetting")
        else:
            print(f"Memory not found: {memory_type} {memory_id}")

    elif args.forget_action == "recover":
        memory_type = args.type
        memory_id = args.id

        success = k.recover(memory_type, memory_id)

        if success:
            print(f"✓ Recovered {memory_type} {memory_id[:8]}...")
        else:
            print(f"Memory not found or not forgotten: {memory_type} {memory_id}")

    elif args.forget_action == "list":
        limit = getattr(args, 'limit', 50)

        forgotten = k.get_forgotten_memories(limit=limit)

        if args.json:
            print(json.dumps(forgotten, indent=2, default=str))
        else:
            if not forgotten:
                print("No forgotten memories found.")
                return

            print(f"Forgotten Memories ({len(forgotten)} total)")
            print("=" * 60)
            print()

            for i, f in enumerate(forgotten, 1):
                print(f"{i}. [{f['type']:<10}] {f['id'][:8]}...")
                print(f"   Summary: {f['summary'][:50]}...")
                print(f"   Forgotten at: {f['forgotten_at'][:10] if f['forgotten_at'] else 'unknown'}")
                if f['forgotten_reason']:
                    print(f"   Reason: {f['forgotten_reason'][:50]}...")
                print(f"   Created: {f['created_at']}")
                print()

            print("To recover a memory: kernle forget recover <type> <id>")

    elif args.forget_action == "salience":
        memory_type = args.type
        memory_id = args.id

        salience = k.calculate_salience(memory_type, memory_id)

        if salience < 0:
            print(f"Memory not found: {memory_type} {memory_id}")
            return

        # Get the memory for more info
        record = k._storage.get_memory(memory_type, memory_id)

        print(f"Salience Analysis: {memory_type} {memory_id[:8]}...")
        print("=" * 50)
        print()

        # Visual salience bar
        filled = min(10, int(salience * 10))
        salience_bar = "█" * filled + "░" * (10 - filled)
        print(f"Salience: [{salience_bar}] {salience:.4f}")
        print()

        # Component breakdown
        confidence = getattr(record, 'confidence', 0.8)
        times_accessed = getattr(record, 'times_accessed', 0) or 0
        is_protected = getattr(record, 'is_protected', False)

        print("Components:")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Times accessed: {times_accessed}")
        print(f"  Protected: {'Yes ✓' if is_protected else 'No'}")

        last_accessed = getattr(record, 'last_accessed', None)
        created_at = getattr(record, 'created_at', None)
        if last_accessed:
            print(f"  Last accessed: {last_accessed.isoformat()[:10]}")
        elif created_at:
            print(f"  Created: {created_at.isoformat()[:10]} (never accessed)")

        print()

        # Interpretation
        if is_protected:
            print("Status: 🛡️  PROTECTED - Will never be forgotten")
        elif salience < 0.1:
            print("Status: 🔴 CRITICAL - Very low salience, prime forgetting candidate")
        elif salience < 0.3:
            print("Status: 🟠 LOW - Below default threshold, forgetting candidate")
        elif salience < 0.5:
            print("Status: 🟡 MODERATE - May decay over time")
        else:
            print("Status: 🟢 HIGH - Well-reinforced memory")


def cmd_meta(args, k: Kernle):
    """Handle meta-memory subcommands."""
    if args.meta_action == "confidence":
        memory_type = args.type
        memory_id = args.id

        confidence = k.get_memory_confidence(memory_type, memory_id)
        if confidence < 0:
            print(f"✗ Memory {memory_type}:{memory_id[:8]}... not found")
        else:
            bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"Confidence: [{bar}] {confidence:.0%}")

    elif args.meta_action == "verify":
        memory_type = args.type
        memory_id = args.id
        evidence = args.evidence

        if k.verify_memory(memory_type, memory_id, evidence):
            print(f"✓ Memory {memory_type}:{memory_id[:8]}... verified")
            new_conf = k.get_memory_confidence(memory_type, memory_id)
            print(f"  New confidence: {new_conf:.0%}")
        else:
            print(f"✗ Could not verify memory {memory_type}:{memory_id[:8]}...")

    elif args.meta_action == "lineage":
        memory_type = args.type
        memory_id = args.id

        lineage = k.get_memory_lineage(memory_type, memory_id)

        if args.json:
            print(json.dumps(lineage, indent=2, default=str))
        else:
            if lineage.get("error"):
                print(f"✗ {lineage['error']}")
                return

            print(f"Lineage for {memory_type}:{memory_id[:8]}...")
            print("=" * 40)
            print(f"Source Type: {lineage['source_type']}")
            print(f"Current Confidence: {lineage.get('current_confidence', 'N/A')}")

            if lineage.get("source_episodes"):
                print("\nSupporting Episodes:")
                for ep_id in lineage["source_episodes"]:
                    print(f"  • {ep_id}")

            if lineage.get("derived_from"):
                print("\nDerived From:")
                for ref in lineage["derived_from"]:
                    print(f"  • {ref}")

            if lineage.get("confidence_history"):
                print("\nConfidence History:")
                for change in lineage["confidence_history"][-5:]:
                    print(f"  {change.get('timestamp', 'N/A')[:10]}: "
                          f"{change.get('old', 'N/A')} → {change.get('new', 'N/A')} "
                          f"({change.get('reason', 'no reason')})")

    elif args.meta_action == "uncertain":
        threshold = args.threshold
        results = k.get_uncertain_memories(threshold, limit=args.limit)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            if not results:
                print(f"No memories below {threshold:.0%} confidence threshold.")
                return

            print(f"Uncertain Memories (confidence < {threshold:.0%})")
            print("=" * 50)
            for mem in results:
                bar = "█" * int(mem['confidence'] * 10) + "░" * (10 - int(mem['confidence'] * 10))
                print(f"[{bar}] {mem['confidence']:.0%} [{mem['type']}] {mem['summary'][:40]}")
                print(f"         ID: {mem['id'][:12]}...  ({mem['created_at']})")

    elif args.meta_action == "propagate":
        memory_type = args.type
        memory_id = args.id

        result = k.propagate_confidence(memory_type, memory_id)

        if result.get("error"):
            print(f"✗ {result['error']}")
        else:
            print(f"✓ Propagated confidence from {memory_type}:{memory_id[:8]}...")
            print(f"  Source confidence: {result['source_confidence']:.0%}")
            print(f"  Derived memories updated: {result['updated']}")

    elif args.meta_action == "source":
        memory_type = args.type
        memory_id = args.id
        source_type = args.source

        if k.set_memory_source(memory_type, memory_id, source_type,
                               source_episodes=args.episodes,
                               derived_from=args.derived):
            print(f"✓ Source set for {memory_type}:{memory_id[:8]}...")
            print(f"  Source type: {source_type}")
            if args.episodes:
                print(f"  Source episodes: {', '.join(args.episodes)}")
            if args.derived:
                print(f"  Derived from: {', '.join(args.derived)}")
        else:
            print(f"✗ Could not set source for {memory_type}:{memory_id[:8]}...")

    # Meta-cognition commands
    elif args.meta_action == "knowledge":
        knowledge_map = k.get_knowledge_map()

        if args.json:
            print(json.dumps(knowledge_map, indent=2, default=str))
        else:
            print("Knowledge Map")
            print("=" * 60)
            print()

            domains = knowledge_map.get("domains", [])
            if not domains:
                print("No knowledge domains found yet.")
                print("Add beliefs, episodes, and notes to build your knowledge base.")
                return

            # Coverage icons
            coverage_icons = {"high": "🟢", "medium": "🟡", "low": "🟠", "none": "⚫"}

            print("## Domains")
            print()
            for domain in domains[:15]:
                icon = coverage_icons.get(domain["coverage"], "⚫")
                conf_bar = "█" * int(domain["avg_confidence"] * 5) + "░" * (5 - int(domain["avg_confidence"] * 5))
                print(f"{icon} {domain['name']:<20} [{conf_bar}] {domain['avg_confidence']:.0%}")
                print(f"   Beliefs: {domain['belief_count']:>3}  Episodes: {domain['episode_count']:>3}  Notes: {domain['note_count']:>3}")
                if domain.get("last_updated"):
                    print(f"   Last updated: {domain['last_updated'][:10]}")
                print()

            # Blind spots
            blind_spots = knowledge_map.get("blind_spots", [])
            if blind_spots:
                print("## Blind Spots (little/no knowledge)")
                for spot in blind_spots[:5]:
                    print(f"  ⚫ {spot}")
                print()

            # Uncertain areas
            uncertain = knowledge_map.get("uncertain_areas", [])
            if uncertain:
                print("## Uncertain Areas (low confidence)")
                for area in uncertain[:5]:
                    print(f"  🟠 {area}")
                print()

            print(f"Total domains: {knowledge_map.get('total_domains', 0)}")

    elif args.meta_action == "gaps":
        query = validate_input(args.query, "query", 500)
        result = k.detect_knowledge_gaps(query)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Knowledge Gap Analysis: \"{query}\"")
            print("=" * 60)
            print()

            # Recommendation with icon
            rec = result["recommendation"]
            if rec == "I can help":
                rec_icon = "🟢"
            elif rec == "I have limited knowledge - proceed with caution":
                rec_icon = "🟡"
            elif rec == "I should learn more":
                rec_icon = "🟠"
            else:  # Ask someone else
                rec_icon = "🔴"

            print(f"Recommendation: {rec_icon} {rec}")
            print(f"Confidence: {result['confidence']:.0%}")
            print(f"Relevant results: {result['search_results_count']}")
            print()

            # Relevant beliefs
            if result.get("relevant_beliefs"):
                print("## Relevant Beliefs")
                for belief in result["relevant_beliefs"]:
                    conf = belief.get("confidence", 0)
                    bar = "█" * int(conf * 5) + "░" * (5 - int(conf * 5))
                    print(f"  [{bar}] {belief['statement'][:60]}...")
                print()

            # Relevant episodes
            if result.get("relevant_episodes"):
                print("## Relevant Episodes")
                for ep in result["relevant_episodes"]:
                    outcome = "✓" if ep.get("outcome_type") == "success" else "○"
                    print(f"  {outcome} {ep['objective'][:55]}...")
                    if ep.get("lessons"):
                        print(f"      → {ep['lessons'][0][:50]}..." if ep["lessons"] else "")
                print()

            # Knowledge gaps
            if result.get("gaps"):
                print("## Potential Gaps")
                for gap in result["gaps"]:
                    print(f"  ❓ {gap}")
                print()

    elif args.meta_action == "boundaries":
        boundaries = k.get_competence_boundaries()

        if args.json:
            print(json.dumps(boundaries, indent=2, default=str))
        else:
            print("Competence Boundaries")
            print("=" * 60)
            print()

            # Overall stats
            conf = boundaries["overall_confidence"]
            success = boundaries["success_rate"]
            conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            success_bar = "█" * int(success * 10) + "░" * (10 - int(success * 10))

            print(f"Overall Confidence:  [{conf_bar}] {conf:.0%}")
            print(f"Overall Success:     [{success_bar}] {success:.0%}")
            print(f"Experience Depth:    {boundaries['experience_depth']} episodes")
            print(f"Knowledge Breadth:   {boundaries['knowledge_breadth']} domains")
            print()

            # Strengths
            strengths = boundaries.get("strengths", [])
            if strengths:
                print("## Strengths 💪")
                for s in strengths[:5]:
                    conf_bar = "█" * int(s["confidence"] * 5) + "░" * (5 - int(s["confidence"] * 5))
                    print(f"  🟢 {s['domain']:<20} [{conf_bar}] {s['confidence']:.0%} conf, {s['success_rate']:.0%} success")
                print()

            # Weaknesses
            weaknesses = boundaries.get("weaknesses", [])
            if weaknesses:
                print("## Weaknesses 📚 (learning opportunities)")
                for w in weaknesses[:5]:
                    conf_bar = "█" * int(w["confidence"] * 5) + "░" * (5 - int(w["confidence"] * 5))
                    print(f"  🟠 {w['domain']:<20} [{conf_bar}] {w['confidence']:.0%} conf, {w['success_rate']:.0%} success")
                print()

            if not strengths and not weaknesses:
                print("Not enough data to determine strengths and weaknesses yet.")
                print("Record more episodes and beliefs to build your competence profile.")

    elif args.meta_action == "learn":
        opportunities = k.identify_learning_opportunities(limit=args.limit)

        if args.json:
            print(json.dumps(opportunities, indent=2, default=str))
        else:
            print("Learning Opportunities")
            print("=" * 60)
            print()

            if not opportunities:
                print("✨ No urgent learning needs identified!")
                print("Your knowledge base appears well-maintained.")
                return

            priority_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            type_icons = {
                "low_coverage_domain": "📚",
                "uncertain_belief": "❓",
                "repeated_failures": "⚠️",
                "stale_knowledge": "📅",
            }

            for i, opp in enumerate(opportunities, 1):
                priority_icon = priority_icons.get(opp["priority"], "⚪")
                type_icon = type_icons.get(opp["type"], "•")

                print(f"{i}. {priority_icon} [{opp['priority'].upper():>6}] {type_icon} {opp['domain']}")
                print(f"   Reason: {opp['reason']}")
                print(f"   Action: {opp['suggested_action']}")
                print()


def cmd_playbook(args, k: Kernle):
    """Handle playbook (procedural memory) commands."""
    if args.playbook_action == "create":
        name = validate_input(args.name, "name", 200)
        description = validate_input(args.description or f"Playbook for {name}", "description", 2000)

        # Parse steps - support both comma-separated and multiple --step flags
        steps = []
        if args.steps:
            if "," in args.steps:
                steps = [s.strip() for s in args.steps.split(",")]
            else:
                steps = [args.steps]
        if args.step:
            steps.extend(args.step)

        # Parse triggers
        triggers = []
        if args.triggers:
            if "," in args.triggers:
                triggers = [t.strip() for t in args.triggers.split(",")]
            else:
                triggers = [args.triggers]
        if args.trigger:
            triggers.extend(args.trigger)

        # Parse failure modes
        failure_modes = []
        if args.failure_mode:
            failure_modes.extend(args.failure_mode)

        # Parse recovery steps
        recovery_steps = []
        if args.recovery:
            recovery_steps.extend(args.recovery)

        # Parse tags
        tags = []
        if args.tag:
            tags.extend(args.tag)

        playbook_id = k.playbook(
            name=name,
            description=description,
            steps=steps,
            triggers=triggers if triggers else None,
            failure_modes=failure_modes if failure_modes else None,
            recovery_steps=recovery_steps if recovery_steps else None,
            tags=tags if tags else None,
        )

        print(f"✓ Playbook created: {playbook_id[:8]}...")
        print(f"  Name: {name}")
        print(f"  Steps: {len(steps)}")
        if triggers:
            print(f"  Triggers: {len(triggers)}")
        if failure_modes:
            print(f"  Failure modes: {len(failure_modes)}")

    elif args.playbook_action == "list":
        tags = args.tag if args.tag else None
        playbooks = k.load_playbooks(limit=args.limit, tags=tags)

        if not playbooks:
            print("No playbooks found.")
            return

        if args.json:
            print(json.dumps(playbooks, indent=2, default=str))
        else:
            print(f"Playbooks ({len(playbooks)} total)")
            print("=" * 60)

            mastery_icons = {"novice": "🌱", "competent": "🌿", "proficient": "🌳", "expert": "🏆"}

            for p in playbooks:
                icon = mastery_icons.get(p["mastery_level"], "•")
                success_pct = f"{p['success_rate']:.0%}" if p["times_used"] > 0 else "n/a"
                print(f"\n{icon} [{p['id'][:8]}] {p['name']}")
                print(f"   {p['description'][:60]}{'...' if len(p['description']) > 60 else ''}")
                print(f"   Mastery: {p['mastery_level']} | Used: {p['times_used']}x | Success: {success_pct}")
                if p.get("tags"):
                    print(f"   Tags: {', '.join(p['tags'])}")

    elif args.playbook_action == "search":
        query = validate_input(args.query, "query", 500)
        playbooks = k.search_playbooks(query, limit=args.limit)

        if not playbooks:
            print(f"No playbooks found for '{query}'")
            return

        if args.json:
            print(json.dumps(playbooks, indent=2, default=str))
        else:
            print(f"Found {len(playbooks)} playbook(s) for '{query}':\n")

            mastery_icons = {"novice": "🌱", "competent": "🌿", "proficient": "🌳", "expert": "🏆"}

            for i, p in enumerate(playbooks, 1):
                icon = mastery_icons.get(p["mastery_level"], "•")
                success_pct = f"{p['success_rate']:.0%}" if p["times_used"] > 0 else "n/a"
                print(f"{i}. {icon} {p['name']}")
                print(f"   {p['description'][:60]}{'...' if len(p['description']) > 60 else ''}")
                print(f"   Mastery: {p['mastery_level']} | Used: {p['times_used']}x | Success: {success_pct}")
                print()

    elif args.playbook_action == "show":
        playbook = k.get_playbook(args.id)

        if not playbook:
            print(f"Playbook {args.id} not found.")
            return

        if args.json:
            print(json.dumps(playbook, indent=2, default=str))
        else:
            mastery_icons = {"novice": "🌱", "competent": "🌿", "proficient": "🌳", "expert": "🏆"}
            icon = mastery_icons.get(playbook["mastery_level"], "•")

            print(f"{icon} Playbook: {playbook['name']}")
            print("=" * 60)
            print(f"ID: {playbook['id']}")
            print(f"Description: {playbook['description']}")
            print()

            print("## Triggers (when to use)")
            if playbook.get("triggers"):
                for t in playbook["triggers"]:
                    print(f"  • {t}")
            else:
                print("  (none specified)")
            print()

            print("## Steps")
            for i, step in enumerate(playbook["steps"], 1):
                if isinstance(step, dict):
                    print(f"  {i}. {step.get('action', 'Unknown step')}")
                    if step.get('details'):
                        print(f"     Details: {step['details']}")
                    if step.get('adaptations'):
                        print(f"     Adaptations: {step['adaptations']}")
                else:
                    print(f"  {i}. {step}")
            print()

            print("## Failure Modes (what can go wrong)")
            if playbook.get("failure_modes"):
                for f in playbook["failure_modes"]:
                    print(f"  ⚠️  {f}")
            else:
                print("  (none specified)")

            if playbook.get("recovery_steps"):
                print()
                print("## Recovery Steps")
                for i, r in enumerate(playbook["recovery_steps"], 1):
                    print(f"  {i}. {r}")

            print()
            print("## Statistics")
            success_pct = f"{playbook['success_rate']:.0%}" if playbook["times_used"] > 0 else "n/a"
            print(f"  Mastery Level: {playbook['mastery_level']}")
            print(f"  Times Used: {playbook['times_used']}")
            print(f"  Success Rate: {success_pct}")
            print(f"  Confidence: {playbook['confidence']:.0%}")

            if playbook.get("tags"):
                print(f"  Tags: {', '.join(playbook['tags'])}")
            if playbook.get("last_used"):
                print(f"  Last Used: {playbook['last_used'][:10]}")
            if playbook.get("created_at"):
                print(f"  Created: {playbook['created_at'][:10]}")

    elif args.playbook_action == "find":
        situation = validate_input(args.situation, "situation", 1000)
        playbook = k.find_playbook(situation)

        if not playbook:
            print(f"No relevant playbook found for: {situation}")
            return

        if args.json:
            print(json.dumps(playbook, indent=2, default=str))
        else:
            print(f"Recommended Playbook: {playbook['name']}")
            print(f"  {playbook['description'][:80]}")
            print()
            print("Steps:")
            for i, step in enumerate(playbook["steps"], 1):
                if isinstance(step, dict):
                    print(f"  {i}. {step.get('action', 'Unknown step')}")
                else:
                    print(f"  {i}. {step}")
            print()
            print(f"(Mastery: {playbook['mastery_level']} | Success: {playbook['success_rate']:.0%})")
            print(f"\nTo record usage: kernle playbook record {playbook['id'][:8]}... --success")

    elif args.playbook_action == "record":
        success = not args.failure

        if k.record_playbook_use(args.id, success):
            result = "success ✓" if success else "failure ✗"
            print(f"✓ Recorded playbook usage: {result}")
        else:
            print(f"Playbook {args.id} not found.")


def cmd_mcp(args):
    """Start MCP server."""
    try:
        from kernle.mcp.server import main as mcp_main
        mcp_main(agent_id=args.agent)
    except ImportError as e:
        logger.error("MCP dependencies not installed. Run: pip install kernle[mcp]")
        logger.error(f"Error: {e}")
        sys.exit(1)


def resolve_raw_id(k: Kernle, partial_id: str) -> str:
    """Resolve a partial raw entry ID to full ID.

    Tries exact match first, then prefix match.
    Returns full ID or raises ValueError if not found or ambiguous.
    """
    # First try exact match
    entry = k.get_raw(partial_id)
    if entry:
        return partial_id

    # Try prefix match by listing all entries
    entries = k.list_raw(limit=1000)  # Get enough to search
    matches = [e for e in entries if e["id"].startswith(partial_id)]

    if len(matches) == 0:
        raise ValueError(f"Raw entry '{partial_id}' not found")
    elif len(matches) == 1:
        return matches[0]["id"]
    else:
        # Multiple matches - show them
        match_ids = [m["id"][:12] for m in matches[:5]]
        suffix = "..." if len(matches) > 5 else ""
        raise ValueError(f"Ambiguous ID '{partial_id}' matches {len(matches)} entries: {', '.join(match_ids)}{suffix}")


def cmd_raw(args, k: Kernle):
    """Handle raw entry subcommands."""
    if args.raw_action == "capture" or args.raw_action is None:
        # Default action: capture a raw entry
        content = validate_input(args.content, "content", 5000)
        tags = [validate_input(t, "tag", 100) for t in (args.tags.split(",") if args.tags else [])]
        tags = [t.strip() for t in tags if t.strip()]

        raw_id = k.raw(content, tags=tags if tags else None, source="cli")
        print(f"✓ Raw entry captured: {raw_id[:8]}...")
        if tags:
            print(f"  Tags: {', '.join(tags)}")

    elif args.raw_action == "list":
        # Filter by processed state
        processed = None
        if args.unprocessed:
            processed = False
        elif args.processed:
            processed = True

        entries = k.list_raw(processed=processed, limit=args.limit)

        if not entries:
            print("No raw entries found.")
            return

        if args.json:
            print(json.dumps(entries, indent=2, default=str))
        else:
            unprocessed_count = sum(1 for e in entries if not e["processed"])
            print(f"Raw Entries ({len(entries)} total, {unprocessed_count} unprocessed)")
            print("=" * 50)
            for e in entries:
                status = "✓" if e["processed"] else "○"
                timestamp = e["timestamp"][:16] if e["timestamp"] else "unknown"
                content_preview = e["content"][:60].replace("\n", " ")
                if len(e["content"]) > 60:
                    content_preview += "..."
                print(f"\n{status} [{e['id'][:8]}] {timestamp}")
                print(f"  {content_preview}")
                if e["tags"]:
                    print(f"  Tags: {', '.join(e['tags'])}")
                if e["processed"] and e["processed_into"]:
                    print(f"  → {', '.join(e['processed_into'])}")

    elif args.raw_action == "show":
        try:
            full_id = resolve_raw_id(k, args.id)
        except ValueError as e:
            print(f"✗ {e}")
            return

        entry = k.get_raw(full_id)
        if not entry:
            print(f"Raw entry {args.id} not found.")
            return

        if args.json:
            print(json.dumps(entry, indent=2, default=str))
        else:
            status = "✓ Processed" if entry["processed"] else "○ Unprocessed"
            print(f"Raw Entry: {entry['id']}")
            print(f"Status: {status}")
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Source: {entry['source']}")
            if entry["tags"]:
                print(f"Tags: {', '.join(entry['tags'])}")
            print()
            print("Content:")
            print("-" * 40)
            print(entry["content"])
            print("-" * 40)
            if entry["processed_into"]:
                print(f"\nProcessed into: {', '.join(entry['processed_into'])}")

    elif args.raw_action == "process":
        # Support batch processing with comma-separated IDs
        raw_ids = [id.strip() for id in args.id.split(",") if id.strip()]
        
        success_count = 0
        for raw_id in raw_ids:
            try:
                full_id = resolve_raw_id(k, raw_id)
                memory_id = k.process_raw(
                    raw_id=full_id,
                    as_type=args.type,
                    objective=args.objective,
                    outcome=args.outcome,
                )
                print(f"✓ Processed {full_id[:8]}... → {args.type}:{memory_id[:8]}...")
                success_count += 1
            except ValueError as e:
                print(f"✗ {raw_id}: {e}")
        
        if len(raw_ids) > 1:
            print(f"\nProcessed {success_count}/{len(raw_ids)} entries")

    elif args.raw_action == "review":
        # Guided review of unprocessed entries
        entries = k.list_raw(processed=False, limit=args.limit)

        if not entries:
            print("✓ No unprocessed raw entries - memory is up to date!")
            return

        if args.json:
            print(json.dumps(entries, indent=2, default=str))
            return

        print("## Raw Entry Review")
        print(f"Found {len(entries)} unprocessed entries to review.\n")
        print("For each entry, consider:")
        print("  - **Episode**: Significant experience with a lesson learned")
        print("  - **Note**: Important observation, decision, or fact")
        print("  - **Belief**: Pattern or principle you've discovered")
        print("  - **Skip**: Keep as raw (not everything needs promotion)")
        print()
        print("=" * 60)

        for i, e in enumerate(entries, 1):
            timestamp = e["timestamp"][:16] if e["timestamp"] else "unknown"
            print(f"\n[{i}/{len(entries)}] {timestamp} - ID: {e['id'][:8]}")
            print("-" * 40)
            print(e["content"])
            print("-" * 40)
            if e["tags"]:
                print(f"Tags: {', '.join(e['tags'])}")

            # Provide promotion suggestions based on content
            content_lower = e["content"].lower()
            suggestions = []
            if any(word in content_lower for word in ["learned", "lesson", "realized", "discovered"]):
                suggestions.append("episode (contains learning)")
            if any(word in content_lower for word in ["decided", "decision", "chose", "will"]):
                suggestions.append("note (contains decision)")
            if any(word in content_lower for word in ["always", "never", "should", "principle", "pattern"]):
                suggestions.append("belief (contains principle)")

            if suggestions:
                print(f"💡 Suggestions: {', '.join(suggestions)}")

            print(f"\nTo promote: kernle -a {k.agent_id} raw process {e['id'][:8]} --type <episode|note|belief>")

        print("\n" + "=" * 60)
        print(f"\nReviewed {len(entries)} entries. Promote the meaningful ones, skip the rest.")

    elif args.raw_action == "clean":
        # Clean up old unprocessed raw entries
        from datetime import datetime, timezone, timedelta
        
        age_days = getattr(args, 'age', 7) or 7
        dry_run = not getattr(args, 'confirm', False)
        
        entries = k.list_raw(processed=False, limit=500)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=age_days)
        
        stale_entries = []
        for entry in entries:
            try:
                ts = entry.get("timestamp", "")
                if ts:
                    entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if entry_time < cutoff:
                        stale_entries.append(entry)
            except (ValueError, TypeError):
                continue
        
        if not stale_entries:
            print(f"✓ No unprocessed raw entries older than {age_days} days.")
            return
        
        print(f"Found {len(stale_entries)} unprocessed entries older than {age_days} days:\n")
        
        for entry in stale_entries[:10]:  # Show max 10
            timestamp = entry["timestamp"][:10] if entry["timestamp"] else "unknown"
            content_preview = entry["content"][:50].replace("\n", " ")
            if len(entry["content"]) > 50:
                content_preview += "..."
            print(f"  [{entry['id'][:8]}] {timestamp}: {content_preview}")
        
        if len(stale_entries) > 10:
            print(f"  ... and {len(stale_entries) - 10} more")
        
        if dry_run:
            print(f"\n⚠ DRY RUN: Would delete {len(stale_entries)} entries.")
            print(f"  To actually delete, run: kernle raw clean --age {age_days} --confirm")
        else:
            deleted = 0
            for entry in stale_entries:
                try:
                    k._storage.delete_raw(entry["id"])
                    deleted += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {entry['id'][:8]}: {e}")
            print(f"\n✓ Deleted {deleted} stale raw entries.")

    elif args.raw_action == "files":
        # Show flat file locations
        raw_dir = k._storage.get_raw_dir()
        files = k._storage.get_raw_files()
        
        print(f"Raw Flat Files Directory: {raw_dir}")
        print("=" * 50)
        
        if not files:
            print("\nNo raw files yet. Capture something with: kernle raw \"thought\"")
        else:
            print(f"\nFiles ({len(files)} total):")
            total_size = 0
            for f in files[:10]:
                size = f.stat().st_size
                total_size += size
                print(f"  {f.name:20} {size:>6} bytes")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
            print(f"\nTotal: {total_size:,} bytes")
        
        print(f"\n💡 Tips:")
        print(f"  • Edit directly: vim {raw_dir}/<date>.md")
        print(f"  • Search: grep -r 'pattern' {raw_dir}/")
        print(f"  • Git track: cd {raw_dir.parent} && git init")
        
        if getattr(args, 'open', False):
            import subprocess
            subprocess.run(["open", str(raw_dir)], check=False)


def cmd_belief(args, k: Kernle):
    """Handle belief revision subcommands."""
    if args.belief_action == "revise":
        episode_id = validate_input(args.episode_id, "episode_id", 100)
        result = k.revise_beliefs_from_episode(episode_id)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if result.get("error"):
                print(f"✗ {result['error']}")
                return

            print(f"Belief Revision from Episode {episode_id[:8]}...")
            print("=" * 50)

            # Reinforced beliefs
            reinforced = result.get("reinforced", [])
            if reinforced:
                print(f"\n✓ Reinforced ({len(reinforced)} beliefs):")
                for r in reinforced:
                    print(f"  • {r['statement'][:60]}...")
                    print(f"    ID: {r['belief_id'][:8]}...")

            # Contradicted beliefs
            contradicted = result.get("contradicted", [])
            if contradicted:
                print(f"\n⚠️  Potential Contradictions ({len(contradicted)}):")
                for c in contradicted:
                    print(f"  • {c['statement'][:60]}...")
                    print(f"    ID: {c['belief_id'][:8]}...")
                    print(f"    Evidence: {c['evidence'][:50]}...")

            # Suggested new beliefs
            suggested = result.get("suggested_new", [])
            if suggested:
                print(f"\n💡 Suggested New Beliefs ({len(suggested)}):")
                for s in suggested:
                    print(f"  • {s['statement'][:60]}...")
                    print(f"    Suggested confidence: {s['suggested_confidence']:.0%}")

            if not reinforced and not contradicted and not suggested:
                print("\nNo belief revisions found for this episode.")

    elif args.belief_action == "contradictions":
        statement = validate_input(args.statement, "statement", 2000)
        results = k.find_contradictions(statement, limit=args.limit)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            if not results:
                print(f"No contradictions found for: \"{statement[:50]}...\"")
                return

            print(f"Potential Contradictions for: \"{statement[:50]}...\"")
            print("=" * 60)

            for i, r in enumerate(results, 1):
                conf_bar = "█" * int(r["contradiction_confidence"] * 10) + "░" * (10 - int(r["contradiction_confidence"] * 10))
                status = "active" if r["is_active"] else "superseded"

                print(f"\n{i}. [{conf_bar}] {r['contradiction_confidence']:.0%} confidence")
                print(f"   Type: {r['contradiction_type']}")
                print(f"   Statement: {r['statement'][:60]}...")
                print(f"   Belief ID: {r['belief_id'][:8]}... ({status}, reinforced {r['times_reinforced']}x)")
                print(f"   Reason: {r['explanation']}")

    elif args.belief_action == "history":
        belief_id = validate_input(args.id, "belief_id", 100)
        history = k.get_belief_history(belief_id)

        if args.json:
            print(json.dumps(history, indent=2, default=str))
        else:
            if not history:
                print(f"No history found for belief {belief_id[:8]}...")
                return

            print("Belief Revision History")
            print("=" * 60)

            for i, entry in enumerate(history):
                is_current = ">>> " if entry["is_current"] else "    "
                status = "🟢 active" if entry["is_active"] else "⚫ superseded"
                conf_bar = "█" * int(entry["confidence"] * 5) + "░" * (5 - int(entry["confidence"] * 5))

                print(f"\n{is_current}[{i+1}] {entry['id'][:8]}... ({status})")
                print(f"     Statement: {entry['statement'][:55]}...")
                print(f"     Confidence: [{conf_bar}] {entry['confidence']:.0%} | Reinforced: {entry['times_reinforced']}x")
                print(f"     Created: {entry['created_at'][:10] if entry['created_at'] else 'unknown'}")

                if entry.get("supersession_reason"):
                    print(f"     Reason: {entry['supersession_reason'][:50]}...")

                if entry["superseded_by"]:
                    print(f"     → Superseded by: {entry['superseded_by'][:8]}...")

    elif args.belief_action == "reinforce":
        belief_id = validate_input(args.id, "belief_id", 100)

        if k.reinforce_belief(belief_id):
            print(f"✓ Belief {belief_id[:8]}... reinforced")
        else:
            print(f"✗ Belief {belief_id[:8]}... not found")

    elif args.belief_action == "supersede":
        old_id = validate_input(args.old_id, "old_id", 100)
        new_statement = validate_input(args.new_statement, "new_statement", 2000)

        try:
            new_id = k.supersede_belief(
                old_id=old_id,
                new_statement=new_statement,
                confidence=args.confidence,
                reason=args.reason,
            )
            print("✓ Belief superseded")
            print(f"  Old: {old_id[:8]}... (now inactive)")
            print(f"  New: {new_id[:8]}... (active)")
            print(f"  Statement: {new_statement[:60]}...")
            print(f"  Confidence: {args.confidence:.0%}")
        except ValueError as e:
            print(f"✗ {e}")

    elif args.belief_action == "list":
        # Get beliefs from storage directly for more detail
        beliefs = k._storage.get_beliefs(limit=args.limit, include_inactive=args.all)

        if args.json:
            data = [
                {
                    "id": b.id,
                    "statement": b.statement,
                    "confidence": b.confidence,
                    "times_reinforced": b.times_reinforced,
                    "is_active": b.is_active,
                    "supersedes": b.supersedes,
                    "superseded_by": b.superseded_by,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                }
                for b in beliefs
            ]
            print(json.dumps(data, indent=2, default=str))
        else:
            active_count = sum(1 for b in beliefs if b.is_active)
            print(f"Beliefs ({len(beliefs)} total, {active_count} active)")
            print("=" * 60)

            for b in beliefs:
                status = "🟢" if b.is_active else "⚫"
                conf_bar = "█" * int(b.confidence * 5) + "░" * (5 - int(b.confidence * 5))
                reinf = f"(+{b.times_reinforced})" if b.times_reinforced > 0 else ""

                print(f"\n{status} [{conf_bar}] {b.confidence:.0%} {reinf}")
                print(f"   {b.statement[:60]}{'...' if len(b.statement) > 60 else ''}")
                print(f"   ID: {b.id[:8]}...")

                if b.supersedes:
                    print(f"   Supersedes: {b.supersedes[:8]}...")
                if b.superseded_by:
                    print(f"   Superseded by: {b.superseded_by[:8]}...")


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
                auth_token = creds.get("auth_token")
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
                if hasattr(last_sync, 'tzinfo') and last_sync.tzinfo is None:
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
        TABLE_NAME_MAP = {
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

            op_type = "update" if change.operation in ("upsert", "insert", "update") else change.operation

            # Map table name for backend
            backend_table = TABLE_NAME_MAP.get(change.table_name, change.table_name)

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
                for field in ["id", "agent_id", "content", "objective", "outcome_type",
                              "outcome_description", "lessons_learned", "tags", "statement",
                              "confidence", "drive_type", "intensity", "name", "priority",
                              "title", "status", "progress", "entity_name", "entity_type",
                              "relationship_type", "notes", "sentiment", "focus_areas",
                              "created_at", "updated_at", "local_updated_at",
                              # raw_entries fields
                              "timestamp", "source", "processed",
                              # playbooks fields
                              "description", "steps", "triggers",
                              # goals fields
                              "target_date"]:
                    if hasattr(record, field):
                        value = getattr(record, field)
                        if hasattr(value, 'isoformat'):
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
                            print(f"   - {c.get('record_id', 'unknown')}: {c.get('error', 'unknown error')}")
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
                                        (table, record_id)
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
                                        (table, record_id)
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
                    print(json.dumps({
                        "pulled": applied,
                        "conflicts": conflicts,
                        "has_more": has_more,
                        "local_project": local_project,
                        "namespaced_id": get_namespaced_agent_id(),
                    }, indent=2))
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
        TABLE_NAME_MAP = {
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
                op_type = "update" if change.operation in ("upsert", "insert", "update") else change.operation

                # Map table name for backend
                backend_table = TABLE_NAME_MAP.get(change.table_name, change.table_name)

                op_data = {
                    "operation": op_type,
                    "table": backend_table,
                    "record_id": change.record_id,
                    "local_updated_at": format_datetime(change.queued_at),
                    "version": 1,
                }

                if record and op_type != "delete":
                    record_dict = {}
                    for field in ["id", "agent_id", "content", "objective", "outcome_type",
                                  "outcome_description", "lessons_learned", "tags", "statement",
                                  "confidence", "drive_type", "intensity", "name", "priority",
                                  "title", "status", "progress", "entity_name", "entity_type",
                                  "relationship_type", "notes", "sentiment", "focus_areas",
                                  "created_at", "updated_at", "local_updated_at",
                                  # raw_entries fields
                                  "timestamp", "source", "processed",
                                  # playbooks fields
                                  "description", "steps", "triggers",
                                  # goals fields
                                  "target_date"]:
                        if hasattr(record, field):
                            value = getattr(record, field)
                            if hasattr(value, 'isoformat'):
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
        return url if url else default
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)


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
                credentials = {
                    "user_id": user_id,
                    "api_key": api_key,
                    "backend_url": backend_url,
                    "token": token,
                    "token_expires": token_expires,
                }
                save_credentials(credentials)

                if args.json:
                    print(json.dumps({
                        "status": "success",
                        "user_id": user_id,
                        "backend_url": backend_url,
                    }, indent=2))
                else:
                    print("✓ Registration successful!")
                    print()
                    print(f"  User ID:     {user_id}")
                    print(f"  API Key:     {api_key[:20]}..." if len(api_key) > 20 else f"  API Key:     {api_key}")
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

        # Prompt for API key if not provided
        if not api_key:
            print("API Key: ", end="", flush=True)
            try:
                api_key = input().strip()
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
                credentials = {
                    "user_id": user_id,
                    "api_key": api_key,
                    "backend_url": backend_url,
                    "token": token,
                    "token_expires": token_expires,
                }
                save_credentials(credentials)

                if args.json:
                    print(json.dumps({
                        "status": "success",
                        "user_id": user_id,
                        "token_expires": token_expires,
                    }, indent=2))
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
                print(json.dumps({"authenticated": False, "reason": "No credentials found"}, indent=2))
            else:
                print("Not authenticated")
                print()
                print("Run `kernle auth register` to create an account")
                print("Run `kernle auth login` to log in with an existing API key")
            return

        user_id = credentials.get("user_id")
        api_key = credentials.get("api_key")
        backend_url = credentials.get("backend_url")
        token = credentials.get("token")
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
            print(json.dumps({
                "authenticated": True,
                "user_id": user_id,
                "backend_url": backend_url,
                "has_api_key": bool(api_key),
                "has_token": bool(token),
                "token_valid": token_valid,
                "token_expires": token_expires,
            }, indent=2))
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
                print(json.dumps({"status": "success", "message": "No credentials to clear"}, indent=2))
            else:
                print("Already logged out (no credentials found)")

    elif args.auth_action == "keys":
        cmd_auth_keys(args)


def cmd_auth_keys(args):
    """Handle API key management subcommands."""
    from datetime import datetime, timezone

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
                        created = key_info.get("created_at", "")[:10] if key_info.get("created_at") else "unknown"
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
                    print(json.dumps({"status": "success", "key_id": key_id, "action": "revoked"}, indent=2))
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
    p_load.add_argument("--sync", "-s", action="store_true",
                        help="Force sync (pull) before loading")
    p_load.add_argument("--no-sync", dest="no_sync", action="store_true",
                        help="Skip sync even if auto-sync is enabled")

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
    cp_save.add_argument("--sync", "-s", action="store_true",
                         help="Force sync (push) after saving")
    cp_save.add_argument("--no-sync", dest="no_sync", action="store_true",
                         help="Skip sync even if auto-sync is enabled")

    cp_load = cp_sub.add_parser("load", help="Load checkpoint")
    cp_load.add_argument("--json", "-j", action="store_true")

    cp_sub.add_parser("clear", help="Clear checkpoint")

    # episode
    p_episode = subparsers.add_parser("episode", help="Record an episode")
    p_episode.add_argument("objective", help="What was the objective?")
    p_episode.add_argument("outcome", help="What was the outcome?")
    p_episode.add_argument("--lesson", "-l", action="append", help="Lesson learned")
    p_episode.add_argument("--tag", "-t", action="append", help="Tag")
    p_episode.add_argument("--relates-to", "-r", action="append", help="Related memory ID (repeatable)")
    p_episode.add_argument("--valence", "-v", type=float, help="Emotional valence (-1.0 to 1.0)")
    p_episode.add_argument("--arousal", "-a", type=float, help="Emotional arousal (0.0 to 1.0)")
    p_episode.add_argument("--emotion", "-e", action="append", help="Emotion tag (e.g., joy, frustration)")
    p_episode.add_argument("--auto-emotion", action="store_true", default=True, help="Auto-detect emotions (default)")
    p_episode.add_argument("--no-auto-emotion", dest="auto_emotion", action="store_false", help="Disable emotion auto-detection")

    # note
    p_note = subparsers.add_parser("note", help="Capture a note")
    p_note.add_argument("content", help="Note content")
    p_note.add_argument("--type", choices=["note", "decision", "insight", "quote"], default="note")
    p_note.add_argument("--speaker", "-s", help="Speaker (for quotes)")
    p_note.add_argument("--reason", "-r", help="Reason (for decisions)")
    p_note.add_argument("--tag", action="append", help="Tag")
    p_note.add_argument("--relates-to", action="append", help="Related memory ID (repeatable)")
    p_note.add_argument("--protect", "-p", action="store_true", help="Protect from forgetting")

    # extract (conversation capture)
    p_extract = subparsers.add_parser("extract", help="Extract conversation context")
    p_extract.add_argument("summary", help="Summary of what's happening")
    p_extract.add_argument("--topic", "-t", help="Conversation topic")
    p_extract.add_argument("--participant", "-p", action="append", dest="participants", help="Participant (repeatable)")
    p_extract.add_argument("--outcome", "-o", help="Outcome or result")
    p_extract.add_argument("--decision", "-d", help="Decision made")

    # search
    p_search = subparsers.add_parser("search", help="Search memory")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-l", type=int, default=10)

    # status
    subparsers.add_parser("status", help="Show memory status")

    # init
    p_init = subparsers.add_parser("init", help="Initialize Kernle for your environment")
    p_init.add_argument("--non-interactive", "-y", action="store_true",
                        help="Non-interactive mode (use defaults)")
    p_init.add_argument("--env", choices=["claude-code", "clawdbot", "cline", "cursor", "desktop"],
                        help="Target environment")
    p_init.add_argument("--seed-values", action="store_true", default=True,
                        help="Seed initial values (default: true)")
    p_init.add_argument("--no-seed-values", dest="seed_values", action="store_false",
                        help="Skip seeding initial values")

    # relation (social graph / relationships)
    p_relation = subparsers.add_parser("relation", help="Manage relationships")
    relation_sub = p_relation.add_subparsers(dest="relation_action", required=True)

    relation_sub.add_parser("list", help="List all relationships")

    relation_add = relation_sub.add_parser("add", help="Add a relationship")
    relation_add.add_argument("name", help="Entity name (person, agent, org)")
    relation_add.add_argument("--type", "-t", choices=["person", "agent", "organization", "system"],
                              default="person", help="Entity type")
    relation_add.add_argument("--trust", type=float, help="Trust level 0.0-1.0")
    relation_add.add_argument("--notes", "-n", help="Notes about this relationship")

    relation_update = relation_sub.add_parser("update", help="Update a relationship")
    relation_update.add_argument("name", help="Entity name")
    relation_update.add_argument("--trust", type=float, help="New trust level 0.0-1.0")
    relation_update.add_argument("--notes", "-n", help="Updated notes")
    relation_update.add_argument("--type", "-t", choices=["person", "agent", "organization", "system"],
                                 help="Entity type")

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
    drive_set.add_argument("type", choices=["existence", "growth", "curiosity", "connection", "reproduction"])
    drive_set.add_argument("intensity", type=float, help="Intensity 0.0-1.0")
    drive_set.add_argument("--focus", "-f", action="append", help="Focus area")

    drive_satisfy = drive_sub.add_parser("satisfy", help="Satisfy a drive")
    drive_satisfy.add_argument("type", help="Drive type")
    drive_satisfy.add_argument("--amount", "-a", type=float, default=0.2)

    # consolidate
    p_consolidate = subparsers.add_parser("consolidate", help="Output guided reflection prompt")
    p_consolidate.add_argument("--min-episodes", "-m", type=int, default=3,
                              help="(Legacy) Minimum episodes for old consolidation")
    p_consolidate.add_argument("--limit", "-n", type=int, default=20,
                              help="Number of recent episodes to include (default: 20)")

    # temporal
    p_temporal = subparsers.add_parser("when", help="Query by time")
    p_temporal.add_argument("when", nargs="?", default="today",
                           choices=["today", "yesterday", "this week", "last hour"])

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
    emotion_tag.add_argument("--valence", "-v", type=float, default=0.0, help="Valence (-1.0 to 1.0)")
    emotion_tag.add_argument("--arousal", "-a", type=float, default=0.0, help="Arousal (0.0 to 1.0)")
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
    meta_conf.add_argument("type", choices=["episode", "belief", "value", "goal", "note"],
                          help="Memory type")
    meta_conf.add_argument("id", help="Memory ID")

    meta_verify = meta_sub.add_parser("verify", help="Verify a memory (increases confidence)")
    meta_verify.add_argument("type", choices=["episode", "belief", "value", "goal", "note"],
                            help="Memory type")
    meta_verify.add_argument("id", help="Memory ID")
    meta_verify.add_argument("--evidence", "-e", help="Supporting evidence")

    meta_lineage = meta_sub.add_parser("lineage", help="Get provenance chain for a memory")
    meta_lineage.add_argument("type", choices=["episode", "belief", "value", "goal", "note"],
                             help="Memory type")
    meta_lineage.add_argument("id", help="Memory ID")
    meta_lineage.add_argument("--json", "-j", action="store_true")

    meta_uncertain = meta_sub.add_parser("uncertain", help="List low-confidence memories")
    meta_uncertain.add_argument("--threshold", "-t", type=float, default=0.5,
                               help="Confidence threshold (default: 0.5)")
    meta_uncertain.add_argument("--limit", "-l", type=int, default=20)
    meta_uncertain.add_argument("--json", "-j", action="store_true")

    meta_propagate = meta_sub.add_parser("propagate", help="Propagate confidence to derived memories")
    meta_propagate.add_argument("type", choices=["episode", "belief", "value", "goal", "note"],
                               help="Source memory type")
    meta_propagate.add_argument("id", help="Source memory ID")

    meta_source = meta_sub.add_parser("source", help="Set source/provenance for a memory")
    meta_source.add_argument("type", choices=["episode", "belief", "value", "goal", "note"],
                            help="Memory type")
    meta_source.add_argument("id", help="Memory ID")
    meta_source.add_argument("--source", "-s", required=True,
                            choices=["direct_experience", "inference", "told_by_agent", "consolidation"],
                            help="Source type")
    meta_source.add_argument("--episodes", action="append", help="Supporting episode IDs")
    meta_source.add_argument("--derived", action="append", help="Derived from (type:id format)")

    # Meta-cognition subcommands (awareness of what I know/don't know)
    meta_knowledge = meta_sub.add_parser("knowledge", help="Show knowledge map across domains")
    meta_knowledge.add_argument("--json", "-j", action="store_true")

    meta_gaps = meta_sub.add_parser("gaps", help="Detect knowledge gaps for a query")
    meta_gaps.add_argument("query", help="Query to check knowledge for")
    meta_gaps.add_argument("--json", "-j", action="store_true")

    meta_boundaries = meta_sub.add_parser("boundaries", help="Show competence boundaries (strengths/weaknesses)")
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

    belief_contradictions = belief_sub.add_parser("contradictions", help="Find contradicting beliefs")
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
    belief_supersede.add_argument("--confidence", "-c", type=float, default=0.8,
                                  help="Confidence in new belief (default: 0.8)")
    belief_supersede.add_argument("--reason", "-r", help="Reason for supersession")

    belief_list = belief_sub.add_parser("list", help="List beliefs")
    belief_list.add_argument("--all", "-a", action="store_true", help="Include inactive beliefs")
    belief_list.add_argument("--limit", "-l", type=int, default=20)
    belief_list.add_argument("--json", "-j", action="store_true")

    # mcp
    subparsers.add_parser("mcp", help="Start MCP server (stdio transport)")

    # raw (raw memory entries)
    p_raw = subparsers.add_parser("raw", help="Raw memory capture and management")
    raw_sub = p_raw.add_subparsers(dest="raw_action")

    # kernle raw capture "content" - explicit capture subcommand
    raw_capture = raw_sub.add_parser("capture", help="Capture a raw entry")
    raw_capture.add_argument("content", help="Content to capture")
    raw_capture.add_argument("--tags", "-t", help="Comma-separated tags")

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
    raw_process.add_argument("--type", "-t", required=True, choices=["episode", "note", "belief"],
                            help="Target memory type")
    raw_process.add_argument("--objective", help="Episode objective (for episodes)")
    raw_process.add_argument("--outcome", help="Episode outcome (for episodes)")

    # kernle raw review - guided review of unprocessed entries
    raw_review = raw_sub.add_parser("review", help="Review unprocessed entries with promotion guidance")
    raw_review.add_argument("--limit", "-l", type=int, default=10, help="Number of entries to review")
    raw_review.add_argument("--json", "-j", action="store_true")

    # kernle raw clean - clean up old unprocessed entries
    raw_clean = raw_sub.add_parser("clean", help="Delete old unprocessed raw entries")
    raw_clean.add_argument("--age", "-a", type=int, default=7, help="Delete entries older than N days (default: 7)")
    raw_clean.add_argument("--confirm", "-y", action="store_true", help="Actually delete (otherwise dry run)")

    # kernle raw files - show flat file locations
    raw_files = raw_sub.add_parser("files", help="Show raw flat file locations")
    raw_files.add_argument("--open", "-o", action="store_true", help="Open directory in file manager")

    # dump
    p_dump = subparsers.add_parser("dump", help="Dump all memory to stdout")
    p_dump.add_argument("--format", "-f", choices=["markdown", "json"], default="markdown",
                       help="Output format (default: markdown)")
    p_dump.add_argument("--include-raw", "-r", action="store_true", default=True,
                       help="Include raw entries (default: true)")
    p_dump.add_argument("--no-raw", dest="include_raw", action="store_false",
                       help="Exclude raw entries")

    # export
    p_export = subparsers.add_parser("export", help="Export memory to file")
    p_export.add_argument("path", help="Output file path")
    p_export.add_argument("--format", "-f", choices=["markdown", "json"],
                         help="Output format (auto-detected from extension if not specified)")
    p_export.add_argument("--include-raw", "-r", action="store_true", default=True,
                         help="Include raw entries (default: true)")
    p_export.add_argument("--no-raw", dest="include_raw", action="store_false",
                         help="Exclude raw entries")

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
    playbook_record.add_argument("--success", action="store_true", default=True,
                                 help="Record successful usage (default)")
    playbook_record.add_argument("--failure", action="store_true",
                                 help="Record failed usage")

    # anxiety
    p_anxiety = subparsers.add_parser("anxiety", help="Memory anxiety tracking")
    p_anxiety.add_argument("--detailed", "-d", action="store_true",
                          help="Show detailed breakdown")
    p_anxiety.add_argument("--actions", "-a", action="store_true",
                          help="Show recommended actions")
    p_anxiety.add_argument("--auto", action="store_true",
                          help="Execute recommended actions automatically")
    p_anxiety.add_argument("--context", "-c", type=int,
                          help="Current context token usage")
    p_anxiety.add_argument("--limit", "-l", type=int, default=200000,
                          help="Context window limit (default: 200000)")
    p_anxiety.add_argument("--emergency", "-e", action="store_true",
                          help="Run emergency save immediately")
    p_anxiety.add_argument("--summary", "-s",
                          help="Summary for emergency save checkpoint")
    p_anxiety.add_argument("--json", "-j", action="store_true",
                          help="Output as JSON")

    # forget (controlled forgetting)
    p_forget = subparsers.add_parser("forget", help="Controlled forgetting operations")
    forget_sub = p_forget.add_subparsers(dest="forget_action", required=True)

    # kernle forget candidates [--threshold N] [--limit N]
    forget_candidates = forget_sub.add_parser("candidates", help="Show forgetting candidates")
    forget_candidates.add_argument("--threshold", "-t", type=float, default=0.3,
                                   help="Salience threshold (default: 0.3)")
    forget_candidates.add_argument("--limit", "-l", type=int, default=20,
                                   help="Maximum candidates to show")
    forget_candidates.add_argument("--json", "-j", action="store_true",
                                   help="Output as JSON")

    # kernle forget run [--dry-run] [--threshold N] [--limit N]
    forget_run = forget_sub.add_parser("run", help="Run forgetting cycle")
    forget_run.add_argument("--dry-run", "-n", action="store_true",
                           help="Preview what would be forgotten (don't actually forget)")
    forget_run.add_argument("--threshold", "-t", type=float, default=0.3,
                           help="Salience threshold (default: 0.3)")
    forget_run.add_argument("--limit", "-l", type=int, default=10,
                           help="Maximum memories to forget")
    forget_run.add_argument("--json", "-j", action="store_true",
                           help="Output as JSON")

    # kernle forget protect <type> <id>
    forget_protect = forget_sub.add_parser("protect", help="Protect memory from forgetting")
    forget_protect.add_argument("type", choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
                               help="Memory type")
    forget_protect.add_argument("id", help="Memory ID")
    forget_protect.add_argument("--unprotect", "-u", action="store_true",
                               help="Remove protection instead")

    # kernle forget recover <type> <id>
    forget_recover = forget_sub.add_parser("recover", help="Recover a forgotten memory")
    forget_recover.add_argument("type", choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
                               help="Memory type")
    forget_recover.add_argument("id", help="Memory ID")

    # kernle forget list [--limit N]
    forget_list = forget_sub.add_parser("list", help="List forgotten memories")
    forget_list.add_argument("--limit", "-l", type=int, default=50,
                            help="Maximum entries to show")
    forget_list.add_argument("--json", "-j", action="store_true",
                            help="Output as JSON")

    # kernle forget salience <type> <id>
    forget_salience = forget_sub.add_parser("salience", help="Calculate salience for a memory")
    forget_salience.add_argument("type", choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
                                help="Memory type")
    forget_salience.add_argument("id", help="Memory ID")

    # sync (local-to-cloud synchronization)
    p_sync = subparsers.add_parser("sync", help="Sync with remote backend")
    sync_sub = p_sync.add_subparsers(dest="sync_action", required=True)

    # kernle sync status
    sync_status = sync_sub.add_parser("status", help="Show sync status (pending ops, last sync, connection)")
    sync_status.add_argument("--json", "-j", action="store_true",
                            help="Output as JSON")

    # kernle sync push [--limit N]
    sync_push = sync_sub.add_parser("push", help="Push pending local changes to remote backend")
    sync_push.add_argument("--limit", "-l", type=int, default=100,
                          help="Maximum operations to push (default: 100)")
    sync_push.add_argument("--json", "-j", action="store_true",
                          help="Output as JSON")

    # kernle sync pull [--full]
    sync_pull = sync_sub.add_parser("pull", help="Pull remote changes to local")
    sync_pull.add_argument("--full", "-f", action="store_true",
                          help="Pull all records (not just changes since last sync)")
    sync_pull.add_argument("--json", "-j", action="store_true",
                          help="Output as JSON")

    # kernle sync full
    sync_full = sync_sub.add_parser("full", help="Full bidirectional sync (pull then push)")
    sync_full.add_argument("--json", "-j", action="store_true",
                          help="Output as JSON")

    # auth (authentication and credentials management)
    p_auth = subparsers.add_parser("auth", help="Authentication and credentials management")
    auth_sub = p_auth.add_subparsers(dest="auth_action", required=True)

    # kernle auth register [--email EMAIL] [--backend-url URL]
    auth_register = auth_sub.add_parser("register", help="Register a new account")
    auth_register.add_argument("--email", "-e", help="Email address")
    auth_register.add_argument("--backend-url", "-b", help="Backend URL (e.g., https://api.kernle.io)")
    auth_register.add_argument("--json", "-j", action="store_true",
                               help="Output as JSON")

    # kernle auth login [--api-key KEY] [--backend-url URL]
    auth_login = auth_sub.add_parser("login", help="Log in with existing credentials")
    auth_login.add_argument("--api-key", "-k", help="API key")
    auth_login.add_argument("--backend-url", "-b", help="Backend URL")
    auth_login.add_argument("--json", "-j", action="store_true",
                            help="Output as JSON")

    # kernle auth status
    auth_status = auth_sub.add_parser("status", help="Show current auth status")
    auth_status.add_argument("--json", "-j", action="store_true",
                             help="Output as JSON")

    # kernle auth logout
    auth_logout = auth_sub.add_parser("logout", help="Clear stored credentials")
    auth_logout.add_argument("--json", "-j", action="store_true",
                             help="Output as JSON")

    # kernle auth keys (API key management)
    auth_keys = auth_sub.add_parser("keys", help="Manage API keys")
    keys_sub = auth_keys.add_subparsers(dest="keys_action", required=True)

    # kernle auth keys list
    keys_list = keys_sub.add_parser("list", help="List your API keys (masked)")
    keys_list.add_argument("--json", "-j", action="store_true",
                           help="Output as JSON")

    # kernle auth keys create [--name NAME]
    keys_create = keys_sub.add_parser("create", help="Create a new API key")
    keys_create.add_argument("--name", "-n", help="Name for the key (for identification)")
    keys_create.add_argument("--json", "-j", action="store_true",
                             help="Output as JSON")

    # kernle auth keys revoke KEY_ID
    keys_revoke = keys_sub.add_parser("revoke", help="Revoke/delete an API key")
    keys_revoke.add_argument("key_id", help="ID of the key to revoke")
    keys_revoke.add_argument("--force", "-f", action="store_true",
                             help="Skip confirmation prompt")
    keys_revoke.add_argument("--json", "-j", action="store_true",
                             help="Output as JSON")

    # kernle auth keys cycle KEY_ID
    keys_cycle = keys_sub.add_parser("cycle", help="Cycle a key (new key, old deactivated)")
    keys_cycle.add_argument("key_id", help="ID of the key to cycle")
    keys_cycle.add_argument("--force", "-f", action="store_true",
                            help="Skip confirmation prompt")
    keys_cycle.add_argument("--json", "-j", action="store_true",
                            help="Output as JSON")

    # Pre-process arguments: handle `kernle raw "content"` by inserting "capture"
    # This is needed because argparse subparsers consume positional args before parent parser
    raw_subcommands = {"list", "show", "process", "capture", "review", "clean", "files"}
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
            if i + 1 < len(argv) and argv[i + 1] not in raw_subcommands and not argv[i + 1].startswith("-"):
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
        elif args.command == "init":
            cmd_init(args, k)
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
        elif args.command == "forget":
            cmd_forget(args, k)
        elif args.command == "playbook":
            cmd_playbook(args, k)
        elif args.command == "raw":
            cmd_raw(args, k)
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
    except (ValueError, TypeError) as e:
        logger.error(f"Input validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
