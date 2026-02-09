"""Memory commands for Kernle CLI — load, checkpoint, episode, note, extract, search."""

import json
import logging
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


def cmd_load(args, k: "Kernle"):
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


def cmd_checkpoint(args, k: "Kernle"):
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
                except Exception as e:
                    logger.debug(f"Failed to parse checkpoint timestamp: {e}")

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


def cmd_episode(args, k: "Kernle"):
    """Record an episode."""
    objective = validate_input(args.objective, "objective", 1000)
    outcome = validate_input(args.outcome, "outcome", 1000)
    lessons = [validate_input(lesson, "lesson", 500) for lesson in (args.lesson or [])]
    tags = [validate_input(t, "tag", 100) for t in (args.tag or [])]
    derived_from = getattr(args, "derived_from", None)
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
            derived_from=derived_from,
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
            derived_from=derived_from,
            source=source,
            context=context,
            context_tags=context_tags,
        )

    print(f"✓ Episode saved: {episode_id[:8]}...")
    if args.lesson:
        print(f"  Lessons: {len(args.lesson)}")
    if derived_from:
        print(f"  Derived from: {len(derived_from)} memories")
    if valence is not None or arousal is not None:
        v = valence or 0.0
        a = arousal or 0.0
        print(f"  Emotion: valence={v:+.2f}, arousal={a:.2f}")
    elif auto_emotion and not has_emotion_args:
        print("  (emotions auto-detected)")


def cmd_note(args, k: "Kernle"):
    """Capture a note."""
    content = validate_input(args.content, "content", 2000)
    speaker = validate_input(args.speaker, "speaker", 200) if args.speaker else None
    reason = validate_input(args.reason, "reason", 1000) if args.reason else None
    tags = [validate_input(t, "tag", 100) for t in (args.tag or [])]
    derived_from = getattr(args, "derived_from", None)
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
        derived_from=derived_from,
        source=source,
        context=context,
        context_tags=context_tags,
    )
    print(f"✓ Note saved: {args.content[:50]}...")
    if args.tag:
        print(f"  Tags: {', '.join(args.tag)}")
    if derived_from:
        print(f"  Derived from: {len(derived_from)} memories")
    if source:
        print(f"  Source: {source}")
    if context:
        print(f"  Context: {context}")


def cmd_extract(args, k: "Kernle"):
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

    # Fold tags into blob text (tags parameter was removed from save_raw)
    blob = f"{content}\n\n[Tags: {', '.join(tags)}]"
    raw_id = k.raw(blob=blob, source="conversation")
    print(f"✓ Extracted: {summary[:50]}...")
    print(
        f"  ID: {raw_id[:8]} (promote later with: kernle raw process {raw_id[:8]} --type <episode|note>)"
    )


def cmd_search(args, k: "Kernle"):
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
