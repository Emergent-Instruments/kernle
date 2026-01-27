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
        )
    else:
        episode_id = k.episode(
            objective=objective,
            outcome=outcome,
            lessons=lessons,
            tags=tags,
        )
    
    print(f"✓ Episode saved: {episode_id[:8]}...")
    if args.lesson:
        print(f"  Lessons: {len(args.lesson)}")
    if valence is not None or arousal is not None:
        v = valence or 0.0
        a = arousal or 0.0
        print(f"  Emotion: valence={v:+.2f}, arousal={a:.2f}")
    elif auto_emotion and not has_emotion_args:
        print(f"  (emotions auto-detected)")


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
                print(f"\nSupporting Episodes:")
                for ep_id in lineage["source_episodes"]:
                    print(f"  • {ep_id}")
            
            if lineage.get("derived_from"):
                print(f"\nDerived From:")
                for ref in lineage["derived_from"]:
                    print(f"  • {ref}")
            
            if lineage.get("confidence_history"):
                print(f"\nConfidence History:")
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
